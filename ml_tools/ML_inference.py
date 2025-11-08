import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Dict, Any, Optional
from abc import ABC, abstractmethod

from .ML_scaler import DragonScaler
from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath
from ._keys import PyTorchInferenceKeys, PyTorchCheckpointKeys, MLTaskKeys


__all__ = [
    "DragonInferenceHandler",
    "multi_inference_regression",
    "multi_inference_classification"
]


class _BaseInferenceHandler(ABC):
    """
    Abstract base class for PyTorch inference handlers.

    Manages common tasks like loading a model's state dictionary, validating
    the target device, and preprocessing input features.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 scaler: Optional[Union[DragonScaler, str, Path]] = None):
        """
        Initializes the handler.

        Args:
            model (nn.Module): An instantiated PyTorch model.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (DragonScaler | str | Path | None): An optional scaler or path to a saved scaler state.
        """
        self.model = model
        self.device = self._validate_device(device)
        self._classification_threshold = 0.5
        self._loaded_threshold: bool = False
        self._loaded_class_map: bool = False
        self._class_map: Optional[dict[str,int]] = None
        self._idx_to_class: Optional[Dict[int, str]] = None
        self._loaded_data_dict: Dict[str, Any] = {} #Store whatever is in the finalized file

        # Load the scaler if a path is provided
        if scaler is not None:
            if isinstance(scaler, (str, Path)):
                self.scaler = DragonScaler.load(scaler)
            else:
                self.scaler = scaler
        else:
            self.scaler = None

        model_p = make_fullpath(state_dict, enforce="file")

        try:
            # Load whatever is in the file
            loaded_data = torch.load(model_p, map_location=self.device)
            
            # Check if it's dictionary or a old weights-only file
            if isinstance(loaded_data, dict):
                self._loaded_data_dict = loaded_data # Store the dict
                
                if PyTorchCheckpointKeys.MODEL_STATE in loaded_data:
                    # It's a new training checkpoint, extract the weights
                    self.model.load_state_dict(loaded_data[PyTorchCheckpointKeys.MODEL_STATE])
                    
                    # attempt to get a custom classification threshold
                    if PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD in loaded_data:
                        try:
                            self._classification_threshold = float(loaded_data[PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD])
                        except Exception as e_int:
                            _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD}' but an error occurred when retrieving it:\n{e_int}")
                            self._classification_threshold = 0.5
                        else:
                            _LOGGER.info(f"'{PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD}' found and set to {self._classification_threshold}")
                            self._loaded_threshold = True
                            
                    # attempt to get a class map
                    if PyTorchCheckpointKeys.CLASS_MAP in loaded_data:
                        try:
                            self._class_map = loaded_data[PyTorchCheckpointKeys.CLASS_MAP]
                        except Exception as e_int:
                            _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.CLASS_MAP}' but an error occurred when retrieving it:\n{e_int}")
                        else:
                            if isinstance(self._class_map, dict):
                                self._loaded_class_map = True
                                self.set_class_map(self._class_map)
                            else:
                                _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.CLASS_MAP}' but it is not a dict: '{type(self._class_map)}'.")
                                self._class_map = None        
                else:
                    # It's a state_dict, load it directly
                    self.model.load_state_dict(loaded_data)
            
            else:
                 # It's an old state_dict (just weights), load it directly
                self.model.load_state_dict(loaded_data)
            
            _LOGGER.info(f"Model state loaded from '{model_p.name}'.")
                
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
        except Exception as e:
            _LOGGER.error(f"Failed to load model state from '{model_p}': {e}")
            raise

    def _validate_device(self, device: str) -> torch.device:
        """Validates the selected device and returns a torch.device object."""
        device_lower = device.lower()
        if "cuda" in device_lower and not torch.cuda.is_available():
            _LOGGER.warning("CUDA not available, switching to CPU.")
            device_lower = "cpu"
        elif device_lower == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
            device_lower = "cpu"
        return torch.device(device_lower)
    
    def set_class_map(self, class_map: Dict[str, int], force_overwrite: bool = False):
        """
        Sets the class name mapping to translate predicted integer labels back into string names.
        
        If a class_map was previously loaded from a model configuration, this
        method will log a warning and refuse to update the value. This
        prevents accidentally overriding a setting from a loaded checkpoint.
        
        To bypass this safety check set `force_overwrite` to `True`.

        Args:
            class_map (Dict[str, int]): The class_to_idx dictionary (e.g., {'cat': 0, 'dog': 1}).
            force_overwrite (bool): If True, allows overwriting a map that was loaded from a configuration file.
        """
        if self._loaded_class_map:
            warning_message = f"A '{PyTorchCheckpointKeys.CLASS_MAP}' was loaded from the model configuration file."
            if not force_overwrite:
                warning_message += " Use 'force_overwrite=True' if you are sure you want to modify it. This will not affect the value from the file."
                _LOGGER.warning(warning_message)
                return
            else:
                warning_message += " Overwriting it for this inference instance."
                _LOGGER.warning(warning_message)
        
        # Store the map and invert it for fast lookup
        self._class_map = class_map
        self._idx_to_class = {v: k for k, v in class_map.items()}
        # Mark as 'loaded' by the user, even if it wasn't from a file, to prevent accidental changes later if _loaded_class_map was false.
        self._loaded_class_map = True # Protect this newly set map
        _LOGGER.info("Class map set for label-to-name translation.")

    @abstractmethod
    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Core batch prediction method. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Core single-sample prediction method. Must be implemented by subclasses."""
        pass


class DragonInferenceHandler(_BaseInferenceHandler):
    """
    Handles loading a PyTorch model's state dictionary and performing inference.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Literal["regression", "binary classification", "multiclass classification", "multitarget regression", "multilabel binary classification"],
                 device: str = 'cpu',
                 scaler: Optional[Union[DragonScaler, str, Path]] = None):
        """
        Initializes the handler for single-target tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model architecture.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str): The type of task.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (DragonScaler | str | Path | None): A DragonScaler instance or the file path to a saved DragonScaler state.
            
        Note: class_map (Dict[int, str]) will be loaded from the model file, to set or override it use `.set_class_map()`.
        """
        # Call the parent constructor to handle model loading, device, and scaler
        super().__init__(model, state_dict, device, scaler)

        if task not in [MLTaskKeys.REGRESSION,
                        MLTaskKeys.BINARY_CLASSIFICATION, 
                        MLTaskKeys.MULTICLASS_CLASSIFICATION,
                        MLTaskKeys.MULTITARGET_REGRESSION,
                        MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
            _LOGGER.error(f"'task' not recognized: '{task}'.")
            raise ValueError()
        self.task = task
        self.target_ids: Optional[list[str]] = None
        self._target_ids_set: bool = False
        
        # attempt to get target name or target names
        if PyTorchCheckpointKeys.TARGET_NAME in self._loaded_data_dict:
            try:
                target_from_dict = [self._loaded_data_dict[PyTorchCheckpointKeys.TARGET_NAME]]
            except Exception as e_int:
                _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.TARGET_NAME}' but an error occurred when retrieving it:\n{e_int}")
                self.target_ids = None
            else:
                self.set_target_ids([target_from_dict]) # type: ignore
        elif PyTorchCheckpointKeys.TARGET_NAMES in self._loaded_data_dict:
            try:
                targets_from_dict = self._loaded_data_dict[PyTorchCheckpointKeys.TARGET_NAMES]
            except Exception as e_int:
                _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.TARGET_NAMES}' but an error occurred when retrieving it:\n{e_int}")
                self.target_ids = None
            else:
                self.set_target_ids(targets_from_dict) # type: ignore

    def _preprocess_input(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts input to a torch.Tensor, applies scaling if a scaler is
        present, and moves it to the correct device.
        """
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        if self.scaler:
            features_tensor = self.scaler.transform(features_tensor)

        return features_tensor.to(self.device)
    
    def set_target_ids(self, target_names: list[str], force_overwrite: bool=False):
        """
        Assigns the provided list of strings as the target variable names.
        If target IDs have already been set, this method will log a warning.

        Args:
            target_names (list[str]): A list of target names.
            force_overwrite (bool): If True, allows the method to overwrite previously set target IDs.
        """
        if self._target_ids_set:
            warning_message = "Target IDs was previously set."
            if not force_overwrite:
                warning_message += " Use `force_overwrite=True` to overwrite."
                _LOGGER.warning(warning_message)
                return
            else:
                warning_message += " Overwriting..."
                _LOGGER.warning(warning_message)

        self.target_ids = target_names
        self._target_ids_set = True
        _LOGGER.info("Target IDs set.")

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core batch prediction method.

        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of input features.

        Returns:
            Dict: A dictionary containing the raw output tensors from the model.
        """
        if features.ndim != 2:
            _LOGGER.error("Input for batch prediction must be a 2D array or tensor.")
            raise ValueError()

        input_tensor = self._preprocess_input(features)

        with torch.no_grad():
            output = self.model(input_tensor)

            if self.task == MLTaskKeys.MULTICLASS_CLASSIFICATION:
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
                
            elif self.task == MLTaskKeys.BINARY_CLASSIFICATION:
                # Assumes model output is [N, 1] (a single logit)
                # Squeeze output from [N, 1] to [N] if necessary
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
                    
                probs = torch.sigmoid(output) # Probability of positive class
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
                
            elif self.task == MLTaskKeys.REGRESSION:
                # For single-target regression, ensure output is flattened
                return {PyTorchInferenceKeys.PREDICTIONS: output.flatten()}
            
            elif self.task == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
                probs = torch.sigmoid(output)
                # Get binary predictions based on the threshold
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            
            elif self.task == MLTaskKeys.MULTITARGET_REGRESSION:
                # The output is already in the correct [batch_size, n_targets] shape
                return {PyTorchInferenceKeys.PREDICTIONS: output}
            
            else:
                # should never happen
                _LOGGER.error(f"Unrecognized task '{self.task}'.")
                raise ValueError()

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core single-sample prediction method for single-target models.

        Args:
            features (np.ndarray | torch.Tensor): A 1D array/tensor of input features.

        Returns:
            Dict: A dictionary containing the raw output tensors for a single sample.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1) # Reshape to a batch of one

        if features.shape[0] != 1:
            _LOGGER.error("The 'predict()' method is for a single sample. Use 'predict_batch()' for multiple samples.")
            raise ValueError()

        batch_results = self.predict_batch(features)

        # Extract the first (and only) result from the batch output
        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results
    
    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays
        and adds string labels for classification tasks if a class_map is set.
        """
        tensor_results = self.predict_batch(features)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        
        # Add string names for classification if map exists
        is_classification = self.task in [
            MLTaskKeys.BINARY_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_CLASSIFICATION
        ]
        
        if is_classification and self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
            int_labels = numpy_results[PyTorchInferenceKeys.LABELS] # This is a (B,) array
            numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [ # type: ignore
                self._idx_to_class.get(label_id, "Unknown")
                for label_id in int_labels
            ]
        
        return numpy_results

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays or scalars
        and adds string labels for classification tasks if a class_map is set.
        """
        tensor_results = self.predict(features)

        if self.task == MLTaskKeys.REGRESSION:
            # .item() implicitly moves to CPU and returns a Python scalar
            return {PyTorchInferenceKeys.PREDICTIONS: tensor_results[PyTorchInferenceKeys.PREDICTIONS].item()}
        
        elif self.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            int_label = tensor_results[PyTorchInferenceKeys.LABELS].item()
            label_name = "Unknown"
            if self._idx_to_class:
                label_name = self._idx_to_class.get(int_label, "Unknown") # type: ignore

            return {
                PyTorchInferenceKeys.LABELS: int_label,
                PyTorchInferenceKeys.LABEL_NAMES: label_name,
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
            
        elif self.task in [MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION, MLTaskKeys.MULTITARGET_REGRESSION]:
            # For multi-target models, the output is always an array.
            numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
            return numpy_results
        else:
            # should never happen
            _LOGGER.error(f"Unrecognized task '{self.task}'.")
            raise ValueError()
    
    def quick_predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper to get the mapping {target_name: prediction} or {target_name: label}
        
        `target_ids` must be implemented.
        """
        if self.target_ids is None:
            _LOGGER.error(f"'target_ids' has not been implemented.")
            raise AttributeError()
        
        if self.task == MLTaskKeys.REGRESSION:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.PREDICTIONS]
            return {self.target_ids[0]: result}
        
        elif self.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.LABELS]
            return {self.target_ids[0]: result}
        
        elif self.task == MLTaskKeys.MULTITARGET_REGRESSION:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.PREDICTIONS].flatten().tolist()
            return {key: value for key, value in zip(self.target_ids, result)}
        
        elif self.task == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.LABELS].flatten().tolist()
            return {key: value for key, value in zip(self.target_ids, result)}
        
        else:
            # should never happen
            _LOGGER.error(f"Unrecognized task '{self.task}'.")
            raise ValueError()
        
    def set_classification_threshold(self, threshold: float, force_overwrite: bool=False):
        """
        Sets the classification threshold for the current inference instance.
        
        If a threshold was previously loaded from a model configuration, this
        method will log a warning and refuse to update the value. This
        prevents accidentally overriding a setting from a loaded checkpoint.
        
        To bypass this safety check set `force_overwrite` to `True`.

        Args:
            threshold (float): The new classification threshold value to set.
            force_overwrite (bool): If True, allows overwriting a threshold that was loaded from a configuration file. 
        """
        if self._loaded_threshold:
            warning_message = f"The current '{PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD}={self._classification_threshold}' was loaded and set from a model configuration file."
            if not force_overwrite:
                warning_message += " Use 'force_overwrite' if you are sure you want to modify it. This will not affect the value from the file."
                _LOGGER.warning(warning_message)
                return
            else:
                warning_message += f" Overwriting it to {threshold}."
                _LOGGER.warning(warning_message)
 
        self._classification_threshold = threshold


def multi_inference_regression(handlers: list[DragonInferenceHandler], 
                               feature_vector: Union[np.ndarray, torch.Tensor], 
                               output: Literal["numpy","torch"]="numpy") -> dict[str,Any]:
    """
    Performs regression inference using multiple models on a single feature vector.

    This function iterates through a list of DragonInferenceHandler objects,
    each configured for a different regression target. It runs a prediction for
    each handler using the same input feature vector and returns the results
    in a dictionary.
    
    The function adapts its behavior based on the input dimensions:
    - 1D input: Returns a dictionary mapping target ID to a single value.
    - 2D input: Returns a dictionary mapping target ID to a list of values.

    Args:
        handlers (list[DragonInferenceHandler]): A list of initialized inference
            handlers. Each handler must have a unique `target_id` and be configured with `task="regression"`.
        feature_vector (Union[np.ndarray, torch.Tensor]): An input sample (1D) or a batch of samples (2D) to be fed into each regression model.
        output (Literal["numpy", "torch"], optional): The desired format for the output predictions.
            - "numpy": Returns predictions as Python scalars or NumPy arrays.
            - "torch": Returns predictions as PyTorch tensors.

    Returns:
        (dict[str, Any]): A dictionary mapping each handler's `target_id` to its
        predicted regression values. 

    Raises:
        AttributeError: If any handler in the list is missing a `target_id`.
        ValueError: If any handler's `task` is not 'regression' or if the input `feature_vector` is not 1D or 2D.
    """
    # check batch dimension
    is_single_sample = feature_vector.ndim == 1
    
    # Reshape a 1D vector to a 2D batch of one for uniform processing.
    if is_single_sample:
        feature_vector = feature_vector.reshape(1, -1)
    
    # Validate that the input is a 2D tensor.
    if feature_vector.ndim != 2:
        _LOGGER.error("Input feature_vector must be a 1D or 2D array/tensor.")
        raise ValueError()
    
    results: dict[str,Any] = dict()
    for handler in handlers:
        # validation
        if handler.target_ids is None:
            _LOGGER.error("All inference handlers must have a 'target_ids' attribute.")
            raise AttributeError()
        if handler.task != MLTaskKeys.REGRESSION:
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_ids[0]}' is for '{handler.task}', only single target regression tasks are supported.")
            raise ValueError()
            
        # inference
        if output == "numpy":
            # This path returns NumPy arrays or standard Python scalars
            numpy_result = handler.predict_batch_numpy(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, convert the 1-element array to a Python scalar
                results[handler.target_ids[0]] = numpy_result.item()
            else:
                # For a batch, return the full NumPy array of predictions
                results[handler.target_ids[0]] = numpy_result

        else:  # output == "torch"
            # This path returns PyTorch tensors on the model's device
            torch_result = handler.predict_batch(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, return the 0-dim tensor
                results[handler.target_ids[0]] = torch_result[0]
            else:
                # For a batch, return the full tensor of predictions
                results[handler.target_ids[0]] = torch_result

    return results


def multi_inference_classification(
    handlers: list[DragonInferenceHandler], 
    feature_vector: Union[np.ndarray, torch.Tensor], 
    output: Literal["numpy","torch"]="numpy"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Performs classification inference on a single sample or a batch.

    This function iterates through a list of DragonInferenceHandler objects,
    each configured for a different classification target. It returns two
    dictionaries: one for the predicted labels and one for the probabilities.

    The function adapts its behavior based on the input dimensions:
    - 1D input: The dictionaries map target ID to a single label and a single probability array.
    - 2D input: The dictionaries map target ID to an array of labels and an array of probability arrays.

    Args:
        handlers (list[DragonInferenceHandler]): A list of initialized inference handlers. Each must have a unique `target_id` and be configured
            with `task="classification"`.
        feature_vector (Union[np.ndarray, torch.Tensor]): An input sample (1D)
            or a batch of samples (2D) for prediction.
        output (Literal["numpy", "torch"], optional): The desired format for the
            output predictions.

    Returns:
        (tuple[dict[str, Any], dict[str, Any]]): A tuple containing two dictionaries:
        1.  A dictionary mapping `target_id` to the predicted label(s).
        2.  A dictionary mapping `target_id` to the prediction probabilities.

    Raises:
        AttributeError: If any handler in the list is missing a `target_id`.
        ValueError: If any handler's `task` is not 'classification' or if the input `feature_vector` is not 1D or 2D.
    """
    # Store if the original input was a single sample
    is_single_sample = feature_vector.ndim == 1
    
    # Reshape a 1D vector to a 2D batch of one for uniform processing
    if is_single_sample:
        feature_vector = feature_vector.reshape(1, -1)
    
    if feature_vector.ndim != 2:
        _LOGGER.error("Input feature_vector must be a 1D or 2D array/tensor.")
        raise ValueError()

    # Initialize two dictionaries for results
    labels_results: dict[str, Any] = dict()
    probs_results: dict[str, Any] = dict()

    for handler in handlers:
        # Validation
        if handler.target_ids is None:
            _LOGGER.error("All inference handlers must have a 'target_id' attribute.")
            raise AttributeError()
        if handler.task not in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_ids[0]}' is for '{handler.task}', but this function only supports binary and multiclass classification.")
            raise ValueError()
            
        # Inference
        if output == "numpy":
            # predict_batch_numpy returns a dict of NumPy arrays
            result = handler.predict_batch_numpy(feature_vector)
        else: # torch
            # predict_batch returns a dict of Torch tensors
            result = handler.predict_batch(feature_vector)
        
        labels = result[PyTorchInferenceKeys.LABELS]
        probabilities = result[PyTorchInferenceKeys.PROBABILITIES]
        
        if is_single_sample:
            # For "numpy", convert the single label to a Python int scalar.
            # For "torch", get the 0-dim tensor label.
            if output == "numpy":
                labels_results[handler.target_ids[0]] = labels.item()
            else: # torch
                labels_results[handler.target_ids[0]] = labels[0]
            
            # The probabilities are an array/tensor of values
            probs_results[handler.target_ids[0]] = probabilities[0]
        else:
            labels_results[handler.target_ids[0]] = labels
            probs_results[handler.target_ids[0]] = probabilities
            
    return labels_results, probs_results


def info():
    _script_info(__all__)
