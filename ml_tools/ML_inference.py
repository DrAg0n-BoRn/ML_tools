import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Dict, Any, Optional
from abc import ABC, abstractmethod

from .ML_scaler import PytorchScaler
from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath
from .keys import PyTorchInferenceKeys

__all__ = [
    "PyTorchInferenceHandler",
    "PyTorchInferenceHandlerMulti",
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
                 scaler: Optional[Union[PytorchScaler, str, Path]] = None):
        """
        Initializes the handler.

        Args:
            model (nn.Module): An instantiated PyTorch model.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (PytorchScaler | str | Path | None): An optional scaler or path to a saved scaler state.
        """
        self.model = model
        self.device = self._validate_device(device)

        # Load the scaler if a path is provided
        if scaler is not None:
            if isinstance(scaler, (str, Path)):
                self.scaler = PytorchScaler.load(scaler)
            else:
                self.scaler = scaler
        else:
            self.scaler = None

        model_p = make_fullpath(state_dict, enforce="file")

        try:
            # Load the state dictionary and apply it to the model structure
            self.model.load_state_dict(torch.load(model_p, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            _LOGGER.info(f"Model state loaded from '{model_p.name}' and set to evaluation mode.")
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
            # Your M-series Mac will appreciate this check!
            _LOGGER.warning("Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
            device_lower = "cpu"
        return torch.device(device_lower)

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

    @abstractmethod
    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Core batch prediction method. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Core single-sample prediction method. Must be implemented by subclasses."""
        pass


class PyTorchInferenceHandler(_BaseInferenceHandler):
    """
    Handles loading a PyTorch model's state dictionary and performing inference
    for single-target regression or classification tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Literal["classification", "regression"],
                 device: str = 'cpu',
                 target_id: Optional[str] = None,
                 scaler: Optional[Union[PytorchScaler, str, Path]] = None):
        """
        Initializes the handler for single-target tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model architecture.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str): The type of task, 'regression' or 'classification'.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            target_id (str | None): An optional identifier for the target.
            scaler (PytorchScaler | str | Path | None): A PytorchScaler instance or the file path to a saved PytorchScaler state.
        """
        # Call the parent constructor to handle model loading, device, and scaler
        super().__init__(model, state_dict, device, scaler)

        if task not in ["classification", "regression"]:
            raise ValueError("`task` must be 'classification' or 'regression'.")
        self.task = task
        self.target_id = target_id

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core batch prediction method for single-target models.

        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of input features.

        Returns:
            A dictionary containing the raw output tensors from the model.
        """
        if features.ndim != 2:
            _LOGGER.error("Input for batch prediction must be a 2D array or tensor.")
            raise ValueError()

        input_tensor = self._preprocess_input(features)

        with torch.no_grad():
            output = self.model(input_tensor)

            if self.task == "classification":
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            else:  # regression
                # For single-target regression, ensure output is flattened
                return {PyTorchInferenceKeys.PREDICTIONS: output.flatten()}

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core single-sample prediction method for single-target models.

        Args:
            features (np.ndarray | torch.Tensor): A 1D array/tensor of input features.

        Returns:
            A dictionary containing the raw output tensors for a single sample.
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
        Convenience wrapper for predict_batch that returns NumPy arrays.
        """
        tensor_results = self.predict_batch(features)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays or scalars.
        """
        tensor_results = self.predict(features)

        if self.task == "regression":
            # .item() implicitly moves to CPU and returns a Python scalar
            return {PyTorchInferenceKeys.PREDICTIONS: tensor_results[PyTorchInferenceKeys.PREDICTIONS].item()}
        else: # classification
            return {
                PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].item(),
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
    
    def quick_predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper to get the mapping {target_name: prediction} or {target_name: label}
        
        `target_id` must be implemented.
        """
        if self.target_id is None:
            _LOGGER.error(f"'target_id' has not been implemented.")
            raise AttributeError()
        
        if self.task == "regression":
            result = self.predict_numpy(features)[PyTorchInferenceKeys.PREDICTIONS]
        else:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.LABELS]
        
        return {self.target_id: result}


class PyTorchInferenceHandlerMulti(_BaseInferenceHandler):
    """
    Handles loading a PyTorch model's state dictionary and performing inference
    for multi-target regression or multi-label classification tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Literal["multi_target_regression", "multi_label_classification"],
                 device: str = 'cpu',
                 target_ids: Optional[list[str]] = None,
                 scaler: Optional[Union[PytorchScaler, str, Path]] = None):
        """
        Initializes the handler for multi-target tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str): The type of task, 'multi_target_regression' or 'multi_label_classification'.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            target_ids (list[str] | None): An optional identifier for the targets.
            scaler (PytorchScaler | str | Path | None): A PytorchScaler instance or the file path to a saved PytorchScaler state.
        """
        super().__init__(model, state_dict, device, scaler)

        if task not in ["multi_target_regression", "multi_label_classification"]:
            _LOGGER.error("`task` must be 'multi_target_regression' or 'multi_label_classification'.")
            raise ValueError()
        self.task = task
        self.target_ids = target_ids

    def predict_batch(self,
                      features: Union[np.ndarray, torch.Tensor],
                      classification_threshold: float = 0.5
                      ) -> Dict[str, torch.Tensor]:
        """
        Core batch prediction method for multi-target models.

        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of input features.
            classification_threshold (float): The threshold to convert probabilities
                into binary predictions for multi-label classification.

        Returns:
            A dictionary containing the raw output tensors from the model.
        """
        if features.ndim != 2:
            _LOGGER.error("Input for batch prediction must be a 2D array or tensor.")
            raise ValueError()

        input_tensor = self._preprocess_input(features)

        with torch.no_grad():
            output = self.model(input_tensor)

            if self.task == "multi_label_classification":
                probs = torch.sigmoid(output)
                # Get binary predictions based on the threshold
                labels = (probs >= classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            else:  # multi_target_regression
                # The output is already in the correct [batch_size, n_targets] shape
                return {PyTorchInferenceKeys.PREDICTIONS: output}

    def predict(self,
                features: Union[np.ndarray, torch.Tensor],
                classification_threshold: float = 0.5
                ) -> Dict[str, torch.Tensor]:
        """
        Core single-sample prediction method for multi-target models.

        Args:
            features (np.ndarray | torch.Tensor): A 1D array/tensor of input features.
            classification_threshold (float): The threshold for multi-label tasks.

        Returns:
            A dictionary containing the raw output tensors for a single sample.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[0] != 1:
            _LOGGER.error("The 'predict()' method is for a single sample. 'Use predict_batch()' for multiple samples.")
            raise ValueError()

        batch_results = self.predict_batch(features, classification_threshold)

        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results

    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self,
                            features: Union[np.ndarray, torch.Tensor],
                            classification_threshold: float = 0.5
                            ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        """
        tensor_results = self.predict_batch(features, classification_threshold)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results

    def predict_numpy(self,
                      features: Union[np.ndarray, torch.Tensor],
                      classification_threshold: float = 0.5
                      ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper for predict that returns NumPy arrays for a single sample.
        Note: For multi-target models, the output is always an array.
        """
        tensor_results = self.predict(features, classification_threshold)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results
    
    def quick_predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper to get the mapping {target_name: prediction} or {target_name: label}
        
        `target_ids` must be implemented.
        """
        if self.target_ids is None:
            _LOGGER.error(f"'target_id' has not been implemented.")
            raise AttributeError()
        
        if self.task == "multi_target_regression":
            result = self.predict_numpy(features)[PyTorchInferenceKeys.PREDICTIONS].flatten().tolist()
        else:
            result = self.predict_numpy(features)[PyTorchInferenceKeys.LABELS].flatten().tolist()
        
        return {key: value for key, value in zip(self.target_ids, result)}


def multi_inference_regression(handlers: list[PyTorchInferenceHandler], 
                               feature_vector: Union[np.ndarray, torch.Tensor], 
                               output: Literal["numpy","torch"]="numpy") -> dict[str,Any]:
    """
    Performs regression inference using multiple models on a single feature vector.

    This function iterates through a list of PyTorchInferenceHandler objects,
    each configured for a different regression target. It runs a prediction for
    each handler using the same input feature vector and returns the results
    in a dictionary.
    
    The function adapts its behavior based on the input dimensions:
    - 1D input: Returns a dictionary mapping target ID to a single value.
    - 2D input: Returns a dictionary mapping target ID to a list of values.

    Args:
        handlers (list[PyTorchInferenceHandler]): A list of initialized inference
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
        if handler.target_id is None:
            _LOGGER.error("All inference handlers must have a 'target_id' attribute.")
            raise AttributeError()
        if handler.task != "regression":
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_id}' is for '{handler.task}', but only 'regression' tasks are supported.")
            raise ValueError()
            
        # inference
        if output == "numpy":
            # This path returns NumPy arrays or standard Python scalars
            numpy_result = handler.predict_batch_numpy(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, convert the 1-element array to a Python scalar
                results[handler.target_id] = numpy_result.item()
            else:
                # For a batch, return the full NumPy array of predictions
                results[handler.target_id] = numpy_result

        else:  # output == "torch"
            # This path returns PyTorch tensors on the model's device
            torch_result = handler.predict_batch(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, return the 0-dim tensor
                results[handler.target_id] = torch_result[0]
            else:
                # For a batch, return the full tensor of predictions
                results[handler.target_id] = torch_result

    return results


def multi_inference_classification(
    handlers: list[PyTorchInferenceHandler], 
    feature_vector: Union[np.ndarray, torch.Tensor], 
    output: Literal["numpy","torch"]="numpy"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Performs classification inference on a single sample or a batch.

    This function iterates through a list of PyTorchInferenceHandler objects,
    each configured for a different classification target. It returns two
    dictionaries: one for the predicted labels and one for the probabilities.

    The function adapts its behavior based on the input dimensions:
    - 1D input: The dictionaries map target ID to a single label and a single probability array.
    - 2D input: The dictionaries map target ID to an array of labels and an array of probability arrays.

    Args:
        handlers (list[PyTorchInferenceHandler]): A list of initialized inference handlers. Each must have a unique `target_id` and be configured
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
        if handler.target_id is None:
            _LOGGER.error("All inference handlers must have a 'target_id' attribute.")
            raise AttributeError()
        if handler.task != "classification":
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_id}' is for '{handler.task}', but this function only supports 'classification'.")
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
                labels_results[handler.target_id] = labels.item()
            else: # torch
                labels_results[handler.target_id] = labels[0]
            
            # The probabilities are an array/tensor of values
            probs_results[handler.target_id] = probabilities[0]
        else:
            labels_results[handler.target_id] = labels
            probs_results[handler.target_id] = probabilities
            
    return labels_results, probs_results


def info():
    _script_info(__all__)
