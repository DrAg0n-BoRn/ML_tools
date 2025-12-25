import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Optional

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, PyTorchCheckpointKeys, MLTaskKeys

from ._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("DragonInference")


__all__ = [
    "DragonInferenceHandler",
]


class DragonInferenceHandler(_BaseInferenceHandler):
    """
    Handles loading a PyTorch model's state dictionary and performing inference for tabular data.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Optional[Literal["regression", 
                               "binary classification", 
                               "multiclass classification", 
                               "multitarget regression", 
                               "multilabel binary classification"]] = None,
                 device: str = 'cpu',
                 scaler: Optional[Union[str, Path]] = None):
        """
        Initializes the handler for single-target tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model architecture.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str, optional): The type of task. If None, it will be detected from file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (str | Path | None): A path to a saved DragonScaler state.
            
        Note: class_map (Dict[int, str]) will be loaded from the model file, to set or override it use `.set_class_map()`.
        """
        # Call the parent constructor to handle model loading, device, and scaler
        # The parent constructor resolves 'task'
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         scaler=scaler, 
                         task=task)
        
        # --- Validation of resolved task ---
        valid_tasks = [
            MLTaskKeys.REGRESSION,
            MLTaskKeys.BINARY_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_CLASSIFICATION,
            MLTaskKeys.MULTITARGET_REGRESSION,
            MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION
        ]
        
        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this inference handler only supports: {valid_tasks}.")
            raise ValueError()

        self.target_ids: Optional[list[str]] = None
        self._target_ids_set: bool = False
        
        # --- Attempt to load target names from FinalizedFileHandler ---
        if self._file_handler.target_names is not None:
            self.set_target_ids(self._file_handler.target_names)
        elif self._file_handler.target_name is not None:
            self.set_target_ids([self._file_handler.target_name])
        else:
            _LOGGER.warning("No target names found in file metadata.")

    def _preprocess_input(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts input to a torch.Tensor, applies FEATURE scaling if a scaler is
        present, and moves it to the correct device.
        """
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        if self.feature_scaler:
            features_tensor = self.feature_scaler.transform(features_tensor)

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

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
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

            # --- Target Scaling Logic (Inverse Transform) ---
            # Only for regression tasks and if a target scaler exists
            if self.target_scaler:
                if self.task not in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                    # raise error
                    _LOGGER.error("Target scaler is only applicable for regression tasks. A target scaler was provided for a non-regression task.")
                    raise ValueError()
                
                # Ensure output is 2D (N, Targets) for the scaler
                original_shape = output.shape
                if output.ndim == 1:
                    output = output.reshape(-1, 1)
                
                # Apply inverse transform (de-scale)
                output = self.target_scaler.inverse_transform(output)
                
                # Restore original shape if necessary (though usually we want 2D or 1D flat)
                if len(original_shape) == 1:
                    output = output.flatten()

            # --- Task Specific Formatting ---
            if self.task == MLTaskKeys.MULTICLASS_CLASSIFICATION:
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
                
            elif self.task == MLTaskKeys.BINARY_CLASSIFICATION:
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
                    
                probs = torch.sigmoid(output) 
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
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            
            elif self.task == MLTaskKeys.MULTITARGET_REGRESSION:
                return {PyTorchInferenceKeys.PREDICTIONS: output}
            
            else:
                _LOGGER.error(f"Unrecognized task '{self.task}'.")
                raise ValueError()

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, np.ndarray]:
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

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
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
    
    def quick_predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
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

