import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Dict, Any, Optional

from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath
from .keys import PyTorchInferenceKeys

__all__ = [
    "PyTorchInferenceHandler",
    "multi_inference_regression",
    "multi_inference_classification"
]

class PyTorchInferenceHandler:
    """
    Handles loading a PyTorch model's state dictionary and performing inference
    for either regression or classification tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Literal["classification", "regression"],
                 device: str = 'cpu',
                 target_id: Optional[str]=None):
        """
        Initializes the handler by loading a model's state_dict.

        Args:
            model (nn.Module): An instantiated PyTorch model with the correct architecture.
            state_dict (str | Path): The path to the saved .pth model state_dict file.
            task (str): The type of task, 'regression' or 'classification'.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            target_id (str | None): Target name as used in the training set.
        """
        self.model = model
        self.task = task
        self.device = self._validate_device(device)
        self.target_id = target_id

        model_p = make_fullpath(state_dict, enforce="file")

        try:
            # Load the state dictionary and apply it to the model structure
            self.model.load_state_dict(torch.load(model_p, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            _LOGGER.info(f"✅ Model state loaded from '{model_p.name}' and set to evaluation mode.")
        except Exception as e:
            _LOGGER.error(f"❌ Failed to load model state from '{model_p}': {e}")
            raise

    def _validate_device(self, device: str) -> torch.device:
        """Validates the selected device and returns a torch.device object."""
        device_lower = device.lower()
        if "cuda" in device_lower and not torch.cuda.is_available():
            _LOGGER.warning("⚠️ CUDA not available, switching to CPU.")
            device_lower = "cpu"
        elif device_lower == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("⚠️ Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
            device_lower = "cpu"
        return torch.device(device_lower)

    def _preprocess_input(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Converts input to a torch.Tensor and moves it to the correct device."""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Ensure tensor is on the correct device
        return features.to(self.device)
    
    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core batch prediction method. Returns results as PyTorch tensors on the model's device.
        """
        if features.ndim != 2:
            raise ValueError("Input for batch prediction must be a 2D array or tensor.")

        input_tensor = self._preprocess_input(features)
        
        with torch.no_grad():
            # Output tensor remains on the model's device (e.g., 'mps' or 'cuda')
            output = self.model(input_tensor)

            if self.task == "classification":
                probs = nn.functional.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            else:  # regression
                return {PyTorchInferenceKeys.PREDICTIONS: output}

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Core single-sample prediction. Returns results as PyTorch tensors on the model's device.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[0] != 1:
            raise ValueError("The predict() method is for a single sample. Use predict_batch() for multiple samples.")

        batch_results = self.predict_batch(features)
        
        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results

    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        """
        tensor_results = self.predict_batch(features)
        # Move tensor to CPU before converting to NumPy
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays or scalars.
        """
        tensor_results = self.predict(features)
        
        if self.task == "regression":
            # .item() implicitly moves to CPU
            return {PyTorchInferenceKeys.PREDICTIONS: tensor_results[PyTorchInferenceKeys.PREDICTIONS].item()}
        else: # classification
            return {
                PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].item(),
                # Move tensor to CPU before converting to NumPy
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }


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
        raise ValueError("Input feature_vector must be a 1D or 2D array/tensor.")
    
    results: dict[str,Any] = dict()
    for handler in handlers:
        # validation
        if handler.target_id is None:
            raise AttributeError("All inference handlers must have a 'target_id' attribute.")
        if handler.task != "regression":
            raise ValueError(
                f"Invalid task type: The handler for target_id '{handler.target_id}' "
                f"is for '{handler.task}', but only 'regression' tasks are supported."
            )
            
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
        raise ValueError("Input feature_vector must be a 1D or 2D array/tensor.")

    # Initialize two dictionaries for results
    labels_results: dict[str, Any] = dict()
    probs_results: dict[str, Any] = dict()

    for handler in handlers:
        # Validation
        if handler.target_id is None:
            raise AttributeError("All inference handlers must have a 'target_id' attribute.")
        if handler.task != "classification":
            raise ValueError(
                f"Invalid task type: The handler for target_id '{handler.target_id}' "
                f"is for '{handler.task}', but this function only supports 'classification'."
            )
            
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
