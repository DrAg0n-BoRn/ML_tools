import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Dict, Any

from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath
from .keys import PyTorchInferenceKeys

__all__ = [
    "PyTorchInferenceHandler"
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
                 device: str = 'cpu'):
        """
        Initializes the handler by loading a model's state_dict.

        Args:
            model (nn.Module): An instantiated PyTorch model with the correct architecture.
            state_dict (str | Path): The path to the saved .pth model state_dict file.
            task (str): The type of task, 'regression' or 'classification'.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
        """
        self.model = model
        self.task = task
        self.device = self._validate_device(device)

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
                # ✅ Move tensor to CPU before converting to NumPy
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
     

def info():
    _script_info(__all__)
