import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Optional, Any
from abc import ABC, abstractmethod

from ..ML_finalize_handler._ML_finalize_handler import FinalizedFileHandler, ClassificationMetadata
from ..ML_scaler import DragonScaler
from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import PyTorchCheckpointKeys, MagicWords, ScalerKeys

_LOGGER = get_logger("Inference Handler")


__all__ = [
    "_CoreInferenceHandler",
    "_ClassificationMixin",
    "_ScalerMixin"
]


class _ClassificationMixin:
    """
    Mixin class to provide classification metadata and threshold management 
    for inference handlers.
    """
    def __init__(self):
        self._classification_threshold: float = 0.5
        self._loaded_threshold: bool = False
        self._loaded_class_map: bool = False
        self._class_map: Optional[dict[str, int]] = None
        self._idx_to_class: Optional[dict[int, str]] = None

    def set_class_map(self, class_map: dict[str, int], force_overwrite: bool = False) -> None:
        if self._loaded_class_map:
            warning_message = f"A '{PyTorchCheckpointKeys.CLASS_MAP}' was loaded from the model configuration file."
            if not force_overwrite:
                warning_message += " Use 'force_overwrite=True' if you are sure you want to modify it. This will not affect the value from the file."
                _LOGGER.warning(warning_message)
                return
            else:
                warning_message += " Overwriting it for this inference instance."
                _LOGGER.warning(warning_message)
        
        self._class_map = class_map
        self._idx_to_class = {v: k for k, v in class_map.items()}
        self._loaded_class_map = True
        _LOGGER.info("Class map set for label-to-name translation.")

    def set_classification_threshold(self, threshold: float, force_overwrite: bool = False) -> None:
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

    def _load_classification_metadata(self, meta: ClassificationMetadata) -> None:
        """Helper to safely unpack standard classification metadata from a configuration object."""
        if meta.classification_threshold is not None:
            self.set_classification_threshold(meta.classification_threshold, force_overwrite=True)
            self._loaded_threshold = True
            
        if meta.class_map is not None:
            self.set_class_map(meta.class_map, force_overwrite=True)


class _ScalerMixin:
    """
    Mixin class to centralize the loading and validation of DragonScalers
    for feature and target transformations.
    """
    def __init__(self):
        self.feature_scaler: Optional[DragonScaler] = None
        self.target_scaler: Optional[DragonScaler] = None
        
    def _load_scalers(self, scaler_path: Optional[Union[str, Path]]) -> None:
        """Helper to load single or unified scalers directly from a file path."""
        if scaler_path is None:
            return
            
        if isinstance(scaler_path, (str, Path)):
            path_obj = make_fullpath(scaler_path, enforce="file")
            loaded_scaler_data = torch.load(path_obj)
            
            if isinstance(loaded_scaler_data, dict) and (ScalerKeys.FEATURE_SCALER in loaded_scaler_data or ScalerKeys.TARGET_SCALER in loaded_scaler_data):
                if ScalerKeys.FEATURE_SCALER in loaded_scaler_data:
                    self.feature_scaler = DragonScaler.load(loaded_scaler_data[ScalerKeys.FEATURE_SCALER], verbose=False)
                    _LOGGER.info("Loaded DragonScaler state for feature scaling.")
                if ScalerKeys.TARGET_SCALER in loaded_scaler_data:
                    self.target_scaler = DragonScaler.load(loaded_scaler_data[ScalerKeys.TARGET_SCALER], verbose=False)
                    _LOGGER.info("Loaded DragonScaler state for target scaling.")
            else:
                _LOGGER.warning("Loaded scaler file does not contain separate feature/target scalers. Assuming it is a feature scaler (legacy format).")
                self.feature_scaler = DragonScaler.load(loaded_scaler_data)
        else:
            _LOGGER.error("Scaler must be a file path (str or Path) to a saved DragonScaler state file.")
            raise ValueError()


class _CoreInferenceHandler(ABC):
    """
    Universal abstract base class for PyTorch inference handlers.
    
    Handles strictly domain-agnostic setup: hardware validation, loading model weights,
    resolving the task, and configuring the evaluation state.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 task: Optional[str] = None):
        self.model = model
        self.device = self._validate_device(device)
        
        # --- 1. Load File Handler ---
        self._file_handler = FinalizedFileHandler(state_dict)
        self._file_handler._verbose = False
        
        # --- 2. Task Resolution ---
        file_task = self._file_handler.task
        
        if task is None:
            if file_task == MagicWords.UNKNOWN:
                _LOGGER.error(f"Task not specified in arguments and not found in file '{make_fullpath(state_dict).name}'.")
                raise ValueError()
            self.task = file_task
            _LOGGER.info(f"Task '{self.task}' detected from file.")
        else:
            if file_task != MagicWords.UNKNOWN and file_task != task:
                _LOGGER.warning(f"Provided task '{task}' differs from file metadata task '{file_task}'. Using provided task '{task}'.")
            self.task = task

        # --- 3. Load Model Weights ---
        try:
            self.model.load_state_dict(self._file_handler.model_state_dict)
        except RuntimeError as e:
            _LOGGER.error(f"State dict mismatch: {e}")
            raise

        # --- 4. Move to Device ---
        self.model.to(self.device)
        self.model.eval()
        _LOGGER.info(f"Model loaded and moved to {self.device} in evaluation mode.")

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

    @abstractmethod
    def predict_batch(self, inputs: Union[np.ndarray, torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """Core batch prediction method. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, single_input: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
        """Core single-sample prediction method. Must be implemented by subclasses."""
        pass
