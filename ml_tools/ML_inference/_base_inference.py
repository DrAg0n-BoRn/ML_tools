import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Optional
from abc import ABC, abstractmethod

from ..ML_finalize_handler import FinalizedFileHandler
from ..ML_scaler import DragonScaler

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import PyTorchCheckpointKeys, ScalerKeys, MagicWords


_LOGGER = get_logger("Inference Handler")


__all__ = [
    "_BaseInferenceHandler",
]


class _BaseInferenceHandler(ABC):
    """
    Abstract base class for PyTorch inference handlers.

    Manages common tasks like loading a model's state dictionary via FinalizedFileHandler,
    validating the target device, and preprocessing input features.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 scaler: Optional[Union[str, Path]] = None,
                 task: Optional[str] = None):
        """
        Initializes the handler.

        Args:
            model (nn.Module): An instantiated PyTorch model.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (str | Path | None): An optional scaler or path to a saved scaler state.
            task (str | None): The specific machine learning task. If None, it attempts to read it from the finalized-file.
        """
        self.model = model
        self.device = self._validate_device(device)
        self._classification_threshold = 0.5
        self._loaded_threshold: bool = False
        self._loaded_class_map: bool = False
        self._class_map: Optional[dict[str,int]] = None
        self._idx_to_class: Optional[dict[int, str]] = None
        
        # --- 1. Load File Handler ---
        # This loads the content on CPU and validates structure
        self._file_handler = FinalizedFileHandler(state_dict)
        
        # Silence warnings of the filehandler internally
        self._file_handler._verbose = False
        
        # --- 2. Task Resolution ---
        file_task = self._file_handler.task
        
        if task is None:
            # User didn't provide task, must be in file
            if file_task == MagicWords.UNKNOWN:
                _LOGGER.error(f"Task not specified in arguments and not found in file '{make_fullpath(state_dict).name}'.")
                raise ValueError()
            self.task = file_task
            _LOGGER.info(f"Task '{self.task}' detected from file.")
        else:
            # User provided task
            if file_task != MagicWords.UNKNOWN and file_task != task:
                _LOGGER.warning(f"Provided task '{task}' differs from file metadata task '{file_task}'. Using provided task '{task}'.")
            self.task = task

        # --- 3. Load Model Weights ---
        # Weights are already loaded in file_handler (on CPU)
        try:
            self.model.load_state_dict(self._file_handler.model_state_dict)
        except RuntimeError as e:
            _LOGGER.error(f"State dict mismatch: {e}")
            raise

        # --- 4. Load Metadata (Thresholds, Class Maps) ---
        if self._file_handler.classification_threshold is not None:
            self._classification_threshold = self._file_handler.classification_threshold
            self._loaded_threshold = True
            
        if self._file_handler.class_map is not None:
            self.set_class_map(self._file_handler.class_map)
            # set_class_map sets _loaded_class_map to True
        
        # --- 5. Move to Device ---
        self.model.to(self.device)
        self.model.eval()
        _LOGGER.info(f"Model loaded and moved to {self.device} in evaluation mode.")

        # --- 6. Load Scalers ---
        self.feature_scaler: Optional[DragonScaler] = None
        self.target_scaler: Optional[DragonScaler] = None

        if scaler is not None:
            if isinstance(scaler, (str, Path)):
                path_obj = make_fullpath(scaler, enforce="file")
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
    
    def set_class_map(self, class_map: dict[str, int], force_overwrite: bool = False):
        """
        Sets the class name mapping to translate predicted integer labels back into string names.
        
        Args:
            class_map (Dict[str, int]): The class_to_idx dictionary.
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
        
        self._class_map = class_map
        self._idx_to_class = {v: k for k, v in class_map.items()}
        self._loaded_class_map = True
        _LOGGER.info("InferenceHandler: Class map set for label-to-name translation.")

    @abstractmethod
    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Core batch prediction method. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Core single-sample prediction method. Must be implemented by subclasses."""
        pass

