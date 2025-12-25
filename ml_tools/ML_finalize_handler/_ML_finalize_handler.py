import torch
import numpy as np

from typing import Union, Any, Optional
from pathlib import Path

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import PyTorchCheckpointKeys, MagicWords


_LOGGER = get_logger("Finalized-File")


__all__ = [
    "FinalizedFileHandler"
]


class FinalizedFileHandler:
    """
    Handles the loading and validation of a finalized-file with PyTorch model artifacts.

    It provides a robust fallback mechanism: if the loaded file does not match 
    the specific finalized-file schema, it is treated as a raw state dictionary (standard PyTorch save format).

    Attributes:
        task (str): The specific machine learning task.
        model_state_dict (dict): The dictionary containing model weights.
        epoch (Optional[int]): The epoch number at which the model was finalized.
        target_name (Optional[str]): The name of the target variable (for single-target tasks).
        target_names (Optional[list[str]]): List of target names (for multi-target/multi-label tasks).
        classification_threshold (Optional[float]): The threshold used for binary classification.
        class_map (Optional[dict[str, int]]): Mapping of class names to integer labels.
        sequence_length (Optional[int]): Length of input sequences (for sequence tasks).
        initial_sequence (Optional[np.ndarray]): The starting sequence data (for sequence tasks).
    """
    def __init__(self, finalized_file_path: Union[str, Path]) -> None:
        """
        Initializes the handler by loading the file and validating its structure.

        Args:
            finalized_file_path (Union[str, Path]): The path to the PyTorch finalized-file.
        """
        self._task: str = MagicWords.UNKNOWN
        self._epoch: Optional[int] = None
        self._classification_threshold: Optional[float] = None
        self._class_map: Optional[dict[str, int]] = None
        self._sequence_length: Optional[int] = None
        self._initial_sequence: Optional[np.ndarray] = None
        self._target_name: Optional[str] = None
        self._target_names: Optional[list[str]] = None
        self._model_state_dict: Optional[dict[str, Any]] = None
        
        # Set warning outputs
        self._verbose: bool=True
        
        # validate path
        pth_path = make_fullpath(finalized_file_path, enforce="file")
        
        # load file
        try:
            pth_file_content = torch.load(pth_path, map_location='cpu')
        except Exception as e:
            _LOGGER.error(f"Failed to load finalized-file from '{pth_path}': {e}")
            raise
        
        # validation
        if not isinstance(pth_file_content, dict):
            _LOGGER.error(f"The loaded content from '{pth_path.name}' is of type '{type(pth_file_content).__name__}', but a dictionary was expected.")
            raise TypeError()
        
        # check for valid finalized-file
        if (PyTorchCheckpointKeys.MODEL_STATE in pth_file_content and
            PyTorchCheckpointKeys.EPOCH in pth_file_content and
            PyTorchCheckpointKeys.TASK in pth_file_content):
            # store state dict
            self._model_state_dict = pth_file_content.get(PyTorchCheckpointKeys.MODEL_STATE)
            
            # populate attributes
            self._epoch = pth_file_content.get(PyTorchCheckpointKeys.EPOCH)
            self._task = pth_file_content.get(PyTorchCheckpointKeys.TASK, MagicWords.UNKNOWN)
            self._target_name = pth_file_content.get(PyTorchCheckpointKeys.TARGET_NAME)
            self._target_names = pth_file_content.get(PyTorchCheckpointKeys.TARGET_NAMES)
            self._classification_threshold = pth_file_content.get(PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD)
            self._class_map = pth_file_content.get(PyTorchCheckpointKeys.CLASS_MAP)
            self._sequence_length = pth_file_content.get(PyTorchCheckpointKeys.SEQUENCE_LENGTH)
            self._initial_sequence = pth_file_content.get(PyTorchCheckpointKeys.INITIAL_SEQUENCE)
            
        else:
            # It is a dict, but missing the keys, assume it is the raw state dict
            _LOGGER.warning(f"File '{pth_path.name}' does not have the required keys for a Dragon-ML finalized-file. Keys found:\n    {list(pth_file_content.keys())}")
            self._model_state_dict = pth_file_content
    
            
        if self._model_state_dict is None:
            _LOGGER.error("Error loading the model state dictionary from the file provided.")
            raise IOError()
        
    def _none_checker(self, attribute: Any, atr_name: str) -> None:
        if attribute is None and self._verbose:
            if self._task != MagicWords.UNKNOWN:
                message = f"Task '{self._task}' does not have a parameter '{atr_name}'."
            else:
                message = f"Property '{atr_name}' was not found in the file."
                
            _LOGGER.warning(message)
        
    @property
    def task(self) -> str:
        """Returns the task type."""
        return self._task
    
    @property
    def model_state_dict(self) -> dict[str, Any]:
        """Returns the model state dictionary."""
        # No need to check for None, as it is guaranteed to be set in __init__
        return self._model_state_dict # type: ignore
    
    @property
    def epoch(self) -> Optional[int]:
        """Returns the number of epochs trained."""
        self._none_checker(self._epoch, PyTorchCheckpointKeys.EPOCH)
        return self._epoch
    
    @property
    def target_name(self) -> Optional[str]:
        """Returns the target name."""
        self._none_checker(self._target_name, PyTorchCheckpointKeys.TARGET_NAME)
        return self._target_name
    
    @property
    def target_names(self) -> Optional[list[str]]:
        """Returns a list of target names."""
        self._none_checker(self._target_names, PyTorchCheckpointKeys.TARGET_NAMES)
        return self._target_names
    
    @property
    def classification_threshold(self) -> Optional[float]:
        """Returns the classification threshold."""
        self._none_checker(self._classification_threshold, PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD)
        return self._classification_threshold
    
    @property
    def class_map(self) -> Optional[dict[str, int]]:
        """Returns the class map (string to integer)."""
        self._none_checker(self._class_map, PyTorchCheckpointKeys.CLASS_MAP)
        return self._class_map
    
    @property
    def initial_sequence(self) -> Optional[np.ndarray]:
        """Returns the initial sequence for sequence prediction."""
        self._none_checker(self._initial_sequence, PyTorchCheckpointKeys.INITIAL_SEQUENCE)
        return self._initial_sequence
    
    @property
    def sequence_length(self) -> Optional[int]:
        """Returns the sequence length for sequence prediction."""
        self._none_checker(self._sequence_length, PyTorchCheckpointKeys.SEQUENCE_LENGTH)
        return self._sequence_length
    
