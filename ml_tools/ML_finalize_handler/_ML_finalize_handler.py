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

    It dynamically maps all metadata contained within the file to object properties.
    It provides a robust fallback mechanism: if the loaded file does not match 
    the specific finalized-file schema, it is treated as a raw state dictionary.
    """
    def __init__(self, finalized_file_path: Union[str, Path]) -> None:
        """
        Initializes the handler by loading the file and validating its structure.

        Args:
            finalized_file_path (Union[str, Path]): The path to the Dragon-ML finalized-file or PyTorch state dictionary.
        """
        self._verbose: bool = True
        self.task: str = MagicWords.UNKNOWN
        self._model_state_dict: Optional[dict[str, Any]] = None
        
        pth_path = make_fullpath(finalized_file_path, enforce="file")
        
        try:
            pth_file_content = torch.load(pth_path, map_location='cpu')
        except Exception as e:
            _LOGGER.error(f"Failed to load finalized-file from '{pth_path}': {e}")
            raise
        
        if not isinstance(pth_file_content, dict):
            _LOGGER.error(f"The loaded content from '{pth_path.name}' is of type '{type(pth_file_content).__name__}', but a dictionary was expected.")
            raise TypeError()
        
        # Check for core finalized-file structure
        has_core_keys = (PyTorchCheckpointKeys.MODEL_STATE in pth_file_content and
                         PyTorchCheckpointKeys.EPOCH in pth_file_content and
                         PyTorchCheckpointKeys.TASK in pth_file_content)
        
        if has_core_keys:
            # Extract strict requirements
            self._model_state_dict = pth_file_content.pop(PyTorchCheckpointKeys.MODEL_STATE)
            self.task = pth_file_content.pop(PyTorchCheckpointKeys.TASK, MagicWords.UNKNOWN)
            # Retain the rest of the dictionary as arbitrary metadata
            self._metadata = pth_file_content 
        else:
            _LOGGER.warning(f"File '{pth_path.name}' does not have the required keys for a Dragon-ML finalized-file. Keys found:\n    {list(pth_file_content.keys())}")
            self._model_state_dict = pth_file_content
            self._metadata = {}
            
        if self._model_state_dict is None:
            _LOGGER.error("Error loading the model state dictionary from the file provided.")
            raise IOError()
    
    @property
    def model_state_dict(self) -> dict[str, Any]:
        """Returns the model state dictionary strictly typed to satisfy linters."""
        return self._model_state_dict # type: ignore

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically handles the retrieval of metadata attributes.
        Called only when the attribute is not found via normal lookup.
        """
        if name in self._metadata:
            return self._metadata[name]
            
        # _none_checker warnings
        if self._verbose:
            if self.task != MagicWords.UNKNOWN:
                _LOGGER.warning(f"Task '{self.task}' does not have a parameter '{name}'.")
            else:
                _LOGGER.warning(f"Property '{name}' was not found in the file.")
                
        return None
