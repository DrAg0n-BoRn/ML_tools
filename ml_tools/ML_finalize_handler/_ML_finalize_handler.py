import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, Any, Optional
from pathlib import Path

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import PyTorchCheckpointKeys, MagicWords, MLTaskKeys


_LOGGER = get_logger("Finalized-File")


__all__ = [
    "FinalizedFileHandler",
    "ClassificationMetadata",
    "SequenceMetadata"
]

@dataclass
class ClassificationMetadata:
    """Strongly typed container for classification and threshold parameters."""
    class_map: Optional[dict[str, int]] = None
    classification_threshold: Optional[float] = None
    idx_to_class: Optional[dict[int, str]] = None


@dataclass
class SequenceMetadata:
    """Strongly typed container for sequence prediction parameters."""
    initial_sequence: Optional[np.ndarray] = None
    sequence_length: Optional[int] = None
    target_types: Optional[dict[str, str]] = None


class FinalizedFileHandler:
    """
    Handles the loading and validation of a finalized-file with PyTorch model artifacts.

    It parses the dictionary into strongly-typed domain metadata objects (Dataclasses),
    providing robust cross-validation and safe fallback mechanisms.
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
        self._metadata: dict[str, Any] = {}
        
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
            
        if self._model_state_dict is None:
            _LOGGER.error("Error loading the model state dictionary from the file provided.")
            raise IOError()
    
    @property
    def model_state_dict(self) -> dict[str, Any]:
        """Returns the model state dictionary strictly typed to satisfy linters."""
        return self._model_state_dict # type: ignore
    
    def parse_targets(self) -> Optional[list[str]]:
        """
        Safely extracts target names, standardizing single and multiple targets 
        into a uniform list of strings.
        """
        if PyTorchCheckpointKeys.TARGET_NAMES in self._metadata:
            return self._metadata[PyTorchCheckpointKeys.TARGET_NAMES]
        
        if PyTorchCheckpointKeys.TARGET_NAME in self._metadata:
            return [self._metadata[PyTorchCheckpointKeys.TARGET_NAME]]
            
        if self._verbose:
            _LOGGER.warning("No target names found in FinalizedFile.")
            
        return None
    
    def parse_classification_metadata(self) -> ClassificationMetadata:
        """
        Extracts and validates classification mappings and thresholds.
        """
        class_map = self._metadata.get(PyTorchCheckpointKeys.CLASS_MAP)
        threshold = self._metadata.get(PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD)
        idx_to_class = {v: k for k, v in class_map.items()} if class_map else None

        # Task-specific validation logic
        requires_map = [
            MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION,
            MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION,
            MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTICLASS_SEGMENTATION,
            MLTaskKeys.OBJECT_DETECTION
        ]
        
        if self.task in requires_map and class_map is None and self._verbose:
            _LOGGER.warning(f"Task '{self.task}' expected a class_map, but none was found.")
            
        requires_threshold = [
            MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION,
            MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION, MLTaskKeys.BINARY_SEGMENTATION
        ]
        
        if self.task in requires_threshold and threshold is None and self._verbose:
            _LOGGER.warning(f"Task '{self.task}' expected a classification_threshold, but none was found.")

        return ClassificationMetadata(
            class_map=class_map,
            classification_threshold=threshold,
            idx_to_class=idx_to_class
        )
    
    def parse_sequence_metadata(self) -> SequenceMetadata:
        """
        Extracts sequence forecasting parameters and enforces dimensionality 
        and length cross-validation.
        """
        seq_len = self._metadata.get(PyTorchCheckpointKeys.SEQUENCE_LENGTH)
        init_seq = self._metadata.get(PyTorchCheckpointKeys.INITIAL_SEQUENCE)
        target_types = self._metadata.get(PyTorchCheckpointKeys.TARGET_TYPES)

        # Validation
        if self.task in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            if seq_len is None and self._verbose:
                _LOGGER.warning(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' not found in model file. Forecasting validation will be skipped.")
                
            if init_seq is None and self._verbose:
                _LOGGER.info("No default 'initial_sequence' found in model file. Must be provided for forecasting.")
                
            if init_seq is not None and seq_len is not None:
                if len(init_seq) != seq_len and self._verbose:
                    _LOGGER.warning(f"Loaded 'initial_sequence' length ({len(init_seq)}) mismatches 'sequence_length' ({seq_len}).")
                    
        return SequenceMetadata(
            initial_sequence=init_seq,
            sequence_length=seq_len,
            target_types=target_types
        )
    
    @property
    def custom_metadata(self) -> dict[str, Any]:
        """
        Provides access to any arbitrary metadata (kwargs) attached to the 
        model during finalization that is not part of the core schema.
        
        Note: The 'model_state_dict' and 'task' keys are extracted during 
        initialization. Access them via their dedicated properties instead of this dictionary.
        """
        return self._metadata
    
    def print_available_keys(self) -> None:
        """
        Prints all keys originally found in the FinalizedFile or raw state dictionary.
        """
        # If it's a valid FinalizedFile, these two keys were popped during initialization
        if self.task != MagicWords.UNKNOWN:
            popped_keys = [PyTorchCheckpointKeys.MODEL_STATE, PyTorchCheckpointKeys.TASK]
            all_keys = popped_keys + list(self._metadata.keys())
            
            _LOGGER.info(f"Keys found in Dragon-ML FinalizedFile:\n" + "\n".join(f"  - {k}" for k in all_keys))
        else:
            # If it's just a raw state dict, nothing was popped
            _LOGGER.info(f"Keys found in raw state dictionary:\n" + "\n".join(f"  - {k}" for k in self._metadata.keys()))
    
    # def __getattr__(self, name: str) -> Any:
    #     """
    #     Dynamically handles the retrieval of metadata attributes.
    #     Called only when the attribute is not found via normal lookup.
    #     """
    #     if name in self._metadata:
    #         return self._metadata[name]
            
    #     # _none_checker warnings
    #     if self._verbose:
    #         if self.task != MagicWords.UNKNOWN:
    #             _LOGGER.warning(f"Task '{self.task}' does not have a parameter '{name}'.")
    #         else:
    #             _LOGGER.warning(f"Property '{name}' was not found in the file.")
                
    #     return None
