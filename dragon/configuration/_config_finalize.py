from typing import Optional
import numpy as np

from .._core import get_logger
from ..path_manager import sanitize_filename
from ..keys._keys import MLTaskKeys, MagicWords, DatasetKeys


_LOGGER = get_logger("Finalized Configuration")


__all__ = [
    # --- Finalize Configs ---
    "FinalizeBinaryClassification",
    "FinalizeBinarySegmentation",
    "FinalizeBinaryImageClassification",
    "FinalizeMultiClassClassification",
    "FinalizeMultiClassImageClassification",
    "FinalizeMultiClassSegmentation",
    "FinalizeMultiLabelBinaryClassification",
    "FinalizeMultiTargetRegression",
    "FinalizeRegression",
    "FinalizeObjectDetection",
    "FinalizeAutoregressiveSequenceSequence",
    "FinalizeAutoregressiveSequenceValue",
    "FinalizeExogenousSequenceSequence",
    "FinalizeExogenousSequenceValue",
    "FinalizeAutoencoder",
    "FinalizeTabularDiffusion"
]

# -------- Finalize classes --------
class _FinalizeModelTraining:
    """
    Base class for finalizing model training.

    This class is not intended to be instantiated directly. Instead, use one of its specific subclasses.
    """
    def __init__(self,
                 filename: str,
                 **kwargs
                 ) -> None:
        self.filename = _validate_string(string=filename, attribute_name="filename", extension=".pth")
        self.target_name: Optional[str] = None
        self.target_names: Optional[list[str]] = None
        self.classification_threshold: Optional[float] = None
        self.class_map: Optional[dict[str,int]] = None
        self.initial_sequence: Optional[np.ndarray] = None
        self.sequence_length: Optional[int] = None
        self.task: str = MagicWords.UNKNOWN
        
        # Dynamically attach any extra arbitrary metadata
        for key, value in kwargs.items():
            setattr(self, key, value)


class FinalizeRegression(_FinalizeModelTraining):
    """Parameters for finalizing a single-target regression model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.task = MLTaskKeys.REGRESSION
    
    
class FinalizeMultiTargetRegression(_FinalizeModelTraining):
    """Parameters for finalizing a multi-target regression model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        safe_names = [_validate_string(string=target_name, attribute_name="All target names") for target_name in target_names]
        self.target_names = safe_names
        self.task = MLTaskKeys.MULTITARGET_REGRESSION


class FinalizeBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 classification_threshold: float,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_CLASSIFICATION


class FinalizeMultiClassClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_CLASSIFICATION
    
    
class FinalizeBinaryImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary image classification model."""
    def __init__(self,
                 filename: str,
                 classification_threshold: float,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            classification_threshold (float): The cutoff threshold for
                classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_IMAGE_CLASSIFICATION


class FinalizeMultiClassImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class image classification model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
    
    
class FinalizeMultiLabelBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-label binary classification model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 classification_threshold: float,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        safe_names = [_validate_string(string=target_name, attribute_name="All target names") for target_name in target_names]
        self.target_names = safe_names
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.task = MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION


class FinalizeBinarySegmentation(_FinalizeModelTraining):
    """Parameters for finalizing a binary segmentation model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int],
                 classification_threshold: float,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'background': 0, 'object': 1}).
            classification_threshold (float): The cutoff threshold for classifying as the positive class (mask).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_SEGMENTATION
    
    
class FinalizeMultiClassSegmentation(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class segmentation model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str, int]): A mapping of class names to their corresponding integer labels.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_SEGMENTATION


class FinalizeObjectDetection(_FinalizeModelTraining):
    """Parameters for finalizing an object detection model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int],
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, **kwargs)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.OBJECT_DETECTION



class _FinalizeSequencePrediction(_FinalizeModelTraining):
    """
    Internal base class for finalizing sequence prediction models.
    Handles strict 2D validation and target name assignment.
    """
    def __init__(self,
                 filename: str,
                 target_types: dict[str, str],
                 last_dataset_sequence: np.ndarray,
                 task: str,
                 **kwargs
                 ) -> None:
        super().__init__(filename=filename, **kwargs)
        
        if not isinstance(target_types, dict) or not target_types:
            _LOGGER.error(f"target_types must be a non-empty dictionary mapping target names to their types ('{DatasetKeys.TARGET_CONTINUOUS}' or '{DatasetKeys.TARGET_CATEGORICAL}').")
            raise ValueError()
        
        target_names = target_types.keys()
        target_literal_types = target_types.values()
        
        # 1. Validate Target Names
        safe_names = [_validate_string(string=t, attribute_name="All target names") for t in target_names]
        self.target_names = safe_names
        
        # 2. Validate Target Types
        for t_type in target_literal_types:
            if t_type not in [DatasetKeys.TARGET_CONTINUOUS, DatasetKeys.TARGET_CATEGORICAL]:
                _LOGGER.error(f"Invalid target type '{t_type}'. Must be '{DatasetKeys.TARGET_CONTINUOUS}' or '{DatasetKeys.TARGET_CATEGORICAL}'.")
                raise ValueError()
        
        self.target_types = target_types
        
        # 3. Validate 2D Sequence
        if not isinstance(last_dataset_sequence, np.ndarray):
            _LOGGER.error(f"The last dataset sequence must be a 2D numpy array, got {type(last_dataset_sequence)}.")
            raise TypeError()
            
        if last_dataset_sequence.ndim != 2:
            _LOGGER.error(f"The last dataset sequence must be a 2D numpy array (sequence_length, num_features), got shape {last_dataset_sequence.shape}.")
            raise ValueError()
        
        self.initial_sequence = last_dataset_sequence
        self.sequence_length = last_dataset_sequence.shape[0]
        self.task = task


class FinalizeAutoregressiveSequenceSequence(_FinalizeSequencePrediction):
    """Parameters for finalizing an autoregressive sequence-to-sequence prediction model."""
    def __init__(self,
                 filename: str,
                 target_types: dict[str, str],
                 last_dataset_sequence: np.ndarray,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_types (dict[str, str]): A dictionary mapping target names to their types ('continuous' or 'categorical').
            last_dataset_sequence (np.ndarray): A 2D array (sequence_length, num_features) from the dataset that will become the initial sequence for predictions.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, 
                         target_types=target_types, 
                         last_dataset_sequence=last_dataset_sequence, 
                         task=MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, 
                         **kwargs)


class FinalizeAutoregressiveSequenceValue(_FinalizeSequencePrediction):
    """Parameters for finalizing an autoregressive sequence-to-value prediction model."""
    def __init__(self,
                 filename: str,
                 target_types: dict[str, str],
                 last_dataset_sequence: np.ndarray,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_types (dict[str, str]): A dictionary mapping target names to their types ('continuous' or 'categorical').
            last_dataset_sequence (np.ndarray): A 2D array (sequence_length, num_features) from the dataset that will become the initial sequence for predictions.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, 
                         target_types=target_types,
                         last_dataset_sequence=last_dataset_sequence, 
                         task=MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE, 
                         **kwargs)


class FinalizeExogenousSequenceSequence(_FinalizeSequencePrediction):
    """Parameters for finalizing an exogenous sequence-to-sequence prediction model."""
    def __init__(self,
                 filename: str,
                 target_types: dict[str, str],
                 last_dataset_sequence: np.ndarray,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters for an exogenous model.
        
        Args:
            filename (str): The name of the file to be saved.
            target_types (dict[str, str]): A dictionary mapping target names to their types ('continuous' or 'categorical').
            last_dataset_sequence (np.ndarray): A 2D array (sequence_length, num_features) from the dataset that will become the initial sequence for predictions.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, 
                         target_types=target_types, 
                         last_dataset_sequence=last_dataset_sequence, 
                         task=MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE, 
                         **kwargs)


class FinalizeExogenousSequenceValue(_FinalizeSequencePrediction):
    """Parameters for finalizing an exogenous sequence-to-value prediction model."""
    def __init__(self,
                 filename: str,
                 target_types: dict[str, str],
                 last_dataset_sequence: np.ndarray,
                 **kwargs
                 ) -> None:
        """Initializes the finalization parameters for an exogenous model.
        
        Args:
            filename (str): The name of the file to be saved.
            target_types (dict[str, str]): A dictionary mapping target names to their types ('continuous' or 'categorical').
            last_dataset_sequence (np.ndarray): A 2D array (sequence_length, num_features) from the dataset that will become the initial sequence for predictions.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename=filename, 
                         target_types=target_types,
                         last_dataset_sequence=last_dataset_sequence, 
                         task=MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE, 
                         **kwargs)


class FinalizeAutoencoder(_FinalizeModelTraining):
    """Parameters for finalizing an autoencoder model."""
    def __init__(self, 
                 filename: str,
                 **kwargs) -> None:
        """Initializes the finalization parameters.
        
        Args:
            filename (str): The name of the file to be saved.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename, **kwargs)
        self.task = MLTaskKeys.AUTOENCODER


class FinalizeTabularDiffusion(_FinalizeModelTraining):
    """Parameters for finalizing a tabular diffusion model."""
    def __init__(self, 
                 filename: str,
                 **kwargs) -> None:
        """Initializes the finalization parameters.
        
        Args:
            filename (str): The name of the file to be saved.
            **kwargs: Additional arbitrary metadata to be attached to the finalized configuration.
        """
        super().__init__(filename, **kwargs)
        self.task = MLTaskKeys.DIFFUSION





#### Helper functions for validation in finalize classes ####
def _validate_string(string: str, attribute_name: str, extension: Optional[str]=None) -> str:
    """Helper for finalize classes"""
    if not isinstance(string, str):
        _LOGGER.error(f"{attribute_name} must be a string.")
        raise TypeError()

    if extension:
        safe_name = sanitize_filename(string)
        
        if not safe_name.endswith(extension):
            safe_name += extension
    else:
        safe_name = string
            
    return safe_name

def _validate_threshold(threshold: float):
    """Helper for finalize classes"""
    if not isinstance(threshold, float):
        _LOGGER.error(f"Classification threshold must be a float.")
        raise TypeError()
    elif threshold < 0.1 or threshold > 0.9:
        _LOGGER.error(f"Classification threshold must be in the range [0.1, 0.9]")
        raise ValueError()
    
    return threshold

def _validate_class_map(map_dict: dict[str, int]):
    """Helper for finalize classes"""
    if not isinstance(map_dict, dict):
        _LOGGER.error(f"Class map must be a dictionary, but got {type(map_dict)}.")
        raise TypeError()
    
    if not map_dict:
        _LOGGER.error("Class map dictionary cannot be empty.")
        raise ValueError()

    for key, val in map_dict.items():
        if not isinstance(key, str):
            _LOGGER.error(f"All keys in the class map must be strings, but found key: {key} ({type(key)}).")
            raise TypeError()
        if not isinstance(val, int):
            _LOGGER.error(f"All values in the class map must be integers, but for key '{key}' found value: {val} ({type(val)}).")
            raise TypeError()
            
    return map_dict

