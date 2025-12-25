from typing import Optional
import numpy as np

from .._core import get_logger
from ..path_manager import sanitize_filename
from ..keys._keys import MLTaskKeys


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
    "FinalizeSequenceSequencePrediction",
    "FinalizeSequenceValuePrediction",
]

# -------- Finalize classes --------
class _FinalizeModelTraining:
    """
    Base class for finalizing model training.

    This class is not intended to be instantiated directly. Instead, use one of its specific subclasses.
    """
    def __init__(self,
                 filename: str,
                 ) -> None:
        self.filename = _validate_string(string=filename, attribute_name="filename", extension=".pth")
        self.target_name: Optional[str] = None
        self.target_names: Optional[list[str]] = None
        self.classification_threshold: Optional[float] = None
        self.class_map: Optional[dict[str,int]] = None
        self.initial_sequence: Optional[np.ndarray] = None
        self.sequence_length: Optional[int] = None
        self.task: str = 'UNKNOWN'


class FinalizeRegression(_FinalizeModelTraining):
    """Parameters for finalizing a single-target regression model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.task = MLTaskKeys.REGRESSION
    
    
class FinalizeMultiTargetRegression(_FinalizeModelTraining):
    """Parameters for finalizing a multi-target regression model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
        """
        super().__init__(filename=filename)
        safe_names = [_validate_string(string=target_name, attribute_name="All target names") for target_name in target_names]
        self.target_names = safe_names
        self.task = MLTaskKeys.MULTITARGET_REGRESSION


class FinalizeBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 classification_threshold: float,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_CLASSIFICATION


class FinalizeMultiClassClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_CLASSIFICATION
    
    
class FinalizeBinaryImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary image classification model."""
    def __init__(self,
                 filename: str,
                 classification_threshold: float,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            classification_threshold (float): The cutoff threshold for
                classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_IMAGE_CLASSIFICATION


class FinalizeMultiClassImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class image classification model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
    
    
class FinalizeMultiLabelBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-label binary classification model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 classification_threshold: float,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
        """
        super().__init__(filename=filename)
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
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            classification_threshold (float): The cutoff threshold for classifying as the positive class (mask).
        """
        super().__init__(filename=filename)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_SEGMENTATION
    
    
class FinalizeMultiClassSegmentation(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class segmentation model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_SEGMENTATION


class FinalizeObjectDetection(_FinalizeModelTraining):
    """Parameters for finalizing an object detection model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.OBJECT_DETECTION


class FinalizeSequenceSequencePrediction(_FinalizeModelTraining):
    """Parameters for finalizing a sequence-to-sequence prediction model."""
    def __init__(self,
                 filename: str,
                 last_training_sequence: np.ndarray,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            last_training_sequence (np.ndarray): The last sequence from the training data, needed to start predictions.
        """
        super().__init__(filename=filename)
        
        if not isinstance(last_training_sequence, np.ndarray):
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got {type(last_training_sequence)}.")
            raise TypeError()
        
        if last_training_sequence.ndim == 1:
            # It's already 1D, (N,). This is valid.
            self.initial_sequence = last_training_sequence
        elif last_training_sequence.ndim == 2:
            # Handle both (1, N) and (N, 1)
            if last_training_sequence.shape[0] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            elif last_training_sequence.shape[1] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            else:
                _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
                raise ValueError()
        else:
            # It's 3D or more, which is not supported
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
            raise ValueError()
        
        # Save the length of the validated 1D sequence
        self.sequence_length = len(self.initial_sequence) # type: ignore
        self.task = MLTaskKeys.SEQUENCE_SEQUENCE


class FinalizeSequenceValuePrediction(_FinalizeModelTraining):
    """Parameters for finalizing a sequence-to-value prediction model."""
    def __init__(self,
                 filename: str,
                 last_training_sequence: np.ndarray,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            last_training_sequence (np.ndarray): The last sequence from the training data, needed to start predictions.
        """
        super().__init__(filename=filename)
        
        if not isinstance(last_training_sequence, np.ndarray):
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got {type(last_training_sequence)}.")
            raise TypeError()
        
        if last_training_sequence.ndim == 1:
            # It's already 1D, (N,). This is valid.
            self.initial_sequence = last_training_sequence
        elif last_training_sequence.ndim == 2:
            # Handle both (1, N) and (N, 1)
            if last_training_sequence.shape[0] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            elif last_training_sequence.shape[1] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            else:
                _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
                raise ValueError()
        else:
            # It's 3D or more, which is not supported
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
            raise ValueError()
        
        # Save the length of the validated 1D sequence
        self.sequence_length = len(self.initial_sequence) # type: ignore
        self.task = MLTaskKeys.SEQUENCE_VALUE


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

