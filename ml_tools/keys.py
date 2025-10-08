class PyTorchLogKeys:
    """
    Used internally for ML scripts module.
    
    Centralized keys for logging and history.
    """
    # --- Epoch Level ---
    TRAIN_LOSS = 'train_loss'
    VAL_LOSS = 'val_loss'

    # --- Batch Level ---
    BATCH_LOSS = 'loss'
    BATCH_INDEX = 'batch'
    BATCH_SIZE = 'size'


class EnsembleKeys:
    """
    Used internally by ensemble_learning.
    """
    # Serializing a trained model metadata.
    MODEL = "model"
    FEATURES = "feature_names"
    TARGET = "target_name"
    
    # Classification keys
    CLASSIFICATION_LABEL = "labels"
    CLASSIFICATION_PROBABILITIES = "probabilities"


class PyTorchInferenceKeys:
    """Keys for the output dictionaries of PyTorchInferenceHandler."""
    # For regression tasks
    PREDICTIONS = "predictions"
    
    # For classification tasks
    LABELS = "labels"
    PROBABILITIES = "probabilities"


class PytorchModelArchitectureKeys:
    """Keys for saving and loading model architecture."""
    MODEL = 'model_class'
    CONFIG = "config"
    SAVENAME = "architecture"


class PytorchArtifactPathKeys:
    """Keys for model artifact paths."""
    FEATURES_PATH = "feature_names_path"
    TARGETS_PATH = "target_names_path"
    ARCHITECTURE_PATH = "model_architecture_path"
    WEIGHTS_PATH = "model_weights_path"
    SCALER_PATH = "scaler_path"


class DatasetKeys:
    """Keys for saving dataset artifacts"""
    FEATURE_NAMES = "feature_names"
    TARGET_NAMES = "target_names"
    SCALER_PREFIX = "scaler_"


class SHAPKeys:
    """Keys for SHAP functions"""
    FEATURE_COLUMN = "feature"
    SHAP_VALUE_COLUMN = "mean_abs_shap_value"
    SAVENAME = "shap_summary"


class _OneHotOtherPlaceholder:
    """Used internally by GUI_tools."""
    OTHER_GUI = "OTHER"
    OTHER_MODEL = "one hot OTHER placeholder"
    OTHER_DICT = {OTHER_GUI: OTHER_MODEL}
