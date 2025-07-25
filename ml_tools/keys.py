class LogKeys:
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


class ModelSaveKeys:
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


class _OneHotOtherPlaceholder:
    """Used internally by GUI_tools."""
    OTHER_GUI = "OTHER"
    OTHER_MODEL = "one hot OTHER placeholder"
    OTHER_DICT = {OTHER_GUI: OTHER_MODEL}
