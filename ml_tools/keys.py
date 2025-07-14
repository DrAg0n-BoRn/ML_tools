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
    Used internally for ensemble_learning module.
    """
    # Serializing a trained model metadata.
    MODEL = "model"
    FEATURES = "feature_names"
    TARGET = "target_name"
    
    # Classification keys
    CLASSIFICATION_LABEL = "label"
    CLASSIFICATION_PROBABILITIES = "probabilities"
