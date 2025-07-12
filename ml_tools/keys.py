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
    
    Keys used for serializing a trained model metadata.
    """
    MODEL = "model"
    FEATURES = "feature_names"
    TARGET = "target_name"
