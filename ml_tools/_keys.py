class MagicWords:
    """General purpose keys"""
    LATEST = "latest"
    CURRENT = "current"
    RENAME = "rename"


class PyTorchLogKeys:
    """
    Used internally for ML scripts module.
    
    Centralized keys for logging and history.
    """
    # --- Epoch Level ---
    TRAIN_LOSS = 'train_loss'
    VAL_LOSS = 'val_loss'
    LEARNING_RATE = 'lr'

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
    LABEL_NAMES = "label_names"


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
    """Keys for saving dataset artifacts. Also used by FeatureSchema"""
    FEATURE_NAMES = "feature_names"
    TARGET_NAMES = "target_names"
    SCALER_PREFIX = "scaler_"
    # Feature Schema
    CONTINUOUS_NAMES = "continuous_feature_names"
    CATEGORICAL_NAMES = "categorical_feature_names"


class SHAPKeys:
    """Keys for SHAP functions"""
    FEATURE_COLUMN = "feature"
    SHAP_VALUE_COLUMN = "mean_abs_shap_value"
    SAVENAME = "shap_summary"


class PyTorchCheckpointKeys:
    """Keys for saving/loading a training checkpoint dictionary."""
    MODEL_STATE = "model_state_dict"
    OPTIMIZER_STATE = "optimizer_state_dict"
    SCHEDULER_STATE = "scheduler_state_dict"
    EPOCH = "epoch"
    BEST_SCORE = "best_score"
    HISTORY = "history"
    CHECKPOINT_NAME = "PyModelCheckpoint"
    # Finalized config
    CLASSIFICATION_THRESHOLD = "classification_threshold"
    CLASS_MAP = "class_map"
    SEQUENCE_LENGTH = "sequence_length"
    INITIAL_SEQUENCE = "initial_sequence"
    TARGET_NAME = "target_name"
    TARGET_NAMES = "target_names"


class UtilityKeys:
    """Keys used for utility modules"""
    MODEL_PARAMS_FILE = "model_parameters"
    TOTAL_PARAMS = "Total Parameters"
    TRAINABLE_PARAMS = "Trainable Parameters"
    PTH_FILE = "pth report "
    MODEL_ARCHITECTURE_FILE = "model_architecture_summary"


class VisionKeys:
    """For vision ML metrics"""
    SEGMENTATION_REPORT = "segmentation_report"
    SEGMENTATION_HEATMAP = "segmentation_metrics_heatmap"
    SEGMENTATION_CONFUSION_MATRIX = "segmentation_confusion_matrix"
    # Object detection
    OBJECT_DETECTION_REPORT = "object_detection_report"


class VisionTransformRecipeKeys:
    """Defines the key names for the transform recipe JSON file."""
    TASK = "task"
    PIPELINE = "pipeline"
    NAME = "name"
    KWARGS = "kwargs"
    PRE_TRANSFORMS = "pre_transforms"
    
    RESIZE_SIZE = "resize_size"
    CROP_SIZE = "crop_size"
    MEAN = "mean"
    STD = "std"


class ObjectDetectionKeys:
    """Used by the object detection dataset"""
    BOXES = "boxes"
    LABELS = "labels"


class MLTaskKeys:
    """Used by the Trainer and InferenceHandlers"""
    REGRESSION = "regression"
    MULTITARGET_REGRESSION = "multitarget regression"
    
    BINARY_CLASSIFICATION = "binary classification"
    MULTICLASS_CLASSIFICATION = "multiclass classification"
    MULTILABEL_BINARY_CLASSIFICATION = "multilabel binary classification"
    
    BINARY_IMAGE_CLASSIFICATION = "binary image classification"
    MULTICLASS_IMAGE_CLASSIFICATION = "multiclass image classification"
    
    BINARY_SEGMENTATION = "binary segmentation"
    MULTICLASS_SEGMENTATION = "multiclass segmentation"
    
    OBJECT_DETECTION = "object detection"
    
    SEQUENCE_SEQUENCE = "sequence-to-sequence"
    SEQUENCE_VALUE = "sequence-to-value"
    
    ALL_BINARY_TASKS = [BINARY_CLASSIFICATION, MULTILABEL_BINARY_CLASSIFICATION, BINARY_IMAGE_CLASSIFICATION, BINARY_SEGMENTATION]


class DragonTrainerKeys:
    VALIDATION_METRICS_DIR = "Validation_Metrics"
    TEST_METRICS_DIR = "Test_Metrics"


class _OneHotOtherPlaceholder:
    """Used internally by GUI_tools."""
    OTHER_GUI = "OTHER"
    OTHER_MODEL = "one hot OTHER placeholder"
    OTHER_DICT = {OTHER_GUI: OTHER_MODEL}
