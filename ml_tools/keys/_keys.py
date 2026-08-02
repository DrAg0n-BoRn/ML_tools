class MagicWords:
    """General purpose keys"""
    BEST = "best"
    CURRENT = "current"
    RENAME = "rename"
    UNKNOWN = "unknown"
    AUTO = "auto"


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


class ScalerKeys:
    """Keys for saving/loading scaler artifacts."""
    MEAN = "mean"
    STD = "std"
    INDICES = "continuous_feature_indices"
    FEATURE_SCALER = "feature_scaler"
    TARGET_SCALER = "target_scaler"


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
    """Keys for the output dictionaries of InferenceHandler classes."""
    # For regression tasks
    PREDICTIONS = "predictions"
    
    # For distribution regression tasks
    VARIANCE = "variance"
    
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
    CLASS_MAP = "class_map"
    CLASSES = "classes"
    ID = "id"
    VALIDATION_SPLIT = "validation_split"
    TEST_SPLIT = "test_split"
    
    SCALER_PREFIX = "scaler_"
    # Feature Schema
    CONTINUOUS_NAMES = "continuous_feature_names"
    CATEGORICAL_NAMES = "categorical_feature_names"
    # Dataset splits
    TRAIN_SUBSET = "train_set"
    VALIDATION_SUBSET = "validation_set"
    TEST_SUBSET = "test_set"
    # Dataset filename
    DATASET_FILENAME = "dataset_bundle"
    
    # Sequence
    SEQUENCE_LENGTH = "sequence_length"
    PREDICTION_MODE = "prediction_mode"
    LAST_SEQUENCE = "last_dataset_sequence"
    TARGET_TYPES = "target_types"
    TARGET_CONTINUOUS = "continuous"
    TARGET_CATEGORICAL = "categorical"


class SHAPKeys:
    """Keys for SHAP functions"""
    FEATURE_COLUMN = "feature"
    SHAP_VALUE_COLUMN = "mean_abs_shap_value"
    SAVENAME = "shap_summary"


class CaptumKeys:
    """Keys for Captum functions"""
    FEATURE_COLUMN = "Feature"
    IMPORTANCE_COLUMN = "Scaled Mean Attribution"
    PERCENT_COLUMN = "Relative Importance(%)"
    SAVENAME = "captum_summary"
    PLOT_NAME = "captum_importance"


class PyTorchCheckpointKeys:
    """Keys for saving/loading a training checkpoint dictionary."""
    MODEL_STATE = "model_state_dict"
    OPTIMIZER_STATE = "optimizer_state_dict"
    SCHEDULER_STATE = "scheduler_state_dict"
    EPOCH = "epoch"
    BEST_SCORE = "best_score"
    HISTORY = "history"
    CHECKPOINT_NAME = "DragonCheckpoint"
    
    ### Finalized config
    # EPOCH
    # MODEL_STATE
    TASK = "task"
    CLASSIFICATION_THRESHOLD = "classification_threshold"
    CLASS_MAP = "class_map"
    SEQUENCE_LENGTH = "sequence_length"
    INITIAL_SEQUENCE = "initial_sequence"
    TARGET_NAME = "target_name"
    TARGET_NAMES = "target_names"
    TARGET_TYPES = "target_types"


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
    SEGMENTATION_HEATMAP = "heatmap"
    SEGMENTATION_CONFUSION_MATRIX = "confusion_matrix"
    SEGMENTATION_RADAR_PLOT = "radar_plot"
    SEGMENTATION_DISTRIBUTION_PLOT = "distribution_plot"
    # Object detection
    OBJECT_DETECTION_REPORT = "object_detection_report"
    # Overlapped image suffix
    OVERLAPPED_SUFFIX = "_overlapped.png"


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
    SCORES = "scores"


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
    
    DIFFUSION = "diffusion"
    AUTOENCODER = "autoencoder"
    
    ALL_BINARY_TASKS = [BINARY_CLASSIFICATION, MULTILABEL_BINARY_CLASSIFICATION, BINARY_IMAGE_CLASSIFICATION, BINARY_SEGMENTATION]
    
    ALL_TASKS = [REGRESSION, MULTITARGET_REGRESSION, 
                 BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_BINARY_CLASSIFICATION, 
                 BINARY_IMAGE_CLASSIFICATION, MULTICLASS_IMAGE_CLASSIFICATION, 
                 BINARY_SEGMENTATION, MULTICLASS_SEGMENTATION, 
                 OBJECT_DETECTION, 
                 SEQUENCE_SEQUENCE, SEQUENCE_VALUE, 
                 DIFFUSION, AUTOENCODER]


class DragonTrainerKeys:
    VALIDATION_METRICS_DIR = "Evaluation_Validation"
    TEST_METRICS_DIR = "Evaluation_Test"
    CHECKPOINT_DIR = "Checkpoints"
    CAPTUM_DIR = "Explain_Captum"
    SHAP_DIR = "Explain_SHAP"
    ATTENTION_DIR = "Explain_Attention"
    

class ParetoOptimizationKeys:
    """Used by the ML optimization pareto module."""
    PARETO_PLOTS_DIR = "Pareto_Plots"
    OPTIMIZATION_PLOTS_DIR = "Optimization_Plots"
    SQL_DATABASE_FILENAME = "OptimizationResults.db"
    HISTORY_PLOTS_DIR = "History"
    
    # Plot Config values
    FONT_PAD = 10
    DPI = 400


class OptimizationToolsKeys:
    """Used by the optimization tools module."""
    OPTIMIZATION_BOUNDS_FILENAME = "optimization_bounds"


class SchemaKeys:
    """Used by the schema module."""
    SCHEMA_FILENAME = "FeatureSchema.json"
    GUI_SCHEMA_FILENAME = "GUISchema.json"
    # Model architecture API
    SCHEMA_DICT = "schema_dict"
    # GUI Schema
    TARGETS = "targets"
    CONTINUOUS = "continuous"
    BINARY = "binary"
    MULTIBINARY = "multibinary"
    CATEGORICAL = "categorical"
    MODEL_NAME = "model_name"
    GUI_NAME = "gui_name"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    MAPPING = "mapping"
    OPTIONAL_LABELS = "optional_labels"


class ChainKeys:
    """Used by the ML chaining module."""
    CHAIN_PREDICTION_PREFIX = "pred_"
