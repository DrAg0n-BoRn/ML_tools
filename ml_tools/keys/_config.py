
class _EvaluationConfig:
    """Set config values for evaluation modules."""
    DPI = 400
    LABEL_PADDING = 10
    # large sizes for SVG layout to accommodate large fonts
    REGRESSION_PLOT_SIZE = (10, 7)
    SEQUENCE_PLOT_SIZE = (10, 7)
    CLASSIFICATION_PLOT_SIZE = (9, 9)
    # Captum plots
    CAPTUM_PLOT_SIZE = (10, 8)
    CAPTUM_FONT_SIZE = 24
    CAPTUM_X_TICK_SIZE = 20
    # Loss plot
    LOSS_PLOT_SIZE = (18, 9)
    LOSS_PLOT_LABEL_SIZE = 24
    LOSS_PLOT_TICK_SIZE = 22
    LOSS_PLOT_LEGEND_SIZE = 24
    # CM settings
    CM_SIZE = (9, 8)    # used for multi label binary classification confusion matrix 
    HEATMAP_WIDTH = 10.0 # default width for classification heatmaps, height is dynamic based on number of classes
    NAME_LIMIT = 15  # max number of characters for feature/label names in plots
    TITLE_LIMIT = 35  # max number of characters for plot titles before wrapping
    # RADAR Plot settings
    RADAR_PLOT_WIDTH = 800
    RADAR_PLOT_HEIGHT = 800
    RADAR_MAX_FEATURES_BEFORE_DYNAMIC_SIZING = 15
    RADAR_MAX_FEATURE_NAME_LENGTH_FOR_MARGIN = 14


class _OneHotOtherPlaceholder:
    """Used internally by GUI_tools."""
    OTHER_GUI = "OTHER"
    OTHER_MODEL = "one hot OTHER placeholder"
    OTHER_DICT = {OTHER_GUI: OTHER_MODEL}
