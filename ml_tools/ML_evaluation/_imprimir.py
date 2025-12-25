from .._core import _imprimir_disponibles

_GRUPOS = [
    # regression
    "regression_metrics",
    "multi_target_regression_metrics",
    # classification
    "classification_metrics",
    "multi_label_classification_metrics",
    # loss
    "plot_losses",
    # feature importance
    "shap_summary_plot",
    "multi_target_shap_summary_plot",
    "plot_attention_importance",
    # sequence
    "sequence_to_value_metrics",
    "sequence_to_sequence_metrics",
    # vision
    "segmentation_metrics",
    "object_detection_metrics",
]

def info():
    _imprimir_disponibles(_GRUPOS)
