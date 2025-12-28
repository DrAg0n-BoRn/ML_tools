from ._regression import (
    regression_metrics,
    multi_target_regression_metrics
)

from ._classification import (
    classification_metrics,
    multi_label_classification_metrics
)

from ._loss import (
    plot_losses,
)

from ._feature_importance import (
    shap_summary_plot,
    multi_target_shap_summary_plot,
    plot_attention_importance
)

from ._sequence import (
    sequence_to_value_metrics,
    sequence_to_sequence_metrics
)

from ._vision import (
    segmentation_metrics,
    object_detection_metrics
)

from .._core import _imprimir_disponibles


__all__ = [
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
    _imprimir_disponibles(__all__)
