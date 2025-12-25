from .._core import _imprimir_disponibles

_GRUPOS = [
    "evaluate_model_classification",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "evaluate_model_regression",
    "get_shap_values",
    "plot_learning_curves"
]

def info():
    _imprimir_disponibles(_GRUPOS)
