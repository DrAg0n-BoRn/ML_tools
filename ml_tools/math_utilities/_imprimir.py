from .._core import _imprimir_disponibles

_GRUPOS = [
    "normalize_mixed_list",
    "threshold_binary_values",
    "threshold_binary_values_batch",
    "discretize_categorical_values",
]

def info():
    _imprimir_disponibles(_GRUPOS)
