from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonInferenceHandler",
    "DragonChainInference",
    "multi_inference_regression",
    "multi_inference_classification"
]

def info():
    _imprimir_disponibles(_GRUPOS)
