from .._core import _imprimir_disponibles

_GRUPOS = [
    "InferenceKeys",
    "CheckpointCallbackKeys",
    "FinalizedFileKeys",
    "TaskKeys",
]

def info():
    _imprimir_disponibles(_GRUPOS)
