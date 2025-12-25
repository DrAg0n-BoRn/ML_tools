from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonTrainer",
    "DragonSequenceTrainer",
    "DragonDetectionTrainer",
]

def info():
    _imprimir_disponibles(_GRUPOS)
