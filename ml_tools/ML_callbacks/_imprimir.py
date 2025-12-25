from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonPatienceEarlyStopping",
    "DragonPrecheltEarlyStopping",
    "DragonModelCheckpoint",
    "DragonScheduler",
    "DragonPlateauScheduler",
]

def info():
    _imprimir_disponibles(_GRUPOS)
