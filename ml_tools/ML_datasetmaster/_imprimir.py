from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonDataset",
    "DragonDatasetMulti",
    # sequence
    "DragonDatasetSequence",
    # vision
    "DragonDatasetVision",
    "DragonDatasetSegmentation",
    "DragonDatasetObjectDetection",
]

def info():
    _imprimir_disponibles(_GRUPOS)
