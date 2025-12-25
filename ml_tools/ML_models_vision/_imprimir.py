from .._core import _imprimir_disponibles

_GRUPOS = [
    # Image Classification
    "DragonResNet",
    "DragonEfficientNet",
    "DragonVGG",
    # Image Segmentation
    "DragonFCN",
    "DragonDeepLabv3",
    # Object Detection
    "DragonFastRCNN",
]

def info():
    _imprimir_disponibles(_GRUPOS)
