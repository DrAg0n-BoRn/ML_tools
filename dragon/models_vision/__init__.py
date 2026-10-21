from ._image_classification import (
    DragonResNet,
    DragonEfficientNet,
    DragonVGG,
)

from ._image_segmentation import (
    DragonFCN,
    DragonDeepLabv3
)

from ._object_detection import (
    DragonFastRCNN,
)

from .._core import _imprimir_disponibles


__all__ = [
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
    _imprimir_disponibles(__all__)
