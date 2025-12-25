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

from ._imprimir import info


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
