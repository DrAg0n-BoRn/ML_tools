from ._datasetmaster import (
    DragonDataset,
    DragonDatasetMulti,
)

from ._sequence_datasetmaster import (
    DragonDatasetSequenceAutoregressive,
    DragonDatasetSequenceExogenous,
)

from ._vision_classification import (
    DragonDatasetImageClassification
)

from ._segmentation_dataset import (
    DragonDatasetSegmentation
)

from ._object_detection_dataset import (
    DragonDatasetObjectDetection
)

from .._core import _imprimir_disponibles


__all__ = [
    # Standard tabular datasets
    "DragonDataset",
    "DragonDatasetMulti",
    # sequence
    "DragonDatasetSequenceAutoregressive",
    "DragonDatasetSequenceExogenous",
    # vision classification
    "DragonDatasetImageClassification",
    # segmentation
    "DragonDatasetSegmentation",
    # object detection
    "DragonDatasetObjectDetection",
]


def info():
    _imprimir_disponibles(__all__)
