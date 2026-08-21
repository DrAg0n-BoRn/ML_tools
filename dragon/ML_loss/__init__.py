from ._regression_loss import (
    LogCoshLoss,
    QuantileLoss,
    WingLoss
)

from ._classification_loss import (
    ClassificationFocalLoss,
    PolyLoss,
    AsymmetricLoss
)

from ._segmentation_loss import (
    DiceLoss,
    GeneralizedDiceLoss,
    SegmentationFocalLoss,
    DiceFocalLoss
)

from ._segmentation_loss_tversky import (
    TverskyLoss,
    FocalTverskyLoss
)


from .._core import _imprimir_disponibles


__all__ = [
    # Regression Losses
    "LogCoshLoss",
    "QuantileLoss",
    "WingLoss",
    # Classification Losses
    "ClassificationFocalLoss",
    "PolyLoss",
    "AsymmetricLoss",
    # Segmentation Losses
    "DiceLoss",
    "GeneralizedDiceLoss",
    "SegmentationFocalLoss",
    "DiceFocalLoss",
    # Segmentation Losses - Tversky
    "TverskyLoss",
    "FocalTverskyLoss"
]


def info():
    _imprimir_disponibles(__all__)
