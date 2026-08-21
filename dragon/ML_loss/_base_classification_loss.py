import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal

from .._core import get_logger

from ._z_helpers import _apply_reduction


_LOGGER = get_logger("Classification Loss")


__all__ = [
    "_BaseClassificationLoss",
]


class _BaseClassificationLoss(nn.Module):
    """
    Base class for classification loss functions. This class provides a common interface
    and shared functionality for various classification loss implementations, including
    Focal Loss and Poly Loss. It handles both binary and multi-class classification scenarios, allowing for flexible loss computation based on the input logits and target labels.
    """
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def _get_base_loss_and_pt(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # key to determine if the task is binary classification or multi-class classification
        is_binary = (logits.ndim == 1) or (logits.ndim == 2 and logits.shape[1] == 1)

        if is_binary:
            logits = logits.view(-1)
            targets = targets.view(-1).float()
            
            base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        else:
            targets = targets.long()
            base_loss = F.cross_entropy(logits, targets, reduction="none")
            
            p_t = torch.exp(-base_loss)

        return base_loss, p_t

    def _compute_custom_loss(self, base_loss: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        _LOGGER.error("The method `_compute_custom_loss` was called but not implemented for this class.")
        raise NotImplementedError()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss, p_t = self._get_base_loss_and_pt(logits, targets)
        custom_loss = self._compute_custom_loss(base_loss, p_t)
        return _apply_reduction(custom_loss, self.reduction)
