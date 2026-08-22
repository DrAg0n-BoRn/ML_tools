from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._z_helpers import _handle_ignore_index


__all__ = [
    "_BaseDiceLoss"
]


class _BaseDiceLoss(nn.Module):
    def __init__(
        self,
        include_background: bool,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.smooth = 1e-6
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.reduction = "mean"

    def _prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor):
        num_classes = logits.shape[1]
        
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        if targets.ndim == logits.ndim and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # Streamlined execution via private helper
        probs, targets_one_hot = _handle_ignore_index(
            probs, targets, self.ignore_index, num_classes
        )
        
        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        spatial_dims = tuple(range(2, logits.ndim))

        intersection = torch.sum(probs * targets_one_hot, dim=spatial_dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=spatial_dims)

        return intersection, cardinality, targets_one_hot, spatial_dims
