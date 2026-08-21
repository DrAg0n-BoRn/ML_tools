import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


__all__ = [
    "_BaseDiceLoss"
]


class _BaseDiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction

    def _prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot: (B, H, W) -> (B, C, H, W)
        # Convert targets to one-hot and permute channel to index 1 dynamically
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        permute_dims = [0, targets_one_hot.ndim - 1] + list(range(1, targets_one_hot.ndim - 1))
        targets_one_hot = targets_one_hot.permute(*permute_dims).float()
        
        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        # Compute over spatial dimensions per sample and per class
        spatial_dims = tuple(range(2, logits.ndim))
        intersection = torch.sum(probs * targets_one_hot, dim=spatial_dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=spatial_dims)

        return intersection, cardinality, targets_one_hot, spatial_dims
