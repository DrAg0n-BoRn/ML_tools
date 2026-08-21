import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .._core import get_logger

from ._z_helpers import _apply_reduction


_LOGGER = get_logger("Segmentation Loss")


__all__ = [
    "TverskyLoss",
    "FocalTverskyLoss",
]


class TverskyLoss(nn.Module):
    """
    Computes the Tversky Loss for imbalanced multi-class image segmentation.
    
    Tversky Loss is a generalization of the Dice coefficient. It introduces two 
    parameters, alpha and beta, which control the magnitude of penalties for 
    false positives (FP) and false negatives (FN). 
    
    Formula:
        $$ \\mathcal{L}_{\\text{Tversky}} = 1 - \\frac{|P \\cap G|}{|P \\cap G| + \\alpha |P \\setminus G| + \\beta |G \\setminus P|} $$
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            alpha (float): Weight of false positives. 
            beta (float): Weight of false negatives.
            smooth (float): A small constant added to the numerator and denominator 
                to avoid division by zero and stabilize gradients. Defaults to 1e-6.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0).
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        
        # validate ratios
        if not (0 <= alpha <= 1):
            _LOGGER.error(f"Alpha must be in [0, 1], got {alpha}")
            raise ValueError()
        if not (0 <= beta <= 1):
            _LOGGER.error(f"Beta must be in [0, 1], got {beta}")
            raise ValueError()
        
        # ratios should sum to 1 for proper weighting
        if not (abs(alpha + beta - 1.0) < 1e-6):
            _LOGGER.error(f"Alpha and Beta should sum to 1. Got alpha={alpha}, beta={beta}")
            raise ValueError()
        
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot and permute channel to index 1 dynamically
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        permute_dims = [0, targets_one_hot.ndim - 1] + list(range(1, targets_one_hot.ndim - 1))
        targets_one_hot = targets_one_hot.permute(*permute_dims).float()

        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        spatial_dims = tuple(range(2, logits.ndim))
        
        true_positives = torch.sum(probs * targets_one_hot, dim=spatial_dims)
        false_positives = torch.sum(probs * (1.0 - targets_one_hot), dim=spatial_dims)
        false_negatives = torch.sum((1.0 - probs) * targets_one_hot, dim=spatial_dims)

        denominator = true_positives + self.alpha * false_positives + self.beta * false_negatives
        tversky_score = (true_positives + self.smooth) / (denominator + self.smooth)
        
        tversky_loss = 1.0 - tversky_score

        loss = _apply_reduction(tversky_loss, self.reduction)
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Computes the Focal Tversky Loss for imbalanced multi-class image segmentation.
    
    This combines the boundary-focusing properties of Focal Loss with the 
    asymmetric false-positive/false-negative weighting of Tversky Loss.
    
    Formula:
        $$ \\mathcal{L}_{\\text{FocalTversky}} = (1 - \\text{TverskyIndex})^\\gamma $$
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            alpha (float): Weight of false positives.
            beta (float): Weight of false negatives.
            gamma (float): Focal parameter to down-weight easy examples. Rule of thumb: gamma > 1 focuses more on hard examples.
            smooth (float): A small constant added to the numerator and denominator 
                to avoid division by zero and stabilize gradients. Defaults to 1e-6.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0).
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            include_background=include_background,
            reduction="none" 
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss_unreduced = self.tversky(logits, targets)
        
        focal_tversky_loss = tversky_loss_unreduced ** self.gamma

        loss = _apply_reduction(focal_tversky_loss, self.reduction)
        return loss
