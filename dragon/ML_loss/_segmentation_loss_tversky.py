from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .._core import get_logger

from ._z_helpers import _apply_reduction, _handle_ignore_index


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
        include_background: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            alpha (float): Weight of false positives. 
            beta (float): Weight of false negatives.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0).
            ignore_index (int | None): Specifies a target value that is ignored and does 
                not contribute to the input gradient.
        """
        super().__init__()
        
        # validate ratios
        if not (0 <= alpha <= 1):
            _LOGGER.error(f"Alpha must be in [0, 1], got {alpha}")
            raise ValueError()
        if not (0 <= beta <= 1):
            _LOGGER.error(f"Beta must be in [0, 1], got {beta}")
            raise ValueError()
        
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1e-6
        self.include_background = include_background
        self.reduction = "mean"
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Squeeze channel dim if present, e.g., (B, 1, H, W) -> (B, H, W)
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
        include_background: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            alpha (float): Weight of false positives.
            beta (float): Weight of false negatives.
            gamma (float): Focal parameter to down-weight easy examples. Rule of thumb: gamma > 1 focuses more on hard examples.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0).
            ignore_index (int | None): A target index to ignore for the loss gradient 
                computation.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = "mean"
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            include_background=include_background,
            ignore_index=ignore_index
        )
        self.tversky.reduction = "none"  # Apply reduction after the focal transformation

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss_unreduced = self.tversky(logits, targets)
        
        focal_tversky_loss = tversky_loss_unreduced ** self.gamma

        loss = _apply_reduction(focal_tversky_loss, self.reduction)
        return loss
