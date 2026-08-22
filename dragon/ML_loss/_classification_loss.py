import torch
import torch.nn.functional as F
from typing import Literal

from .._core import get_logger

from ._base_classification_loss import _BaseClassificationLoss
from ._z_helpers import _apply_reduction


_LOGGER = get_logger("Classification Loss")


__all__ = [
    "ClassificationFocalLoss",
    "PolyLoss",
    "AsymmetricLoss"
]


class ClassificationFocalLoss(_BaseClassificationLoss):
    """
    Computes the Focal Loss for binary and multi-class classification tasks.
    
    Focal Loss applies a modulating factor to the standard cross-entropy, 
    down-weighting well-classified examples and focusing gradients on hard, 
    misclassified samples.
    
    Formula:
        $$ \\mathcal{L}_{\\text{Focal}} = (1 - p_t)^\\gamma \\cdot \\mathcal{L}_{\\text{CE}} $$
    """
    def __init__(self, 
                 gamma: float = 2.0, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            gamma (float): The focusing parameter that dictates the rate at which 
                easy examples are down-weighted. A value of 0 is equivalent to 
                standard Cross-Entropy.
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__(reduction=reduction)
        self.gamma = gamma

    def _compute_custom_loss(self, base_loss: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        return ((1.0 - p_t) ** self.gamma) * base_loss


class PolyLoss(_BaseClassificationLoss):
    """
    Computes the PolyLoss for binary and multi-class classification tasks.
    
    PolyLoss operates on the Taylor expansion formulation of Cross-Entropy, 
    adjusting the leading polynomial term to improve gradient dynamics and 
    overall classification margins.
    
    Formula:
        $$ \\mathcal{L}_{\\text{Poly}} = \\mathcal{L}_{\\text{CE}} + \\epsilon (1 - p_t) $$
    """
    def __init__(self, 
                 epsilon: float = 2.0, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            epsilon (float): The coefficient weighting the first polynomial term. 
                Adjusts the perturbation applied to the standard cross-entropy loss.
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__(reduction=reduction)
        self.epsilon = epsilon

    def _compute_custom_loss(self, base_loss: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        return base_loss + self.epsilon * (1.0 - p_t)


class AsymmetricLoss(torch.nn.Module):
    """
    Computes the Asymmetric Loss (ASL) for binary and multi-class classification tasks.
    
    ASL decouples the decay rates of positive and negative samples, handling 
    imbalances by strictly penalizing hard negatives while maintaining 
    contributions from positive samples. It also implements probability shifting 
    (clipping) for negative samples to filter out easy negatives completely.
    
    Formula:
        $$ \\mathcal{L}_{\\text{ASL}} = - y (1 - p_+)^{\\gamma_+} \\log(p_+) - (1 - y) (1 - p_-)^{\\gamma_-} \\log(p_-) $$
    """
    def __init__(self, 
                 gamma_pos: float = 1.0, 
                 gamma_neg: float = 4.0, 
                 clip: float = 0.05, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            gamma_pos (float): The focusing parameter for positive samples.
            gamma_neg (float): The focusing parameter for negative samples, typically 
                set higher than gamma_pos to aggressively down-weight easy negatives.
            clip (float): Probability margin for shifting negative samples. Probabilities 
                below this threshold are zeroed out (treated as perfectly classified).
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        
        # warn if gamma_neg is less than gamma_pos
        if gamma_neg < gamma_pos:
            _LOGGER.warning(
                f"gamma_neg ({gamma_neg}) is less than gamma_pos ({gamma_pos}). This may lead to suboptimal performance, use with caution."
            )
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        is_binary = logits.ndim == 1 or logits.shape[1] == 1

        if is_binary:
            # Retain original shape, squeezing channel dim if present
            if logits.ndim > 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            
            # Match targets shape if they have an extra channel dim
            if targets.ndim == logits.ndim + 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
                
            targets = targets.float()
            
            probs = torch.sigmoid(logits)
            probs_pos = probs
            probs_neg = 1.0 - probs
            
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
            
            loss_pos = targets * torch.log(probs_pos.clamp(min=1e-8)) * ((1.0 - probs_pos) ** self.gamma_pos)
            loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=1e-8)) * ((1.0 - probs_neg) ** self.gamma_neg)
            
            custom_loss = -(loss_pos + loss_neg)
            
        else:
            # Squeeze targets channel dimension if present for multi-class
            if targets.ndim == logits.ndim and targets.shape[1] == 1:
                targets = targets.squeeze(1)
                
            num_classes = logits.shape[1]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            
            # Permute one-hot target to match logits shape (B, C, ...)
            permute_dims = [0, targets_one_hot.ndim - 1] + list(range(1, targets_one_hot.ndim - 1))
            targets_one_hot = targets_one_hot.permute(*permute_dims).float()
            
            probs = torch.softmax(logits, dim=1)
            
            probs_pos = probs
            probs_neg = 1.0 - probs
            
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
            
            loss_pos = targets_one_hot * torch.log(probs_pos.clamp(min=1e-8)) * ((1.0 - probs_pos) ** self.gamma_pos)
            loss_neg = (1.0 - targets_one_hot) * torch.log(probs_neg.clamp(min=1e-8)) * ((1.0 - probs_neg) ** self.gamma_neg)
            
            custom_loss = -(loss_pos + loss_neg).sum(dim=1)

        loss = _apply_reduction(custom_loss, self.reduction)
        return loss
