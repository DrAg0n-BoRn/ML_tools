import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .._core import get_logger

from ._z_helpers import _apply_reduction


_LOGGER = get_logger("Regression Loss")


__all__ = [
    "LogCoshLoss",
    "QuantileLoss",
    "WingLoss",
]


class LogCoshLoss(nn.Module):
    """
    Computes the Log-Cosh Loss for regression tasks.
    
    Log-Cosh is approximately equal to $(x^2)/2$ for small $x$ and to $|x| - \\log(2)$ 
    for large $x$. This means it works like MSE, but is highly robust to outliers 
    like MAE. It is twice continuously differentiable everywhere.
    
    Formula:
        $$ \\mathcal{L} = \\log(\\cosh(\\hat{y} - y)) $$
    """
    def __init__(self, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = logits - targets
        # Mathematically equivalent to log(cosh(x)) but numerically stable
        loss = diff + F.softplus(-2.0 * diff) - math.log(2.0)
        return _apply_reduction(loss, self.reduction)


class QuantileLoss(nn.Module):
    """
    Computes the Quantile (Pinball) Loss for regression tasks.
    
    Instead of estimating the mean, this loss function estimates a specific 
    quantile (e.g., the median if tau=0.5, or the 90th percentile if tau=0.9).
    
    Formula:
        $$ \\mathcal{L} = \\max(\\tau(y - \\hat{y}), (\\tau - 1)(y - \\hat{y})) $$
    """
    def __init__(self, 
                 tau: float = 0.5, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            tau (float): The target quantile between 0 and 1.
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        if not (0.0 < tau < 1.0):
            _LOGGER.error(f"Quantile tau must be between 0 and 1, got {tau}.")
            raise ValueError()
            
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = targets - logits
        loss = torch.max(self.tau * diff, (self.tau - 1.0) * diff)
        return _apply_reduction(loss, self.reduction)


class WingLoss(nn.Module):
    """
    Computes the Wing Loss for regression tasks requiring high precision.
    
    Wing Loss behaves like a logarithmic function for small errors (amplifying 
    gradients to encourage precise refinement) and like L1 loss for large errors 
    (providing robustness against outliers).
    
    Formula:
        $$ \\mathcal{L} = \\begin{cases} w \\ln(1 + |x|/\\epsilon) & \\text{if } |x| < w \\ |x| - C & \\text{otherwise} \\end{cases} $$
    """
    def __init__(self, 
                 w: float = 10.0, 
                 epsilon: float = 2.0, 
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            w (float): The threshold that separates the linear and logarithmic parts. Rule of thumb: set w to the maximum expected error.
            epsilon (float): Controls the curvature of the logarithmic part.
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.reduction = reduction
        # C is a constant that smoothly links the linear and non-linear parts
        self.c = self.w - self.w * math.log(1.0 + self.w / self.epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(logits - targets)
        
        log_loss = self.w * torch.log(1.0 + abs_diff / self.epsilon)
        l1_loss = abs_diff - self.c
        
        loss = torch.where(abs_diff < self.w, log_loss, l1_loss)
        return _apply_reduction(loss, self.reduction)
