import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Literal

from .._core import get_logger

from ._z_helpers import _apply_reduction
from ._base_dice_loss import _BaseDiceLoss


_LOGGER = get_logger("Segmentation Loss")


__all__ = [
    "DiceLoss",
    "GeneralizedDiceLoss",
    "SegmentationFocalLoss",
    "DiceFocalLoss",
]


class DiceLoss(_BaseDiceLoss):
    """
    Computes the Dice Similarity Coefficient loss for multi-class segmentation.
    
    The Dice Loss evaluates the spatial overlap between the predicted probabilities 
    and the ground truth one-hot encoded masks. It is highly robust to class 
    imbalance as it considers the intersection over the union of regions.
    
    Formula:
        $$ \\mathcal{L}_{\\text{Dice}} = 1 - \\frac{2 |X \\cap Y| + \\text{smooth}}{|X| + |Y| + \\text{smooth}} $$
    """
    def __init__(
        self,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            smooth (float): A small constant added to the numerator and denominator 
                to avoid division by zero and stabilize gradients. Defaults to 1e-6.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0). 
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__(smooth=smooth, include_background=include_background, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection, cardinality, _, _ = self._prepare_inputs(logits, targets)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        loss = _apply_reduction(dice_loss, self.reduction)
        
        return loss


class GeneralizedDiceLoss(_BaseDiceLoss):
    """
    Computes the Generalized Dice Loss for multi-class segmentation.
    
    This variation dynamically assigns class weights inversely proportional to the 
    squared area of the class in the ground truth. This ensures that small objects 
    contribute significantly to the loss, preventing them from being overwhelmed 
    by larger background structures.
    
    Formula:
        $$ \\mathcal{L}_{\\text{GDL}} = 1 - 2 \\frac{\\sum_{c=1}^C w_c \\sum_{n} p_{nc} g_{nc}}{\\sum_{c=1}^C w_c \\sum_{n} (p_{nc} + g_{nc})} $$
    where:
        $$ w_c = \\frac{1}{(\\sum_{n} g_{nc} + \\epsilon)^2} $$
    """
    def __init__(
        self,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            smooth (float): A small constant added to the numerator and denominator 
                to avoid division by zero and stabilize gradients. Defaults to 1e-6.
            include_background (bool): If False, the loss is computed only for the 
                foreground classes (assumes background is at channel index 0). 
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__(smooth=smooth, include_background=include_background, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection, cardinality, targets_one_hot, spatial_dims = self._prepare_inputs(logits, targets)

        volumes = torch.sum(targets_one_hot, dim=spatial_dims)
        weights = 1.0 / (volumes ** 2 + self.smooth)

        numerator = torch.sum(weights * intersection, dim=1)
        denominator = torch.sum(weights * cardinality, dim=1)

        gdl_score = (2.0 * numerator + self.smooth) / (denominator + self.smooth)
        gdl_loss = 1.0 - gdl_score

        loss = _apply_reduction(gdl_loss, self.reduction)
        return loss


class SegmentationFocalLoss(nn.Module):
    """
    Computes the Focal Loss for multi-class image segmentation.
    
    Focal Loss dynamically scales the cross-entropy loss based on the prediction 
    confidence. It down-weights the contribution of easily classified examples 
    (often background) and focuses the model on hard, misclassified examples.
    
    Formula:
        $$ \\mathcal{L}_{\\text{Focal}} = -\\alpha (1 - p_t)^\\gamma \\log(p_t) $$
    """
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            alpha (Optional[Union[float, torch.Tensor]]): A weighting factor to 
                address class frequency imbalances. Can be a scalar for uniform 
                scaling, or a 1D tensor of length `num_classes` containing 
                class weights.
            gamma (float): The focusing parameter that dictates the rate at which 
                easy examples are down-weighted. A value of 0 is equivalent to 
                standard Cross-Entropy.
            ignore_index (int): Specifies a target value that is ignored and does 
                not contribute to the input gradient. Defaults to -100 (PyTorch default).
            reduction (str): Specifies the reduction to apply to the output:
                - 'none': No reduction, returns per-sample loss.
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha if (self.alpha is not None and self.alpha.numel() > 1) else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None and self.alpha.numel() == 1:
            focal_loss = self.alpha * focal_loss

        loss = _apply_reduction(focal_loss, self.reduction)
        return loss


class DiceFocalLoss(nn.Module):
    """
    Computes a composite loss by combining Dice Loss and Focal Loss for imbalanced multi-class image segmentation.
    
    This function leverages both region-based (Dice) and voxel-based (Focal) 
    optimizations. The Dice component ensures robust handling of imbalanced object 
    sizes and spatial overlap, while the Focal component aggressively penalizes 
    hard-to-classify edge pixels and boundaries.
    
    Formula:
        $$ \\mathcal{L}_{\\text{Total}} = w_{\\text{dice}} \\mathcal{L}_{\\text{Dice}} + w_{\\text{focal}} \\mathcal{L}_{\\text{Focal}} $$
    """
    def __init__(
        self,
        weight_dice: float = 0.5,
        weight_focal: float = 0.5,
        dice_type: Literal["standard", "generalized"] = "generalized",
        focal_alpha: Optional[Union[float, torch.Tensor]] = None,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-6,
        include_background: bool = True,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """
        Args:
            weight_dice (float): The multiplier weight for the Dice Loss component. 
            weight_focal (float): The multiplier weight for the Focal Loss component.
            dice_type (str): Specifies which Dice Loss variant to use:
                - 'standard': Uses the standard Dice Loss.
                - 'generalized': Uses the Generalized Dice Loss, which applies class weights inversely proportional to the squared volume of each class.
            focal_alpha (Optional[Union[float, torch.Tensor]]): A weighting factor to 
                address class frequency imbalances. Can be a scalar for uniform 
                scaling, or a 1D tensor of length `num_classes` containing 
                class weights.
            focal_gamma (float): The focusing parameter that dictates the rate at which 
                easy examples are down-weighted. A value of 0 is equivalent to 
                standard Cross-Entropy.
            dice_smooth (float): The smoothing constant passed to the underlying 
                DiceLoss. Defaults to 1e-6.
            include_background (bool): Whether to include the background class in 
                the Dice Loss computation.
            ignore_index (int): A target index to ignore for the Focal Loss gradient 
                computation. Defaults to -100 (PyTorch default).
            reduction (str): The reduction method applied to the combined loss:
                - 'mean': Returns the mean loss across the batch.
                - 'sum': Returns the sum of losses across the batch.
        """
        super().__init__()
        
        # validate weights
        if weight_dice <= 0 or weight_focal <= 0:
            _LOGGER.error("Weights for Dice and Focal Loss must be positive.")
            raise ValueError()
        
        # ensure weights sum to 1
        if not (abs(weight_dice + weight_focal - 1.0) < 1e-6):
            _LOGGER.error(f"weight_dice and weight_focal should sum to 1. Got weight_dice={weight_dice}, weight_focal={weight_focal}")
            raise ValueError()
        
        # Validate reduction for this special case combining two losses and want to ensure consistent behavior
        if reduction not in ["mean", "sum"]:
            _LOGGER.error(f"Unsupported reduction type: {reduction}. Must be 'mean' or 'sum'.")
            raise ValueError()
        
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        if dice_type == "standard":
            self.dice = DiceLoss(
                smooth=dice_smooth,
                include_background=include_background,
                reduction=reduction,
            )
        elif dice_type == "generalized":
            self.dice = GeneralizedDiceLoss(
                smooth=dice_smooth,
                include_background=include_background,
                reduction=reduction,
            )
        else:
            _LOGGER.error(f"Unsupported dice_type: {dice_type}.")
            raise ValueError()
        
        self.focal = SegmentationFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice(logits, targets)
        focal = self.focal(logits, targets)
        return self.weight_dice * dice + self.weight_focal * focal
