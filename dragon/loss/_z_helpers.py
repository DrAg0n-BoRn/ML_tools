import torch
import torch.nn.functional as F
from typing import Optional

from .._core import get_logger


_LOGGER = get_logger("Custom Loss")


__all__ = [
    "_apply_reduction",
    "_handle_ignore_index"
]


def _apply_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        _LOGGER.error(f"Unsupported reduction mode: {reduction}")
        raise ValueError()


def _handle_ignore_index(
    probs: torch.Tensor, 
    targets: torch.Tensor, 
    ignore_index: Optional[int], 
    num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streamlines execution by only applying mask overhead if ignore_index is provided.
    Returns the (potentially masked) probabilities and one-hot encoded targets.
    """
    if ignore_index is None:
        if num_classes == 1:
            targets_one_hot = targets.unsqueeze(1).float()
        else:
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            permute_dims = [0, targets_one_hot.ndim - 1] + list(range(1, targets_one_hot.ndim - 1))
            targets_one_hot = targets_one_hot.permute(*permute_dims).float()
        return probs, targets_one_hot

    # Execute masking overhead only when necessary
    valid_mask = (targets != ignore_index).unsqueeze(1)
    safe_targets = torch.where(
        targets == ignore_index, 
        torch.zeros_like(targets), 
        targets
    ).long()

    if num_classes == 1:
        targets_one_hot = safe_targets.unsqueeze(1).float()
    else:
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes)
        permute_dims = [0, targets_one_hot.ndim - 1] + list(range(1, targets_one_hot.ndim - 1))
        targets_one_hot = targets_one_hot.permute(*permute_dims).float()

    probs = probs * valid_mask
    targets_one_hot = targets_one_hot * valid_mask

    return probs, targets_one_hot
