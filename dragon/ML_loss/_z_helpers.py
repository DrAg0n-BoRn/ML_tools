import torch

from .._core import get_logger


_LOGGER = get_logger("Custom Loss")


__all__ = [
    "_apply_reduction",
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
