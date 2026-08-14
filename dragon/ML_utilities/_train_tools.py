import torch

from .._core import get_logger


_LOGGER = get_logger("ML Utilities")


__all__ = [
    "validate_torch_device"
]


def validate_torch_device(device: str):
    """
    Validates the specified PyTorch device string and returns a valid torch.device object.
    
    Args:
        device (str): The device string to validate (e.g., "cuda:0", "mps", "cpu").
    """
    device_lower = device.lower()
    if "cuda" in device_lower and not torch.cuda.is_available():
        _LOGGER.warning("CUDA not available, switching to CPU.")
        device = "cpu"
    elif device_lower == "mps" and not torch.backends.mps.is_available():
        _LOGGER.warning("Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
        device = "cpu"
    elif device_lower == "cpu":
        pass  # CPU is always available
    else:
        # For any other device string, we will attempt to create a torch.device and catch errors
        try:
            torch.device(device)
        except Exception as e:
            _LOGGER.error(f"Invalid device string '{device}': {e}. Defaulting to CPU.")
            device = "cpu"
    
    return torch.device(device)

