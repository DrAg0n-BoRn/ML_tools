from typing import Union, Dict, Type, Callable
from PIL import ImageOps, Image

from ._logger import _LOGGER
from ._script_info import _script_info
from .keys import VisionTransformRecipeKeys


__all__ = [
    "TRANSFORM_REGISTRY",
    "ResizeAspectFill"
]

# --- Custom Vision Transform Class ---
class ResizeAspectFill:
    """
    Custom transformation to make an image square by padding it to match the
    longest side, preserving the aspect ratio. The image is finally centered.

    Args:
        pad_color (Union[str, int]): Color to use for the padding.
                                     Defaults to "black".
    """
    def __init__(self, pad_color: Union[str, int] = "black") -> None:
        self.pad_color = pad_color
        # Store kwargs to allow for recreation
        self.__setattr__(VisionTransformRecipeKeys.KWARGS, {"pad_color": pad_color})
        # self._kwargs = {"pad_color": pad_color}

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            _LOGGER.error(f"Expected PIL.Image.Image, got {type(image).__name__}")
            raise TypeError()

        w, h = image.size
        if w == h:
            return image

        # Determine padding to center the image
        if w > h:
            top_padding = (w - h) // 2
            bottom_padding = w - h - top_padding
            padding = (0, top_padding, 0, bottom_padding)
        else: # h > w
            left_padding = (h - w) // 2
            right_padding = h - w - left_padding
            padding = (left_padding, 0, right_padding, 0)

        return ImageOps.expand(image, padding, fill=self.pad_color)
    

#NOTE: Add custom transforms here.
TRANSFORM_REGISTRY: Dict[str, Type[Callable]] = {
    "ResizeAspectFill": ResizeAspectFill,
}

def info():
    _script_info(__all__)
