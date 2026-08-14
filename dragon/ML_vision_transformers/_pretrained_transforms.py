from pathlib import Path
from typing import Union
from torch import nn

from ..serde import serialize_object_filename
from ..path_manager import sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Pretrained Transforms Saver")


__all__ = [
    "save_pretrained_transforms"
]


def save_pretrained_transforms(model: nn.Module, 
                               output_dir: Union[str, Path], 
                               file_identifier: str):
    """
    Used for wrapper vision models when initialized with pre-trained weights.

    Checks a model for the 'self._pretrained_default_transforms' attribute, if found, serializes the returned **transform object** as a .joblib file.
        
    To use with a Vision Inference Handler, just deserialize the object and pass it to the handler's `transform_source` argument.

    Args:
        model (nn.Module): The model instance to check.
        output_dir (str | Path): The directory where the transform file will be saved.
        file_identifier (str): A string to identify the model, used in the filename.
    """
    output_filename = f"pretrained_model_transformations_{sanitize_filename(file_identifier).replace('.', '_')}.joblib"

    # 1. Check for the "secret attribute"
    if not hasattr(model, '_pretrained_default_transforms'):
        _LOGGER.warning(f"Model of type {type(model).__name__} does not have the required attribute. No transformations saved.")
        return

    # 2. Get the transform object
    try:
        transform_obj = model._pretrained_default_transforms
    except Exception as e:
        _LOGGER.error(f"Error calling the required attribute '_pretrained_default_transforms' on model: {e}")
        return

    # 3. Check if the object is actually there
    if transform_obj is None:
        _LOGGER.warning(f"Model {type(model).__name__} has the required attribute but returned None. No transforms saved.")
        return

    # 4. Serialize and save using serde
    try:
        serialize_object_filename(
            obj=transform_obj,
            save_dir=output_dir,
            filename=output_filename,
            verbose=True,
            raise_on_error=True
        )
        # _LOGGER.info(f"Successfully saved pretrained transforms to '{output_dir}'.")
    except Exception as e:
        _LOGGER.error(f"Failed to serialize transformations: {e}")
        raise
