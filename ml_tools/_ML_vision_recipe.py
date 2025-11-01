import json
import torch
from torchvision import transforms
from typing import Dict, Any, List, Callable, Union
from pathlib import Path

from .ML_vision_transformers import TRANSFORM_REGISTRY
from ._logger import _LOGGER
from .keys import VisionTransformRecipeKeys
from .path_manager import make_fullpath


def save_recipe(recipe: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a transform recipe dictionary to a JSON file.

    Args:
        recipe (Dict[str, Any]): The recipe dictionary to save.
        filepath (str): The path to the output .json file.
    """
    final_filepath = filepath.with_suffix(".json")
    
    try:
        with open(final_filepath, 'w') as f:
            json.dump(recipe, f, indent=4)
        _LOGGER.info(f"Transform recipe saved as '{final_filepath.name}'.")
    except Exception as e:
        _LOGGER.error(f"Failed to save recipe to '{final_filepath}': {e}")
        raise


def load_recipe_and_build_transform(filepath: Union[str,Path]) -> transforms.Compose:
    """
    Loads a transform recipe from a .json file and reconstructs the
    torchvision.transforms.Compose pipeline.

    Args:
        filepath (str): Path to the saved transform recipe .json file.

    Returns:
        transforms.Compose: The reconstructed transformation pipeline.
        
    Raises:
        ValueError: If a transform name in the recipe is not found in
                    torchvision.transforms or the custom TRANSFORM_REGISTRY.
    """
    # validate filepath
    final_filepath = make_fullpath(filepath, enforce="file")
    
    try:
        with open(final_filepath, 'r') as f:
            recipe = json.load(f)
    except Exception as e:
        _LOGGER.error(f"Failed to load recipe from '{final_filepath}': {e}")
        raise
        
    pipeline_steps: List[Callable] = []
    
    if VisionTransformRecipeKeys.PIPELINE not in recipe:
        _LOGGER.error("Recipe file is invalid: missing 'pipeline' key.")
        raise ValueError("Invalid recipe format.")

    for step in recipe[VisionTransformRecipeKeys.PIPELINE]:
        t_name = step[VisionTransformRecipeKeys.NAME]
        t_kwargs = step[VisionTransformRecipeKeys.KWARGS]
        
        transform_class: Any = None

        # 1. Check standard torchvision transforms
        if hasattr(transforms, t_name):
            transform_class = getattr(transforms, t_name)
        # 2. Check custom transforms
        elif t_name in TRANSFORM_REGISTRY:
            transform_class = TRANSFORM_REGISTRY[t_name]
        # 3. Not found
        else:
            _LOGGER.error(f"Unknown transform '{t_name}' in recipe. Not found in torchvision.transforms or TRANSFORM_REGISTRY.")
            raise ValueError(f"Unknown transform name: {t_name}")
            
        # Instantiate the transform
        try:
            pipeline_steps.append(transform_class(**t_kwargs))
        except Exception as e:
            _LOGGER.error(f"Failed to instantiate transform '{t_name}' with kwargs {t_kwargs}: {e}")
            raise
            
    _LOGGER.info(f"Successfully loaded and built transform pipeline from '{final_filepath.name}'.")
    return transforms.Compose(pipeline_steps)
