from typing import Union, Dict, Type, Callable, Optional, Any, List, Literal
from PIL import ImageOps, Image
from torchvision import transforms
from pathlib import Path
import json

from ._logger import _LOGGER
from ._script_info import _script_info
from ._keys import VisionTransformRecipeKeys
from .path_manager import make_fullpath


__all__ = [
    "TRANSFORM_REGISTRY",
    "ResizeAspectFill",
    "create_offline_augmentations"
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
        # Store kwargs to allow for re-creation
        self.__setattr__(VisionTransformRecipeKeys.KWARGS, {"pad_color": pad_color})

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


#############################################################
#NOTE: Add custom transforms.
TRANSFORM_REGISTRY: Dict[str, Type[Callable]] = {
    "ResizeAspectFill": ResizeAspectFill,
}
#############################################################


def create_offline_augmentations(
    input_directory: Union[str, Path],
    output_directory: Union[str, Path],
    results_per_image: int,
    recipe: Optional[Dict[str, Any]] = None,
    save_format: Literal["WEBP", "JPEG", "PNG", "BMP", "TIF"] = "WEBP",
    save_quality: int = 80
) -> None:
    """
    Reads all valid images from an input directory, applies augmentations,
    and saves the new images to an output directory (offline augmentation).

    Skips subdirectories in the input path.

    Args:
        input_directory (Union[str, Path]): Path to the directory of source images.
        output_directory (Union[str, Path]): Path to save the augmented images.
        results_per_image (int): The number of augmented versions to create
                                 for each source image.
        recipe (Optional[Dict[str, Any]]): A transform recipe dictionary. If None,
                                           a default set of strong, random
                                           augmentations will be used.
        save_format (str): The format to save images (e.g., "WEBP", "JPEG", "PNG").
                           Defaults to "WEBP" for good compression.
        save_quality (int): The quality for lossy formats (1-100). Defaults to 80.
    """
    VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    
    # --- 1. Validate Paths ---
    in_path = make_fullpath(input_directory, enforce="directory")
    out_path = make_fullpath(output_directory, make=True, enforce="directory")
    
    _LOGGER.info(f"Starting offline augmentation:\n\tInput: {in_path}\n\tOutput: {out_path}")

    # --- 2. Find Images ---
    image_files = [
        f for f in in_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMG_EXTENSIONS
    ]
    
    if not image_files:
        _LOGGER.warning(f"No valid image files found in {in_path}.")
        return

    _LOGGER.info(f"Found {len(image_files)} images to process.")

    # --- 3. Define Transform Pipeline ---
    transform_pipeline: transforms.Compose
    
    if recipe:
        _LOGGER.info("Building transformations from provided recipe.")
        try:
            transform_pipeline = _build_transform_from_recipe(recipe)
        except Exception as e:
            _LOGGER.error(f"Failed to build transform from recipe: {e}")
            return
    else:
        _LOGGER.info("No recipe provided. Using default random augmentation pipeline.")
        # Default "random" pipeline
        transform_pipeline = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3)
        ])

    # --- 4. Process Images ---
    total_saved = 0
    format_upper = save_format.upper()
    
    for img_path in image_files:
        _LOGGER.debug(f"Processing {img_path.name}...")
        try:
            original_image = Image.open(img_path).convert("RGB")
            
            for i in range(results_per_image):
                new_stem = f"{img_path.stem}_aug_{i+1:03d}"
                output_path = out_path / f"{new_stem}.{format_upper.lower()}"
                
                # Apply transform
                transformed_image = transform_pipeline(original_image)
                
                # Save
                transformed_image.save(
                    output_path, 
                    format=format_upper, 
                    quality=save_quality,
                    optimize=True # Add optimize flag
                )
                total_saved += 1
                
        except Exception as e:
            _LOGGER.warning(f"Failed to process or save augmentations for {img_path.name}: {e}")

    _LOGGER.info(f"Offline augmentation complete. Saved {total_saved} new images.")


def _build_transform_from_recipe(recipe: Dict[str, Any]) -> transforms.Compose:
    """Internal helper to build a transform pipeline from a recipe dict."""
    pipeline_steps: List[Callable] = []
    
    if VisionTransformRecipeKeys.PIPELINE not in recipe:
        _LOGGER.error("Recipe dict is invalid: missing 'pipeline' key.")
        raise ValueError("Invalid recipe format.")

    for step in recipe[VisionTransformRecipeKeys.PIPELINE]:
        t_name = step.get(VisionTransformRecipeKeys.NAME)
        t_kwargs = step.get(VisionTransformRecipeKeys.KWARGS, {})
        
        if not t_name:
            _LOGGER.error(f"Invalid transform step, missing 'name': {step}")
            continue
        
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
            
    return transforms.Compose(pipeline_steps)


def _save_recipe(recipe: Dict[str, Any], filepath: Path) -> None:
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


def _load_recipe_and_build_transform(filepath: Union[str,Path]) -> transforms.Compose:
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


def info():
    _script_info(__all__)
