from typing import Union, Callable, Optional, Any, Literal
from PIL import Image
from torchvision import transforms
from pathlib import Path

from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys
from ..path_manager import make_fullpath

from ._core_transforms import TRANSFORM_REGISTRY


_LOGGER = get_logger("Offline Augmentation")


__all__ = [
    "create_offline_augmentations"
]

def create_offline_augmentations(
    input_directory: Union[str, Path],
    output_directory: Union[str, Path],
    results_per_image: int,
    recipe: Optional[dict[str, Any]] = None,
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


def _build_transform_from_recipe(recipe: dict[str, Any]) -> transforms.Compose:
    """Internal helper to build a transform pipeline from a recipe dict."""
    pipeline_steps: list[Callable] = []
    
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

