from typing import Union, Dict, Type, Callable, Optional, Any, List, Literal, Tuple
from PIL import ImageOps, Image
from torchvision import transforms
from pathlib import Path
import json
import random

from ._logger import _LOGGER
from ._script_info import _script_info
from ._keys import VisionTransformRecipeKeys
from .path_manager import make_fullpath


__all__ = [
    "TRANSFORM_REGISTRY",
    "ResizeAspectFill",
    "LetterboxResize",
    "HistogramEqualization",
    "RandomHistogramEqualization",
    "create_offline_augmentations"
]

# --- Custom Vision Transform Class ---
class ResizeAspectFill:
    """
    Pre-Transform
    
    Custom transformation to make an image square by padding it to match the
    longest side, preserving the aspect ratio. The image is finally centered.

    Args:
        pad_color (Union[str, int]): Color to use for the padding.
                                     Defaults to "black".
    """
    def __init__(self, pad_color: Union[str, int] = "black") -> None:
        self.pad_color = pad_color
        # Important: Store keyword to allow for re-creation
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


class LetterboxResize:
    """
    Pre-Transform
    
    Resizes an image to fit within a target size (e.g., 640x640) while
    maintaining its aspect ratio. The remaining space is padded to meet
    the target size, and the image is centered.

    Args:
        target_size (Union[int, Tuple[int, int]]): The target (width, height)
            to resize to. If an int, it's used for both width and height.
        pad_color (Union[str, int]): Color to use for the padding.
                                     Defaults to "black".
    
    Note:
        This is extremely common for object detection models that require fixed-size, square inputs but where distorting the
        aspect ratio would harm geometric accuracy.
    """
    def __init__(
        self, 
        target_size: Union[int, Tuple[int, int]], 
        pad_color: Union[str, int] = "black"
    ) -> None:
        
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
            
        self.pad_color = pad_color
        # Store kwargs to allow for re-creation
        self.__setattr__(VisionTransformRecipeKeys.KWARGS, {
            "target_size": target_size, 
            "pad_color": pad_color
        })

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            _LOGGER.error(f"Expected PIL.Image.Image, got {type(image).__name__}")
            raise TypeError()

        w, h = image.size
        tw, th = self.target_size
        
        if (w, h) == (tw, th):
            return image

        # Calculate resize ratio
        r = min(tw / w, th / h)
        
        # New dimensions
        new_w = int(round(w * r))
        new_h = int(round(h * r))

        # Resize
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Calculate padding
        left_padding = (tw - new_w) // 2
        right_padding = tw - new_w - left_padding
        top_padding = (th - new_h) // 2
        bottom_padding = th - new_h - top_padding
        
        padding = (left_padding, top_padding, right_padding, bottom_padding)
        
        return ImageOps.expand(resized_image, padding, fill=self.pad_color)


class HistogramEqualization:
    """
    Augmentation / Pre-Transform
    
    Applies histogram equalization to the image, spreading out the
    most frequent pixel intensity values to improve contrast.

    Note:
        This is useful as a pre-processing step for datasets with
        consistently poor lighting or low contrast (e.g., some medical imaging or security camera footage).
    """
    def __init__(self) -> None:
        # Store kwargs to allow for re-creation
        self.__setattr__(VisionTransformRecipeKeys.KWARGS, {})

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            _LOGGER.error(f"Expected PIL.Image.Image, got {type(image).__name__}")
            raise TypeError()
            
        return ImageOps.equalize(image)


class RandomHistogramEqualization:
    """
    Augmentation
    
    Randomly applies histogram equalization to the image with a
    given probability `p`.

    Args:
        p (float): The probability of applying the equalization.
                   Defaults to 0.5.

    Note:
        This is useful as a data augmentation to make the model robust
        to a wide variety of lighting conditions and contrast levels,
        without forcing all images to be equalized.
    """
    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability 'p' must be between 0.0 and 1.0, got {p}")
        self.p = p
        # Store kwargs to allow for re-creation
        self.__setattr__(VisionTransformRecipeKeys.KWARGS, {"p": p})

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            _LOGGER.error(f"Expected PIL.Image.Image, got {type(image).__name__}")
            raise TypeError()

        if random.random() < self.p:
            return ImageOps.equalize(image)
        
        return image


#############################################################
#NOTE: Add custom transforms.
TRANSFORM_REGISTRY: Dict[str, Type[Callable]] = {
    "ResizeAspectFill": ResizeAspectFill, 
    "LetterboxResize": LetterboxResize,
    "HistogramEqualization": HistogramEqualization,
    "RandomHistogramEqualization": RandomHistogramEqualization,
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
