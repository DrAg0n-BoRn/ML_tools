import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union

from ..path_manager import make_fullpath

from .._core import get_logger


_LOGGER = get_logger("Mask Ops")


__all__ = [
    "count_mask_pixels_by_class",
]


def count_mask_pixels_by_class(
        directory: Union[str, Path], 
        class_map: dict[str, int],
        verbose: int = 2
    ) -> dict[str, dict[str, int]]:
    """
    Calculates the pixel distribution for specific classes across a directory of segmentation masks.
    
    This function scans the provided directory for valid lossless image formats (PNG, TIF, TIFF, BMP),
    opens each image, and ensures it is in a valid 1-channel mask format ('L' for Luminance/Grayscale 
    or 'P' for Palette/Indexed). It then counts the exact number of pixels belonging to each class 
    defined in the `class_map` and maps these counts to the image's filename.

    Files with invalid image modes (e.g., RGB) or unreadable files are logged.
    
    Args:
        directory (Union[str, Path]): The path to the directory containing the segmentation mask images.
        class_map (dict[str, int]): A dictionary mapping string class names to their exact integer 
            pixel values (e.g., {"Background": 0, "Road": 1, "Car": 2}).
            
    Returns:
        dict[str, dict[str, int]]: A nested dictionary where the top-level keys are the image 
            filenames (without extensions) and the values are dictionaries containing the pixel 
            counts for each class. 
            Example: 
            {
                "image_001": {"Background": 50000, "Road": 15000, "Car": 536},
                "image_002": {"Background": 65536, "Road": 0, "Car": 0}
            }
    """
    dir_path = make_fullpath(directory, enforce="directory", make=False)
    pixel_counts = {}
    invalid_files = []

    for file_path in dir_path.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in {'.png', '.tif', '.tiff', '.bmp'}:
            continue
        
        try:
            with Image.open(file_path) as img:
                # Explicitly enforce L (Grayscale) or P (Palette) modes
                if img.mode not in {'L', 'P'}:
                    if verbose >= 3:
                        _LOGGER.warning(f"File '{file_path.name}' is not a valid mask format (Mode: {img.mode}). Expected 'L' or 'P'. Skipping.")
                    invalid_files.append(file_path.name)
                    continue
                
                # For both 'P' and 'L', np.array safely extracts the raw 8-bit integer indices
                mask_array = np.array(img)
    
        except Exception as e:
            if verbose >= 3:
                _LOGGER.warning(f"Failed to read '{file_path.name}': {e}")
            invalid_files.append(file_path.name)
            continue

        class_counts = {}
        for class_name, class_value in class_map.items():
            class_counts[class_name] = int(np.sum(mask_array == class_value))
            
        pixel_counts[file_path.stem] = class_counts
    
    if verbose >= 2:
        _LOGGER.info(f"Processed {len(pixel_counts)} valid mask files in '{dir_path}'.")
    
    if invalid_files and verbose >= 1:
        _LOGGER.warning(f"Skipped {len(invalid_files)} invalid files: {invalid_files}")
    
    return pixel_counts
