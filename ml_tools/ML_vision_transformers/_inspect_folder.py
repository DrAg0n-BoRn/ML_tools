from typing import Union
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from ..IO_tools import custom_logger
from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("Vision Inspection")


__all__ = [
    "inspect_folder"
]


def inspect_folder(path: Union[str, Path]) -> None:
    """
    Logs a report of the types, sizes, and channels of image files
    found in the directory and its subdirectories.
    
    A JSON log is also saved at the same level as the root directory inspected, 
    with detailed information about the inspection results, including any non-image files or permission issues encountered.
    
    This is a utility method to help diagnose potential dataset
    issues (e.g., mixed image modes, corrupted files).

    Args:
        path (str, Path): The directory path to inspect.
    """
    path_obj = make_fullpath(path, make=False, enforce="directory")

    non_image_files = set()
    permission_denied_files = set()
    img_types = set()
    img_sizes = set()
    img_channels = set()
    img_counter = 0
    non_image_counter = 0

    _LOGGER.info(f"Inspecting folder: {path_obj}...")
    
    # Use rglob to recursively find all files
    for filepath in path_obj.rglob('*'):
        if filepath.is_file():
            try:
                # Using PIL to open is a reliable check. It's lazy and only reads the header.
                with Image.open(filepath) as img:
                    img_types.add(img.format)
                    img_sizes.add(img.size)
                    img_channels.update(img.getbands())
                    img_counter += 1
            except PermissionError:
                _LOGGER.warning(f"Permission denied: {filepath.name}")
                permission_denied_files.add(str(filepath))
            except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError):
                non_image_files.add(str(filepath))
                non_image_counter += 1

    if non_image_counter > 0:
        # Show a sample of non-image files to avoid flooding the logs
        sample = list(non_image_files)[:5]
        _LOGGER.warning(
            f"Found {non_image_counter} non-image or corrupted files. "
            f"Samples ignored: {sample}"
            f"{' ...' if len(non_image_files) > 5 else ''}"
        )

    report = (
        f"\n--- Inspection Report for '{path_obj.name}' ---\n"
        f"Total valid images found: {img_counter}\n"
        f"Image formats: {img_types or 'None'}\n"
        f"Image sizes (WxH): {img_sizes or 'None'}\n"
        f"Image channels (bands): {img_channels or 'None'}\n"
        f"--------------------------------------"
    )
    
    _LOGGER.info(report)

    # Compile all details into a dictionary for JSON logging
    log_data = {
        "inspected_directory": str(path_obj),
        "summary": {
            "total_valid_images": img_counter,
            "total_invalid_files": non_image_counter,
            "total_permission_denied": len(permission_denied_files)
        },
        "image_details": {
            "formats": list(img_types),
            "sizes_w_h": [list(size) for size in img_sizes],
            "channels": list(img_channels)
        },
        "warnings_and_errors": {
            "corrupted_or_non_image_files": list(non_image_files),
            "permission_denied_files": list(permission_denied_files)
        }
    }

    # Save log at the same level as the root directory inspected
    save_dir = path_obj.parent
    log_name = f"inspection_report_{path_obj.name}"
    
    custom_logger(
        data=log_data,
        save_directory=save_dir,
        log_name=log_name,
        dict_as='json'
    )
