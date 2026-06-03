from typing import Union, Literal, Optional
from pathlib import Path
from PIL import Image

from .._core import get_logger
from ..path_manager import make_fullpath


_LOGGER = get_logger("Vision Tiling")


__all__ = [
    "make_tiled_dataset",
]


VALID_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def make_tiled_dataset(
    input_dir: Union[str, Path], 
    mask_dir: Optional[Union[str, Path]] = None, 
    window_size: int = 512, 
    ratio_strategy: Literal["pad-white", "pad-black", "shift"] = "shift", 
    stride: float = 0.8,
    drop_empty_masks_by_value: Optional[Union[int, tuple]] = None,
) -> None:
    """
    Slices high-resolution images (and optional paired masks) into smaller PNG images, overlapping 
    square tiles suitable for training deep learning models.
    
    The function creates a new directory named `<input_dir>_tiled` at the same level 
    as the input directory, containing 'images' and 'masks' subdirectories if masks are provided. Mask tiling 
    is strictly synchronized with image tiling to preserve pixel-perfect ground truth alignment.
    
    Args:
        input_dir (str | Path): Path to the directory containing source images 
            (supported formats: .tif, .tiff, .png, .jpg, .jpeg).
        mask_dir (str | Path | None): Optional path to the directory containing corresponding ground-truth masks. Filenames (stem) must match the source 
            images. If None, only main images will be processed.
        window_size (int): The width and height of the square output tiles in pixels.
        ratio_strategy (Literal["pad-white", "pad-black", "shift"]): Strategy 
            for handling edge tiles when dimensions are not perfectly divisible by the window size. 
            - "shift": Pulls the final sliding window backward to perfectly align with the image boundary, avoiding artificial padding.
            - "pad-white": Fills out-of-bounds areas with 255 (white).
            - "pad-black": Fills out-of-bounds areas with 0 (black).
        stride (float): The sliding window step size, expressed as a fraction 
            of the window_size (e.g., 0.8 means an 80% step, yielding 20% overlap). 
        drop_empty_masks_by_value (int | tuple | None): If set, any tile whose corresponding mask patch consists entirely of this value (or tuple of values for multi-channel masks) 
            will be considered "empty" and skipped during saving. This is useful for excluding tiles that contain no relevant features (e.g., all background). 
            For single-channel masks, provide an int (e.g., 0). For multi-channel masks, provide a tuple matching the number of channels (e.g., (0, 0, 0) for RGB).
    """
    input_path = make_fullpath(input_dir, make=False, enforce="directory")
    output_dir = input_path.parent / f"{input_path.name}_tiled"
    
    image_out_dir = output_dir / "images" if mask_dir else output_dir
    image_out_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = make_fullpath(mask_dir, make=False, enforce="directory") if mask_dir else None
    mask_out_dir = None
    if mask_path:
        mask_out_dir = output_dir / "masks"
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
    
    if not image_files:
        _LOGGER.error(f"No valid images found in {input_dir}")
        raise FileNotFoundError()

    step_size = max(1, int(window_size * stride))
    pad_color = 255 if ratio_strategy == "pad-white" else 0
    
    skipped_empty_tiles = 0

    for img_file in image_files:
        img = Image.open(img_file)
        width, height = img.size
        
        
        # Validate window size against image dimensions
        if width < window_size or height < window_size:
            _LOGGER.warning(f"Image {img_file.name} ({width}x{height}) is smaller than window_size {window_size}. Skipping.")
            img.close()
            continue
        
        # check for corresponding mask if mask_dir is provided
        mask = None
        if mask_path:
            mask_file_candidates = [mask_path / f"{img_file.stem}{ext}" for ext in VALID_EXTENSIONS]
            mask_file = next((m for m in mask_file_candidates if m.exists()), None)
            
            if not mask_file:
                _LOGGER.warning(f"No corresponding mask found for {img_file.name}, skipping image.")
                img.close()
                continue
            mask = Image.open(mask_file)
            
            # validate dimensions
            if mask.size != img.size:
                _LOGGER.warning(f"Dimension mismatch between image {img_file.name} ({img.size}) and its mask {mask_file.name} ({mask.size}), skipping image.")
                img.close()
                mask.close()
                continue

        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                
                crop_x, crop_y = x, y
                
                if ratio_strategy == "shift":
                    if crop_x + window_size > width:
                        crop_x = max(0, width - window_size)
                    if crop_y + window_size > height:
                        crop_y = max(0, height - window_size)
                        
                box = (crop_x, crop_y, min(crop_x + window_size, width), min(crop_y + window_size, height))
                
                img_patch = img.crop(box)
                mask_patch = mask.crop(box) if mask else None

                if ratio_strategy in ["pad-white", "pad-black"] and (img_patch.size[0] < window_size or img_patch.size[1] < window_size):
                    new_img_patch = Image.new(img.mode, (window_size, window_size), color=pad_color)
                    new_img_patch.paste(img_patch, (0, 0))
                    img_patch = new_img_patch
                    
                    if mask and mask_patch:
                        new_mask_patch = Image.new(mask.mode, (window_size, window_size), color=0)
                        new_mask_patch.paste(mask_patch, (0, 0))
                        mask_patch = new_mask_patch

                patch_name = f"{img_file.stem}_y{crop_y}_x{crop_x}"
                
                # --- check for empty masks ---
                save_patch = True
                if drop_empty_masks_by_value is not None and mask_patch is not None:
                    extrema = mask_patch.getextrema()
                    mask_background_value = drop_empty_masks_by_value
                    
                    if isinstance(extrema[0], tuple):  # Multi-channel (e.g., RGB)
                        # Broadcast int to tuple if needed, e.g., 0 becomes (0, 0, 0)
                        bg_val = mask_background_value if isinstance(mask_background_value, tuple) else (mask_background_value,) * len(extrema)
                        # It is empty ONLY if all channels consist entirely of the background value
                        is_empty = all(b_min == b_max == bg for (b_min, b_max), bg in zip(extrema, bg_val)) # type: ignore
                        save_patch = not is_empty
                    else:  # Single-channel (e.g., Grayscale/L)
                        # It is empty ONLY if the min and max are exactly the background value
                        is_empty = (extrema[0] == extrema[1] == mask_background_value)
                        save_patch = not is_empty
                
                if save_patch:
                    img_patch.save(image_out_dir / f"{patch_name}.png", format="png")
                    
                    if mask_patch and mask_out_dir:
                        mask_patch.save(mask_out_dir / f"{patch_name}.png", format="png")
                else:
                    skipped_empty_tiles += 1
                    
                if ratio_strategy == "shift" and crop_x == width - window_size:
                    break
            if ratio_strategy == "shift" and crop_y == height - window_size:
                break
                
        img.close()
        if mask:
            mask.close()
            
    if skipped_empty_tiles > 0:
        _LOGGER.info(f"Total empty tiles skipped: {skipped_empty_tiles}")
    
    _LOGGER.info(f"Tiling completed. Output saved to {output_dir.name}")
