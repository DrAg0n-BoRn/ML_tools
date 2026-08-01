import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Any
from PIL import Image
from abc import abstractmethod

from ..ML_vision_transformers._core_transforms import _load_recipe_and_build_transform

from .._core import get_logger
from ..keys._keys import VisionKeys
from ..path_manager import make_fullpath

from ..ML_inference._base_inference import _CoreInferenceHandler, _ClassificationMixin


_LOGGER = get_logger("Vision Inference")


__all__ = [
    "_BaseVisionInferenceHandler"
]


_VISION_PALETTE_50 = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255),
    (100, 255, 255), (150, 50, 50), (50, 150, 50), (50, 50, 150), (150, 150, 50),
    (150, 50, 150), (50, 150, 150), (200, 100, 50), (50, 200, 100), (100, 50, 200),
    (200, 50, 100), (100, 200, 50), (50, 100, 200), (255, 150, 0), (0, 255, 150),
    (150, 0, 255), (255, 0, 150), (150, 255, 0), (0, 150, 255), (100, 0, 0),
    (0, 100, 0), (0, 0, 100), (100, 100, 0), (100, 0, 100), (0, 100, 100)
]


class _BaseVisionInferenceHandler(_CoreInferenceHandler, _ClassificationMixin):
    """
    Abstract base class for PyTorch vision inference handlers.
    Manages image transformations, directory loading, and file predictions.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 task: Optional[str] = None,
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        
        # 1. Initialize Universal Core
        _CoreInferenceHandler.__init__(self, model=model, state_dict=state_dict, device=device, task=task)
        
        # 2. Initialize Classification Mixin and load cleanly typed metadata
        _ClassificationMixin.__init__(self)
        self._load_classification_metadata(self._file_handler.parse_classification_metadata())

        self._transform: Optional[Callable] = None
        self._is_transformed: bool = False
        
        # --- Model specific channels ---
        self.expected_in_channels: int = 3
        if hasattr(model, 'in_channels'):
            self.expected_in_channels = model.in_channels # type: ignore
            _LOGGER.info(f"Model expects {self.expected_in_channels} input channels.")
        else:
            _LOGGER.warning("Could not determine 'in_channels' from model. Defaulting to 3 (RGB). Modify with '.expected_in_channels'.")
        
        if transform_source:
            self.set_transform(transform_source)
            self._is_transformed = True
        
        # Initialize color map for visualization only if not already set
        if not hasattr(self, '_color_map'):
            self._initialize_color_map()
    
    def _initialize_color_map(self) -> None:
        """Generates and saves a fixed color palette for the labels."""
        self._color_map = {} # Reset and initialize the attribute
        
        if self._idx_to_class is not None and len(self._idx_to_class) > 0:
            all_labels = sorted(list(self._idx_to_class.keys()))
        else:
            # Fallback if no class map is present
            all_labels = list(range(100))
            
        np.random.seed(42)
        for i, label in enumerate(all_labels):
            if i < len(_VISION_PALETTE_50):
                self._color_map[label] = _VISION_PALETTE_50[i]
            else:
                self._color_map[label] = tuple(np.random.randint(0, 255, 3).tolist())

    def set_class_map(self, class_map: dict[str, int], force_overwrite: bool = False) -> None:
        """
        Sets the class name mapping and regenerates the color palette to match the new labels.
        
        Args:
            class_map (dict[str, int]): Mapping of class names to integer labels.
            force_overwrite (bool): If True, overwrites any existing class map.
        """
        super().set_class_map(class_map, force_overwrite)
        self._initialize_color_map()

    def set_transform(self, transform_source: Union[str, Path, Callable]):
        """
        Sets or updates the inference transformation pipeline.
        """
        if self._is_transformed:
            _LOGGER.warning("Transformations were previously applied. Applying new transformations...")
            
        if isinstance(transform_source, (str, Path)):
            _LOGGER.info(f"Loading transform from recipe file: '{transform_source}'")
            try:
                self._transform = _load_recipe_and_build_transform(transform_source)
            except Exception as e:
                _LOGGER.error(f"Failed to load transform from recipe '{transform_source}': {e}")
                raise
        elif isinstance(transform_source, Callable):
            _LOGGER.info("Inference transform has been set from a direct Callable.")
            self._transform = transform_source
        else:
            _LOGGER.error(f"Invalid transform_source type: {type(transform_source)}. Must be str, Path, or Callable.")
            raise TypeError()

    def predict_from_pil(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """
        Applies the stored transform to a single PIL image and returns the overlapped image and prediction results.
        
        Args:
            image (Image.Image): Input PIL image.
        Returns:
            tuple[Image.Image, dict[str, Any]]: A tuple containing the overlapped PIL image and the prediction results as a dictionary.
        """
        if self._transform is None:
            _LOGGER.error("Cannot predict from PIL image: No transform has been set. Call .set_transform() or provide transform_source in __init__.")
            raise RuntimeError()

        try:
            transformed_image = self._transform(image)
        except Exception as e:
            _LOGGER.error(f"Error applying transform to PIL image: {e}")
            raise
            
        if not isinstance(transformed_image, torch.Tensor):
            _LOGGER.error("The provided transform did not return a torch.Tensor. Does it include transforms.ToTensor()?")
            raise ValueError()
            
        if transformed_image.ndim != 3:
            _LOGGER.warning(f"Expected transform to output a 3D (C, H, W) tensor, but got {transformed_image.ndim}D. Attempting to proceed.")
            if transformed_image.ndim == 4 and transformed_image.shape[0] == 1:
                transformed_image = transformed_image.squeeze(0)
                _LOGGER.warning("Removed an extra batch dimension.")
            else:
                _LOGGER.error(f"Transform must output a 3D (C, H, W) tensor, got {transformed_image.shape}.")
                raise ValueError()

        results = self.predict_numpy(transformed_image)
        overlapped_image = self._create_overlapped_image(image, results)

        return overlapped_image, results

    def predict_from_file(self, 
                          image_path: Union[str, Path], 
                          save_overlay: bool = True,
                          verbose: int = 2) -> dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction.
        
        Args:
            image_path (Union[str, Path]): Path to the image file.
            save_overlay (bool): If True, saves the overlapped image in the same directory with a suffix.
            verbose (int): Verbosity level for logging.
            
        Returns:
            dict[str, Any]: Dictionary containing prediction results.
        """
        full_path = make_fullpath(image_path, make=False, enforce="file")
        
        try:
            pil_mode: str
            if self.expected_in_channels == 1:
                pil_mode = "L"
            elif self.expected_in_channels == 4:
                pil_mode = "RGBA"
            else:
                if self.expected_in_channels != 3 and verbose >= 1:
                    _LOGGER.warning(f"Model expects {self.expected_in_channels} channels. PIL conversion is limited, defaulting to 3 channels (RGB).")
                pil_mode = "RGB"
                
            image = Image.open(full_path).convert(pil_mode)
        except Exception as e:
            _LOGGER.error(f"Failed to load and convert image from '{image_path}': {e}")
            raise

        overlapped_image, results = self.predict_from_pil(image)
        
        if save_overlay:
            overlay_path = full_path.parent / f"{full_path.stem}{VisionKeys.OVERLAPPED_SUFFIX}"
            try:
                overlapped_image.save(overlay_path)
                if verbose >= 2:
                    _LOGGER.info(f"Saved overlapped image to '{overlay_path}'.")
            except Exception as e:
                _LOGGER.error(f"Failed to save overlapped image to '{overlay_path}': {e}")

        return results
    
    def _iterate_directory_for_images(self, dir_path: Path, valid_extensions: Optional[list[str]] = None) -> list[Path]:
        """
        Private helper method to scan a directory for image files matching the specified extensions.
        
        Excludes files that already have the overlapped suffix to avoid reprocessing.
        """
        if valid_extensions is None:
            valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"]
        
        valid_extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in valid_extensions]
        
        found_images = [
            p for p in dir_path.iterdir() 
            if p.is_file() 
            and p.suffix.lower() in valid_extensions
            and not p.name.endswith(VisionKeys.OVERLAPPED_SUFFIX)
        ]
    
        return found_images
    
    def predict_from_directory(
        self, 
        directory_path: Union[str, Path], 
        valid_extensions: Optional[list[str]] = None,
        verbose: int = 2
    ) -> None:
        """
        Scans a directory for images matching the target formats and saves overlapped predictions.
        
        Images that already have the 'overlapped' suffix will be ignored to prevent reprocessing.
        
        Args:
            directory_path (Union[str, Path]): Path to the directory containing images.
            valid_extensions (Optional[list[str]]): List of valid image file extensions. Defaults to common formats.
            verbose (int): Verbosity level for logging.
        """
        dir_path = make_fullpath(directory_path, make=False, enforce="directory")
        
        found_images = self._iterate_directory_for_images(dir_path, valid_extensions)
        
        if not found_images:
            if verbose >= 1:
                _LOGGER.warning(f"No images found in '{dir_path}' matching extensions: {valid_extensions}")
            return
            
        if verbose >= 2:
            _LOGGER.info(f"Found {len(found_images)} images in '{dir_path}'. Processing...")
            
        for img_path in found_images:
            try:
                inner_verbose = 3 if verbose >= 3 else (0 if verbose <= 0 else 1)
                _ = self.predict_from_file(img_path, save_overlay=True, verbose=inner_verbose)
            except Exception as e:
                _LOGGER.error(f"Failed to process image '{img_path.name}': {e}")
            
        if verbose >= 2:
            _LOGGER.info(f"Directory processing completed for '{dir_path}'.")
    
    @staticmethod
    def clear_overlapped_images(directory_path: Union[str, Path], verbose: int = 2) -> None:
        """
        Deletes all files ending with the overlapped suffix in the target directory and its subdirectories.
        
        Args:
            directory_path (Union[str, Path]): Path to the directory to clean (subdirectories will be searched).
            verbose (int): Verbosity level for logging.
        """
        dir_path = make_fullpath(directory_path, make=False, enforce="directory")
        pattern = f"*{VisionKeys.OVERLAPPED_SUFFIX}"
        
        overlapped_files = list(dir_path.rglob(pattern))
        
        if not overlapped_files:
            if verbose >= 2:
                _LOGGER.info(f"No overlapped images found in '{dir_path}' to delete.")
            return
            
        deleted_count = 0
        for file_path in overlapped_files:
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    _LOGGER.error(f"Failed to delete '{file_path}': {e}")
                    
        if verbose >= 2:
            _LOGGER.info(f"Successfully deleted {deleted_count} overlapped images from '{dir_path}'.")

    @abstractmethod
    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """Convenience wrapper for predict that returns NumPy arrays."""
        pass

    @abstractmethod
    def _create_overlapped_image(self, original_image: Image.Image, predictions: dict[str, Any]) -> Image.Image:
        """Helper method to create a PIL Image with the predictions overlapped."""
        pass
