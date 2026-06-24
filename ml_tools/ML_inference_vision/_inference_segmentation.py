import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Optional, Callable
from PIL import Image

from ..ML_vision_transformers._core_transforms import _load_recipe_and_build_transform

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys, VisionKeys
from ..path_manager import make_fullpath

from ..ML_inference._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("Segmentation Inference")


__all__ = [
    "DragonSegmentationInference"
]


class DragonSegmentationInference(_BaseInferenceHandler):
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference 
    for image segmentation tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Optional[Literal["binary segmentation", "multiclass segmentation"]] = None,
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        """
        Initializes the vision segmentation inference handler.

        Args:
            model (nn.Module): An instantiated PyTorch model from ML_vision_models.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str, optional): The type of segmentation task. If None, detected from file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            transform_source (str | Path | Callable | None): 
                - A path to a .json recipe file (str or Path).
                - A pre-built transformation pipeline (Callable).
                - None, in which case .set_transform() must be called explicitly to set transformations.
        """
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         scaler=None, 
                         task=task)

        # --- Validate Task ---
        valid_tasks = [
            MLTaskKeys.BINARY_SEGMENTATION, 
            MLTaskKeys.MULTICLASS_SEGMENTATION
        ]
        
        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
            raise ValueError()

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

    def _preprocess_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """
        Validates input and moves it to the correct device.
        Expects 4D Tensor (B, C, H, W).
        """
        if not isinstance(inputs, torch.Tensor):
            _LOGGER.error(f"Input for {self.task} must be a torch.Tensor.")
            raise ValueError()
            
        if inputs.ndim != 4:
             _LOGGER.error(f"Input tensor for {self.task} must be 4D (B, C, H, W). Got {inputs.ndim}D.")
             raise ValueError()
        
        return inputs.float().to(self.device) 
        
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

    def predict_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Core batch prediction method for image segmentation models.
        """
        processed_inputs = self._preprocess_batch(inputs)
        
        with torch.no_grad():
            output = self.model(processed_inputs)
            
            if self.task == MLTaskKeys.BINARY_SEGMENTATION:
                probs = torch.sigmoid(output)
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,       
                    PyTorchInferenceKeys.PROBABILITIES: probs  
                }
                
            elif self.task == MLTaskKeys.MULTICLASS_SEGMENTATION:
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            
            else:
                _LOGGER.error(f"Unknown task: {self.task}")
                raise ValueError()

    def predict(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Core single-sample prediction method.
        """
        if not isinstance(single_input, torch.Tensor) or single_input.ndim != 3:
             _LOGGER.error(f"Input for predict() must be a 3D tensor (C, H, W). Got {single_input.ndim}D.")
             raise ValueError()
        
        batched_input = single_input.unsqueeze(0)
        batch_results = self.predict_batch(batched_input)

        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results

    def predict_batch_numpy(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        """
        tensor_results = self.predict_batch(inputs)
        return {key: value.cpu().numpy() for key, value in tensor_results.items()}

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays.
        """
        tensor_results = self.predict(single_input)
        return {
            PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].cpu().numpy(),
            PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
        }
        
    def _create_overlapped_image(self, original_image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Helper method to create a PIL Image with the predicted mask overlapped on the 
        original image converted to grayscale.
        """
        # Ensure mask is 2D (H, W). Binary segmentation might return (1, H, W)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize(original_image.size, Image.Resampling.NEAREST)
        mask_resized = np.array(mask_pil)

        # Convert base image to grayscale, then to RGBA for blending
        base_img = original_image.convert("L").convert("RGBA")
        
        # Initialize an empty transparent overlay
        overlay = np.zeros((original_image.size[1], original_image.size[0], 4), dtype=np.uint8)
        
        if self.task == MLTaskKeys.BINARY_SEGMENTATION:
            # Red overlay for positive class
            overlay[mask_resized == 1] = [255, 0, 0, 128]
        else:
            # Random distinct colors for multiclass
            unique_labels = np.unique(mask_resized)
            np.random.seed(42)
            color_map = {label: list(np.random.randint(0, 255, 3)) + [128] for label in unique_labels if label != 0}
            
            for label, color in color_map.items():
                overlay[mask_resized == label] = color
                
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        
        # Alpha composite and convert back to RGB
        return Image.alpha_composite(base_img, overlay_img).convert("RGB")    
            
    def predict_from_pil(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """
        Applies the stored transform to a single PIL image and returns the overlapped image and prediction results.
        
        Args:
            image (PIL.Image): The input image for prediction.
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
        overlapped_image = self._create_overlapped_image(image, results[PyTorchInferenceKeys.LABELS])

        return overlapped_image, results

    def predict_from_file(self, 
                          image_path: Union[str, Path], 
                          save_overlay: bool = True,
                          verbose: int = 2) -> dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction. Optionally saves the overlapped image.
        
        Args:
            image_path (str | Path): Path to the input image file.
            save_overlay (bool): Whether to save the overlapped image with mask predictions.
            verbose (int): Level of verbosity for logging.
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
            # Save the overlapped image to PNG format
            overlay_path = full_path.parent / f"{full_path.stem}{VisionKeys.OVERLAPPED_SUFFIX}"
            
            try:
                overlapped_image.save(overlay_path)
                if verbose >= 2:
                    _LOGGER.info(f"Saved overlapped image to '{overlay_path}'.")
            except Exception as e:
                _LOGGER.error(f"Failed to save overlapped image to '{overlay_path}': {e}")

        return results
    
    def predict_from_directory(
        self, 
        directory_path: Union[str, Path], 
        valid_extensions: Optional[list[str]] = None,
        verbose: int = 2
    ) -> None:
        """
        Scans a directory for images matching the target formats and saves overlapped 
        predictions for each image found.
        
        Args:
            directory_path (str | Path): Path to the directory containing images.
            valid_extensions (list[str], optional): List of accepted file extensions. If None, defaults to:
                ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"
            verbose (int): Level of verbosity for logging.
        """
        if valid_extensions is None:
            valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"]
            
        # Ensure extensions are lowercase and start with a dot
        valid_extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in valid_extensions]
        
        dir_path = make_fullpath(directory_path, make=False, enforce="directory")
        
        # Exclude files that end with '_overlapped' to prevent reprocessing previous predictions
        found_images = [
            p for p in dir_path.iterdir() 
            if p.is_file() 
            and p.suffix.lower() in valid_extensions
            and not p.stem.endswith(VisionKeys.OVERLAPPED_SUFFIX)
        ]
        
        if not found_images:
            if verbose >= 1:
                _LOGGER.warning(f"No images found in '{dir_path}' matching extensions: {valid_extensions}")
            return
            
        if verbose >= 2:
            _LOGGER.info(f"Found {len(found_images)} images in '{dir_path}'. Processing...")
            
        for img_path in found_images:
            try:
                # predict_from_file already handles the save_overlay logic and saving to disk
                if verbose >= 3:
                    inner_verbose = 3
                elif verbose <= 0:
                    inner_verbose = 0
                else:
                    inner_verbose = 1
                _ = self.predict_from_file(img_path, save_overlay=True, verbose=inner_verbose)
            except Exception as e:
                _LOGGER.error(f"Failed to process image '{img_path.name}': {e}")
                    
        if verbose >= 2:
            _LOGGER.info(f"Directory processing completed for '{dir_path}'.")
