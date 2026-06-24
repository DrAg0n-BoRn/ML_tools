import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Any, Optional, Callable
from PIL import Image, ImageDraw

from ..ML_vision_transformers._core_transforms import _load_recipe_and_build_transform

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys, ObjectDetectionKeys, VisionKeys
from ..path_manager import make_fullpath

from ..ML_inference._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("Object Detection Inference")


__all__ = [
    "DragonObjectDetectionInference"
]


class DragonObjectDetectionInference(_BaseInferenceHandler):
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference 
    for object detection tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        """
        Initializes the vision object detection inference handler.

        Args:
            model (nn.Module): An instantiated PyTorch model from ML_vision_models.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
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
                         task=MLTaskKeys.OBJECT_DETECTION)

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

    def _preprocess_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> list[torch.Tensor]:
        """
        Validates input and moves it to the correct device.
        Expects List[Tensor(C, H, W)].
        """
        if not isinstance(inputs, list) or not all(isinstance(t, torch.Tensor) for t in inputs):
            _LOGGER.error("Input for object_detection must be a List[torch.Tensor].")
            raise ValueError()
        
        return [t.float().to(self.device) for t in inputs]
        
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
        Core batch prediction method for object detection models.
        """
        processed_inputs = self._preprocess_batch(inputs)
        
        with torch.no_grad():
            output = self.model(processed_inputs)
            
            if self.task == MLTaskKeys.OBJECT_DETECTION:
                return {
                    PyTorchInferenceKeys.PREDICTIONS: output
                }
            else:
                _LOGGER.error(f"Unknown task: {self.task}. Cannot perform prediction.")
                raise ValueError()

    def predict(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Core single-sample prediction method.
        """
        if not isinstance(single_input, torch.Tensor) or single_input.ndim != 3:
             _LOGGER.error(f"Input for predict() must be a 3D tensor (C, H, W). Got {single_input.ndim}D.")
             raise ValueError()
        
        batched_input = [single_input]
        batch_results = self.predict_batch(batched_input)

        return batch_results[PyTorchInferenceKeys.PREDICTIONS][0]

    def predict_batch_numpy(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        """
        tensor_results = self.predict_batch(inputs)
        
        numpy_results = []
        for pred_dict in tensor_results[PyTorchInferenceKeys.PREDICTIONS]:
            np_dict = {key: value.cpu().numpy() for key, value in pred_dict.items()}
            numpy_results.append(np_dict)
            
        return {PyTorchInferenceKeys.PREDICTIONS: numpy_results}

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays.
        """
        tensor_results = self.predict(single_input)
        
        numpy_results = {
            key: value.cpu().numpy() for key, value in tensor_results.items()
        }
            
        return numpy_results
    
    def _create_overlapped_image(self, original_image: Image.Image, predictions: dict[str, Any]) -> Image.Image:
        """
        Helper method to create a PIL Image with the predicted bounding boxes overlapped on the 
        original image.
        """
        result_image = original_image.copy()
        draw = ImageDraw.Draw(result_image)
        
        if ObjectDetectionKeys.BOXES in predictions:
            boxes = predictions[ObjectDetectionKeys.BOXES]
            labels = predictions.get(ObjectDetectionKeys.LABELS, [])
            scores = predictions.get(ObjectDetectionKeys.SCORES, [])
            
            # Generate distinct colors for multiclass
            unique_labels = np.unique(labels) if len(labels) > 0 else []
            np.random.seed(42)
            color_map = {label: tuple(np.random.randint(0, 255, 3).tolist()) for label in unique_labels}
            
            for i, box in enumerate(boxes):
                label_val = labels[i] if i < len(labels) else None
                color = color_map.get(label_val, "red") if label_val is not None else "red"
                
                draw.rectangle(box.tolist(), outline=color, width=3)
                
                if i < len(labels) and i < len(scores):
                    label_str = str(labels[i])
                    score_str = f"{scores[i]:.2f}"
                    text = f"{label_str}: {score_str}"
                    draw.text((box[0], max(0, box[1] - 15)), text, fill=color)
                    
        return result_image
            
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
        overlapped_image = self._create_overlapped_image(image, results)

        return overlapped_image, results

    def predict_from_file(self, 
                          image_path: Union[str, Path], 
                          save_overlay: bool = True,
                          verbose: int = 2) -> dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction. Optionally saves the overlapped image.
        
        Args:
            image_path (str | Path): Path to the input image file.
            save_overlay (bool): Whether to save the overlapped image with bounding box predictions.
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
