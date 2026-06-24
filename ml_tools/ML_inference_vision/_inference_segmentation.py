import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Optional, Callable
from PIL import Image

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys

from ._base_vision_inference import _BaseVisionInferenceHandler


_LOGGER = get_logger("Segmentation Inference")


__all__ = [
    "DragonSegmentationInference"
]


class DragonSegmentationInference(_BaseVisionInferenceHandler):
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
        
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         task=task,
                         transform_source=transform_source)

        # --- Validate Task ---
        valid_tasks = [
            MLTaskKeys.BINARY_SEGMENTATION, 
            MLTaskKeys.MULTICLASS_SEGMENTATION
        ]
        
        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
            raise ValueError()

    def _preprocess_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """
        Validates input and moves it to the correct device.
        Expects 4D Tensor (B, C, H, W), a 3D Tensor, or a list of 3D Tensors.
        """
        if isinstance(inputs, list):
            if not all(isinstance(t, torch.Tensor) for t in inputs):
                _LOGGER.error("All elements in the input list must be torch.Tensor.")
                raise ValueError()
            inputs = torch.stack(inputs)

        if not isinstance(inputs, torch.Tensor):
            _LOGGER.error(f"Input for {self.task} must be a torch.Tensor or a list of torch.Tensor.")
            raise ValueError()
            
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        elif inputs.ndim != 4:
             _LOGGER.error(f"Input tensor for {self.task} must be 4D (B, C, H, W) or 3D. Got {inputs.ndim}D.")
             raise ValueError()
        
        return inputs.float().to(self.device)

    def predict_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Core batch prediction method for image segmentation models.
        
        Args:
            inputs (Union[torch.Tensor, list[torch.Tensor]]): A batch of images as a 4D tensor (B, C, H, W) or a list of 3D tensors (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted labels and probabilities.
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
        
        Args:
            single_input (torch.Tensor): A single image as a 3D tensor (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted labels and probabilities for the single input.
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
        
        Args:
            inputs (Union[torch.Tensor, list[torch.Tensor]]): A batch of images as a 4D tensor (B, C, H, W) or a list of 3D tensors (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted labels and probabilities as NumPy arrays.
        """
        tensor_results = self.predict_batch(inputs)
        return {key: value.cpu().numpy() for key, value in tensor_results.items()}

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays.
        
        Args:
            single_input (torch.Tensor): A single image as a 3D tensor (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted labels and probabilities as NumPy arrays for the single input.
        """
        tensor_results = self.predict(single_input)
        return {
            PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].cpu().numpy(),
            PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
        }
        
    def _create_overlapped_image(self, original_image: Image.Image, predictions: dict[str, Any]) -> Image.Image:
        """
        Helper method to create a PIL Image with the predicted mask overlapped on the 
        original image converted to grayscale.
        """
        if PyTorchInferenceKeys.LABELS not in predictions:
            _LOGGER.error(f"Predictions dictionary must contain {PyTorchInferenceKeys.LABELS} to generate an overlapped image.")
            return original_image
            
        mask = predictions[PyTorchInferenceKeys.LABELS]
        
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
            unique_labels = np.unique(mask_resized)
            for label in unique_labels:
                if label == 0:
                    continue
                
                # Fetch color from centralized map, fallback if missing, and append alpha
                base_color = self._color_map.get(label, tuple(np.random.randint(0, 255, 3).tolist()))
                color = list(base_color) + [128]
                overlay[mask_resized == label] = color
                
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        
        # Alpha composite and convert back to RGB
        return Image.alpha_composite(base_img, overlay_img).convert("RGB") 
