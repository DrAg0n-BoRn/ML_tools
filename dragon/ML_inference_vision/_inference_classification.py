import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Optional, Callable
from PIL import Image, ImageDraw

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys

from ._base_vision_inference import _BaseVisionInferenceHandler


_LOGGER = get_logger("Vision Classification Inference")


__all__ = [
    "DragonVisionClassificationInference"
]


class DragonVisionClassificationInference(_BaseVisionInferenceHandler):
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference 
    for image classification tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Optional[Literal["binary image classification", "multiclass image classification"]] = None,
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        """
        Initializes the vision classification inference handler.

        Args:
            model (nn.Module): An instantiated PyTorch model.
            state_dict (str | Path): Path to the saved .pth model state_dict file or a FinalizedFile format.
            task (str, optional): The type of classification task. If None, detected from file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            transform_source (str | Path | Callable | None): 
                - A path to a .json recipe file (str or Path).
                - A pre-built transformation pipeline (Callable).
                - None, in which case .set_transform() must be called explicitly to set transformations.
        """
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         task=task,
                         device=device, 
                         transform_source=transform_source)

        # --- Validate Task ---
        valid_tasks = [
            MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
        ]
        
        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
            raise ValueError()

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
        
    def predict_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """Core batch prediction method for image classification models."""
        processed_inputs = self._preprocess_batch(inputs)
        
        with torch.no_grad():
            output = self.model(processed_inputs)
            
            if self.task == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION:
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,
                    PyTorchInferenceKeys.PROBABILITIES: probs
                }
            
            elif self.task == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
                    
                probs = torch.sigmoid(output)
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,       
                    PyTorchInferenceKeys.PROBABILITIES: probs  
                }
            
            else:
                _LOGGER.error(f"Unknown task: {self.task}. Cannot perform prediction.")
                raise ValueError()

    def predict(self, single_input: torch.Tensor) -> dict[str, Any]:
        """Core single-sample prediction method."""
        if not isinstance(single_input, torch.Tensor) or single_input.ndim != 3:
             _LOGGER.error(f"Input for predict() must be a 3D tensor (C, H, W). Got {single_input.ndim}D.")
             raise ValueError()
        
        batched_input = single_input.unsqueeze(0)
        batch_results = self.predict_batch(batched_input)

        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results

    def predict_batch_numpy(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """Convenience wrapper for predict_batch that returns NumPy arrays."""
        tensor_results = self.predict_batch(inputs)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        
        if self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
            int_labels = numpy_results[PyTorchInferenceKeys.LABELS]
            numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
                self._idx_to_class.get(label_id, "Unknown")
                for label_id in int_labels
            ]
        
        return numpy_results

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """Convenience wrapper for predict that returns NumPy arrays/scalars."""
        tensor_results = self.predict(single_input)
        
        int_label = tensor_results[PyTorchInferenceKeys.LABELS].item()
        label_name = "Unknown"
        if self._idx_to_class:
            label_name = self._idx_to_class.get(int_label, "Unknown")

        return {
            PyTorchInferenceKeys.LABELS: int_label,
            PyTorchInferenceKeys.LABEL_NAMES: label_name,
            PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
        }

    def _create_overlapped_image(self, original_image: Image.Image, predictions: dict[str, Any]) -> Image.Image:
        """Helper method to create a PIL Image with the predicted class and probability written on it."""
        result_image = original_image.copy()
        draw = ImageDraw.Draw(result_image)
        
        label_id = predictions.get(PyTorchInferenceKeys.LABELS, -1)
        label_name = predictions.get(PyTorchInferenceKeys.LABEL_NAMES, "Unknown")
        prob = predictions.get(PyTorchInferenceKeys.PROBABILITIES, 0.0)
        
        if isinstance(prob, np.ndarray):
            prob_val = float(np.max(prob)) if self.task == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION else float(prob)
        else:
            prob_val = float(prob)
            
        text = f"{label_name} ({prob_val:.2f})"
        
        # Fetch color from initialized map, fallback to a random color if missing
        color = self._color_map.get(label_id, tuple(np.random.randint(0, 255, 3).tolist()))
        
        # Simple text bounding box for readability
        text_bbox = draw.textbbox((10, 10), text)
        
        # Expand bounding box slightly for padding
        padded_bbox = [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5]
        
        # Draw a black background with a colored outline for the text box
        draw.rectangle(padded_bbox, fill="black", outline=color, width=2)
        draw.text((10, 10), text, fill=color)
        
        return result_image
