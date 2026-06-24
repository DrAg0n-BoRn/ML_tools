import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Any, Optional, Callable
from PIL import Image, ImageDraw

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys, ObjectDetectionKeys

from ._base_vision_inference import _BaseVisionInferenceHandler


_LOGGER = get_logger("Object Detection Inference")


__all__ = [
    "DragonObjectDetectionInference"
]


class DragonObjectDetectionInference(_BaseVisionInferenceHandler):
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference 
    for object detection tasks.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         task=MLTaskKeys.OBJECT_DETECTION,
                         transform_source=transform_source)

    def _preprocess_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> list[torch.Tensor]:
        """
        Validates input and moves it to the correct device.
        Expects List[Tensor(C, H, W)] or a single Tensor.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                inputs = list(inputs)
            elif inputs.ndim == 3:
                inputs = [inputs]
            else:
                _LOGGER.error(f"Input tensor must be 3D or 4D. Got {inputs.ndim}D.")
                raise ValueError()
        elif not isinstance(inputs, list) or not all(isinstance(t, torch.Tensor) for t in inputs):
            _LOGGER.error("Input for object_detection must be a torch.Tensor or a List[torch.Tensor].")
            raise ValueError()
        
        return [t.float().to(self.device) for t in inputs]

    def predict_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Core batch prediction method for object detection models.
        
        Args:
            inputs (Union[torch.Tensor, list[torch.Tensor]]): A batch of images as a 4D tensor (B, C, H, W) or a list of 3D tensors (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted bounding boxes, labels, and scores.
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
        
        Args:
            single_input (torch.Tensor): A single image as a 3D tensor (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted bounding boxes, labels, and scores for the single input.
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
        
        Args:
            inputs (Union[torch.Tensor, list[torch.Tensor]]): A batch of images as a 4D tensor (B, C, H, W) or a list of 3D tensors (C, H, W).
            
        Returns:
            dict[str, Any]: A dictionary containing predicted bounding boxes, labels, and scores as NumPy arrays.
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
        
        Args:
            single_input (torch.Tensor): A single image as a 3D tensor (C, H, W).
        Returns:
            dict[str, Any]: A dictionary containing predicted bounding boxes, labels, and scores as NumPy arrays for the single input.
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
            
            for i, box in enumerate(boxes):
                label_val = labels[i] if i < len(labels) else None
                
                # Fetch color from initialized map, fallback to red if missing
                if label_val is not None:
                    color = self._color_map.get(label_val, tuple(np.random.randint(0, 255, 3).tolist()))
                else:
                    color = "red"
                
                draw.rectangle(box.tolist(), outline=color, width=3)
                
                if i < len(labels) and i < len(scores):
                    label_str = str(labels[i])
                    score_str = f"{scores[i]:.2f}"
                    text = f"{label_str}: {score_str}"
                    draw.text((box[0], max(0, box[1] - 15)), text, fill=color)
                    
        return result_image
