import torch
from torch import nn
from pathlib import Path
from typing import Union, Literal, Any, Optional, Callable
from PIL import Image

from ..ML_vision_transformers._core_transforms import _load_recipe_and_build_transform

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys

from ..ML_inference._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("Vision Classification Inference")


__all__ = [
    "DragonVisionClassificationInference"
]


class DragonVisionClassificationInference(_BaseInferenceHandler):
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
            model (nn.Module): An instantiated PyTorch model from ML_vision_models.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str, optional): The type of classification task. If None, detected from file.
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
            MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
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
        Core batch prediction method for image classification models.
        """
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
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        
        if self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
            int_labels = numpy_results[PyTorchInferenceKeys.LABELS]
            numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
                self._idx_to_class.get(label_id, "Unknown")
                for label_id in int_labels
            ]
        
        return numpy_results

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays/scalars.
        """
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
            
    def predict_from_pil(self, image: Image.Image) -> dict[str, Any]:
        """
        Applies the stored transform to a single PIL image and returns the prediction.
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

        return self.predict_numpy(transformed_image)

    def predict_from_file(self, image_path: Union[str, Path]) -> dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction.
        """
        try:
            pil_mode: str
            if self.expected_in_channels == 1:
                pil_mode = "L"
            elif self.expected_in_channels == 4:
                pil_mode = "RGBA"
            else:
                if self.expected_in_channels != 3:
                    _LOGGER.warning(f"Model expects {self.expected_in_channels} channels. PIL conversion is limited, defaulting to 3 channels (RGB).")
                pil_mode = "RGB"
                
            image = Image.open(image_path).convert(pil_mode)
        except Exception as e:
            _LOGGER.error(f"Failed to load and convert image from '{image_path}': {e}")
            raise

        return self.predict_from_pil(image)
