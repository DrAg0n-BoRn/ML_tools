import torch
from torch import nn
import numpy as np  #numpy array return value
from pathlib import Path
from typing import Union, Literal, Dict, Any, List, Optional, Callable
from PIL import Image
from torchvision import transforms

from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath
from .keys import PyTorchInferenceKeys, PyTorchCheckpointKeys
from ._ML_vision_recipe import load_recipe_and_build_transform


__all__ = [
    "PyTorchVisionInferenceHandler"
]


class PyTorchVisionInferenceHandler:
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference.

    This class is specifically for vision models, which typically expect
    4D Tensors (B, C, H, W) or Lists of Tensors as input.
    It does NOT use a scaler, as preprocessing (e.g., normalization)
    is assumed to be part of the input transform pipeline.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Literal["image_classification", "image_segmentation", "object_detection"],
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None,
                 class_map: Optional[Dict[str, int]] = None):
        """
        Initializes the vision inference handler.

        Args:
            model (nn.Module): An instantiated PyTorch model from ML_vision_models.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str): The type of vision task.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            transform_source (str | Path | Callable | None): 
                - A path to a .json recipe file (str or Path).
                - A pre-built transformation pipeline (Callable).
                - None, in which case .set_transform() must be called explicitly to set transformations.
            idx_to_class (Dict[int, str] | None): Sets the class name mapping to translate predicted integer labels back into string names. (For image classification and object detection)
        """
        self._model = model
        self._device = self._validate_device(device)
        self._transform: Optional[Callable] = None
        self._is_transformed: bool = False
        self._idx_to_class: Optional[Dict[int, str]] = None
        if class_map is not None:
            self.set_class_map(class_map)

        if task not in ["image_classification", "image_segmentation", "object_detection"]:
            _LOGGER.error(f"`task` must be 'image_classification', 'image_segmentation', or 'object_detection'. Got '{task}'.")
            raise ValueError("Invalid task type.")
        self.task = task
        
        self.expected_in_channels: int = 3 # Default to RGB
        if hasattr(model, 'in_channels'):
            self.expected_in_channels = model.in_channels # type: ignore
            _LOGGER.info(f"Model expects {self.expected_in_channels} input channels.")
        else:
            _LOGGER.warning("Could not determine 'in_channels' from model. Defaulting to 3 (RGB). Modify with '.expected_in_channels'.")
        
        if transform_source:
            self.set_transform(transform_source)
            self._is_transformed = True
        
        model_p = make_fullpath(state_dict, enforce="file")

        try:
            # Load whatever is in the file
            loaded_data = torch.load(model_p, map_location=self._device)

            # Check if it's a new checkpoint dictionary or an old weights-only file
            if isinstance(loaded_data, dict) and PyTorchCheckpointKeys.MODEL_STATE in loaded_data:
                # It's a new training checkpoint, extract the weights
                self._model.load_state_dict(loaded_data[PyTorchCheckpointKeys.MODEL_STATE])
            else:
                # It's an old-style file (or just a state_dict), load it directly
                self._model.load_state_dict(loaded_data)
            
            _LOGGER.info(f"Model state loaded from '{model_p.name}'.")
                
            self._model.to(self._device)
            self._model.eval()  # Set the model to evaluation mode
        except Exception as e:
            _LOGGER.error(f"Failed to load model state from '{model_p}': {e}")
            raise

    def _validate_device(self, device: str) -> torch.device:
        """Validates the selected device and returns a torch.device object."""
        device_lower = device.lower()
        if "cuda" in device_lower and not torch.cuda.is_available():
            _LOGGER.warning("CUDA not available, switching to CPU.")
            device_lower = "cpu"
        elif device_lower == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
            device_lower = "cpu"
        return torch.device(device_lower)

    def _preprocess_batch(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Validates input and moves it to the correct device.
        - For Classification/Segmentation: Expects 4D Tensor (B, C, H, W).
        - For Object Detection: Expects List[Tensor(C, H, W)].
        """
        if self.task == "object_detection":
            if not isinstance(inputs, list) or not all(isinstance(t, torch.Tensor) for t in inputs):
                _LOGGER.error("Input for object_detection must be a List[torch.Tensor].")
                raise ValueError("Invalid input type for object detection.")
            # Move each tensor in the list to the device
            return [t.float().to(self._device) for t in inputs]
        
        else: # Classification or Segmentation
            if not isinstance(inputs, torch.Tensor):
                _LOGGER.error(f"Input for {self.task} must be a torch.Tensor.")
                raise ValueError(f"Invalid input type for {self.task}.")
                
            if inputs.ndim != 4:
                 _LOGGER.error(f"Input tensor for {self.task} must be 4D (B, C, H, W). Got {inputs.ndim}D.")
                 raise ValueError("Input tensor must be 4D.")
            
            return inputs.float().to(self._device)
        
    def set_transform(self, transform_source: Union[str, Path, Callable]):
        """
        Sets or updates the inference transformation pipeline from a recipe file or a direct Callable.

        Args:
            transform_source (str, Path, Callable):
                - A path to a .json recipe file (str or Path).
                - A pre-built transformation pipeline (Callable).
        """
        if self._is_transformed:
            _LOGGER.warning("Transformations were previously applied. Applying new transformations...")
            
        if isinstance(transform_source, (str, Path)):
            _LOGGER.info(f"Loading transform from recipe file: '{transform_source}'")
            try:
                # Use the new loader function
                self._transform = load_recipe_and_build_transform(transform_source)
            except Exception as e:
                _LOGGER.error(f"Failed to load transform from recipe '{transform_source}': {e}")
                raise
        elif isinstance(transform_source, Callable):
            _LOGGER.info("Inference transform has been set from a direct Callable.")
            self._transform = transform_source
        else:
            _LOGGER.error(f"Invalid transform_source type: {type(transform_source)}. Must be str, Path, or Callable.")
            raise TypeError("transform_source must be a file path or a Callable.")
        
    def set_class_map(self, class_map: Dict[str, int]):
        """
        Sets the class name mapping to translate predicted integer labels
        back into string names.

        Args:
            class_map (Dict[str, int]): The class_to_idx dictionary
                (e.g., {'cat': 0, 'dog': 1}) from the VisionDatasetMaker.
        """
        if self._idx_to_class is not None:
            _LOGGER.warning("Class to index mapping was previously given. Setting new mapping...")
        # Invert the dictionary for fast lookup
        self._idx_to_class = {v: k for k, v in class_map.items()}
        _LOGGER.info("Class map set for label-to-name translation.")

    def predict_batch(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, Any]:
        """
        Core batch prediction method for vision models.
        All preprocessing (resizing, normalization) should be done *before*
        calling this method.

        Args:
            inputs (torch.Tensor | List[torch.Tensor]):
                - For 'image_classification' or 'image_segmentation', 
                  a 4D torch.Tensor (B, C, H, W).
                - For 'object_detection', a List of 3D torch.Tensors 
                  [(C, H, W), ...], each with its own size.

        Returns:
            A dictionary containing the output tensors.
            - Classification: {labels, probabilities}
            - Segmentation: {labels, probabilities} (labels is the mask)
            - Object Detection: {predictions} (List of dicts)
        """
        
        processed_inputs = self._preprocess_batch(inputs)
        
        with torch.no_grad():
            if self.task == "image_classification":
                # --- Image Classification ---
                # 1. Predict
                output = self._model(processed_inputs) # (B, num_classes)
                
                # 2. Post-process
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,       # (B,)
                    PyTorchInferenceKeys.PROBABILITIES: probs  # (B, num_classes)
                }

            elif self.task == "image_segmentation":
                # --- Image Segmentation ---
                # 1. Predict
                output = self._model(processed_inputs) # (B, num_classes, H, W)
                
                # 2. Post-process
                probs = torch.softmax(output, dim=1) # Probs across class channel
                labels = torch.argmax(probs, dim=1)  # (B, H, W) segmentation map
                return {
                    PyTorchInferenceKeys.LABELS: labels,       # (B, H, W)
                    PyTorchInferenceKeys.PROBABILITIES: probs  # (B, num_classes, H, W)
                }

            elif self.task == "object_detection":
                # --- Object Detection ---
                # 1. Predict (model is in eval mode, expects only images)
                # Output is List[Dict[str, Tensor('boxes', 'labels', 'scores')]]
                predictions = self._model(processed_inputs) 
                
                # 2. Post-process: Wrap in our standard key
                return {
                    PyTorchInferenceKeys.PREDICTIONS: predictions
                }
            
            else:
                # This should be unreachable due to __init__ check
                raise ValueError(f"Unknown task: {self.task}")

    def predict(self, single_input: torch.Tensor) -> Dict[str, Any]:
        """
        Core single-sample prediction method for vision models.
        All preprocessing (resizing, normalization) should be done *before*
        calling this method.

        Args:
            single_input (torch.Tensor):
                - A 3D torch.Tensor (C, H, W) for any task.

        Returns:
            A dictionary containing the output tensors for a single sample.
            - Classification: {labels, probabilities} (label is 0-dim)
            - Segmentation: {labels, probabilities} (label is 2D mask)
            - Object Detection: {boxes, labels, scores} (single dict)
        """
        if not isinstance(single_input, torch.Tensor) or single_input.ndim != 3:
             _LOGGER.error(f"Input for predict() must be a 3D tensor (C, H, W). Got {single_input.ndim}D.")
             raise ValueError("Input must be a 3D tensor.")
        
        # --- 1. Batch the input based on task ---
        if self.task == "object_detection":
            batched_input = [single_input] # List of one tensor
        else:
            batched_input = single_input.unsqueeze(0) # (1, C, H, W)

        # --- 2. Call batch prediction ---
        batch_results = self.predict_batch(batched_input)

        # --- 3. Un-batch the results ---
        if self.task == "object_detection":
            # batch_results['predictions'] is a List[Dict]. We want the first (and only) Dict.
            return batch_results[PyTorchInferenceKeys.PREDICTIONS][0]
        else:
            # 'labels' and 'probabilities' are tensors. Get the 0-th element.
            # (B, ...) -> (...)
            single_results = {key: value[0] for key, value in batch_results.items()}
            return single_results

    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, Any]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays. With Labels if set.
        
        Returns:
            Dict: A dictionary containing the outputs as NumPy arrays.
            - Obj. Detection: {predictions: List[Dict[str, np.ndarray]]}
            - Classification: {labels: int, label_names: str, probabilities: np.ndarray}
            - Segmentation: {labels: np.ndarray, probabilities: np.ndarray}
        """
        tensor_results = self.predict_batch(inputs)
        
        if self.task == "object_detection":
            # Output is List[Dict[str, Tensor]]
            # Convert each tensor inside each dict to numpy
            numpy_results = []
            for pred_dict in tensor_results[PyTorchInferenceKeys.PREDICTIONS]:
                # Convert all tensors to numpy
                np_dict = {key: value.cpu().numpy() for key, value in pred_dict.items()}
                
                # Add string names if map exists
                if self._idx_to_class and PyTorchInferenceKeys.LABELS in np_dict:
                    np_dict[PyTorchInferenceKeys.LABEL_NAMES] = [
                        self._idx_to_class.get(label_id, "Unknown") 
                        for label_id in np_dict[PyTorchInferenceKeys.LABELS]
                    ]
                numpy_results.append(np_dict)
            return {PyTorchInferenceKeys.PREDICTIONS: numpy_results}
        else:
            # Output is Dict[str, Tensor]
            numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
            
            # Add string names for classification if map exists
            if self.task == "image_classification" and self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
                int_labels = numpy_results[PyTorchInferenceKeys.LABELS]
                numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
                    self._idx_to_class.get(label_id, "Unknown")
                    for label_id in int_labels
                ]
            
            return numpy_results

    def predict_numpy(self, single_input: torch.Tensor) -> Dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays/scalars.

        Returns:
            Dict: A dictionary containing the outputs as NumPy arrays/scalars.
            - Obj. Detection: {boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray}
            - Classification: {labels: int, label_names: str, probabilities: np.ndarray}
            - Segmentation: {labels: np.ndarray, probabilities: np.ndarray}
        """
        tensor_results = self.predict(single_input)
        
        if self.task == "object_detection":
            # Output is Dict[str, Tensor]
            # Convert each tensor to numpy
            numpy_results = {
                key: value.cpu().numpy() for key, value in tensor_results.items()
            }
            
            # Add string names if map exists
            if self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
                int_labels = numpy_results[PyTorchInferenceKeys.LABELS]
                
                numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
                    self._idx_to_class.get(label_id, "Unknown")
                    for label_id in int_labels
                ]
                
            return numpy_results
            
        elif self.task == "image_classification":
            # Output is Dict[str, Tensor(0-dim) or Tensor(1-dim)]
            int_label = tensor_results[PyTorchInferenceKeys.LABELS].item()
            label_name = "Unknown"
            if self._idx_to_class:
                label_name = self._idx_to_class.get(int_label, "Unknown")

            return {
                PyTorchInferenceKeys.LABELS: int_label,
                PyTorchInferenceKeys.LABEL_NAMES: label_name,
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
        else: # image_segmentation
            # Output is Dict[str, Tensor(2D) or Tensor(3D)]
            return {
                PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].cpu().numpy(),
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
            
    def predict_from_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction.

        Args:
            image_path (str | Path): The file path to the input image.

        Returns:
            Dict: A dictionary containing the prediction results. See `predict_numpy()` for task-specific output structures.
        """
        if self._transform is None:
            _LOGGER.error("Cannot predict from file: No transform has been set. Call .set_transform() or provide transform_source in __init__.")
            raise RuntimeError("Inference transform is not set.")
        
        try:
            # --- Use expected_in_channels to set PIL mode ---
            pil_mode: str
            if self.expected_in_channels == 1:
                pil_mode = "L"  # Grayscale
            elif self.expected_in_channels == 4:
                pil_mode = "RGBA" # RGB + Alpha
            else:
                if self.expected_in_channels != 3: # 2, 5+ channels not supported by PIL convert
                    _LOGGER.warning(f"Model expects {self.expected_in_channels} channels. PIL conversion is limited, defaulting to 3 channels (RGB). The transformations must convert it to the desired channel dimensions.")
                # Default to RGB. If 2-channels are needed, the transform recipe *must* be responsible for handling the conversion from a 3-channel PIL image.
                pil_mode = "RGB"
                
            image = Image.open(image_path).convert(pil_mode)
        except Exception as e:
            _LOGGER.error(f"Failed to load and convert image from '{image_path}': {e}")
            raise

        # Apply the transformation pipeline (e.g., resize, crop, ToTensor, normalize)
        try:
            transformed_image = self._transform(image)
        except Exception as e:
            _LOGGER.error(f"Error applying transform to image: {e}")
            raise
            
        # --- Validation ---
        if not isinstance(transformed_image, torch.Tensor):
            _LOGGER.error("The provided transform did not return a torch.Tensor. Does it include transforms.ToTensor()?")
            raise ValueError("Transform pipeline must output a torch.Tensor.")
            
        if transformed_image.ndim != 3:
            _LOGGER.warning(f"Expected transform to output a 3D (C, H, W) tensor, but got {transformed_image.ndim}D. Attempting to proceed.")
            # .predict() which expects a 3D tensor
            if transformed_image.ndim == 4 and transformed_image.shape[0] == 1:
                transformed_image = transformed_image.squeeze(0) # Fix if user's transform adds a batch dim
                _LOGGER.warning("Removed an extra batch dimension.")
            else:
                raise ValueError(f"Transform must output a 3D (C, H, W) tensor, got {transformed_image.shape}.")

        # Use the existing single-item predict method
        return self.predict_numpy(transformed_image)


def info():
    _script_info(__all__)
