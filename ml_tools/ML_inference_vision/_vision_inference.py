import torch
from torch import nn
from pathlib import Path
from typing import Union, Literal, Any, Optional, Callable
from PIL import Image

from ..ML_vision_transformers._core_transforms import _load_recipe_and_build_transform

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys

from ..ML_inference._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("DragonVisionInference")


__all__ = [
    "DragonVisionInferenceHandler"
]


class DragonVisionInferenceHandler(_BaseInferenceHandler):
    """
    Handles loading a PyTorch vision model's state dictionary and performing inference.

    This class is specifically for vision models, which typically expect
    4D Tensors (B, C, H, W) or Lists of Tensors as input.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 task: Optional[Literal["binary image classification", "multiclass image classification", "binary segmentation", "multiclass segmentation", "object detection"]] = None,
                 device: str = 'cpu',
                 transform_source: Optional[Union[str, Path, Callable]] = None):
        """
        Initializes the vision inference handler.

        Args:
            model (nn.Module): An instantiated PyTorch model from ML_vision_models.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str, optional): The type of vision task. If None, detected from file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            transform_source (str | Path | Callable | None): 
                - A path to a .json recipe file (str or Path).
                - A pre-built transformation pipeline (Callable).
                - None, in which case .set_transform() must be called explicitly to set transformations.
        
        Note: class_map (Dict[int, str]) will be loaded from the model file, to set or override it use `.set_class_map()`.
        """
        # Parent initializes FinalizedFileHandler, loads task, weights, etc.
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         scaler=None, 
                         task=task)

        # --- Validate Task ---
        valid_tasks = [
            MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, 
            MLTaskKeys.BINARY_SEGMENTATION, 
            MLTaskKeys.MULTICLASS_SEGMENTATION, 
            MLTaskKeys.OBJECT_DETECTION
        ]
        
        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
            raise ValueError()

        self._transform: Optional[Callable] = None
        self._is_transformed: bool = False
        
        # --- Model specific channels ---
        self.expected_in_channels: int = 3 # Default to RGB
        if hasattr(model, 'in_channels'):
            self.expected_in_channels = model.in_channels # type: ignore
            _LOGGER.info(f"Model expects {self.expected_in_channels} input channels.")
        else:
            _LOGGER.warning("Could not determine 'in_channels' from model. Defaulting to 3 (RGB). Modify with '.expected_in_channels'.")
        
        if transform_source:
            self.set_transform(transform_source)
            self._is_transformed = True

    def _preprocess_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Validates input and moves it to the correct device.
        - For Classification/Segmentation: Expects 4D Tensor (B, C, H, W).
        - For Object Detection: Expects List[Tensor(C, H, W)].
        """
        if self.task == MLTaskKeys.OBJECT_DETECTION:
            if not isinstance(inputs, list) or not all(isinstance(t, torch.Tensor) for t in inputs):
                _LOGGER.error("Input for object_detection must be a List[torch.Tensor].")
                raise ValueError("Invalid input type for object detection.")
            # Move each tensor in the list to the device
            return [t.float().to(self.device) for t in inputs]
        
        else: # Classification or Segmentation
            if not isinstance(inputs, torch.Tensor):
                _LOGGER.error(f"Input for {self.task} must be a torch.Tensor.")
                raise ValueError(f"Invalid input type for {self.task}.")
                
            if inputs.ndim != 4: # type: ignore
                 _LOGGER.error(f"Input tensor for {self.task} must be 4D (B, C, H, W). Got {inputs.ndim}D.") # type: ignore
                 raise ValueError("Input tensor must be 4D.")
            
            return inputs.float().to(self.device) 
        
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
                # Use the loader function
                self._transform = _load_recipe_and_build_transform(transform_source)
            except Exception as e:
                _LOGGER.error(f"Failed to load transform from recipe '{transform_source}': {e}")
                raise
        elif isinstance(transform_source, Callable):
            _LOGGER.info("Inference transform has been set from a direct Callable.")
            self._transform = transform_source
        else:
            _LOGGER.error(f"Invalid transform_source type: {type(transform_source)}. Must be str, Path, or Callable.")
            raise TypeError("transform_source must be a file path or a Callable.")

    def predict_batch(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Core batch prediction method for vision models.
        All preprocessing (resizing, normalization) should be done *before* calling this method.

        Args:
            inputs (torch.Tensor | List[torch.Tensor]):
                - For binary/multiclass image classification or binary/multiclass image segmentation tasks, 
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
            # get outputs
            output = self.model(processed_inputs)
            if self.task == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION:
                # process
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1)
                return {
                    PyTorchInferenceKeys.LABELS: labels,       # (B,)
                    PyTorchInferenceKeys.PROBABILITIES: probs  # (B, num_classes)
                }
            
            elif self.task == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                # Assumes model output is [N, 1] (a single logit)
                # Squeeze output from [N, 1] to [N] if necessary
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
                    
                probs = torch.sigmoid(output) # Probability of positive class
                labels = (probs >= self._classification_threshold).int()
                return {
                    PyTorchInferenceKeys.LABELS: labels,       
                    PyTorchInferenceKeys.PROBABILITIES: probs  
                }
            
            elif self.task == MLTaskKeys.BINARY_SEGMENTATION:
                # Assumes model output is [N, 1, H, W] (logits for positive class)
                probs = torch.sigmoid(output) # Shape [N, 1, H, W]
                labels = (probs >= self._classification_threshold).int() # Shape [N, 1, H, W]
                return {
                    PyTorchInferenceKeys.LABELS: labels,       
                    PyTorchInferenceKeys.PROBABILITIES: probs  
                }
                
            elif self.task == MLTaskKeys.MULTICLASS_SEGMENTATION:
                # output shape [N, C, H, W]
                probs = torch.softmax(output, dim=1)
                labels = torch.argmax(probs, dim=1) # shape [N, H, W]
                return {
                    PyTorchInferenceKeys.LABELS: labels,       # (N, H, W)
                    PyTorchInferenceKeys.PROBABILITIES: probs  # (N, num_classes, H, W)
                }

            elif self.task == MLTaskKeys.OBJECT_DETECTION:
                return {
                    PyTorchInferenceKeys.PREDICTIONS: output
                }
            
            else:
                # This should be unreachable due to validation
                raise ValueError(f"Unknown task: {self.task}")

    def predict(self, single_input: torch.Tensor) -> dict[str, Any]:
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
            - Segmentation: {labels, probabilities} (label is a 2D (multiclass) or 3D (binary) mask)
            - Object Detection: {boxes, labels, scores} (single dict)
        """
        if not isinstance(single_input, torch.Tensor) or single_input.ndim != 3:
             _LOGGER.error(f"Input for predict() must be a 3D tensor (C, H, W). Got {single_input.ndim}D.")
             raise ValueError()
        
        # --- 1. Batch the input based on task ---
        if self.task == MLTaskKeys.OBJECT_DETECTION:
            batched_input = [single_input] # List of one tensor
        else:
            batched_input = single_input.unsqueeze(0)

        # --- 2. Call batch prediction ---
        batch_results = self.predict_batch(batched_input)

        # --- 3. Un-batch the results ---
        if self.task == MLTaskKeys.OBJECT_DETECTION:
            # batch_results['predictions'] is a List[Dict]. We want the first (and only) Dict.
            return batch_results[PyTorchInferenceKeys.PREDICTIONS][0]
        else:
            # 'labels' and 'probabilities' are tensors. Get the 0-th element.
            # (B, ...) -> (...)
            single_results = {key: value[0] for key, value in batch_results.items()}
            return single_results

    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, inputs: Union[torch.Tensor, list[torch.Tensor]]) -> dict[str, Any]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays. With Labels if set.
        
        Returns:
            Dict: A dictionary containing the outputs as NumPy arrays.
            - Obj. Detection: {predictions: List[Dict[str, np.ndarray]]}
            - Classification: {labels: np.ndarray, label_names: List[str], probabilities: np.ndarray}
            - Segmentation: {labels: np.ndarray, probabilities: np.ndarray}
        """
        tensor_results = self.predict_batch(inputs)
        
        if self.task == MLTaskKeys.OBJECT_DETECTION:
            # Output is List[Dict[str, Tensor]]
            # Convert each tensor inside each dict to numpy
            numpy_results = []
            for pred_dict in tensor_results[PyTorchInferenceKeys.PREDICTIONS]:
                # Convert all tensors to numpy
                np_dict = {key: value.cpu().numpy() for key, value in pred_dict.items()}
                
                # 3D pixel to string map unnecessary 
                # if self._idx_to_class and PyTorchInferenceKeys.LABELS in np_dict:
                #     np_dict[PyTorchInferenceKeys.LABEL_NAMES] = [
                #         self._idx_to_class.get(label_id, "Unknown") 
                #         for label_id in np_dict[PyTorchInferenceKeys.LABELS]
                #     ]
                numpy_results.append(np_dict)
            return {PyTorchInferenceKeys.PREDICTIONS: numpy_results}
        
        else:
            # Output is Dict[str, Tensor] (for Classification or Segmentation)
            numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
            
            # Add string names for classification if map exists
            is_image_classification = self.task in [
                MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
                MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
            ]
            
            if is_image_classification and self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
                int_labels = numpy_results[PyTorchInferenceKeys.LABELS] # This is a (B,) array
                numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
                    self._idx_to_class.get(label_id, "Unknown")
                    for label_id in int_labels
                ]
            
            return numpy_results

    def predict_numpy(self, single_input: torch.Tensor) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays/scalars.

        Returns:
            Dict: A dictionary containing the outputs as NumPy arrays/scalars.
            - Obj. Detection: {boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, label_names: List[str]}
            - Classification: {labels: int, label_names: str, probabilities: np.ndarray}
            - Segmentation: {labels: np.ndarray, probabilities: np.ndarray}
        """
        tensor_results = self.predict(single_input)
        
        if self.task == MLTaskKeys.OBJECT_DETECTION:
            # Output is Dict[str, Tensor]
            # Convert each tensor to numpy
            numpy_results = {
                key: value.cpu().numpy() for key, value in tensor_results.items()
            }
            
            # Add string names if map exists
            # if self._idx_to_class and PyTorchInferenceKeys.LABELS in numpy_results:
            #     int_labels = numpy_results[PyTorchInferenceKeys.LABELS]
                
            #     numpy_results[PyTorchInferenceKeys.LABEL_NAMES] = [
            #         self._idx_to_class.get(label_id, "Unknown")
            #         for label_id in int_labels
            #     ]
                
            return numpy_results
            
        elif self.task in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
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
        else: # image_segmentation (binary or multiclass)
            # Output is Dict[str, Tensor(2D) or Tensor(3D)]
            return {
                PyTorchInferenceKeys.LABELS: tensor_results[PyTorchInferenceKeys.LABELS].cpu().numpy(),
                PyTorchInferenceKeys.PROBABILITIES: tensor_results[PyTorchInferenceKeys.PROBABILITIES].cpu().numpy()
            }
            
    def predict_from_pil(self, image: Image.Image) -> dict[str, Any]:
        """
        Applies the stored transform to a single PIL image and returns the prediction.

        Args:
            image (PIL.Image.Image): The input PIL image.

        Returns:
            Dict: A dictionary containing the prediction results. See `predict_numpy()` for task-specific output structures.
        """
        if self._transform is None:
            _LOGGER.error("Cannot predict from PIL image: No transform has been set. Call .set_transform() or provide transform_source in __init__.")
            raise RuntimeError("Inference transform is not set.")

        # Apply the transformation pipeline (e.g., resize, crop, ToTensor, normalize)
        try:
            transformed_image = self._transform(image)
        except Exception as e:
            _LOGGER.error(f"Error applying transform to PIL image: {e}")
            raise
            
        # --- Validation ---
        if not isinstance(transformed_image, torch.Tensor):
            _LOGGER.error("The provided transform did not return a torch.Tensor. Does it include transforms.ToTensor()?")
            raise ValueError("Transform pipeline must output a torch.Tensor.")
            
        if transformed_image.ndim != 3:
            _LOGGER.warning(f"Expected transform to output a 3D (C, H, W) tensor, but got {transformed_image.ndim}D. Attempting to proceed.")
            # .predict_numpy() -> .predict() which expects a 3D tensor
            if transformed_image.ndim == 4 and transformed_image.shape[0] == 1:
                transformed_image = transformed_image.squeeze(0) # Fix if user's transform adds a batch dim
                _LOGGER.warning("Removed an extra batch dimension.")
            else:
                raise ValueError(f"Transform must output a 3D (C, H, W) tensor, got {transformed_image.shape}.")

        # Use the existing single-item predict method
        return self.predict_numpy(transformed_image)

    def predict_from_file(self, image_path: Union[str, Path]) -> dict[str, Any]:
        """
        Loads a single image from a file, applies the stored transform, and returns the prediction.

        This is a convenience wrapper that loads the image and calls `predict_from_pil()`.

        Args:
            image_path (str | Path): The file path to the input image.

        Returns:
            Dict: A dictionary containing the prediction results. See `predict_numpy()` for task-specific output structures.
        """
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

        # Call the PIL-based prediction method
        return self.predict_from_pil(image)

