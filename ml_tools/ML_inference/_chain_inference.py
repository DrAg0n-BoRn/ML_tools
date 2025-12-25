import torch
import numpy as np
from typing import Union, Any

from ..keys._keys import MLTaskKeys, PyTorchInferenceKeys
from .._core import get_logger

from ._dragon_inference import DragonInferenceHandler


_LOGGER = get_logger("DragonChainInference")


__all__ = [
    "DragonChainInference"
]


class DragonChainInference:
    """
    Orchestrates a sequential chain of DragonInferenceHandlers.
    
    The output of each handler in the chain is appended to the feature vector 
    passed to the next handler. This allows for 'Chain of Regression' or 
    'Classifier Chains' where subsequent models depend on the predictions 
    of previous models.
    """
    def __init__(self, handlers: list[DragonInferenceHandler]):
        """
        Args:
            handlers (List[DragonInferenceHandler]): An ordered list of inference handlers. 
                The order matters: output of handlers[0] feeds into handlers[1], etc.
        """
        if not handlers:
            _LOGGER.error("Initialization failed: handlers list is empty.")
            raise ValueError()
        
        self.handlers = handlers
        self._validate_chain()
        
        # process target ids internally
        all_ids = []
        for h in self.handlers:
            if h.target_ids:
                all_ids.extend(h.target_ids)
        self._all_target_ids = all_ids

    def _validate_chain(self):
        """Ensures all handlers have valid target_ids and warns about duplicates."""
        seen_targets = set()
        for i, h in enumerate(self.handlers):
            if h.target_ids is None:
                _LOGGER.error(f"Handler at index {i} is missing 'target_ids'.")
                raise AttributeError()
            
            for tid in h.target_ids:
                if tid in seen_targets:
                    _LOGGER.warning(f"Duplicate target_id '{tid}' detected in chain at index {i}. Previous predictions for this target will be overwritten in the final output.")
                seen_targets.add(tid)

    @property
    def target_ids(self) -> list[str]:
        """Returns a unified list of all target_ids in the chain order."""
        return self._all_target_ids

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the inference chain on a batch of features.

        Args:
            features (np.ndarray | torch.Tensor): The initial input features (2D).

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping every target_id in the chain 
            to its predicted tensor.
        """
        # We perform operations on CPU or let the handlers manage device transfer internally.
        # To maintain the "growing tensor", we keep current_features as the object to pass.
        # Note: DragonInferenceHandler accepts numpy or torch, so we can pass the output 
        # of one (torch) directly to the next.
        
        # Ensure input is 2D
        if isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)
        elif isinstance(features, torch.Tensor) and features.ndim == 1:
            features = features.unsqueeze(0)

        current_features = features
        results: dict[str, torch.Tensor] = {}

        for i, handler in enumerate(self.handlers):
            # 1. Predict
            # We assume handler handles device movement internally via _preprocess_input
            output_dict = handler.predict_batch(current_features)
            
            # 2. Extract the "primary" result for chaining
            # Classification -> Labels, Regression -> Predictions
            if handler.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
                step_output = output_dict[PyTorchInferenceKeys.LABELS]
            else:
                step_output = output_dict[PyTorchInferenceKeys.PREDICTIONS]
            
            # 3. Store results mapped to target_ids
            # Ensure step_output is 2D (N, Targets) for zipping, even if it's (N,)
            if step_output.ndim == 1:
                step_output_reshaped = step_output.unsqueeze(1)
            else:
                step_output_reshaped = step_output

            if handler.target_ids:
                for idx, target_name in enumerate(handler.target_ids):
                    # We store the 1D slice (N,) for the dictionary
                    results[target_name] = step_output_reshaped[:, idx]

            # 4. Augment features for the next step (if there is a next step)
            if i < len(self.handlers) - 1:
                # We need to concatenate. 
                # Ensure current_features is a Tensor (it might be numpy on first iter)
                if isinstance(current_features, np.ndarray):
                    current_features = torch.from_numpy(current_features).float()
                
                # Ensure step_output is on the same device/type as current_features
                # Usually handlers return tensors on their device.
                # We move prediction to current_features device for concatenation
                step_output_reshaped = step_output_reshaped.to(current_features.device)
                
                # If step_output is Int (Labels), it must be float to concatenate with features.
                step_output_reshaped = step_output_reshaped.float()
                
                current_features = torch.cat([current_features, step_output_reshaped], dim=1)

        return results
    
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the chain on a single sample, returning PyTorch Tensors.
        
        Args:
            features: 1D array/tensor.
            
        Returns:
            Dict containing 0-dim tensors (scalars) for the predictions.
        """
        # Ensure input is 2D batch of 1
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        elif isinstance(features, torch.Tensor):
            if features.ndim == 1:
                features = features.unsqueeze(0)
        
        # Call batch prediction (returns Tensors)
        batch_results = self.predict_batch(features)

        # Extract single sample (0-dim tensors)
        single_results = {k: v[0] for k, v in batch_results.items()}
        return single_results

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        Convenience wrapper that returns NumPy arrays instead of Tensors.
        Useful for final consumption of the chain results.
        """
        tensor_results = self.predict_batch(features)
        return {k: v.cpu().numpy() for k, v in tensor_results.items()}

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
        """
        Runs the chain on a single sample, returning Python scalars or NumPy arrays.
        
        Args:
            features: 1D array/tensor.
            
        Returns:
            Dict containing standard Python types (float, int) or NumPy arrays.
        """
        # Get tensor results first
        tensor_results = self.predict(features)
        
        # Convert to NumPy/Scalar
        numpy_results = {}
        for k, v in tensor_results.items():
            if v.numel() == 1:
                numpy_results[k] = v.item()  # Convert to standard Python scalar
            else:
                numpy_results[k] = v.cpu().numpy() # Convert to array
                
        return numpy_results

