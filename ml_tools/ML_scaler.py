import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional

from ._logger import _LOGGER
from ._script_info import _script_info
from .path_manager import make_fullpath
from ._keys import ScalerKeys


__all__ = [
    "DragonScaler"
]


class DragonScaler:
    """
    Standardizes continuous features/targets by subtracting the mean and 
    dividing by the standard deviation.
    """
    def __init__(self,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 continuous_feature_indices: Optional[List[int]] = None):
        """
        Initializes the scaler.
        """
        self.mean_ = mean
        self.std_ = std
        self.continuous_feature_indices = continuous_feature_indices

    @classmethod
    def fit(cls, dataset: Dataset, continuous_feature_indices: List[int], batch_size: int = 64) -> 'DragonScaler':
        """
        Fits the scaler using a PyTorch Dataset (Method A).
        """
        if not continuous_feature_indices:
            _LOGGER.error("No continuous feature indices provided. Scaler will not be fitted.")
            return cls()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        running_sum, running_sum_sq = None, None
        count = 0
        num_continuous_features = len(continuous_feature_indices)

        for features, _ in loader:
            if running_sum is None:
                device = features.device
                running_sum = torch.zeros(num_continuous_features, device=device)
                running_sum_sq = torch.zeros(num_continuous_features, device=device)

            continuous_features = features[:, continuous_feature_indices].to(device)
            
            running_sum += torch.sum(continuous_features, dim=0)
            running_sum_sq += torch.sum(continuous_features**2, dim=0) # type: ignore
            count += continuous_features.size(0)

        if count == 0:
             _LOGGER.error("Dataset is empty. Scaler cannot be fitted.")
             return cls(continuous_feature_indices=continuous_feature_indices)

        # Calculate mean
        mean = running_sum / count

        # Calculate standard deviation
        if count < 2:
            _LOGGER.warning(f"Only one sample found. Standard deviation cannot be calculated and is set to 1.")
            std = torch.ones_like(mean)
        else:
            # var = E[X^2] - (E[X])^2
            var = (running_sum_sq / count) - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-8)) # Clamp for numerical stability

        _LOGGER.info(f"Scaler fitted on {count} samples for {num_continuous_features} features.")
        return cls(mean=mean, std=std, continuous_feature_indices=continuous_feature_indices)

    @classmethod
    def fit_tensor(cls, data: torch.Tensor) -> 'DragonScaler':
        """
        Fits the scaler directly on a Tensor (Method B).
        Useful for targets or small datasets already in memory.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        num_features = data.shape[1]
        indices = list(range(num_features))
        
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        
        # Handle constant values (std=0) to prevent division by zero
        std = torch.where(std == 0, torch.tensor(1.0, device=data.device), std)
        
        return cls(mean=mean, std=std, continuous_feature_indices=indices)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies standardization.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            # If not fitted, return as is
            return data

        data_clone = data.clone()
        
        # Ensure mean and std are on the same device as the data
        mean = self.mean_.to(data.device)
        std = self.std_.to(data.device)
        
        # Extract the columns to be scaled
        features_to_scale = data_clone[:, self.continuous_feature_indices]
        
        # Apply scaling, adding epsilon to std to prevent division by zero
        scaled_features = (features_to_scale - mean) / (std + 1e-8)
        
        # Place the scaled features back into the cloned tensor
        data_clone[:, self.continuous_feature_indices] = scaled_features
        
        return data_clone

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the standardization transformation.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            return data
            
        data_clone = data.clone()
        
        mean = self.mean_.to(data.device)
        std = self.std_.to(data.device)
        
        features_to_inverse = data_clone[:, self.continuous_feature_indices]
        
        # Apply inverse scaling
        original_scale_features = (features_to_inverse * (std + 1e-8)) + mean
        
        data_clone[:, self.continuous_feature_indices] = original_scale_features
        
        return data_clone

    def _get_state(self):
        """Helper to get state dict."""
        return {
            ScalerKeys.MEAN: self.mean_,
            ScalerKeys.STD: self.std_,
            ScalerKeys.INDICES: self.continuous_feature_indices
        }

    def save(self, filepath: Union[str, Path], verbose: bool=True):
        """
        Saves the scaler's state. 
        """
        path_obj = make_fullpath(filepath, make=True, enforce="file")
        state = self._get_state()
        torch.save(state, path_obj)
        if verbose:
            _LOGGER.info(f"DragonScaler state saved as '{path_obj.name}'.")

    @classmethod
    def load(cls, filepath_or_state: Union[str, Path, dict], verbose: bool=True) -> 'DragonScaler':
        """
        Loads a scaler's state from a .pth file OR a dictionary.
        """
        if isinstance(filepath_or_state, (str, Path)):
            path_obj = make_fullpath(filepath_or_state, enforce="file")
            state = torch.load(path_obj)
            source_name = path_obj.name
        else:
            state = filepath_or_state
            source_name = "dictionary"
            
        # Handle cases where the state might be None (scaler was not fitted)
        if state is None:
            _LOGGER.warning(f"Loaded DragonScaler state is None from '{source_name}'. Returning unfitted scaler.")
            return DragonScaler()

        if verbose:
            _LOGGER.info(f"DragonScaler state loaded from '{source_name}'.")
            
        return DragonScaler(
            mean=state[ScalerKeys.MEAN],
            std=state[ScalerKeys.STD],
            continuous_feature_indices=state[ScalerKeys.INDICES]
        )
    
    def __repr__(self) -> str:
        if self.continuous_feature_indices:
            num_features = len(self.continuous_feature_indices)
            return f"DragonScaler(fitted for {num_features} columns)"
        return "DragonScaler(not fitted)"

def info():
    _script_info(__all__)