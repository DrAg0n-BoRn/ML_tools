import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Optional

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import ScalerKeys


_LOGGER = get_logger("DragonScaler")


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
                 continuous_feature_indices: Optional[list[int]] = None):
        """
        Initializes the scaler.
        """
        self.mean_ = mean
        self.std_ = std
        self.continuous_feature_indices = continuous_feature_indices

    @classmethod
    def fit(cls, dataset: Dataset, continuous_feature_indices: list[int], batch_size: int = 64, verbose: int = 3) -> 'DragonScaler':
        """
        Fits the scaler using a PyTorch Dataset (Method A) using Batched Welford's Algorithm.
        """
        if not continuous_feature_indices:
            _LOGGER.error("No continuous feature indices provided. Scaler will not be fitted.")
            raise ValueError()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Welford's Algorithm State
        n_total = 0
        mean_global = None
        m2_global = None # Sum of squares of differences from the mean
        
        num_continuous_features = len(continuous_feature_indices)

        for features, _ in loader:
            # Extract continuous features for this batch
            x_batch = features[:, continuous_feature_indices].to(features.device).float()
            
            n_batch = x_batch.shape[0]
            if n_batch == 0:
                continue
                
            # Batch statistics
            mean_batch = torch.mean(x_batch, dim=0)
            # Sum of squared differences from the batch mean
            m2_batch = torch.sum((x_batch - mean_batch) ** 2, dim=0)

            if n_total == 0:
                # Initialize global stats with first batch
                mean_global = mean_batch
                m2_global = m2_batch
                n_total = n_batch
            else:
                # Batched Welford's Update
                # Combine existing global stats (A) with new batch stats (B)
                delta = mean_batch - mean_global # type: ignore
                new_n_total = n_total + n_batch
                
                # Update M2 (Sum of Squares)
                # Formula: M2_X = M2_A + M2_B + delta^2 * (n_A * n_B / n_X)
                m2_global += m2_batch + (delta ** 2) * (n_total * n_batch / new_n_total)
                
                # Update Mean
                # Formula: mean_X = mean_A + delta * (n_B / n_X)
                mean_global += delta * (n_batch / new_n_total)
                
                n_total = new_n_total

        if n_total == 0:
            _LOGGER.error("Dataset is empty. Scaler cannot be fitted.")
            return cls(continuous_feature_indices=continuous_feature_indices)

        # Finalize Standard Deviation
        # Unbiased estimator (divide by n-1)
        if n_total < 2:
            if verbose >= 1:
                _LOGGER.warning(f"Only one sample found. Standard deviation set to 1.")
            std = torch.ones_like(mean_global) # type: ignore
        else:
            variance = m2_global / (n_total - 1)
            std = torch.sqrt(torch.clamp(variance, min=1e-8))
        
        if verbose >= 2:
            _LOGGER.info(f"Scaler fitted on {n_total} samples for {num_continuous_features} features (Welford's).")
        return cls(mean=mean_global, std=std, continuous_feature_indices=continuous_feature_indices)

    @classmethod
    def fit_tensor(cls, data: torch.Tensor, verbose: int = 3) -> 'DragonScaler':
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
        
        if verbose >= 2:
            _LOGGER.info(f"Scaler fitted on tensor with {data.shape[0]} samples for {num_features} features.")
        
        return cls(mean=mean, std=std, continuous_feature_indices=indices)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies standardization.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            # If not fitted, return as is
            return data
        
        data_clone = data.clone()
        
        # Robust check: If data is 1D, assume it's a single feature/target column and reshape it to (N, 1) for the operation, then reshape back.
        input_is_1d = (data_clone.ndim == 1)
        if input_is_1d:
            data_clone = data_clone.view(-1, 1)

        # Ensure mean and std are on the same device as the data
        mean = self.mean_.to(data.device)
        std = self.std_.to(data.device)
        
        # Extract the columns to be scaled
        features_to_scale = data_clone[:, self.continuous_feature_indices]
        
        # Apply scaling, adding epsilon to std to prevent division by zero
        scaled_features = (features_to_scale - mean) / (std + 1e-8)
        
        # Place the scaled features back into the cloned tensor
        data_clone[:, self.continuous_feature_indices] = scaled_features
        
        if input_is_1d:
            return data_clone.view(-1)
        
        return data_clone

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the standardization transformation.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            return data
        
        data_clone = data.clone()
        
        input_is_1d = (data_clone.ndim == 1)
        if input_is_1d:
            data_clone = data_clone.view(-1, 1)
        
        mean = self.mean_.to(data.device)
        std = self.std_.to(data.device)
        
        features_to_inverse = data_clone[:, self.continuous_feature_indices]
        
        # Apply inverse scaling
        original_scale_features = (features_to_inverse * (std + 1e-8)) + mean
        
        data_clone[:, self.continuous_feature_indices] = original_scale_features
        
        if input_is_1d:
            return data_clone.view(-1)
        
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

