import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional
from ._logger import _LOGGER
from ._script_info import _script_info
from .path_manager import make_fullpath

__all__ = [
    "PytorchScaler"
]

class PytorchScaler:
    """
    Standardizes continuous features in a PyTorch dataset by subtracting the
    mean and dividing by the standard deviation.

    The scaler is fitted on a training dataset and can then be saved and
    loaded for consistent transformation during inference.
    """
    def __init__(self,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 continuous_feature_indices: Optional[List[int]] = None):
        """
        Initializes the scaler.

        Args:
            mean (torch.Tensor, optional): The mean of the features to scale.
            std (torch.Tensor, optional): The standard deviation of the features.
            continuous_feature_indices (List[int], optional): The column indices of the features to standardize.
        """
        self.mean_ = mean
        self.std_ = std
        self.continuous_feature_indices = continuous_feature_indices

    @classmethod
    def fit(cls, dataset: Dataset, continuous_feature_indices: List[int], batch_size: int = 64) -> 'PytorchScaler':
        """
        Fits the scaler by computing the mean and std dev from a dataset using a
        fast, single-pass, vectorized algorithm.

        Args:
            dataset (Dataset): The PyTorch Dataset to fit on.
            continuous_feature_indices (List[int]): The column indices of the
                features to standardize.
            batch_size (int): The batch size for iterating through the dataset.

        Returns:
            PytorchScaler: A new, fitted instance of the scaler.
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

        _LOGGER.info(f"Scaler fitted on {count} samples for {num_continuous_features} continuous features.")
        return cls(mean=mean, std=std, continuous_feature_indices=continuous_feature_indices)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies standardization to the specified continuous features.

        Args:
            data (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The transformed data tensor.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            _LOGGER.error("Scaler has not been fitted. Returning original data.")
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

        Args:
            data (torch.Tensor): The scaled data tensor.

        Returns:
            torch.Tensor: The original-scale data tensor.
        """
        if self.mean_ is None or self.std_ is None or self.continuous_feature_indices is None:
            _LOGGER.error("Scaler has not been fitted. Returning original data.")
            return data
            
        data_clone = data.clone()
        
        mean = self.mean_.to(data.device)
        std = self.std_.to(data.device)
        
        features_to_inverse = data_clone[:, self.continuous_feature_indices]
        
        # Apply inverse scaling
        original_scale_features = (features_to_inverse * (std + 1e-8)) + mean
        
        data_clone[:, self.continuous_feature_indices] = original_scale_features
        
        return data_clone

    def save(self, filepath: Union[str, Path], verbose: bool=True):
        """
        Saves the scaler's state (mean, std, indices) to a .pth file.

        Args:
            filepath (str | Path): The path to save the file.
        """
        path_obj = make_fullpath(filepath, make=True, enforce="file")
        state = {
            'mean': self.mean_,
            'std': self.std_,
            'continuous_feature_indices': self.continuous_feature_indices
        }
        torch.save(state, path_obj)
        if verbose:
            _LOGGER.info(f"PytorchScaler state saved as '{path_obj.name}'.")

    @staticmethod
    def load(filepath: Union[str, Path], verbose: bool=True) -> 'PytorchScaler':
        """
        Loads a scaler's state from a .pth file.

        Args:
            filepath (str | Path): The path to the saved scaler file.

        Returns:
            PytorchScaler: An instance of the scaler with the loaded state.
        """
        path_obj = make_fullpath(filepath, enforce="file")
        state = torch.load(path_obj)
        if verbose:
            _LOGGER.info(f"PytorchScaler state loaded from '{path_obj.name}'.")
        return PytorchScaler(
            mean=state['mean'],
            std=state['std'],
            continuous_feature_indices=state['continuous_feature_indices']
        )
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the scaler."""
        if self.continuous_feature_indices:
            num_features = len(self.continuous_feature_indices)
            return f"PytorchScaler(fitted for {num_features} features)"
        return "PytorchScaler(not fitted)"


def info():
    _script_info(__all__)
