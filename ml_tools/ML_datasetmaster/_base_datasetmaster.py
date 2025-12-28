import torch
from torch.utils.data import Dataset
import pandas
import numpy
from typing import Union, Optional
from abc import ABC
from pathlib import Path

from ..IO_tools import save_list_strings, save_json
from ..ML_scaler import DragonScaler
from ..schema import FeatureSchema

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import DatasetKeys, ScalerKeys


_LOGGER = get_logger("DragonDataset")


__all__ = [
    "_BaseDatasetMaker",
    "_PytorchDataset",
]


# --- Internal Helper Class ---
class _PytorchDataset(Dataset):
    """
    Internal helper class to create a PyTorch Dataset.
    Converts numpy/pandas data into tensors for model consumption.
    """
    def __init__(self, features: Union[numpy.ndarray, pandas.DataFrame], 
                 labels: Union[numpy.ndarray, pandas.Series, pandas.DataFrame],
                 labels_dtype: torch.dtype,
                 features_dtype: torch.dtype = torch.float32,
                 feature_names: Optional[list[str]] = None,
                 target_names: Optional[list[str]] = None):
        
        if isinstance(features, numpy.ndarray):
            self.features = torch.tensor(features, dtype=features_dtype)
        else: # It's a pandas.DataFrame
            self.features = torch.tensor(features.to_numpy(), dtype=features_dtype)

        if isinstance(labels, numpy.ndarray):
            self.labels = torch.tensor(labels, dtype=labels_dtype)
        elif isinstance(labels, (pandas.Series, pandas.DataFrame)):
            self.labels = torch.tensor(labels.to_numpy(), dtype=labels_dtype)
        else:
            self.labels = torch.tensor(labels, dtype=labels_dtype)
            
        self._feature_names = feature_names
        self._target_names = target_names
        self._classes: list[str] = []
        self._class_map: dict[str,int] = dict()
        self._feature_scaler: Optional[DragonScaler] = None
        self._target_scaler: Optional[DragonScaler] = None
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    @property
    def feature_names(self):
        if self._feature_names is not None:
            return self._feature_names
        else:
            _LOGGER.error(f"Dataset {self.__class__} has not been initialized with any feature names.")
            raise ValueError()
        
    @property
    def target_names(self):
        if self._target_names is not None:
            return self._target_names
        else:
            _LOGGER.error(f"Dataset {self.__class__} has not been initialized with any target names.")
            raise ValueError()

    @property
    def classes(self):
        return self._classes
    
    @property
    def class_map(self):
        return self._class_map
    
    @property
    def feature_scaler(self):
        return self._feature_scaler
    
    @property
    def target_scaler(self):
        return self._target_scaler


# --- Abstract Base Class ---
class _BaseDatasetMaker(ABC):
    """
    Abstract base class for dataset makers. Contains shared logic.
    """
    def __init__(self):
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None
        
        self.feature_scaler: Optional[DragonScaler] = None
        self.target_scaler: Optional[DragonScaler] = None
        
        self._id: Optional[str] = None
        self._feature_names: list[str] = []
        self._target_names: list[str] = []
        self._X_train_shape = (0,0)
        self._X_val_shape = (0,0)
        self._X_test_shape = (0,0)
        self._y_train_shape = (0,)
        self._y_val_shape = (0,)
        self._y_test_shape = (0,)
        self.class_map: dict[str, int] = dict()
        self.classes: list[str] = list()
        
    def _prepare_feature_scaler(self, 
                        X_train: pandas.DataFrame, 
                        y_train: Union[pandas.Series, pandas.DataFrame], 
                        X_val: pandas.DataFrame,
                        X_test: pandas.DataFrame, 
                        label_dtype: torch.dtype, 
                        schema: FeatureSchema,
                        verbose:int = 3) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Internal helper to fit and apply a DragonScaler for FEATURES using a FeatureSchema."""
        continuous_feature_indices: Optional[list[int]] = None

        # Get continuous feature indices *from the schema*
        if schema.continuous_feature_names:
            if verbose >= 2:
                _LOGGER.info("Getting continuous feature indices from schema.")
            try:
                # Convert columns to a standard list for .index()
                train_cols_list = X_train.columns.to_list()
                # Map names from schema to column indices in the training DataFrame
                continuous_feature_indices = [train_cols_list.index(name) for name in schema.continuous_feature_names]
            except ValueError as e: 
                _LOGGER.error(f"Feature name from schema not found in training data columns:\n{e}")
                raise ValueError()
        else:
            if verbose >= 2:
                _LOGGER.info("No continuous features listed in schema. Feature scaler will not be fitted.")

        X_train_values = X_train.to_numpy()
        X_val_values = X_val.to_numpy()
        X_test_values = X_test.to_numpy()

        # continuous_feature_indices is derived
        if self.feature_scaler is None and continuous_feature_indices:
            if verbose >= 3:
                _LOGGER.info("Fitting a new DragonScaler on training features.")
            temp_train_ds = _PytorchDataset(X_train_values, y_train, label_dtype) 
            self.feature_scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices, verbose=verbose)

        if self.feature_scaler and self.feature_scaler.mean_ is not None:
            if verbose >= 3:
                _LOGGER.info("Applying scaler transformation to train, validation, and test feature sets.")
            X_train_tensor = self.feature_scaler.transform(torch.tensor(X_train_values, dtype=torch.float32))
            X_val_tensor = self.feature_scaler.transform(torch.tensor(X_val_values, dtype=torch.float32))
            X_test_tensor = self.feature_scaler.transform(torch.tensor(X_test_values, dtype=torch.float32))
            return X_train_tensor.numpy(), X_val_tensor.numpy(), X_test_tensor.numpy()
        
        if verbose >= 2:
            _LOGGER.info("Feature scaling transformation complete.")

        return X_train_values, X_val_values, X_test_values
    
    def _prepare_target_scaler(self,
                               y_train: Union[pandas.Series, pandas.DataFrame],
                               y_val: Union[pandas.Series, pandas.DataFrame],
                               y_test: Union[pandas.Series, pandas.DataFrame],
                               verbose: int = 3) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Internal helper to fit and apply a DragonScaler for TARGETS."""
        
        y_train_arr = y_train.to_numpy() if isinstance(y_train, (pandas.Series, pandas.DataFrame)) else y_train
        y_val_arr = y_val.to_numpy() if isinstance(y_val, (pandas.Series, pandas.DataFrame)) else y_val
        y_test_arr = y_test.to_numpy() if isinstance(y_test, (pandas.Series, pandas.DataFrame)) else y_test
        
        # --- Ensure targets are 2D (N, 1) if they are currently 1D (N,) ---
        if y_train_arr.ndim == 1: y_train_arr = y_train_arr.reshape(-1, 1)
        if y_val_arr.ndim == 1:   y_val_arr = y_val_arr.reshape(-1, 1)
        if y_test_arr.ndim == 1:  y_test_arr = y_test_arr.reshape(-1, 1)
        # ------------------------------------------------------------------

        if self.target_scaler is None:
            if verbose >= 2:
                _LOGGER.info("Fitting a new DragonScaler on training targets.")
            # Convert to float tensor for calculation
            y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
            self.target_scaler = DragonScaler.fit_tensor(y_train_tensor, verbose=verbose)
            
        if self.target_scaler and self.target_scaler.mean_ is not None:
            if verbose >= 3:
                 _LOGGER.info("Applying scaler transformation to train, validation, and test targets.")
            y_train_tensor = self.target_scaler.transform(torch.tensor(y_train_arr, dtype=torch.float32))
            y_val_tensor = self.target_scaler.transform(torch.tensor(y_val_arr, dtype=torch.float32))
            y_test_tensor = self.target_scaler.transform(torch.tensor(y_test_arr, dtype=torch.float32))
            return y_train_tensor.numpy(), y_val_tensor.numpy(), y_test_tensor.numpy()

        return y_train_arr, y_val_arr, y_test_arr
    
    def _attach_scalers_to_datasets(self):
        """Helper to attach the master scalers to the child datasets."""
        for ds in [self._train_ds, self._val_ds, self._test_ds]:
            if ds is not None:
                ds._feature_scaler = self.feature_scaler # type: ignore
                ds._target_scaler = self.target_scaler # type: ignore

    @property
    def train_dataset(self) -> Dataset:
        if self._train_ds is None: 
            _LOGGER.error("Train Dataset not yet created.")
            raise RuntimeError()
        return self._train_ds
    
    @property
    def validation_dataset(self) -> Dataset:
        if self._val_ds is None: 
            _LOGGER.error("Validation Dataset not yet created.")
            raise RuntimeError()
        return self._val_ds

    @property
    def test_dataset(self) -> Dataset:
        if self._test_ds is None: 
            _LOGGER.error("Test Dataset not yet created.")
            raise RuntimeError()
        return self._test_ds

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names
    
    @property
    def target_names(self) -> list[str]:
        return self._target_names
    
    @property
    def number_of_features(self) -> int:
        return len(self._feature_names)
    
    @property
    def number_of_targets(self) -> int:
        return len(self._target_names)

    @property
    def id(self) -> Optional[str]:
        return self._id

    @id.setter
    def id(self, dataset_id: str):
        if not isinstance(dataset_id, str): raise ValueError("ID must be a string.")
        self._id = dataset_id

    def dataframes_info(self) -> None:
        print("--- DataFrame Shapes After Split ---")
        print(f"  X_train shape: {self._X_train_shape}, y_train shape: {self._y_train_shape}")
        print(f"  X_val shape:   {self._X_val_shape}, y_val shape:   {self._y_val_shape}")
        print(f"  X_test shape:  {self._X_test_shape}, y_test shape:  {self._y_test_shape}")
        print("------------------------------------")
    
    def save_feature_names(self, directory: Union[str, Path], verbose: bool=True) -> None:
        save_list_strings(list_strings=self._feature_names,
                          directory=directory,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=verbose)
        
    def save_target_names(self, directory: Union[str, Path], verbose: bool=True) -> None:
        save_list_strings(list_strings=self._target_names,
                          directory=directory,
                          filename=DatasetKeys.TARGET_NAMES,
                          verbose=verbose)

    def save_scaler(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """
        Saves both feature and target scalers (if they exist) to a single .pth file
        using a dictionary structure.
        """
        if self.feature_scaler is None and self.target_scaler is None:
            _LOGGER.warning("No scalers (feature or target) were fitted. Nothing to save.")
            return

        if not self.id: 
            _LOGGER.error("Must set the dataset `id` before saving scaler.")
            raise ValueError()
        
        save_path = make_fullpath(directory, make=True, enforce="directory")
        sanitized_id = sanitize_filename(self.id)
        filename = f"{DatasetKeys.SCALER_PREFIX}{sanitized_id}.pth"
        filepath = save_path / filename
        
        # Construct the consolidated dictionary
        combined_state = {}
        
        print_message = "Saved "
        
        if self.feature_scaler:
            combined_state[ScalerKeys.FEATURE_SCALER] = self.feature_scaler._get_state()
            print_message += "feature scaler "
            
        if self.target_scaler:
            if self.feature_scaler:
                print_message += "and "
            combined_state[ScalerKeys.TARGET_SCALER] = self.target_scaler._get_state()
            print_message += "target scaler "
            
        torch.save(combined_state, filepath)
        
        if verbose:
            _LOGGER.info(f"{print_message}to '{filepath.name}'.")
            
    def save_class_map(self, directory: Union[str,Path], verbose: bool=True) -> None:
        """  
        Saves the class map dictionary to a JSON file.
        
        Args:
            directory (str | Path): Directory to save the class map.
            verbose (bool): Whether to print log messages.
        """
        if not self.class_map:
            _LOGGER.warning(f"No class_map defined. Skipping.")
            return
        
        log_name = f"Class_to_Index_{self.id}" if self.id else "Class_to_Index"
        
        save_json(data=self.class_map,
                  directory=directory,
                  filename=log_name,
                  verbose=False)
        
        if verbose:
            _LOGGER.info(f"Class map for '{self.id}' saved as '{log_name}.json'.")

    def save_artifacts(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """
        Saves all dataset artifacts: feature names, target names, scalers, and class map (if applicable).
        
        Args:
            directory (str | Path): Directory to save artifacts.
            verbose (bool): Whether to print log messages.
        """
        self.save_feature_names(directory=directory, verbose=verbose)
        self.save_target_names(directory=directory, verbose=verbose)
        if self.feature_scaler is not None or self.target_scaler is not None:
            self.save_scaler(directory=directory, verbose=verbose)
        if self.class_map:
            self.save_class_map(directory=directory, verbose=verbose)

