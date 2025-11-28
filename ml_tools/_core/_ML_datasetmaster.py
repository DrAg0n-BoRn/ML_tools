import torch
from torch.utils.data import Dataset
import pandas
import numpy
from sklearn.model_selection import train_test_split
from typing import Literal, Union, List, Optional, Tuple
from abc import ABC
from pathlib import Path

from ._path_manager import make_fullpath, sanitize_filename
from ._logger import get_logger
from ._script_info import _script_info
from ._custom_logger import save_list_strings
from ._ML_scaler import DragonScaler
from ._keys import DatasetKeys, MLTaskKeys, ScalerKeys
from ._schema import FeatureSchema
from ._custom_logger import custom_logger


_LOGGER = get_logger("DragonDataset")


__all__ = [
    "DragonDataset",
    "DragonDatasetMulti"
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
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None):
        
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
        self._classes: List[str] = []
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
        self._feature_names: List[str] = []
        self._target_names: List[str] = []
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
                        schema: FeatureSchema) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Internal helper to fit and apply a DragonScaler for FEATURES using a FeatureSchema."""
        continuous_feature_indices: Optional[List[int]] = None

        # Get continuous feature indices *from the schema*
        if schema.continuous_feature_names:
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
            _LOGGER.info("No continuous features listed in schema. Feature scaler will not be fitted.")

        X_train_values = X_train.to_numpy()
        X_val_values = X_val.to_numpy()
        X_test_values = X_test.to_numpy()

        # continuous_feature_indices is derived
        if self.feature_scaler is None and continuous_feature_indices:
            _LOGGER.info("Fitting a new DragonScaler on training features.")
            temp_train_ds = _PytorchDataset(X_train_values, y_train, label_dtype) 
            self.feature_scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices)

        if self.feature_scaler and self.feature_scaler.mean_ is not None:
            _LOGGER.info("Applying scaler transformation to train, validation, and test feature sets.")
            X_train_tensor = self.feature_scaler.transform(torch.tensor(X_train_values, dtype=torch.float32))
            X_val_tensor = self.feature_scaler.transform(torch.tensor(X_val_values, dtype=torch.float32))
            X_test_tensor = self.feature_scaler.transform(torch.tensor(X_test_values, dtype=torch.float32))
            return X_train_tensor.numpy(), X_val_tensor.numpy(), X_test_tensor.numpy()

        return X_train_values, X_val_values, X_test_values
    
    def _prepare_target_scaler(self,
                               y_train: Union[pandas.Series, pandas.DataFrame],
                               y_val: Union[pandas.Series, pandas.DataFrame],
                               y_test: Union[pandas.Series, pandas.DataFrame]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Internal helper to fit and apply a DragonScaler for TARGETS."""
        
        y_train_arr = y_train.to_numpy() if isinstance(y_train, (pandas.Series, pandas.DataFrame)) else y_train
        y_val_arr = y_val.to_numpy() if isinstance(y_val, (pandas.Series, pandas.DataFrame)) else y_val
        y_test_arr = y_test.to_numpy() if isinstance(y_test, (pandas.Series, pandas.DataFrame)) else y_test

        if self.target_scaler is None:
            _LOGGER.info("Fitting a new DragonScaler on training targets.")
            # Convert to float tensor for calculation
            y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
            self.target_scaler = DragonScaler.fit_tensor(y_train_tensor)
            
        if self.target_scaler and self.target_scaler.mean_ is not None:
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
                ds._feature_scaler = self.feature_scaler
                ds._target_scaler = self.target_scaler

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
        
        if self.feature_scaler:
            combined_state[ScalerKeys.FEATURE_SCALER] = self.feature_scaler._get_state()
            
        if self.target_scaler:
            combined_state[ScalerKeys.TARGET_SCALER] = self.target_scaler._get_state()
            
        torch.save(combined_state, filepath)
        
        if verbose:
            _LOGGER.info(f"Scalers for dataset '{self.id}' saved as '{filepath.name}'.")
            
    def save_class_map(self, directory: Union[str,Path], verbose: bool=True) -> None:
        if not self.class_map:
            _LOGGER.warning(f"No class_map defined. Skipping.")
            return
        
        log_name = f"Class_to_Index_{self.id}" if self.id else "Class_to_Index"
        
        custom_logger(data=self.class_map,
                      save_directory=directory,
                      log_name=log_name,
                      add_timestamp=False,
                      dict_as="json")
        if verbose:
            _LOGGER.info(f"Class map for '{self.id}' saved as '{log_name}.json'.")

    def save_artifacts(self, directory: Union[str, Path], verbose: bool=True) -> None:
        self.save_feature_names(directory=directory, verbose=verbose)
        self.save_target_names(directory=directory, verbose=verbose)
        if self.feature_scaler is not None or self.target_scaler is not None:
            self.save_scaler(directory=directory, verbose=verbose)
        if self.class_map:
            self.save_class_map(directory=directory, verbose=verbose)


# Single target dataset
class DragonDataset(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with a single target column.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 schema: FeatureSchema,
                 kind: Literal["regression", "binary classification", "multiclass classification"],
                 feature_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 target_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 validation_size: float = 0.2,
                 test_size: float = 0.1,
                 class_map: Optional[dict[str,int]]=None,
                 random_state: int = 42):
        """
        Args:
            pandas_df (pandas.DataFrame): 
                The pre-processed input DataFrame containing all columns. (features and single target).
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            kind (str): 
                The type of ML task. Must be one of:
                - "regression"
                - "binary classification"
                - "multiclass classification" 
            validation_size (float):
                The proportion of the *original* dataset to allocate to the validation split.
            test_size (float): 
                The proportion of the dataset to allocate to the test split (can be 0).
            class_map (dict[str,int] | None): Optional class map for the target classes in classification tasks. Can be set later using `.set_class_map()`.
            random_state (int): 
                The seed for the random number of generator for reproducibility.
            feature_scaler: Strategy for feature scaling ("fit", "none", or DragonScaler).
                - "fit": Fit a new DragonScaler on continuous features.
                - "none": Do not scale data (e.g., for TabularTransformer).
                - DragonScaler instance: Use a pre-fitted scaler to transform data.
            target_scaler: Strategy for target scaling. ONLY applies for "regression" tasks.
        """
        super().__init__()
        
        # --- Validation for split sizes ---
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError()
        elif validation_size <= 0.0:
            _LOGGER.error(f"Invalid validation split of {validation_size}.")
            raise ValueError()
        
        # --- 1. Identify features (from schema) ---
        self._feature_names = list(schema.feature_names)
        
        # --- 2. Infer target (by set difference) ---
        all_cols_set = set(pandas_df.columns)
        feature_cols_set = set(self._feature_names)
        target_cols_set = all_cols_set - feature_cols_set
        
        if len(target_cols_set) == 0:
            _LOGGER.error("No target column found. The schema's features match the DataFrame's columns exactly.")
            raise ValueError()
        if len(target_cols_set) > 1:
            _LOGGER.error(f"Ambiguous target. Found {len(target_cols_set)} columns not in the schema: {list(target_cols_set)}. One target required.")
            raise ValueError()
            
        target_name = list(target_cols_set)[0]
        self._target_names = [target_name]
        self._id = target_name
        
        # --- 3. Split Data ---
        features_df = pandas_df[self._feature_names]
        target_series = pandas_df[target_name]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, target_series, test_size=test_size, random_state=random_state
        )
        val_split_size = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_split_size, random_state=random_state
        )
        
        self._X_train_shape, self._X_val_shape, self._X_test_shape = X_train.shape, X_val.shape, X_test.shape
        self._y_train_shape, self._y_val_shape, self._y_test_shape = y_train.shape, y_val.shape, y_test.shape
        
        # --- label_dtype logic ---
        if kind == MLTaskKeys.REGRESSION or kind == MLTaskKeys.BINARY_CLASSIFICATION:
            label_dtype = torch.float32
        elif kind == MLTaskKeys.MULTICLASS_CLASSIFICATION:
            label_dtype = torch.int64
        else:
            _LOGGER.error(f"Invalid 'kind' {kind}.")
            raise ValueError()
        self.kind = kind

        # --- 4. Scale Features ---
        if feature_scaler == "fit":
            self.feature_scaler = None # To be created
            _apply_f_scaling = True
        elif feature_scaler == "none":
            self.feature_scaler = None
            _apply_f_scaling = False
        elif isinstance(feature_scaler, DragonScaler):
            self.feature_scaler = feature_scaler
            _apply_f_scaling = True
        else:
            _LOGGER.error("Invalid feature_scaler argument.")
            raise ValueError()

        if _apply_f_scaling:
            X_train_final, X_val_final, X_test_final = self._prepare_feature_scaler(
                X_train, y_train, X_val, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final, X_val_final, X_test_final = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()

        # --- 5. Scale Targets (Regression Only) ---
        if kind == MLTaskKeys.REGRESSION:
            if target_scaler == "fit":
                self.target_scaler = None
                _apply_t_scaling = True
            elif target_scaler == "none":
                self.target_scaler = None
                _apply_t_scaling = False
            elif isinstance(target_scaler, DragonScaler):
                self.target_scaler = target_scaler
                _apply_t_scaling = True
            else:
                _LOGGER.error("Invalid target_scaler argument.")
                raise ValueError()

            if _apply_t_scaling:
                y_train_final, y_val_final, y_test_final = self._prepare_target_scaler(y_train, y_val, y_test)
            else:
                y_train_final = y_train.to_numpy() if isinstance(y_train, (pandas.Series, pandas.DataFrame)) else y_train
                y_val_final = y_val.to_numpy() if isinstance(y_val, (pandas.Series, pandas.DataFrame)) else y_val
                y_test_final = y_test.to_numpy() if isinstance(y_test, (pandas.Series, pandas.DataFrame)) else y_test
        else:
            # No scaling for Classification targets
            y_train_final = y_train.to_numpy() if isinstance(y_train, (pandas.Series, pandas.DataFrame)) else y_train
            y_val_final = y_val.to_numpy() if isinstance(y_val, (pandas.Series, pandas.DataFrame)) else y_val
            y_test_final = y_test.to_numpy() if isinstance(y_test, (pandas.Series, pandas.DataFrame)) else y_test

        # --- 6. Create Datasets ---
        self._train_ds = _PytorchDataset(X_train_final, y_train_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._val_ds = _PytorchDataset(X_val_final, y_val_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        
        # --- 7. Attach scalers to datasets ---
        self._attach_scalers_to_datasets()
        
        # --- 8. Set class map if classification ---
        if self.kind != MLTaskKeys.REGRESSION:
            if class_map is None:
                self.class_map = dict()
            else:
                self.set_class_map(class_map)
        else:
            self.class_map = dict()

    def set_class_map(self, class_map: dict[str, int], force_overwrite: bool=False) -> None:
        if self.kind == MLTaskKeys.REGRESSION:
            _LOGGER.warning(f"Class Map is for classifications tasks only.")
            return
        
        if self.class_map:
            warning_message = f"Class map was previously set."
            if not force_overwrite:
                warning_message += " Use `force_overwrite=True` to set new values."
                _LOGGER.warning(warning_message)
                return
            else:
                warning_message += ". Setting new values..."
                _LOGGER.warning(warning_message)
        
        self.class_map = class_map
        try:
            sorted_items = sorted(class_map.items(), key=lambda item: item[1])
            class_list = [item[0] for item in sorted_items]
        except Exception as e:
            _LOGGER.error(f"Could not sort class map. Ensure it is a dict of {str: int}. Error: {e}")
            raise TypeError()
        else:
            self.classes = class_list
        
        if self._train_ds: self._train_ds._classes, self._train_ds._class_map = class_list, class_map # type: ignore
        if self._val_ds: self._val_ds._classes, self._val_ds._class_map = class_list, class_map # type: ignore
        if self._test_ds: self._test_ds._classes, self._test_ds._class_map = class_list, class_map # type: ignore
            
        _LOGGER.info(f"Class map set for dataset '{self.id}' and its subsets:\n{class_map}")

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Target: {self.target_names[0]}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Feature Scaler: {'Fitted' if self.feature_scaler else 'None'}\n"
        s += f"  Target Scaler: {'Fitted' if self.target_scaler else 'None'}\n"
        
        if self._train_ds: s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
        if self._val_ds: s += f"  Validation Samples: {len(self._val_ds)}\n" # type: ignore
        if self._test_ds: s += f"  Test Samples: {len(self._test_ds)}\n" # type: ignore
        return s


# --- Multi-Target Class ---
class DragonDatasetMulti(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with 
    multiple target columns.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 target_columns: List[str],
                 schema: FeatureSchema,
                 kind: Literal["multitarget regression", "multilabel binary classification"],
                 feature_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 target_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 validation_size: float = 0.2,
                 test_size: float = 0.1,
                 random_state: int = 42):
        """
        Args:
            pandas_df (pandas.DataFrame): 
                The pre-processed input DataFrame with *all* columns
                (features and targets).
            target_columns (list[str]): 
                List of target column names.
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            kind (str):
                The type of multi-target ML task. Must be one of:
                - "multitarget regression"
                - "multilabel binary classification"
            validation_size (float):
                The proportion of the dataset to allocate to the validation split.
            test_size (float): 
                The proportion of the dataset to allocate to the test split.
            random_state (int): 
                The seed for the random number generator for reproducibility.
            feature_scaler: Strategy for feature scaling.
                - "fit": Fit a new DragonScaler on continuous features.
                - "none": Do not scale data (e.g., for TabularTransformer).
                - DragonScaler instance: Use a pre-fitted scaler to transform data.
            target_scaler: Strategy for target scaling (Regression only).
        """
        super().__init__()
        
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError()
        elif validation_size <= 0.0:
            _LOGGER.error(f"Invalid validation split of {validation_size}.")
            raise ValueError()

        if kind not in [MLTaskKeys.MULTITARGET_REGRESSION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
            _LOGGER.error(f"Invalid 'kind' {kind}.")
            raise ValueError()
        self.kind = kind
        
        # --- 1. Get features and targets from schema/args ---
        self._feature_names = list(schema.feature_names)
        self._target_names = target_columns
        self._id = f"{len(self._target_names)}-targets"
        
        # --- 2. Validation ---
        all_cols_set = set(pandas_df.columns)
        feature_cols_set = set(self._feature_names)
        target_cols_set = set(self._target_names)

        overlap = feature_cols_set.intersection(target_cols_set)
        if overlap:
            _LOGGER.error(f"Features and targets are not mutually exclusive. Overlap: {list(overlap)}")
            raise ValueError()

        schema_plus_targets = feature_cols_set.union(target_cols_set)
        if (all_cols_set - schema_plus_targets):
            _LOGGER.warning(f"Columns in DataFrame but not in schema or targets: {list(all_cols_set - schema_plus_targets)}")
            
        if (schema_plus_targets - all_cols_set):
            _LOGGER.error(f"Columns in schema/targets but not in DataFrame: {list(schema_plus_targets - all_cols_set)}")
            raise ValueError()

        # --- 3. Split Data ---
        features_df = pandas_df[self._feature_names]
        target_df = pandas_df[self._target_names]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, target_df, test_size=test_size, random_state=random_state
        )
        val_split_size = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_split_size, random_state=random_state
        )

        self._X_train_shape, self._X_val_shape, self._X_test_shape = X_train.shape, X_val.shape, X_test.shape
        self._y_train_shape, self._y_val_shape, self._y_test_shape = y_train.shape, y_val.shape, y_test.shape
        
        label_dtype = torch.float32 

        # --- 4. Scale Features ---
        if feature_scaler == "fit":
            self.feature_scaler = None
            _apply_f_scaling = True
        elif feature_scaler == "none":
            self.feature_scaler = None
            _apply_f_scaling = False
        elif isinstance(feature_scaler, DragonScaler):
            self.feature_scaler = feature_scaler
            _apply_f_scaling = True
        else:
            _LOGGER.error("Invalid feature_scaler argument.")
            raise ValueError()

        if _apply_f_scaling:
            X_train_final, X_val_final, X_test_final = self._prepare_feature_scaler(
                X_train, y_train, X_val, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final, X_val_final, X_test_final = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()

        # --- 5. Scale Targets ---
        if kind == MLTaskKeys.MULTITARGET_REGRESSION:
            if target_scaler == "fit":
                self.target_scaler = None
                _apply_t_scaling = True
            elif target_scaler == "none":
                self.target_scaler = None
                _apply_t_scaling = False
            elif isinstance(target_scaler, DragonScaler):
                self.target_scaler = target_scaler
                _apply_t_scaling = True
            else:
                _LOGGER.error("Invalid target_scaler argument.")
                raise ValueError()

            if _apply_t_scaling:
                y_train_final, y_val_final, y_test_final = self._prepare_target_scaler(y_train, y_val, y_test)
            else:
                y_train_final, y_val_final, y_test_final = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()
        else:
             y_train_final, y_val_final, y_test_final = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

        # --- 6. Create Datasets ---
        self._train_ds = _PytorchDataset(X_train_final, y_train_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._val_ds = _PytorchDataset(X_val_final, y_val_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test_final, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        
        # --- 7. Attach scalers to datasets ---
        self._attach_scalers_to_datasets()

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Targets: {self.number_of_targets}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Feature Scaler: {'Fitted' if self.feature_scaler else 'None'}\n"
        s += f"  Target Scaler: {'Fitted' if self.target_scaler else 'None'}\n"
        
        if self._train_ds: s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
        if self._val_ds: s += f"  Validation Samples: {len(self._val_ds)}\n" # type: ignore
        if self._test_ds: s += f"  Test Samples: {len(self._test_ds)}\n" # type: ignore
        return s


def info():
    _script_info(__all__)
