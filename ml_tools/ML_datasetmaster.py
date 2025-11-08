import torch
from torch.utils.data import Dataset
import pandas
import numpy
from sklearn.model_selection import train_test_split
from typing import Literal, Union, List, Optional
from abc import ABC
from pathlib import Path

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .custom_logger import save_list_strings
from .ML_scaler import DragonScaler
from ._keys import DatasetKeys, MLTaskKeys
from ._schema import FeatureSchema
from .custom_logger import custom_logger


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
        """
        integer labels for classification.
        
        float labels for regression.
        """
        
        if isinstance(features, numpy.ndarray):
            self.features = torch.tensor(features, dtype=features_dtype)
        else: # It's a pandas.DataFrame
            self.features = torch.tensor(features.to_numpy(), dtype=features_dtype)

        if isinstance(labels, numpy.ndarray):
            self.labels = torch.tensor(labels, dtype=labels_dtype)
        elif isinstance(labels, (pandas.Series, pandas.DataFrame)):
            self.labels = torch.tensor(labels.to_numpy(), dtype=labels_dtype)
        else:
             # Fallback for other types (though your type hints don't cover this)
            self.labels = torch.tensor(labels, dtype=labels_dtype)
            
        self._feature_names = feature_names
        self._target_names = target_names
        self._classes: List[str] = []
        self._class_map: dict[str,int] = dict()

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


# --- Abstract Base Class ---
class _BaseDatasetMaker(ABC):
    """
    Abstract base class for dataset makers. Contains shared logic for
    splitting, scaling, and accessing datasets to reduce code duplication.
    """
    def __init__(self):
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None
        self.scaler: Optional[DragonScaler] = None
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
        
    def _prepare_scaler(self, 
                        X_train: pandas.DataFrame, 
                        y_train: Union[pandas.Series, pandas.DataFrame], 
                        X_val: pandas.DataFrame,
                        X_test: pandas.DataFrame, 
                        label_dtype: torch.dtype, 
                        schema: FeatureSchema):
        """Internal helper to fit and apply a DragonScaler using a FeatureSchema."""
        continuous_feature_indices: Optional[List[int]] = None

        # Get continuous feature indices *from the schema*
        if schema.continuous_feature_names:
            _LOGGER.info("Getting continuous feature indices from schema.")
            try:
                # Convert columns to a standard list for .index()
                train_cols_list = X_train.columns.to_list()
                # Map names from schema to column indices in the training DataFrame
                continuous_feature_indices = [train_cols_list.index(name) for name in schema.continuous_feature_names]
            except ValueError as e: #
                _LOGGER.error(f"Feature name from schema not found in training data columns:\n{e}")
                raise ValueError()
        else:
            _LOGGER.info("No continuous features listed in schema. Scaler will not be fitted.")

        X_train_values = X_train.to_numpy()
        X_val_values = X_val.to_numpy()
        X_test_values = X_test.to_numpy()

        # continuous_feature_indices is derived
        if self.scaler is None and continuous_feature_indices:
            _LOGGER.info("Fitting a new DragonScaler on training data.")
            temp_train_ds = _PytorchDataset(X_train_values, y_train, label_dtype) # type: ignore
            self.scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices)

        if self.scaler and self.scaler.mean_ is not None:
            _LOGGER.info("Applying scaler transformation to train, validation, and test feature sets.")
            X_train_tensor = self.scaler.transform(torch.tensor(X_train_values, dtype=torch.float32))
            X_val_tensor = self.scaler.transform(torch.tensor(X_val_values, dtype=torch.float32))
            X_test_tensor = self.scaler.transform(torch.tensor(X_test_values, dtype=torch.float32))
            return X_train_tensor.numpy(), X_val_tensor.numpy(), X_test_tensor.numpy()

        return X_train_values, X_val_values, X_test_values

    @property
    def train_dataset(self) -> Dataset:
        if self._train_ds is None: raise RuntimeError("Dataset not yet created.")
        return self._train_ds
    
    @property
    def validation_dataset(self) -> Dataset:
        if self._val_ds is None: raise RuntimeError("Dataset not yet created.")
        return self._val_ds

    @property
    def test_dataset(self) -> Dataset:
        if self._test_ds is None: raise RuntimeError("Dataset not yet created.")
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
        """Saves a list of feature names as a text file"""
        save_list_strings(list_strings=self._feature_names,
                          directory=directory,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=verbose)
        
    def save_target_names(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """Saves a list of target names as a text file"""
        save_list_strings(list_strings=self._target_names,
                          directory=directory,
                          filename=DatasetKeys.TARGET_NAMES,
                          verbose=verbose)

    def save_scaler(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """
        Saves the fitted DragonScaler's state to a .pth file.

        The filename is automatically generated based on the dataset id.
        
        Args:
            directory (str | Path): The directory where the scaler will be saved.
        """
        if not self.scaler: 
            _LOGGER.error("No scaler was fitted or provided.")
            raise RuntimeError()
        if not self.id: 
            _LOGGER.error("Must set the dataset `id` before saving scaler.")
            raise ValueError()
        save_path = make_fullpath(directory, make=True, enforce="directory")
        sanitized_id = sanitize_filename(self.id)
        filename = f"{DatasetKeys.SCALER_PREFIX}{sanitized_id}.pth"
        filepath = save_path / filename
        self.scaler.save(filepath, verbose=False)
        if verbose:
            _LOGGER.info(f"Scaler for dataset '{self.id}' saved as '{filepath.name}'.")
            
    def save_class_map(self, directory: Union[str,Path], verbose: bool=True) -> None:
        """
        Saves the class to index mapping {str: int} to a directory.
        """
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
        """
        Convenience method to save feature names, target names, and the scaler (if a scaler was fitted)
        """
        self.save_feature_names(directory=directory, verbose=verbose)
        self.save_target_names(directory=directory, verbose=verbose)
        if self.scaler is not None:
            self.save_scaler(directory=directory, verbose=verbose)
        if self.class_map:
            self.save_class_map(directory=directory, verbose=verbose)


# Single target dataset
class DragonDataset(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with a single target column.

    This class takes a DataFrame, and a FeatureSchema, automatically splits and converts them into PyTorch Datasets.
    It can also create and apply a DragonScaler using the schema.
    
    Attributes:
        `scaler` -> DragonScaler | None
        `train_dataset` -> PyTorch Dataset
        `validation_dataset` -> PyTorch Dataset
        `test_dataset`  -> PyTorch Dataset
        `feature_names` -> list[str]
        `target_names`  -> list[str]
        `id` -> str
        
    The ID can be manually set to any string if needed, it is the target name by default.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 schema: FeatureSchema,
                 kind: Literal["regression", "binary classification", "multiclass classification"],
                 scaler: Union[Literal["fit"], Literal["none"], DragonScaler],
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
            scaler ("fit" | "none" | DragonScaler): 
                Strategy for data scaling:
                - "fit": Fit a new DragonScaler on continuous features.
                - "none": Do not scale data (e.g., for TabularTransformer).
                - DragonScaler instance: Use a pre-fitted scaler to transform data.
            validation_size (float):
                The proportion of the *original* dataset to allocate to the validation split.
            test_size (float): 
                The proportion of the dataset to allocate to the test split (can be 0).
            class_map (dict[str,int] | None): Optional class map for the target classes in classification tasks. Can be set later using `.set_class_map()`.
            random_state (int): 
                The seed for the random number of generator for reproducibility.
            
        """
        super().__init__()
        
        # --- Validation for split sizes ---
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError()
        elif validation_size <= 0.0:
            _LOGGER.error(f"Invalid validation split of {validation_size}.")
            raise ValueError()
        
        _apply_scaling: bool = False
        if scaler == "fit":
            self.scaler = None # To be created
            _apply_scaling = True
        elif scaler == "none":
            self.scaler = None
        elif isinstance(scaler, DragonScaler):
            self.scaler = scaler # Use the provided one
            _apply_scaling = True
        else:
            _LOGGER.error(f"Invalid 'scaler' argument. Must be 'fit', 'none', or a DragonScaler instance.")
            raise ValueError()
        
        # --- 1. Identify features (from schema) ---
        self._feature_names = list(schema.feature_names)
        
        # --- 2. Infer target (by set difference) ---
        all_cols_set = set(pandas_df.columns)
        feature_cols_set = set(self._feature_names)
        
        target_cols_set = all_cols_set - feature_cols_set
        
        if len(target_cols_set) == 0:
            _LOGGER.error("No target column found. The schema's features match the DataFrame's columns exactly.")
            raise ValueError("No target column found in DataFrame.")
        if len(target_cols_set) > 1:
            _LOGGER.error(f"Ambiguous target. Found {len(target_cols_set)} columns not in the schema: {list(target_cols_set)}. One target required.")
            raise ValueError("Ambiguous target: More than one non-feature column found.")
            
        target_name = list(target_cols_set)[0]
        self._target_names = [target_name]
        self._id = target_name
        
        # --- 3. Split Data ---
        features_df = pandas_df[self._feature_names]
        target_series = pandas_df[target_name]
        
        # First split: (Train + Val) vs TesT
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, 
            target_series, 
            test_size=test_size, 
            random_state=random_state
        )
        # Calculate validation split size relative to the (Train + Val) set
        val_split_size = validation_size / (1.0 - test_size)
        
        # Second split: Train vs Val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=val_split_size, 
            random_state=random_state
        )
        
        self._X_train_shape, self._X_val_shape, self._X_test_shape = X_train.shape, X_val.shape, X_test.shape
        self._y_train_shape, self._y_val_shape, self._y_test_shape = y_train.shape, y_val.shape, y_test.shape
        
        # --- label_dtype logic ---
        if kind == MLTaskKeys.REGRESSION or kind == MLTaskKeys.BINARY_CLASSIFICATION:
            label_dtype = torch.float32
        elif kind == MLTaskKeys.MULTICLASS_CLASSIFICATION:
            label_dtype = torch.int64
        else:
            _LOGGER.error(f"Invalid 'kind' {kind}. Must be '{MLTaskKeys.REGRESSION}', '{MLTaskKeys.BINARY_CLASSIFICATION}', or '{MLTaskKeys.MULTICLASS_CLASSIFICATION}'.")
            raise ValueError()
        self.kind = kind

        # --- 4. Scale (using the schema) ---
        if _apply_scaling:
            X_train_final, X_val_final, X_test_final = self._prepare_scaler(
                X_train, y_train, X_val, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final = X_train.to_numpy()
            X_val_final = X_val.to_numpy()
            X_test_final = X_test.to_numpy()
        
        # --- 5. Create Datasets ---
        self._train_ds = _PytorchDataset(X_train_final, y_train, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._val_ds = _PytorchDataset(X_val_final, y_val, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        
        # --- 6. create class map if given ---
        if self.kind != MLTaskKeys.REGRESSION:
            if class_map is None:
                self.class_map = dict()
            else:
                self.set_class_map(class_map)
        else:
            self.class_map = dict()

    def set_class_map(self, class_map: dict[str, int], force_overwrite: bool=False) -> None:
        """
        Sets a map of class_name -> integer_label.
        
        This is used by the InferenceHandler and to finalize the model after training.

        Args:
            class_map (Dict[str, int]): A dictionary mapping the integer label
                to its string name.
                Example: {'cat': 0, 'dog': 1, 'bird': 2}
            force_overwrite (bool): Required to overwrite a previously set class map.
        """
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
        
        if self._train_ds:
            self._train_ds._classes = class_list # type: ignore
            self._train_ds._class_map = class_map # type: ignore
        if self._val_ds:
            self._val_ds._classes = class_list # type: ignore
            self._val_ds._class_map = class_map # type: ignore
        if self._test_ds:
            self._test_ds._classes = class_list # type: ignore
            self._test_ds._class_map = class_map # type: ignore
            
        _LOGGER.info(f"Class map set for dataset '{self.id}' and its subsets:\n{class_map}")

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Target: {self.target_names[0]}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Scaler: {'Fitted' if self.scaler else 'None'}\n"
        
        if self._train_ds:
            s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
        if self._val_ds:
            s += f"  Validation Samples: {len(self._val_ds)}\n" # type: ignore
        if self._test_ds:
            s += f"  Test Samples: {len(self._test_ds)}\n" # type: ignore
            
        return s


# --- Multi-Target Class ---
class DragonDatasetMulti(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with 
    multiple target columns.

    This class takes a *full* DataFrame, a *FeatureSchema*, and a list of
    *target_columns*. It validates that the schema's features and the
    target columns are mutually exclusive and together account for all
    columns in the DataFrame.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 target_columns: List[str],
                 schema: FeatureSchema,
                 kind: Literal["multitarget regression", "multilabel binary classification"],
                 scaler: Union[Literal["fit"], Literal["none"], DragonScaler],
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
            scaler ("fit" | "none" | DragonScaler): 
                Strategy for data scaling:
                - "fit": Fit a new DragonScaler on continuous features.
                - "none": Do not scale data (e.g., for TabularTransformer).
                - DragonScaler instance: Use a pre-fitted scaler to transform data.
            validation_size (float):
                The proportion of the dataset to allocate to the validation split.
            test_size (float): 
                The proportion of the dataset to allocate to the test split.
            random_state (int): 
                The seed for the random number generator for reproducibility.
                
        ## Note:
        For multi-binary classification, the most common PyTorch loss function is nn.BCEWithLogitsLoss. 
        This loss function requires the labels to be torch.float32 which is the same type required for multi-regression tasks.
        """
        super().__init__()
        
        # --- Validation for split sizes ---
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError("validation_size and test_size sum must be < 1.0")
        elif validation_size <= 0.0:
            _LOGGER.error(f"Invalid validation split of {validation_size}.")
            raise ValueError()
            
        # --- Validate kind parameter ---
        if kind not in [MLTaskKeys.MULTITARGET_REGRESSION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
            _LOGGER.error(f"Invalid 'kind' {kind}. Must be '{MLTaskKeys.MULTITARGET_REGRESSION}' or '{MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION}'.")
            raise ValueError()
        
        _apply_scaling: bool = False
        if scaler == "fit":
            self.scaler = None
            _apply_scaling = True
        elif scaler == "none":
            self.scaler = None
        elif isinstance(scaler, DragonScaler):
            self.scaler = scaler # Use the provided one
            _apply_scaling = True
        else:
            _LOGGER.error(f"Invalid 'scaler' argument. Must be 'fit', 'none', or a DragonScaler instance.")
            raise ValueError()
        
        # --- 1. Get features and targets from schema/args ---
        self._feature_names = list(schema.feature_names)
        self._target_names = target_columns
        
        # --- 2. Validation ---
        all_cols_set = set(pandas_df.columns)
        feature_cols_set = set(self._feature_names)
        target_cols_set = set(self._target_names)

        overlap = feature_cols_set.intersection(target_cols_set)
        if overlap:
            _LOGGER.error(f"Features and targets are not mutually exclusive. Overlap: {list(overlap)}")
            raise ValueError("Features and targets overlap.")

        schema_plus_targets = feature_cols_set.union(target_cols_set)
        missing_cols = all_cols_set - schema_plus_targets
        if missing_cols:
            _LOGGER.warning(f"Columns in DataFrame but not in schema or targets: {list(missing_cols)}")
            
        extra_cols = schema_plus_targets - all_cols_set
        if extra_cols:
            _LOGGER.error(f"Columns in schema/targets but not in DataFrame: {list(extra_cols)}")
            raise ValueError("Schema/target definition mismatch with DataFrame.")

        # --- 3. Split Data ---
        features_df = pandas_df[self._feature_names]
        target_df = pandas_df[self._target_names]
        
        # First split: (Train + Val) vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df,
            target_df, 
            test_size=test_size, 
            random_state=random_state
        )

        # Calculate validation split size relative to the (Train + Val) set
        val_split_size = validation_size / (1.0 - test_size)
            
        # Second split: Train vs Val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=val_split_size, 
            random_state=random_state
        )

        self._X_train_shape, self._X_val_shape, self._X_test_shape = X_train.shape, X_val.shape, X_test.shape
        self._y_train_shape, self._y_val_shape, self._y_test_shape = y_train.shape, y_val.shape, y_test.shape
        
        # Multi-target for regression or multi-binary
        label_dtype = torch.float32 

        # --- 4. Scale (using the schema) ---
        if _apply_scaling:
            X_train_final, X_val_final, X_test_final = self._prepare_scaler(
                X_train, y_train, X_val, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final = X_train.to_numpy()
            X_val_final = X_val.to_numpy()
            X_test_final = X_test.to_numpy()
        
        # --- 5. Create Datasets ---
        # _PytorchDataset now correctly handles y_train (a DataFrame)
        self._train_ds = _PytorchDataset(X_train_final, y_train, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._val_ds = _PytorchDataset(X_val_final, y_val, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Targets: {self.number_of_targets}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Scaler: {'Fitted' if self.scaler else 'None'}\n"
        
        if self._train_ds:
            s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
        if self._val_ds:
            s += f"  Validation Samples: {len(self._val_ds)}\n" # type: ignore
        if self._test_ds:
            s += f"  Test Samples: {len(self._test_ds)}\n" # type: ignore
            
        return s


def info():
    _script_info(__all__)
