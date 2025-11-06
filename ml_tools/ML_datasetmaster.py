import torch
from torch.utils.data import Dataset
import pandas
import numpy
from sklearn.model_selection import train_test_split
from typing import Literal, Union, Tuple, List, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from pathlib import Path

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .custom_logger import save_list_strings
from .ML_scaler import DragonScaler
from .keys import DatasetKeys, MLTaskKeys
from ._schema import FeatureSchema


__all__ = [
    "DragonDataset",
    "DragonDatasetMulti",
    "DragonDatasetSequence"
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


# --- Abstract Base Class ---
class _BaseDatasetMaker(ABC):
    """
    Abstract base class for dataset makers. Contains shared logic for
    splitting, scaling, and accessing datasets to reduce code duplication.
    """
    def __init__(self):
        self._train_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None
        self.scaler: Optional[DragonScaler] = None
        self._id: Optional[str] = None
        self._feature_names: List[str] = []
        self._target_names: List[str] = []
        self._X_train_shape = (0,0)
        self._X_test_shape = (0,0)
        self._y_train_shape = (0,)
        self._y_test_shape = (0,)
        
    def _prepare_scaler(self, 
                        X_train: pandas.DataFrame, 
                        y_train: Union[pandas.Series, pandas.DataFrame], 
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
        X_test_values = X_test.to_numpy()

        # continuous_feature_indices is derived
        if self.scaler is None and continuous_feature_indices:
            _LOGGER.info("Fitting a new DragonScaler on training data.")
            temp_train_ds = _PytorchDataset(X_train_values, y_train, label_dtype) # type: ignore
            self.scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices)

        if self.scaler and self.scaler.mean_ is not None:
            _LOGGER.info("Applying scaler transformation to train and test feature sets.")
            X_train_tensor = self.scaler.transform(torch.tensor(X_train_values, dtype=torch.float32))
            X_test_tensor = self.scaler.transform(torch.tensor(X_test_values, dtype=torch.float32))
            return X_train_tensor.numpy(), X_test_tensor.numpy()

        return X_train_values, X_test_values

    @property
    def train_dataset(self) -> Dataset:
        if self._train_ds is None: raise RuntimeError("Dataset not yet created.")
        return self._train_ds

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

    def save_artifacts(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """
        Convenience method to save feature names, target names, and the scaler (if a scaler was fitted)
        """
        self.save_feature_names(directory=directory, verbose=verbose)
        self.save_target_names(directory=directory, verbose=verbose)
        if self.scaler is not None:
            self.save_scaler(directory=directory, verbose=verbose)


# Single target dataset
class DragonDataset(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with a single target column.

    This class takes a DataFrame, and a FeatureSchema, automatically splits and converts them into PyTorch Datasets.
    It can also create and apply a DragonScaler using the schema.
    
    Attributes:
        `scaler` -> DragonScaler | None
        `train_dataset` -> PyTorch Dataset
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
                 test_size: float = 0.2,
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
            test_size (float): 
                The proportion of the dataset to allocate to the test split.
            random_state (int): 
                The seed for the random number of generator for reproducibility.
            
        """
        super().__init__()
        
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

        X_train, X_test, y_train, y_test = train_test_split(
            features_df, 
            target_series, 
            test_size=test_size, 
            random_state=random_state
        )
        self._X_train_shape, self._X_test_shape = X_train.shape, X_test.shape
        self._y_train_shape, self._y_test_shape = y_train.shape, y_test.shape
        
        # --- label_dtype logic ---
        if kind == MLTaskKeys.REGRESSION or kind == MLTaskKeys.BINARY_CLASSIFICATION:
            label_dtype = torch.float32
        elif kind == MLTaskKeys.MULTICLASS_CLASSIFICATION:
            label_dtype = torch.int64
        else:
            _LOGGER.error(f"Invalid 'kind' {kind}. Must be '{MLTaskKeys.REGRESSION}', '{MLTaskKeys.BINARY_CLASSIFICATION}', or '{MLTaskKeys.MULTICLASS_CLASSIFICATION}'.")
            raise ValueError()

        # --- 4. Scale (using the schema) ---
        if _apply_scaling:
            X_train_final, X_test_final = self._prepare_scaler(
                X_train, y_train, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final = X_train.to_numpy()
            X_test_final = X_test.to_numpy()
        
        # --- 5. Create Datasets ---
        self._train_ds = _PytorchDataset(X_train_final, y_train, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
    
    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Target: {self.target_names[0]}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Scaler: {'Fitted' if self.scaler else 'None'}\n"
        
        if self._train_ds:
            s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
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
    
    Targets dtype is torch.float32
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 target_columns: List[str],
                 schema: FeatureSchema,
                 kind: Literal["multitarget regression", "multilabel binary classification"],
                 scaler: Union[Literal["fit"], Literal["none"], DragonScaler],
                 test_size: float = 0.2,
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
            test_size (float): 
                The proportion of the dataset to allocate to the test split.
            random_state (int): 
                The seed for the random number generator for reproducibility.
                
        ## Note:
        For multi-binary classification, the most common PyTorch loss function is nn.BCEWithLogitsLoss. 
        This loss function requires the labels to be torch.float32 which is the same type required for regression (multi-regression) tasks.
        """
        super().__init__()
        
        # --- Validate new kind parameter ---
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

        X_train, X_test, y_train, y_test = train_test_split(
            features_df,
            target_df, 
            test_size=test_size, 
            random_state=random_state
        )
        self._X_train_shape, self._X_test_shape = X_train.shape, X_test.shape
        self._y_train_shape, self._y_test_shape = y_train.shape, y_test.shape
        
        # Multi-target for regression or multi-binary
        label_dtype = torch.float32 

        # --- 4. Scale (using the schema) ---
        if _apply_scaling:
            X_train_final, X_test_final = self._prepare_scaler(
                X_train, y_train, X_test, label_dtype, schema
            )
        else:
            _LOGGER.info("Features have not been scaled as specified.")
            X_train_final = X_train.to_numpy()
            X_test_final = X_test.to_numpy()
        
        # --- 5. Create Datasets ---
        # _PytorchDataset now correctly handles y_train (a DataFrame)
        self._train_ds = _PytorchDataset(X_train_final, y_train, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Targets: {self.number_of_targets}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Scaler: {'Fitted' if self.scaler else 'None'}\n"
        
        if self._train_ds:
            s += f"  Train Samples: {len(self._train_ds)}\n" # type: ignore
        if self._test_ds:
            s += f"  Test Samples: {len(self._test_ds)}\n" # type: ignore
            
        return s


# --- Private Base Class ---
class _BaseMaker(ABC):
    """
    Abstract Base Class for extra dataset makers.
    """
    def __init__(self):
        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None

    @abstractmethod
    def get_datasets(self) -> Tuple[Dataset, ...]:
        """
        The primary method to retrieve the final, processed PyTorch datasets.
        Must be implemented by all subclasses.
        """
        pass


# --- SequenceMaker ---
class DragonDatasetSequence(_BaseMaker):
    """
    Creates windowed PyTorch datasets from time-series data.
    
    Pipeline:
    
    1. `.split_data()`: Separate time series into training and testing portions.
    2. `.normalize_data()`: Normalize the data. The scaler will be fitted on the training portion.
    3. `.generate_windows()`: Create the windowed sequences from the split and normalized data.
    4. `.get_datasets()`: Return Pytorch train and test datasets.
    """
    def __init__(self, data: Union[pandas.DataFrame, pandas.Series, numpy.ndarray], sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.scaler = None
        
        if isinstance(data, pandas.DataFrame):
            self.time_axis = data.index.values
            self.sequence = data.iloc[:, 0].values.astype(numpy.float32)
        elif isinstance(data, pandas.Series):
            self.time_axis = data.index.values
            self.sequence = data.values.astype(numpy.float32)
        elif isinstance(data, numpy.ndarray):
            self.time_axis = numpy.arange(len(data))
            self.sequence = data.astype(numpy.float32)
        else:
            _LOGGER.error("Data must be a pandas DataFrame/Series or a numpy array.")
            raise TypeError()
            
        self.train_sequence = None
        self.test_sequence = None
        
        self._is_split = False
        self._is_normalized = False
        self._are_windows_generated = False

    def normalize_data(self) -> 'DragonDatasetSequence':
        """
        Normalizes the sequence data using DragonScaler. Must be called AFTER 
        splitting to prevent data leakage from the test set.
        """
        if not self._is_split:
            _LOGGER.error("Data must be split BEFORE normalizing. Call .split_data() first.")
            raise RuntimeError()

        if self.scaler:
            _LOGGER.warning("Data has already been normalized.")
            return self

        # 1. DragonScaler requires a Dataset to fit. Create a temporary one.
        # The scaler expects 2D data [n_samples, n_features].
        train_features = self.train_sequence.reshape(-1, 1) # type: ignore

        # _PytorchDataset needs labels, so we create dummy ones.
        dummy_labels = numpy.zeros(len(train_features))
        temp_train_ds = _PytorchDataset(train_features, dummy_labels, labels_dtype=torch.float32)

        # 2. Fit the DragonScaler on the temporary training dataset.
        # The sequence is a single feature, so its index is [0].
        _LOGGER.info("Fitting DragonScaler on the training data...")
        self.scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices=[0])

        # 3. Transform sequences using the fitted scaler.
        # The transform method requires a tensor, so we convert, transform, and convert back.
        train_tensor = torch.tensor(self.train_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore
        test_tensor = torch.tensor(self.test_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore

        self.train_sequence = self.scaler.transform(train_tensor).numpy().flatten()
        self.test_sequence = self.scaler.transform(test_tensor).numpy().flatten()

        self._is_normalized = True
        _LOGGER.info("Sequence data normalized using DragonScaler.")
        return self

    def split_data(self, test_size: float = 0.2) -> 'DragonDatasetSequence':
        """Splits the sequence into training and testing portions."""
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return self

        split_idx = int(len(self.sequence) * (1 - test_size))
        self.train_sequence = self.sequence[:split_idx]
        self.test_sequence = self.sequence[split_idx - self.sequence_length:]
        
        self.train_time_axis = self.time_axis[:split_idx]
        self.test_time_axis = self.time_axis[split_idx:]

        self._is_split = True
        _LOGGER.info(f"Sequence split into training ({len(self.train_sequence)} points) and testing ({len(self.test_sequence)} points).")
        return self

    def generate_windows(self, sequence_to_sequence: bool = False) -> 'DragonDatasetSequence':
        """
        Generates overlapping windows for features and labels.
        
        "sequence-to-sequence": Label vectors are of the same size as the feature vectors instead of a single future prediction.
        """
        if not self._is_split:
            _LOGGER.error("Cannot generate windows before splitting data. Call .split_data() first.")
            raise RuntimeError()

        self._train_dataset = self._create_windowed_dataset(self.train_sequence, sequence_to_sequence) # type: ignore
        self._test_dataset = self._create_windowed_dataset(self.test_sequence, sequence_to_sequence) # type: ignore
        
        self._are_windows_generated = True
        _LOGGER.info("Feature and label windows generated for train and test sets.")
        return self

    def _create_windowed_dataset(self, data: numpy.ndarray, use_sequence_labels: bool) -> Dataset:
        """Efficiently creates windowed features and labels using numpy."""
        if len(data) <= self.sequence_length:
            _LOGGER.error("Data length must be greater than the sequence_length to create at least one window.")
            raise ValueError()
            
        if not use_sequence_labels:
            features = data[:-1]
            labels = data[self.sequence_length:]
            
            n_windows = len(features) - self.sequence_length + 1
            bytes_per_item = features.strides[0]
            strided_features = numpy.lib.stride_tricks.as_strided(
                features, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item)
            )
            return _PytorchDataset(strided_features, labels, labels_dtype=torch.float32)
        
        else:
            x_data = data[:-1]
            y_data = data[1:]
            
            n_windows = len(x_data) - self.sequence_length + 1
            bytes_per_item = x_data.strides[0]
            
            strided_x = numpy.lib.stride_tricks.as_strided(x_data, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item))
            strided_y = numpy.lib.stride_tricks.as_strided(y_data, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item))
            
            return _PytorchDataset(strided_x, strided_y, labels_dtype=torch.float32)

    def denormalize(self, data: Union[torch.Tensor, numpy.ndarray]) -> numpy.ndarray:
        """Applies inverse transformation using the stored DragonScaler."""
        if self.scaler is None:
            _LOGGER.error("Data was not normalized. Cannot denormalize.")
            raise RuntimeError()

        # Ensure data is a torch.Tensor
        if isinstance(data, numpy.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32)
        else:
            tensor_data = data

        # Reshape for the scaler [n_samples, n_features]
        if tensor_data.ndim == 1:
            tensor_data = tensor_data.view(-1, 1)

        # Apply inverse transform and convert back to a flat numpy array
        original_scale_tensor = self.scaler.inverse_transform(tensor_data)
        return original_scale_tensor.cpu().numpy().flatten()

    def plot(self, predictions: Optional[numpy.ndarray] = None):
        """Plots the original training and testing data, with optional predictions."""
        if not self._is_split:
            _LOGGER.error("Cannot plot before splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        plt.figure(figsize=(15, 6))
        plt.title("Time Series Data")
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        plt.plot(self.train_time_axis, self.scaler.inverse_transform(self.train_sequence.reshape(-1, 1)), label='Train Data') # type: ignore
        plt.plot(self.test_time_axis, self.scaler.inverse_transform(self.test_sequence[self.sequence_length-1:].reshape(-1, 1)), label='Test Data') # type: ignore

        if predictions is not None:
            pred_time_axis = self.test_time_axis[:len(predictions)]
            plt.plot(pred_time_axis, predictions, label='Predictions', c='red')

        plt.legend()
        plt.show()

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Returns the final train and test datasets."""
        if not self._are_windows_generated:
            _LOGGER.error("Windows have not been generated. Call .generate_windows() first.")
            raise RuntimeError()
        return self._train_dataset, self._test_dataset
    
    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}>:\n"
        s += f"  Sequence Length (Window): {self.sequence_length}\n"
        s += f"  Total Data Points: {len(self.sequence)}\n"
        s += "  --- Status ---\n"
        s += f"  Split: {self._is_split}\n"
        s += f"  Normalized: {self._is_normalized}\n"
        s += f"  Windows Generated: {self._are_windows_generated}\n"
        
        if self._are_windows_generated:
            train_len = len(self._train_dataset) if self._train_dataset else 0 # type: ignore
            test_len = len(self._test_dataset) if self._test_dataset else 0 # type: ignore
            s += f"  Datasets (Train/Test): {train_len} / {test_len} windows\n"
            
        return s


def info():
    _script_info(__all__)
