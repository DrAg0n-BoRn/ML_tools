import torch
from torch.utils.data import Dataset
import pandas
import numpy
from typing import Literal, Union, Optional
from pathlib import Path

from ..ML_scaler import DragonScaler
from ..IO_tools import save_json
from ..schema import FeatureSchema
from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import DatasetKeys, MLTaskKeys, ScalerKeys

from ._base_datasetmaster import _BaseDatasetMaker

_LOGGER = get_logger("Sequence Dataset")


__all__ = [
    "DragonDatasetSequence"
]


class _PytorchSequenceDataset(Dataset):
    """
    Memory-efficient sequence Dataset that mirrors the _PytorchDataset API.
    Maintains overlapping windows as lightweight numpy strided views rather 
    than materializing dense PyTorch tensors in memory.
    """
    def __init__(self, 
                 features: numpy.ndarray, 
                 labels: numpy.ndarray,
                 labels_dtype: torch.dtype,
                 features_dtype: torch.dtype = torch.float32,
                 feature_names: Optional[list[str]] = None,
                 target_names: Optional[list[str]] = None,
                 target_types: Optional[dict[str, str]] = None):
        
        # 1. Store as lightweight numpy views instead of torch.tensor()
        # This prevents the initial RAM explosion.
        self._features_np = features
        self._labels_np = labels
        
        self.features_dtype = features_dtype
        self.labels_dtype = labels_dtype
        
        self._target_types = target_types
        
        # 2. Mirror all internal attributes from _PytorchDataset except 'classes'
        self._feature_names = feature_names
        self._target_names = target_names
        self._class_map: dict[str, dict[str, int]] = dict() # adapted to multivariate categorical targets
        self._feature_scaler: Optional[DragonScaler] = None
        self._target_scaler: Optional[DragonScaler] = None
        
    def __len__(self):
        return len(self._features_np)

    def __getitem__(self, index):
        # 3. Lazily cast to tensor only when the Dataloader fetches a batch.
        # This guarantees we only materialize the memory required for a single batch.
        x = torch.tensor(self._features_np[index], dtype=self.features_dtype)
        y = torch.tensor(self._labels_np[index], dtype=self.labels_dtype)
        return x, y
    
    @property
    def features(self):
        # 4. Intercept the `.features` call used by `save_dataset_bundle`.
        # torch.as_tensor() creates a zero-copy PyTorch view of the numpy array.
        # When saved, PyTorch natively preserves the strides, keeping the .pth file tiny.
        return torch.as_tensor(self._features_np, dtype=self.features_dtype)
    
    @property
    def labels(self):
        return torch.as_tensor(self._labels_np, dtype=self.labels_dtype)
        
    @property
    def feature_names(self):
        if self._feature_names is not None:
            return self._feature_names
        else:
            _LOGGER.error(f"Dataset {self.__class__} has not been initialized with any feature names.")
            raise AttributeError()
        
    @property
    def target_names(self):
        if self._target_names is not None:
            return self._target_names
        else:
            _LOGGER.error(f"Dataset {self.__class__} has not been initialized with any target names.")
            raise AttributeError()
    
    @property
    def target_types(self) -> dict[str, str]:
        if self._target_types is not None:
            return self._target_types
        else:
            _LOGGER.error(f"Dataset {self.__class__} has not been initialized with target types.")
            raise AttributeError()
    
    @property
    def feature_scaler(self):
        return self._feature_scaler
    
    @property
    def target_scaler(self):
        return self._target_scaler


class DragonDatasetSequence(_BaseDatasetMaker):
    """
    Creates windowed PyTorch datasets from multivariate sequential data. 
    
    Supports univariate and multivariate targets, with options for sequence-to-sequence or sequence-to-value prediction modes.
    
    Targets are extracted directly from the feature columns based on the provided list,
    allowing columns to serve simultaneously as historical inputs and future outputs.
    """
    def __init__(self, 
                 pandas_df: pandas.DataFrame,
                 targets: list[str],
                 schema: FeatureSchema,
                 prediction_mode: Literal["sequence-to-sequence", "sequence-to-value"],
                 sequence_length: int,
                 feature_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 validation_size: Optional[float] = 0.2,
                 test_size: Optional[float] = 0.1,
                 verbose: int = 2):
        """
        Initializes the DragonDatasetSequence with chronological splitting, scaling, and windowing.
        
        Args:
            pandas_df (pandas.DataFrame): The input DataFrame containing features and targets.
            targets (list[str]): List of column names to be used as targets.
            schema (FeatureSchema): Schema defining the expected features and their types.
            prediction_mode (str): Either "sequence-to-sequence" or "sequence-to-value".
            sequence_length (int): The length of the input sequences (window size).
            feature_scaler (str | DragonScaler): "fit" to fit a new scaler, "none" for no scaling, or an existing DragonScaler instance.
            validation_size (float | None): Fraction of data to use for validation.
            test_size (float | None): Fraction of data to use for testing.
            verbose (int): Verbosity level for logging (0: silent, 1: warnings, 2: info, 3: detailed).
        """
        
        super().__init__()
        
        validation_size = validation_size or 0.0
        test_size = test_size or 0.0
        
        # --- 1. Validation ---
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError()
        elif validation_size < 0.0 or test_size < 0.0:
            _LOGGER.error("Split sizes cannot be negative.")
            raise ValueError()
        
        if sequence_length <= 0:
            _LOGGER.error(f"sequence_length must be strictly greater than 0. Received: {sequence_length}")
            raise ValueError()
        
        if not targets:
            _LOGGER.error("The 'targets' list cannot be empty. You must specify at least one target column.")
            raise ValueError()

        if prediction_mode not in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            _LOGGER.error(f"Unrecognized prediction mode: '{prediction_mode}'.")
            raise ValueError()

        self.prediction_mode = prediction_mode
        self.sequence_length = sequence_length
        self.validation_split = validation_size
        self.test_split = test_size
        
        self._feature_names = list(schema.feature_names)
        self._target_names = targets
        self._id = f"Seq_{len(self._target_names)}targets"

        # Ensure all schema columns exist in the DF
        df_cols_set = set(pandas_df.columns)
        schema_cols_set = set(self._feature_names)
        if not schema_cols_set.issubset(df_cols_set):
            missing = schema_cols_set - df_cols_set
            _LOGGER.error(f"Required FeatureSchema columns not found in DataFrame: {list(missing)}")
            raise ValueError()

        # Ensure all targets exist in schema
        target_cols_set = set(self._target_names)
        if not target_cols_set.issubset(schema_cols_set):
            missing = target_cols_set - schema_cols_set
            _LOGGER.error(f"Targets not found in FeatureSchema: {list(missing)}")
            raise ValueError()

        # Map target names to their column indices for fast slicing later
        self._target_indices = [self._feature_names.index(t) for t in self._target_names]
        
        # <-- Determine target types from the schema -->
        self._target_types = {
            t: DatasetKeys.TARGET_CATEGORICAL if t in schema.categorical_feature_names else DatasetKeys.TARGET_CONTINUOUS
            for t in self._target_names
        }
        
        # Populate class_map for categorical targets
        self.class_map: dict[str, dict[str, int]] = {}
        self.classes = list()  # vestige from base class, not used in sequence datasets

        if schema.categorical_mappings:
            for target_col in self._target_names:
                if self._target_types[target_col] == DatasetKeys.TARGET_CATEGORICAL:
                    if target_col in schema.categorical_mappings:
                        self.class_map[target_col] = schema.categorical_mappings[target_col]
                    elif verbose >= 1:
                        _LOGGER.warning(f"Categorical target '{target_col}' lacks a mapping in FeatureSchema.categorical_mappings.")

        # --- 2. Chronological Splitting ---
        # Align column order with schema strictly
        features_df = pandas_df[self._feature_names]
        
        if features_df.isna().any().any():
            _LOGGER.error("Input DataFrame contains NaN values in the required feature columns. Please impute or drop missing values before generating the dataset.")
            raise ValueError()
        
        # Verify that all target columns have a numeric dtype
        for t in self._target_names:
            if not pandas.api.types.is_numeric_dtype(features_df[t]):
                _LOGGER.error(f"Target column '{t}' is not numeric. Ensure categorical targets are numerically encoded before dataset creation.")
                raise TypeError()
        
        total_size = len(features_df)
        
        test_split_idx = int(total_size * (1 - test_size))
        val_split_idx = int(total_size * (1 - test_size - validation_size))
        
        # --- Early Fail-Safe for Sequence Length ---
        if val_split_idx <= self.sequence_length:
            _LOGGER.error(
                f"The training split size ({val_split_idx} rows) must be strictly greater than "
                f"the sequence length ({self.sequence_length}). Provide more data, reduce the "
                f"sequence length, or decrease the validation/test split percentages."
            )
            raise ValueError()
        
        
        # Train sequence is from beginning to validation index
        X_train = features_df.iloc[:val_split_idx]
        
        # Extract the absolute last sequence of the entire dataset
        if total_size >= self.sequence_length:
            self._last_dataset_sequence = features_df.iloc[-self.sequence_length:].to_numpy(dtype=numpy.float32)
        else:
            _LOGGER.error("The total dataset size is smaller than the specified sequence length. Cannot extract a valid last sequence.")
            raise ValueError()
        
        # Validation and Test sequences start `sequence_length` BEFORE their split index 
        # to ensure the first prediction exactly matches the split boundary. 
        # Clamp to 0 to prevent negative slicing, and skip padding if the split size is 0.
        start_val = max(0, val_split_idx - self.sequence_length) if self.validation_split > 0 else val_split_idx
        start_test = max(0, test_split_idx - self.sequence_length) if self.test_split > 0 else test_split_idx
        
        X_val = features_df.iloc[start_val : test_split_idx]
        X_test = features_df.iloc[start_test :]
        
        # We pass dummy targets to the base scaler method since our targets are inside X
        dummy_y = pandas.Series(0.0, index=X_train.index)

        # --- 3. Scale Features (and thereby Targets) ---
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
            X_train_np, X_val_np, X_test_np = self._prepare_feature_scaler(
                X_train, dummy_y, X_val, X_test, label_dtype=torch.float32, schema=schema, verbose=verbose
            )
            
            # Extract target scaler directly from the fitted feature scaler to maintain mathematical consistency
            if self.feature_scaler and self.feature_scaler.mean_ is not None and self.feature_scaler.std_ is not None:
                continuous_targets = []
                target_means = []
                target_stds = []
                
                # Map absolute feature indices to relative target indices
                for relative_idx, abs_idx in enumerate(self._target_indices):
                    if self.feature_scaler.continuous_feature_indices and abs_idx in self.feature_scaler.continuous_feature_indices:
                        scaler_idx = self.feature_scaler.continuous_feature_indices.index(abs_idx)
                        continuous_targets.append(relative_idx)
                        target_means.append(self.feature_scaler.mean_[scaler_idx].item())
                        target_stds.append(self.feature_scaler.std_[scaler_idx].item())
                
                if continuous_targets:
                    # Create a dedicated target scaler scoped ONLY to the model's output size
                    self.target_scaler = DragonScaler(
                        mean=torch.tensor(target_means),
                        std=torch.tensor(target_stds),
                        continuous_feature_indices=continuous_targets
                    )
                else:
                    self.target_scaler = None
            else:
                self.target_scaler = None
        else:
            if verbose >= 2:
                _LOGGER.info("Features have not been scaled as specified.")
            # Explicit dtype prevents PyTorch from copying the array to cast it later
            X_train_np = X_train.to_numpy(dtype=numpy.float32)
            X_val_np = X_val.to_numpy(dtype=numpy.float32)
            X_test_np = X_test.to_numpy(dtype=numpy.float32)
            
            self.target_scaler = None


        # --- 4. Generate Windows ---
        self._train_ds: Optional[_PytorchSequenceDataset] = self._create_windowed_dataset(X_train_np, verbose=verbose)
        self._val_ds: Optional[_PytorchSequenceDataset] = self._create_windowed_dataset(X_val_np, verbose=verbose)
        self._test_ds: Optional[_PytorchSequenceDataset] = self._create_windowed_dataset(X_test_np, verbose=verbose)

        # Update base maker shapes
        self._X_train_shape = self._train_ds.features.shape if self._train_ds else (0,0)
        self._X_val_shape = self._val_ds.features.shape if self._val_ds else (0,0)
        self._X_test_shape = self._test_ds.features.shape if self._test_ds else (0,0)

        self._attach_scalers_to_datasets()

        if verbose >= 2:
            _LOGGER.info("Multivariate feature and label windows successfully generated.")

    def _create_windowed_dataset(self, data: numpy.ndarray, verbose: int = 3) -> _PytorchSequenceDataset:
        """Efficiently creates 2D windowed features and extracts targets using numpy strides."""
        if len(data) <= self.sequence_length:
            # Only warn if there is actually data, ignoring intentionally empty (length 0) arrays from 0.0 splits
            if verbose >= 1 and len(data) > 0:
                _LOGGER.warning(f"Data length ({len(data)}) not greater than sequence_length ({self.sequence_length}). Returning empty dataset.")
            
            # Maintain strict 3D/2D dimensional shapes even when empty to prevent downstream indexing errors
            num_features = data.shape[1]
            num_targets = len(self._target_indices)
            
            empty_features = numpy.empty((0, self.sequence_length, num_features), dtype=numpy.float32)
            
            if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
                empty_labels = numpy.empty((0, num_targets), dtype=numpy.float32)
            else:
                empty_labels = numpy.empty((0, self.sequence_length, num_targets), dtype=numpy.float32)
                
            return _PytorchSequenceDataset(empty_features, empty_labels, 
                               labels_dtype=torch.float32,
                               feature_names=self._feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        n_windows = len(data) - self.sequence_length + 1
        row_stride, col_stride = data.strides
        
        # Stride 2D array (N, F) into 3D array (N_windows, Seq_len, F)
        strided_data = numpy.lib.stride_tricks.as_strided(
            data, 
            shape=(n_windows, self.sequence_length, data.shape[1]), 
            strides=(row_stride, row_stride, col_stride)
        )
        
        if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
            # Drop the last window for inputs because there is no "next step" to predict
            features = strided_data[:-1]
            # Targets are the vectors at sequence_length offset, selecting only target columns
            labels = data[self.sequence_length:, self._target_indices]
            
        else: # SEQUENCE_SEQUENCE
            # Drop the last window for inputs
            features = strided_data[:-1]
            # Targets are the shifted windows, selecting only target columns
            labels = strided_data[1:, :, self._target_indices]
        
        # propagate class_map for categorical targets to the dataset instance
        subset_dataset = _PytorchSequenceDataset(features, labels,
                               labels_dtype=torch.float32,
                               feature_names=self._feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        subset_dataset._class_map = self.class_map
        
        return subset_dataset
    
    @property
    def last_dataset_sequence(self) -> numpy.ndarray:
        """
        Returns the final unscaled window of the entire dataset.
        
        Ideal for forecasting the actual unknown future (initial sequence in inference).
        """
        return self._last_dataset_sequence
    
    @property
    def target_types(self) -> dict[str, str]:
        """
        Returns the target types for the dataset. 
        
        Mapping of target names to their types, either 'continuous' or 'categorical'.
        """
        return self._target_types

    def save_dataset_bundle(self, directory: Union[str, Path], verbose: bool = True) -> None:
        """
        Saves the train, validation, and test sets along with all metadata 
        to a single .pth file using dictionary serialization.
        
        Args:
            directory (Union[str, Path]): The directory where the bundle will be saved. 
                Parent directories will be created automatically if they do not exist.
            verbose (bool): Whether to output log messages indicating a successful save.
        """
        save_path = make_fullpath(directory, make=True, enforce="directory")
        safe_mode = self.prediction_mode.replace("-", "_").replace(" ", "_")
        
        dataset_name_suffix = f"{self._id}_{safe_mode}"
        filename = f"{DatasetKeys.DATASET_FILENAME}_{dataset_name_suffix}.pth"
        filepath = save_path / filename

        bundle = {
            DatasetKeys.TRAIN_SUBSET: {
                "features": self.train_dataset.features if self._train_ds else None, # type: ignore
                "labels": self.train_dataset.labels if self._train_ds else None # type: ignore
            },
            DatasetKeys.VALIDATION_SUBSET: {
                "features": self.validation_dataset.features if self._val_ds else None, # type: ignore
                "labels": self.validation_dataset.labels if self._val_ds else None # type: ignore
            },
            DatasetKeys.TEST_SUBSET: {
                "features": self.test_dataset.features if self._test_ds else None, # type: ignore
                "labels": self.test_dataset.labels if self._test_ds else None # type: ignore
            },
            DatasetKeys.FEATURE_NAMES: self.feature_names,
            DatasetKeys.TARGET_NAMES: self.target_names,
            DatasetKeys.TARGET_TYPES: self.target_types,
            DatasetKeys.VALIDATION_SPLIT: self.validation_split,
            DatasetKeys.TEST_SPLIT: self.test_split,
            DatasetKeys.PREDICTION_MODE: self.prediction_mode,
            DatasetKeys.SEQUENCE_LENGTH: self.sequence_length,
            DatasetKeys.ID: self.id,
            ScalerKeys.FEATURE_SCALER: self.feature_scaler._get_state() if self.feature_scaler else None,
            ScalerKeys.TARGET_SCALER: self.target_scaler._get_state() if self.target_scaler else None,
            DatasetKeys.CLASS_MAP: self.class_map,
            DatasetKeys.CLASSES: list(), # vestige from base class, not used in sequence datasets
            DatasetKeys.LAST_SEQUENCE: self.last_dataset_sequence
        }
        
        torch.save(bundle, filepath)
        if verbose:
            _LOGGER.info(f"Sequence dataset bundle saved to '{filepath.name}'.")
            
        ### Save JSON report
        report_filename = f"{DatasetKeys.JSON_REPORT_PREFIX}_{dataset_name_suffix}.json"
        train_split = round(1.0 - self.validation_split - self.test_split, 2)

        report_data = {
            "dataset_id": self.id,
            "prediction_mode": self.prediction_mode,
            "sequence_length": self.sequence_length,
            "number_of_features": self.number_of_features,
            "number_of_targets": self.number_of_targets,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "target_types": self.target_types,
            "split_sizes": {
                "train": train_split,
                "validation": self.validation_split,
                "test": self.test_split
            },
            "number_of_windows": {
                "train": self._X_train_shape[0],
                "validation": self._X_val_shape[0],
                "test": self._X_test_shape[0]
            },
            "class_map": self.class_map,
            "scalers": {
                "feature_scaler": self.feature_scaler is not None,
                "target_scaler": self.target_scaler is not None
            }
        }

        save_json(
            data=report_data,
            directory=save_path,
            filename=report_filename,
            verbose=verbose
        )
        

    @classmethod
    def from_bundle(cls, filepath: Union[str, Path], verbose: bool = False) -> 'DragonDatasetSequence':
        """
        Alternative constructor to instantiate a dataset object from a saved bundle.
        
        Bypasses standard initialization to reconstruct the entire state from a `.pth` 
        file. This includes restoring metadata, reloading `DragonScaler` states, and 
        rebuilding the Custom Dataset instances for the train, validation, and test 
        subsets. If a directory is provided instead of a file, it will attempt to 
        automatically resolve the `.pth` file using the default naming pattern.
        
        Args:
            filepath (Union[str, Path]): The direct path to the `.pth` file, or a 
                directory containing exactly one matching dataset bundle.
            verbose (bool): Whether to log the loading process.
        
        Returns:
            DragonDatasetSequence: An instance of the class fully populated with the loaded datasets, scalers, and metadata.
        """
        target_filepath = make_fullpath(filepath, make=False)
        
        if not target_filepath.is_file():
            if target_filepath.is_dir():
                expected_pattern = f"{DatasetKeys.DATASET_FILENAME}_Seq_*.pth"
                matching_files = list(target_filepath.glob(expected_pattern))
                if not matching_files:
                    raise FileNotFoundError(f"No files matching pattern '{expected_pattern}' found.")
                elif len(matching_files) > 1:
                    raise FileNotFoundError(f"Multiple files matching pattern '{expected_pattern}' found. Specify exact file.")
                else:
                    target_filepath = matching_files[0]
            else:
                raise FileNotFoundError(f"Provided path '{target_filepath}' is invalid.")
            
        bundle = torch.load(target_filepath, weights_only=False)
        instance = cls.__new__(cls)
        
        # 1. Restore Base Attributes
        instance._train_ds = None
        instance._val_ds = None
        instance._test_ds = None
        instance._X_train_shape = (0,0)
        instance._X_val_shape = (0,0)
        instance._X_test_shape = (0,0)
        instance._y_train_shape = (0,)
        instance._y_val_shape = (0,)
        instance._y_test_shape = (0,)

        # 2. Restore Metadata
        instance.prediction_mode = bundle.get(DatasetKeys.PREDICTION_MODE)
        instance.sequence_length = bundle.get(DatasetKeys.SEQUENCE_LENGTH)
        instance.validation_split = bundle.get(DatasetKeys.VALIDATION_SPLIT, 0.0)
        instance.test_split = bundle.get(DatasetKeys.TEST_SPLIT, 0.0)
        instance._feature_names = bundle.get(DatasetKeys.FEATURE_NAMES, [])
        instance._target_names = bundle.get(DatasetKeys.TARGET_NAMES, [])
        instance._id = bundle.get(DatasetKeys.ID, "")
        instance.class_map = bundle.get(DatasetKeys.CLASS_MAP, {})
        instance.classes = list()  # vestige from base class, not used in sequence datasets
        instance._target_types = bundle.get(DatasetKeys.TARGET_TYPES, {})
        instance._last_dataset_sequence = bundle.get(DatasetKeys.LAST_SEQUENCE)

        # 3. Reconstruct Scaler
        f_scaler_state = bundle.get(ScalerKeys.FEATURE_SCALER)
        t_scaler_state = bundle.get(ScalerKeys.TARGET_SCALER)

        if f_scaler_state:
            instance.feature_scaler = DragonScaler.load(f_scaler_state, verbose=False)
        else:
            instance.feature_scaler = None

        if t_scaler_state:
            instance.target_scaler = DragonScaler.load(t_scaler_state, verbose=False)
        else:
            instance.target_scaler = None

        # 4. Reconstruct Datasets
        def _build_ds(split_key: str):
            split_data = bundle.get(split_key)
            if split_data and split_data.get("features") is not None and split_data.get("labels") is not None:
                features = split_data["features"]
                labels = split_data["labels"]
                ds = _PytorchSequenceDataset(
                    features=features.numpy(),
                    labels=labels.numpy(),
                    labels_dtype=labels.dtype,
                    features_dtype=features.dtype,
                    feature_names=instance._feature_names,
                    target_names=instance._target_names,
                    target_types=instance._target_types,
                )
                ds._feature_scaler = instance.feature_scaler
                ds._target_scaler = instance.target_scaler
                return ds, features.shape, labels.shape
            return None, (0,0), (0,)

        instance._train_ds, instance._X_train_shape, instance._y_train_shape = _build_ds(DatasetKeys.TRAIN_SUBSET)
        instance._val_ds, instance._X_val_shape, instance._y_val_shape = _build_ds(DatasetKeys.VALIDATION_SUBSET)
        instance._test_ds, instance._X_test_shape, instance._y_test_shape = _build_ds(DatasetKeys.TEST_SUBSET)
        
        if verbose:
            _LOGGER.info(
                f"Dataset loaded from '{target_filepath.name}' with ID '{instance.id}'.\n"
                f"{repr(instance)}"
            )
        
        return instance

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__} (ID: '{self.id}')>\n"
        s += f"  Prediction Mode: {self.prediction_mode}\n"
        s += f"  Sequence Length (Window): {self.sequence_length}\n"
        s += f"  Targets: {len(self._target_names)}\n"
        s += f"  Features: {self.number_of_features}\n"
        s += f"  Feature-Scaler: {'Present' if self.feature_scaler else 'None'}\n"
        s += f"  Target-Scaler: {'Present' if self.target_scaler else 'None'}\n"
        
        if self._train_ds: s += f"  Train Windows: {len(self._train_ds)}\n"
        if self._val_ds: s += f"  Validation Windows: {len(self._val_ds)}\n"
        if self._test_ds: s += f"  Test Windows: {len(self._test_ds)}\n"
        return s
