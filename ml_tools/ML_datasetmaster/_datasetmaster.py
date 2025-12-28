import torch
import pandas
from sklearn.model_selection import train_test_split
from typing import Literal, Union, Optional

from ..ML_scaler import DragonScaler
from ..schema import FeatureSchema

from .._core import get_logger
from ..keys._keys import MLTaskKeys

from ._base_datasetmaster import _BaseDatasetMaker, _PytorchDataset


_LOGGER = get_logger("DragonDataset")


__all__ = [
    "DragonDataset",
    "DragonDatasetMulti"
]


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
                 random_state: int = 42,
                 verbose: int = 2):
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
            verbose (int): Verbosity level for logging.
                - 0: Errors only
                - 1: Warnings
                - 2: Info
                - 3: Detailed process info
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
                X_train, y_train, X_val, X_test, label_dtype, schema, verbose=verbose
            )
        else:
            if verbose >= 2:
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
                y_train_final, y_val_final, y_test_final = self._prepare_target_scaler(y_train, y_val, y_test, verbose=verbose)
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
                if verbose >= 1:
                    _LOGGER.warning("No class map provided for classification task at initialization. Use `.set_class_map()`.")
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
                 target_columns: list[str],
                 schema: FeatureSchema,
                 kind: Literal["multitarget regression", "multilabel binary classification"],
                 feature_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 target_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                 validation_size: float = 0.2,
                 test_size: float = 0.1,
                 random_state: int = 42,
                 verbose: int = 2):
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
            verbose (int): Verbosity level for logging.
                - 0: Errors only
                - 1: Warnings
                - 2: Info
                - 3: Detailed process info
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
            if verbose >= 1:
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
                X_train, y_train, X_val, X_test, label_dtype, schema, verbose=verbose
            )
        else:
            if verbose >= 2:
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
                y_train_final, y_val_final, y_test_final = self._prepare_target_scaler(y_train, y_val, y_test, verbose=verbose)
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

