import torch
from torch.utils.data import Dataset, Subset
import pandas
import numpy
from sklearn.model_selection import train_test_split
from typing import Literal, Union, Tuple, List, Optional
from abc import ABC, abstractmethod
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .custom_logger import save_list_strings
from .ML_scaler import PytorchScaler
from .keys import DatasetKeys


__all__ = [
    "DatasetMaker",
    "DatasetMakerMulti",
    "VisionDatasetMaker",
    "SequenceMaker",
    "ResizeAspectFill",
]


# --- Internal Helper Class ---
class _PytorchDataset(Dataset):
    """
    Internal helper class to create a PyTorch Dataset.
    Converts numpy/pandas data into tensors for model consumption.
    """
    def __init__(self, features: Union[numpy.ndarray, pandas.DataFrame], 
                 labels: Union[numpy.ndarray, pandas.Series],
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
        else:
            self.features = torch.tensor(features.values, dtype=features_dtype)

        if isinstance(labels, numpy.ndarray):
            self.labels = torch.tensor(labels, dtype=labels_dtype)
        else:
            self.labels = torch.tensor(labels.values, dtype=labels_dtype)
            
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


# --- Abstract Base Class (New) ---
# --- Abstract Base Class (Corrected) ---
class _BaseDatasetMaker(ABC):
    """
    Abstract base class for dataset makers. Contains shared logic for
    splitting, scaling, and accessing datasets to reduce code duplication.
    """
    def __init__(self):
        self._train_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None
        self.scaler: Optional[PytorchScaler] = None
        self._id: Optional[str] = None
        self._feature_names: List[str] = []
        self._target_names: List[str] = []
        self._X_train_shape = (0,0)
        self._X_test_shape = (0,0)
        self._y_train_shape = (0,)
        self._y_test_shape = (0,)

    def _prepare_scaler(self, X_train: pandas.DataFrame, y_train: Union[pandas.Series, pandas.DataFrame], X_test: pandas.DataFrame, label_dtype: torch.dtype, continuous_feature_columns: Optional[Union[List[int], List[str]]]):
        """Internal helper to fit and apply a PytorchScaler."""
        continuous_feature_indices: Optional[List[int]] = None
        if continuous_feature_columns:
            if all(isinstance(c, str) for c in continuous_feature_columns):
                name_to_idx = {name: i for i, name in enumerate(self._feature_names)}
                try:
                    continuous_feature_indices = [name_to_idx[name] for name in continuous_feature_columns] # type: ignore
                except KeyError as e:
                    _LOGGER.error(f"Feature column '{e.args[0]}' not found.")
                    raise ValueError()
            elif all(isinstance(c, int) for c in continuous_feature_columns):
                continuous_feature_indices = continuous_feature_columns # type: ignore
            else:
                _LOGGER.error("'continuous_feature_columns' must be a list of all strings or all integers.")
                raise TypeError()

        X_train_values = X_train.values
        X_test_values = X_test.values

        if self.scaler is None and continuous_feature_indices:
            _LOGGER.info("Fitting a new PytorchScaler on training data.")
            temp_train_ds = _PytorchDataset(X_train_values, y_train, label_dtype) # type: ignore
            self.scaler = PytorchScaler.fit(temp_train_ds, continuous_feature_indices)

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

    def save_scaler(self, save_dir: Union[str, Path], verbose: bool=True) -> None:
        """
        Saves the fitted PytorchScaler's state to a .pth file.

        The filename is automatically generated based on the dataset id.
        
        Args:
            save_dir (str | Path): The directory where the scaler will be saved.
        """
        if not self.scaler: 
            _LOGGER.error("No scaler was fitted or provided.")
            raise RuntimeError()
        if not self.id: 
            _LOGGER.error("Must set the dataset `id` before saving scaler.")
            raise ValueError()
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        sanitized_id = sanitize_filename(self.id)
        filename = f"{DatasetKeys.SCALER_PREFIX}{sanitized_id}.pth"
        filepath = save_path / filename
        self.scaler.save(filepath, verbose=False)
        if verbose:
            _LOGGER.info(f"Scaler for dataset '{self.id}' saved as '{filepath.name}'.")


# Single target dataset
class DatasetMaker(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with a single target column.

    This class takes a DataFrame, automatically splits it into training and
    testing sets, and converts them into PyTorch Datasets. It assumes the
    target variable is the last column. It can also create, apply, and
    save a PytorchScaler for standardizing continuous features.
    
    Attributes:
        `scaler` -> PytorchScaler | None
        `train_dataset` -> PyTorch Dataset
        `test_dataset`  -> PyTorch Dataset
        `feature_names` -> list[str]
        `target_names`  -> list[str]
        `id` -> str
        
    The ID can be manually set to any string if needed, it is the target name by default.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 kind: Literal["regression", "classification"],
                 test_size: float = 0.2,
                 random_state: int = 42,
                 scaler: Optional[PytorchScaler] = None,
                 continuous_feature_columns: Optional[Union[List[int], List[str]]] = None):
        """
        Args:
            pandas_df (pandas.DataFrame): The pre-processed input DataFrame with numerical data.
            kind (Literal["regression", "classification"]): The type of ML task. This determines the data type of the labels.
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): The seed for the random number generator for reproducibility.
            scaler (PytorchScaler | None): A pre-fitted PytorchScaler instance.
            continuous_feature_columns (List[int] | List[str] | None): Column indices or names of continuous features to scale. If provided creates a new PytorchScaler.
        """
        super().__init__()
        self.scaler = scaler
        
        # --- 1. Identify features and target (single-target logic) ---
        features = pandas_df.iloc[:, :-1]
        target = pandas_df.iloc[:, -1]
        self._feature_names = features.columns.tolist()
        self._target_names = [str(target.name)]
        self._id = self._target_names[0]

        # --- 2. Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        self._X_train_shape, self._X_test_shape = X_train.shape, X_test.shape
        self._y_train_shape, self._y_test_shape = y_train.shape, y_test.shape
        
        label_dtype = torch.float32 if kind == "regression" else torch.int64

        # --- 3. Scale ---
        X_train_final, X_test_final = self._prepare_scaler(
            X_train, y_train, X_test, label_dtype, continuous_feature_columns
        )
        
        # --- 4. Create Datasets ---
        self._train_ds = _PytorchDataset(X_train_final, y_train.values, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test.values, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)


# --- New Multi-Target Class ---
class DatasetMakerMulti(_BaseDatasetMaker):
    """
    Dataset maker for pre-processed, numerical pandas DataFrames with a multiple target columns.

    This class takes a DataFrame, automatically splits it into training and testing sets, and converts them into PyTorch Datasets.
    """
    def __init__(self,
                 pandas_df: pandas.DataFrame,
                 target_columns: List[str],
                 test_size: float = 0.2,
                 random_state: int = 42,
                 scaler: Optional[PytorchScaler] = None,
                 continuous_feature_columns: Optional[Union[List[int], List[str]]] = None):
        """
        Args:
            pandas_df (pandas.DataFrame): The pre-processed input DataFrame with numerical data.
            target_columns (list[str]): List of target column names.
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): The seed for the random number generator for reproducibility.
            scaler (PytorchScaler | None): A pre-fitted PytorchScaler instance.
            continuous_feature_columns (List[int] | List[str] | None): Column indices or names of continuous features to scale. If provided creates a new PytorchScaler.
        """
        super().__init__()
        self.scaler = scaler

        self._target_names = target_columns
        self._feature_names = [col for col in pandas_df.columns if col not in target_columns]
        features = pandas_df[self._feature_names]
        target = pandas_df[self._target_names]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        self._X_train_shape, self._X_test_shape = X_train.shape, X_test.shape
        self._y_train_shape, self._y_test_shape = y_train.shape, y_test.shape
        
        label_dtype = torch.float32

        X_train_final, X_test_final = self._prepare_scaler(
            X_train, y_train, X_test, label_dtype, continuous_feature_columns
        )
        
        self._train_ds = _PytorchDataset(X_train_final, y_train, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)
        self._test_ds = _PytorchDataset(X_test_final, y_test, labels_dtype=label_dtype, feature_names=self._feature_names, target_names=self._target_names)


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


# --- VisionDatasetMaker ---
class VisionDatasetMaker(_BaseMaker):
    """
    Creates processed PyTorch datasets for computer vision tasks from an
    image folder directory.
    
    Uses online augmentations per epoch (image augmentation without creating new files).
    """
    def __init__(self, full_dataset: ImageFolder):
        super().__init__()
        self.full_dataset = full_dataset
        self.labels = [s[1] for s in self.full_dataset.samples]
        self.class_map = full_dataset.class_to_idx
        
        self._is_split = False
        self._are_transforms_configured = False

    @classmethod
    def from_folder(cls, root_dir: str) -> 'VisionDatasetMaker':
        """Creates a maker instance from a root directory of images."""
        initial_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = ImageFolder(root=root_dir, transform=initial_transform)
        _LOGGER.info(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes.")
        return cls(full_dataset)
        
    @staticmethod
    def inspect_folder(path: Union[str, Path]):
        """
        Logs a report of the types, sizes, and channels of image files
        found in the directory and its subdirectories.
        """
        path_obj = make_fullpath(path)

        non_image_files = set()
        img_types = set()
        img_sizes = set()
        img_channels = set()
        img_counter = 0

        _LOGGER.info(f"Inspecting folder: {path_obj}...")
        # Use rglob to recursively find all files
        for filepath in path_obj.rglob('*'):
            if filepath.is_file():
                try:
                    # Using PIL to open is a more reliable check
                    with Image.open(filepath) as img:
                        img_types.add(img.format)
                        img_sizes.add(img.size)
                        img_channels.update(img.getbands())
                        img_counter += 1
                except (IOError, SyntaxError):
                    non_image_files.add(filepath.name)

        if non_image_files:
            _LOGGER.warning(f"Non-image or corrupted files found and ignored: {non_image_files}")

        report = (
            f"\n--- Inspection Report for '{path_obj.name}' ---\n"
            f"Total images found: {img_counter}\n"
            f"Image formats: {img_types or 'None'}\n"
            f"Image sizes (WxH): {img_sizes or 'None'}\n"
            f"Image channels (bands): {img_channels or 'None'}\n"
            f"--------------------------------------"
        )
        print(report)

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   stratify: bool = True, random_state: Optional[int] = None) -> 'VisionDatasetMaker':
        """Splits the dataset into training, validation, and optional test sets."""
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return self

        if val_size + test_size >= 1.0:
            _LOGGER.error("The sum of val_size and test_size must be less than 1.")
            raise ValueError()

        indices = list(range(len(self.full_dataset)))
        labels_for_split = self.labels if stratify else None

        train_indices, val_test_indices = train_test_split(
            indices, test_size=(val_size + test_size), random_state=random_state, stratify=labels_for_split
        )

        if test_size > 0:
            val_test_labels = [self.labels[i] for i in val_test_indices]
            stratify_val_test = val_test_labels if stratify else None
            val_indices, test_indices = train_test_split(
                val_test_indices, test_size=(test_size / (val_size + test_size)), 
                random_state=random_state, stratify=stratify_val_test
            )
            self._test_dataset = Subset(self.full_dataset, test_indices)
            _LOGGER.info(f"Test set created with {len(self._test_dataset)} images.")
        else:
            val_indices = val_test_indices
        
        self._train_dataset = Subset(self.full_dataset, train_indices)
        self._val_dataset = Subset(self.full_dataset, val_indices)
        self._is_split = True
        
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, resize_size: int = 256, crop_size: int = 224, 
                             mean: List[float] = [0.485, 0.456, 0.406], 
                             std: List[float] = [0.229, 0.224, 0.225],
                             extra_train_transforms: Optional[List] = None) -> 'VisionDatasetMaker':
        """Configures and applies the image transformations (augmentations)."""
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()

        base_train_transforms = [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip()]
        if extra_train_transforms:
            base_train_transforms.extend(extra_train_transforms)
        
        final_transforms = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        val_transform = transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(crop_size), *final_transforms])
        train_transform = transforms.Compose([*base_train_transforms, *final_transforms])

        self._train_dataset.dataset.transform = train_transform # type: ignore
        self._val_dataset.dataset.transform = val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.dataset.transform = val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Image transforms configured and applied.")
        return self

    def get_datasets(self) -> Tuple[Dataset, ...]:
        """Returns the final train, validation, and optional test datasets."""
        if not self._is_split:
            _LOGGER.error("Data has not been split. Call .split_data() first.")
            raise RuntimeError()
        if not self._are_transforms_configured:
            _LOGGER.warning("Transforms have not been configured. Using default ToTensor only.")

        if self._test_dataset:
            return self._train_dataset, self._val_dataset, self._test_dataset
        return self._train_dataset, self._val_dataset


# --- SequenceMaker ---
class SequenceMaker(_BaseMaker):
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

    def normalize_data(self) -> 'SequenceMaker':
        """
        Normalizes the sequence data using PytorchScaler. Must be called AFTER 
        splitting to prevent data leakage from the test set.
        """
        if not self._is_split:
            _LOGGER.error("Data must be split BEFORE normalizing. Call .split_data() first.")
            raise RuntimeError()

        if self.scaler:
            _LOGGER.warning("Data has already been normalized.")
            return self

        # 1. PytorchScaler requires a Dataset to fit. Create a temporary one.
        # The scaler expects 2D data [n_samples, n_features].
        train_features = self.train_sequence.reshape(-1, 1) # type: ignore

        # _PytorchDataset needs labels, so we create dummy ones.
        dummy_labels = numpy.zeros(len(train_features))
        temp_train_ds = _PytorchDataset(train_features, dummy_labels, labels_dtype=torch.float32)

        # 2. Fit the PytorchScaler on the temporary training dataset.
        # The sequence is a single feature, so its index is [0].
        _LOGGER.info("Fitting PytorchScaler on the training data...")
        self.scaler = PytorchScaler.fit(temp_train_ds, continuous_feature_indices=[0])

        # 3. Transform sequences using the fitted scaler.
        # The transform method requires a tensor, so we convert, transform, and convert back.
        train_tensor = torch.tensor(self.train_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore
        test_tensor = torch.tensor(self.test_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore

        self.train_sequence = self.scaler.transform(train_tensor).numpy().flatten()
        self.test_sequence = self.scaler.transform(test_tensor).numpy().flatten()

        self._is_normalized = True
        _LOGGER.info("Sequence data normalized using PytorchScaler.")
        return self

    def split_data(self, test_size: float = 0.2) -> 'SequenceMaker':
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

    def generate_windows(self, sequence_to_sequence: bool = False) -> 'SequenceMaker':
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
        """Applies inverse transformation using the stored PytorchScaler."""
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


# --- Custom Vision Transform Class ---
class ResizeAspectFill:
    """
    Custom transformation to make an image square by padding it to match the
    longest side, preserving the aspect ratio. The image is finally centered.

    Args:
        pad_color (Union[str, int]): Color to use for the padding.
                                     Defaults to "black".
    """
    def __init__(self, pad_color: Union[str, int] = "black") -> None:
        self.pad_color = pad_color

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            _LOGGER.error(f"Expected PIL.Image.Image, got {type(image).__name__}")
            raise TypeError()

        w, h = image.size
        if w == h:
            return image

        # Determine padding to center the image
        if w > h:
            top_padding = (w - h) // 2
            bottom_padding = w - h - top_padding
            padding = (0, top_padding, 0, bottom_padding)
        else: # h > w
            left_padding = (h - w) // 2
            right_padding = h - w - left_padding
            padding = (left_padding, 0, right_padding, 0)

        return ImageOps.expand(image, padding, fill=self.pad_color)


def info():
    _script_info(__all__)
