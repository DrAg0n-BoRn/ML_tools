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

__all__ = [
    "DatasetMaker",
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
                 features_dtype: torch.dtype = torch.float32):
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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Streamlined DatasetMaker version
class DatasetMaker:
    """
    A simplified dataset maker for pre-processed, numerical pandas DataFrames.

    This class takes a DataFrame, automatically splits it into training and
    testing sets, and converts them into PyTorch Datasets. It assumes the
    target variable is the last column. It can also create, apply, and
    save a PytorchScaler for standardizing continuous features.
    
    Attributes:
        `scaler` -> PytorchScaler | None
        `train_dataset` -> PyTorch Dataset
        `test_dataset`  -> PyTorch Dataset
        `feature_names` -> list[str]
        `target_name`   -> str
        `id` -> str | None
        
    The ID can be manually set to any string if needed, it is `None` by default.
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
        # Validation
        if not isinstance(pandas_df, pandas.DataFrame):
            raise TypeError("Input must be a pandas.DataFrame.")
        if kind not in ["regression", "classification"]:
            raise ValueError("`kind` must be 'regression' or 'classification'.")

        # 1. Identify features and target
        features = pandas_df.iloc[:, :-1]
        target = pandas_df.iloc[:, -1]

        self._feature_names = features.columns.tolist()
        self._target_name = str(target.name)
        
        #set id
        self._id: Optional[str] = None
        # set scaler
        self.scaler = scaler

        # 2. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

        self._X_train_shape = X_train.shape
        self._X_test_shape = X_test.shape
        self._y_train_shape = y_train.shape
        self._y_test_shape = y_test.shape
        
        # 3. Handle Column to Index Conversion
        continuous_feature_indices: Optional[List[int]] = None
        if continuous_feature_columns:
            if all(isinstance(c, str) for c in continuous_feature_columns):
                name_to_idx = {name: i for i, name in enumerate(self._feature_names)}
                try:
                    continuous_feature_indices = [name_to_idx[name] for name in continuous_feature_columns] # type: ignore
                except KeyError as e:
                    raise ValueError(f"Feature column '{e.args[0]}' not found in DataFrame.")
            elif all(isinstance(c, int) for c in continuous_feature_columns):
                continuous_feature_indices = continuous_feature_columns # type: ignore
            else:
                raise TypeError("`continuous_feature_columns` must be a list of all strings or all integers.")
        
        # 4. Handle Scaling
        X_train_values = X_train.values
        X_test_values = X_test.values
        
        # If no scaler is provided, fit a new one from the training data
        if self.scaler is None:
            if continuous_feature_indices:
                _LOGGER.info("Feature indices provided. Fitting a new PytorchScaler on training data.")
                # A temporary dataset is needed for the PytorchScaler.fit method
                temp_label_dtype = torch.float32 if kind == "regression" else torch.int64
                temp_train_ds = _PytorchDataset(X_train_values, y_train.values, labels_dtype=temp_label_dtype)
                self.scaler = PytorchScaler.fit(temp_train_ds, continuous_feature_indices)
        
        # If a scaler exists (either passed in or just fitted), apply it
        if self.scaler and self.scaler.mean_ is not None:
            _LOGGER.info("Applying scaler transformation to train and test feature sets.")
            X_train_tensor = self.scaler.transform(torch.tensor(X_train_values, dtype=torch.float32))
            X_test_tensor = self.scaler.transform(torch.tensor(X_test_values, dtype=torch.float32))
            # Convert back to numpy for the _PytorchDataset class
            X_train_values = X_train_tensor.numpy()
            X_test_values = X_test_tensor.numpy()

        # 5. Convert to final PyTorch Datasets
        label_dtype = torch.float32 if kind == "regression" else torch.int64
        self._train_ds = _PytorchDataset(X_train_values, y_train.values, labels_dtype=label_dtype)
        self._test_ds = _PytorchDataset(X_test_values, y_test.values, labels_dtype=label_dtype)

    @property
    def train_dataset(self) -> Dataset:
        """Returns the training PyTorch dataset."""
        return self._train_ds

    @property
    def test_dataset(self) -> Dataset:
        """Returns the testing PyTorch dataset."""
        return self._test_ds

    @property
    def feature_names(self) -> list[str]:
        """Returns the list of feature column names."""
        return self._feature_names

    @property
    def target_name(self) -> str:
        """Returns the name of the target column."""
        return self._target_name
    
    @property
    def id(self) -> Optional[str]:
        """Returns the object identifier if any."""
        return self._id
    
    @id.setter
    def id(self, dataset_id: str):
        """Sets the ID value"""
        if not isinstance(dataset_id, str):
            raise ValueError(f"Dataset ID '{type(dataset_id)}' is not a string.")
        self._id = dataset_id

    def dataframes_info(self) -> None:
        """Prints the shape information of the split pandas DataFrames."""
        print("--- Original DataFrame Shapes After Split ---")
        print(f"  X_train shape: {self._X_train_shape}")
        print(f"  y_train shape: {self._y_train_shape}\n")
        print(f"  X_test shape:  {self._X_test_shape}")
        print(f"  y_test shape:  {self._y_test_shape}")
        print("-------------------------------------------")
        
    def save_feature_names(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """Saves a list of feature names as a text file"""
        save_list_strings(list_strings=self._feature_names,
                          directory=directory,
                          filename="feature_names",
                          verbose=verbose)
        
    def save_scaler(self, save_dir: Union[str, Path]):
        """
        Saves the fitted PytorchScaler's state to a .pth file.

        The filename is automatically generated based on the target name.
        
        Args:
            save_dir (str | Path): The directory where the scaler will be saved.
        """
        if not self.scaler:
            _LOGGER.error("❌ No scaler was fitted or provided.")
            return

        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        # Sanitize the target name for use in a filename
        sanitized_target = sanitize_filename(self.target_name)
        filename = f"scaler_{sanitized_target}.pth"
        
        filepath = save_path / filename
        self.scaler.save(filepath)


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
        _LOGGER.info(report)

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   stratify: bool = True, random_state: Optional[int] = None) -> 'VisionDatasetMaker':
        """Splits the dataset into training, validation, and optional test sets."""
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return self

        if val_size + test_size >= 1.0:
            raise ValueError("The sum of val_size and test_size must be less than 1.")

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
            raise RuntimeError("Transforms must be configured AFTER splitting data. Call .split_data() first.")

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
            raise RuntimeError("Data has not been split. Call .split_data() first.")
        if not self._are_transforms_configured:
            _LOGGER.warning("⚠️ Transforms have not been configured. Using default ToTensor only.")

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
            raise TypeError("Data must be a pandas DataFrame/Series or a numpy array.")
            
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
            raise RuntimeError("Data must be split BEFORE normalizing. Call .split_data() first.")

        if self.scaler:
            _LOGGER.warning("⚠️ Data has already been normalized.")
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
        _LOGGER.info("✅ Sequence data normalized using PytorchScaler.")
        return self

    def split_data(self, test_size: float = 0.2) -> 'SequenceMaker':
        """Splits the sequence into training and testing portions."""
        if self._is_split:
            _LOGGER.warning("⚠️ Data has already been split.")
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
            raise RuntimeError("Cannot generate windows before splitting data. Call .split_data() first.")

        self._train_dataset = self._create_windowed_dataset(self.train_sequence, sequence_to_sequence) # type: ignore
        self._test_dataset = self._create_windowed_dataset(self.test_sequence, sequence_to_sequence) # type: ignore
        
        self._are_windows_generated = True
        _LOGGER.info("Feature and label windows generated for train and test sets.")
        return self

    def _create_windowed_dataset(self, data: numpy.ndarray, use_sequence_labels: bool) -> Dataset:
        """Efficiently creates windowed features and labels using numpy."""
        if len(data) <= self.sequence_length:
            raise ValueError("Data length must be greater than the sequence_length to create at least one window.")
            
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
            raise RuntimeError("Data was not normalized. Cannot denormalize.")

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
            raise RuntimeError("Cannot plot before splitting data. Call .split_data() first.")
        
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
            raise RuntimeError("Windows have not been generated. Call .generate_windows() first.")
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
            raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

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
