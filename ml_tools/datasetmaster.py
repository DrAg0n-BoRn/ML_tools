import torch
from torch.utils.data import Dataset, Subset
from torch import nn
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal, Union, Tuple, List, Optional
from imblearn.combine import SMOTETomek
from abc import ABC, abstractmethod
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from .utilities import _script_info, make_fullpath
from .logger import _LOGGER


# --- public-facing API ---
__all__ = [
    "DatasetMaker",
    "VisionDatasetMaker",
    "SequenceMaker",
    "ResizeAspectFill",
]


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


# --- Internal Helper Class ---
class _PytorchDataset(Dataset):
    """
    Internal helper class to create a PyTorch Dataset.
    Converts numpy/pandas data into tensors for model consumption.
    """
    def __init__(self, features: Union[numpy.ndarray, pandas.DataFrame], 
                 labels: Union[numpy.ndarray, pandas.Series],
                 features_dtype: torch.dtype = torch.float32, 
                 labels_dtype: torch.dtype = torch.int64):
        
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


# --- Private Base Class ---
class _BaseMaker(ABC):
    """
    Abstract Base Class for all dataset makers.
    Ensures a consistent API across the library.
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


# --- Refactored DatasetMaker ---
class DatasetMaker(_BaseMaker):
    """
    Creates processed PyTorch datasets from a Pandas DataFrame using a fluent, step-by-step interface.
    
    Recommended pipeline:
    
    - Full Control (step-by-step):
        1. Process categorical features `.process_categoricals()`
        2. Split train-test datasets `.split_data()`
        3. Normalize continuous features `.normalize_continuous()`; `.denormalize()` becomes available.
        4. [Optional][Classification only] Balance classes `.balance_data()`
        5. Get PyTorch datasets: `train, test = .get_datasets()`
        6. [Optional] Inspect the processed data as DataFrames `X_train, X_test, y_train, y_test = .inspect_dataframes()`

    - Automated (single call):
    ```python
    maker = DatasetMaker(df, label_col='target')
    maker.process() # uses simplified arguments
    train_ds, test_ds = maker.get_datasets()
    ```
    """
    def __init__(self, pandas_df: pandas.DataFrame, label_col: str):
        super().__init__()
        if not isinstance(pandas_df, pandas.DataFrame):
            raise TypeError("Input must be a pandas.DataFrame.")
        if label_col not in pandas_df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.labels = pandas_df[label_col]
        self.features = pandas_df.drop(columns=label_col)
        self.labels_map = None
        self.scaler = None

        self._is_split = False
        self._is_balanced = False
        self._is_normalized = False
        self._is_categoricals_processed = False
        
        self.features_train = None
        self.features_test = None
        self.labels_train = None
        self.labels_test = None
        
        self.continuous_columns = None 

    def process_categoricals(self, method: Literal["one-hot", "embed"] = "one-hot", 
                             cat_features: Union[list[str], None] = None, **kwargs) -> 'DatasetMaker':
        """
        Encodes categorical features using the specified method.

        Args:
            method (str, optional): 'one-hot' (default) or 'embed'.
            cat_features (list, optional): A list of categorical column names. 
                If None, they will be inferred from the DataFrame's dtypes.
            **kwargs: Additional keyword arguments to pass to the underlying
                pandas.get_dummies() or torch.nn.Embedding() functions.
                For 'one-hot' encoding, it is often recommended to add
                `drop_first=True` to help reduce multicollinearity.
        """
        if self._is_split:
            raise RuntimeError("Categoricals must be processed before splitting data to avoid data leakage.")
        
        if cat_features is None:
            cat_columns = self.features.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        else:
            cat_columns = cat_features

        if not cat_columns:
            _LOGGER.info("No categorical features to process.")
            self._is_categoricals_processed = True
            return self

        continuous_df = self.features.drop(columns=cat_columns)
        # store continuous column names
        self.continuous_columns = continuous_df.columns.tolist()
        
        categorical_df = self.features[cat_columns].copy()

        if method == "one-hot":
            processed_cats = pandas.get_dummies(categorical_df, dtype=numpy.int32, **kwargs)
        elif method == "embed":
            processed_cats = self._embed_categorical(categorical_df, **kwargs)
        else:
            raise ValueError("`method` must be 'one-hot' or 'embed'.")

        self.features = pandas.concat([continuous_df, processed_cats], axis=1)
        self._is_categoricals_processed = True
        _LOGGER.info("Categorical features processed.")
        return self

    def normalize_continuous(self, method: Literal["standard", "minmax"] = "standard") -> 'DatasetMaker':
        """Normalizes all numeric features and saves the scaler."""
        if not self._is_split:
            raise RuntimeError("Continuous features must be normalized AFTER splitting data. Call .split_data() first.")
        if self._is_normalized:
            _LOGGER.warning("⚠️ Data has already been normalized.")
            return self

        # Use continuous features columns
        self.scaler_columns = self.continuous_columns
        if not self.scaler_columns:
            _LOGGER.info("No continuous features to normalize.")
            self._is_normalized = True
            return self

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Normalization `method` must be 'standard' or 'minmax'.")

        # Fit on training data only, then transform both
        self.features_train[self.scaler_columns] = self.scaler.fit_transform(self.features_train[self.scaler_columns]) # type: ignore
        self.features_test[self.scaler_columns] = self.scaler.transform(self.features_test[self.scaler_columns]) # type: ignore
        
        self._is_normalized = True
        _LOGGER.info(f"Continuous features normalized using {self.scaler.__class__.__name__}. Scaler stored in `self.scaler`.")
        return self

    def split_data(self, test_size: float = 0.2, stratify: bool = False, random_state: Optional[int] = None) -> 'DatasetMaker':
        """Splits the data into training and testing sets."""
        if self._is_split:
            _LOGGER.warning("⚠️ Data has already been split.")
            return self

        if self.labels.dtype == 'object' or self.labels.dtype.name == 'category':
            labels_numeric = self.labels.astype("category")
            self.labels_map = {code: val for code, val in enumerate(labels_numeric.cat.categories)}
            self.labels = pandas.Series(labels_numeric.cat.codes, index=self.labels.index)
            _LOGGER.info("Labels have been encoded. Mapping stored in `self.labels_map`.")

        stratify_array = self.labels if stratify else None
        
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state, stratify=stratify_array
        )
        
        self._is_split = True
        _LOGGER.info(f"Data split into training ({len(self.features_train)} samples) and testing ({len(self.features_test)} samples).")
        return self

    def balance_data(self, resampler=None, **kwargs) -> 'DatasetMaker':
        """
        Only useful for classification tasks.
        
        Balances the training data using a specified resampler. 
        
        Defaults to `SMOTETomek`.
        """
        if not self._is_split:
            raise RuntimeError("❌ Cannot balance data before it has been split. Call .split_data() first.")
        if self._is_balanced:
            _LOGGER.warning("⚠️ Training data has already been balanced.")
            return self

        if resampler is None:
            resampler = SMOTETomek(**kwargs)

        _LOGGER.info(f"Balancing training data with {resampler.__class__.__name__}...")
        self.features_train, self.labels_train = resampler.fit_resample(self.features_train, self.labels_train) # type: ignore
        
        self._is_balanced = True
        _LOGGER.info(f"Balancing complete. New training set size: {len(self.features_train)} samples.")
        return self

    def process(self, test_size: float = 0.2, cat_method: Literal["one-hot", "embed"] = "one-hot", normalize_method: Literal["standard", "minmax"] = "standard", 
                balance: bool = False, random_state: Optional[int] = None) -> 'DatasetMaker':
        """Runs a standard, fully automated preprocessing pipeline."""
        _LOGGER.info("--- 🤖 Running Automated Processing Pipeline ---")
        self.process_categoricals(method=cat_method)
        self.split_data(test_size=test_size, stratify=True, random_state=random_state)
        self.normalize_continuous(method=normalize_method)
        if balance:
            self.balance_data()
        _LOGGER.info("--- 🤖 Automated Processing Complete ---")
        return self
        
    def denormalize(self, data: Union[torch.Tensor, numpy.ndarray, pandas.DataFrame]) -> Union[numpy.ndarray, pandas.DataFrame]:
        """
        Applies inverse transformation to denormalize data, preserving DataFrame
        structure if provided.

        Args:
            data: The normalized data to be transformed back to its original scale.
                Can be a PyTorch Tensor, NumPy array, or Pandas DataFrame.
                If a DataFrame, it must contain the columns that were originally scaled.

        Returns:
            The denormalized data. Returns a Pandas DataFrame if the input was a
            DataFrame, otherwise returns a NumPy array.
        """
        if self.scaler is None:
            raise RuntimeError("Data was not normalized. Cannot denormalize.")

        if isinstance(data, pandas.DataFrame):
            # If input is a DataFrame, denormalize in place and return a copy
            if not all(col in data.columns for col in self.scaler_columns): # type: ignore
                raise ValueError(f"Input DataFrame is missing one or more required columns for denormalization. Required: {self.scaler_columns}")
            
            output_df = data.copy()
            output_df[self.scaler_columns] = self.scaler.inverse_transform(data[self.scaler_columns]) # type: ignore
            return output_df
        
        # Handle tensor or numpy array input
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else: # It's already a numpy array
            data_np = data

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        if data_np.shape[1] != len(self.scaler_columns): # type: ignore
            raise ValueError(f"Input array has {data_np.shape[1]} columns, but scaler was fitted on {len(self.scaler_columns)} columns.") # type: ignore

        return self.scaler.inverse_transform(data_np)

    def get_datasets(self) -> Tuple[_PytorchDataset, _PytorchDataset]:
        """Primary method to get the final PyTorch Datasets."""
        if not self._is_split:
            raise RuntimeError("Data has not been split yet. Call .split_data() or .process() first.")
        
        self._train_dataset = _PytorchDataset(self.features_train, self.labels_train) # type: ignore
        self._test_dataset = _PytorchDataset(self.features_test, self.labels_test) # type: ignore
        
        return self._train_dataset, self._test_dataset
    
    def inspect_dataframes(self) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]:
        """Utility method to inspect the processed data as Pandas DataFrames."""
        if not self._is_split:
             raise RuntimeError("Data has not been split yet. Call .split_data() or .process() first.")
        return self.features_train, self.features_test, self.labels_train, self.labels_test # type: ignore

    @staticmethod
    def _embed_categorical(cat_df: pandas.DataFrame, random_state: Optional[int] = None, **kwargs) -> pandas.DataFrame:
        """Internal helper to perform embedding on categorical features."""
        embedded_tensors = []
        new_columns = []
        for col in cat_df.columns:
            cat_series = cat_df[col].astype("category")
            num_categories = len(cat_series.cat.categories)
            embedding_dim = min(50, (num_categories + 1) // 2)
            
            if random_state:
                torch.manual_seed(random_state)
            
            embedder = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim, **kwargs)
            
            with torch.no_grad():
                codes = torch.LongTensor(cat_series.cat.codes.values)
                embedded_tensors.append(embedder(codes))
            
            new_columns.extend([f"{col}_{i+1}" for i in range(embedding_dim)])
        
        with torch.no_grad():
            full_tensor = torch.cat(embedded_tensors, dim=1)
        return pandas.DataFrame(full_tensor.numpy(), columns=new_columns, index=cat_df.index)


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

    def normalize_data(self, method: Literal["standard", "minmax"] = "minmax") -> 'SequenceMaker':
        """
        Normalizes the sequence data. Must be called AFTER splitting to prevent data leakage from the test set.
        """
        if not self._is_split:
            raise RuntimeError("Data must be split BEFORE normalizing. Call .split_data() first.")
        
        if self.scaler:
            _LOGGER.warning("⚠️ Data has already been normalized.")
            return self
            
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError("Normalization `method` must be 'standard' or 'minmax'.")

        # Fit scaler ONLY on the training data
        self.scaler.fit(self.train_sequence.reshape(-1, 1)) # type: ignore
        
        # Transform both train and test data using the fitted scaler
        self.train_sequence = self.scaler.transform(self.train_sequence.reshape(-1, 1)).flatten() # type: ignore
        self.test_sequence = self.scaler.transform(self.test_sequence.reshape(-1, 1)).flatten() # type: ignore
        
        self._is_normalized = True
        _LOGGER.info(f"Sequence data normalized using {self.scaler.__class__.__name__}. Scaler was fit on the training set only.")
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

    def _create_windowed_dataset(self, data: numpy.ndarray, use_sequence_labels: bool) -> _PytorchDataset:
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
        """Applies inverse transformation using the stored scaler."""
        if self.scaler is None:
            raise RuntimeError("Data was not normalized. Cannot denormalize.")
        
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().detach().numpy()
        else:
            data_np = data
            
        return self.scaler.inverse_transform(data_np.reshape(-1, 1)).flatten()

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

    def get_datasets(self) -> Tuple[_PytorchDataset, _PytorchDataset]:
        """Returns the final train and test datasets."""
        if not self._are_windows_generated:
            raise RuntimeError("Windows have not been generated. Call .generate_windows() first.")
        return self._train_dataset, self._test_dataset


def info():
    _script_info(__all__)
