import torch
from torch.utils.data import Dataset
import pandas
import numpy
from typing import Literal, Union
import matplotlib.pyplot as plt
from pathlib import Path

from ..ML_scaler import DragonScaler

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import DatasetKeys, MLTaskKeys, SequenceDatasetKeys, ScalerKeys

from ._base_datasetmaster import _PytorchDataset


_LOGGER = get_logger("DragonSequenceDataset")


__all__ = [
    "DragonDatasetSequence"
]


# --- SequenceMaker ---
class DragonDatasetSequence:
    """
    Creates windowed PyTorch datasets from a univariate (one feature) sequential data.
    
    Automatic Pipeline:
    
    1. Split Data: Separate data into training, validation, and testing portions.
    2. Normalize Data: Normalize the data. The scaler will be fitted on the training portion.
    3. Generate Windows: Create the windowed sequences from the normalized splits.
    """
    def __init__(self, 
                 prediction_mode: Literal["sequence-to-sequence", "sequence-to-value"],
                 data: Union[pandas.DataFrame, pandas.Series, numpy.ndarray], 
                 sequence_length: int,
                 validation_size: float = 0.2,
                 test_size: float = 0.1,
                 verbose: int = 2):
        """
        Initializes the dataset manager and automatically processes the data.
        
        The constructor runs the full pipeline:
        1. Splits the data chronologically (train, validation, test).
        2. Fits a DragonScaler on the training split.
        3. Normalizes all splits using the fitted scaler.
        4. Generates windowed datasets for training, validation, and testing.

        Args:
            prediction_mode: The type of sequence task.
            data: The input univariate time-series data.
                - If pandas.DataFrame: The index is used for the time axis
                  and the *first column* is used as the sequence.
                - If pandas.Series: The index is used for the time axis.
                - If numpy.ndarray: A simple integer range is used for the time axis.
            sequence_length (int): The number of time steps in each input window (X).
            validation_size (float): The fraction of data to hold out for validation.
            test_size (float): The fraction of data to hold out for testing.
            verbose (int): Verbosity level for logging.
                - 0: Errors only
                - 1: Warnings
                - 2: Info
                - 3: Detailed process info
        """
        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None
        self.sequence_length = sequence_length
        self.scaler = None
        
        if not prediction_mode in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            _LOGGER.error(f"Unrecognized prediction mode: '{prediction_mode}'.")
            raise ValueError()
        else:
            self.prediction_mode = prediction_mode
        
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
        self.val_sequence = None
        self.test_sequence = None
        
        self.train_time_axis = None
        self.val_time_axis = None 
        self.test_time_axis = None 
        
        self._is_split = False
        self._is_normalized = False
        self._are_windows_generated = False
        
        # Automation
        self._split_data(validation_size=validation_size, test_size=test_size, verbose=verbose)
        self._normalize_data(verbose=verbose)
        self._generate_windows(verbose=verbose)
    
    def _split_data(self, validation_size: float = 0.2, test_size: float = 0.1, verbose: int = 3) -> None:
        """
        Splits the sequence chronologically into training, validation, and testing portions.
        
        To prevent windowing errors, the validation and test sets include an overlap of `sequence_length` from the preceding data.
        """
        if self._is_split:
            if verbose >= 1:
                _LOGGER.warning("Data has already been split.")
            return
            
        if (validation_size + test_size) >= 1.0:
            _LOGGER.error(f"The sum of validation_size ({validation_size}) and test_size ({test_size}) must be less than 1.0.")
            raise ValueError()

        total_size = len(self.sequence)
        
        # Calculate split indices
        test_split_idx = int(total_size * (1 - test_size))
        val_split_idx = int(total_size * (1 - test_size - validation_size))
        
        # --- Create sequences ---
        # Train sequence is from the beginning to the validation index
        self.train_sequence = self.sequence[:val_split_idx]
        
        # Validation sequence starts `sequence_length` before its split index for windowing
        self.val_sequence = self.sequence[val_split_idx - self.sequence_length : test_split_idx]
        
        # Test sequence starts `sequence_length` before its split index for windowing
        self.test_sequence = self.sequence[test_split_idx - self.sequence_length:]
        
        # --- Create time axes ---
        self.train_time_axis = self.time_axis[:val_split_idx]
        # The "plottable" validation/test time axes start from their respective split indices
        self.val_time_axis = self.time_axis[val_split_idx : test_split_idx]
        self.test_time_axis = self.time_axis[test_split_idx:]

        self._is_split = True
        if verbose >= 2:
            _LOGGER.info(f"Sequence split into training ({len(self.train_sequence)}), validation ({len(self.val_sequence)}), and testing ({len(self.test_sequence)}) points.")
    
    def _normalize_data(self, verbose: int = 3) -> None:
        """
        Normalizes the sequence data using DragonScaler. Must be called AFTER splitting to prevent data leakage from the test set.
        """
        if not self._is_split:
            _LOGGER.error("Data must be split BEFORE normalizing.")
            raise RuntimeError()

        if self.scaler:
            if verbose >= 1:
                _LOGGER.warning("Data has already been normalized.")
            return

        # 1. DragonScaler requires a Dataset to fit. Create a temporary one.
        # The scaler expects 2D data [n_samples, n_features].
        train_features = self.train_sequence.reshape(-1, 1) # type: ignore

        # _PytorchDataset needs labels, so we create dummy ones.
        dummy_labels = numpy.zeros(len(train_features))
        temp_train_ds = _PytorchDataset(train_features, dummy_labels, labels_dtype=torch.float32)

        # 2. Fit the DragonScaler on the temporary training dataset.
        # The sequence is a single feature, so its index is [0].
        if verbose >= 3:
            _LOGGER.info("Fitting DragonScaler on the training data...")
        self.scaler = DragonScaler.fit(temp_train_ds, continuous_feature_indices=[0], verbose=verbose)

        # 3. Transform sequences using the fitted scaler.
        # The transform method requires a tensor, so we convert, transform, and convert back.
        train_tensor = torch.tensor(self.train_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore
        val_tensor = torch.tensor(self.val_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore
        test_tensor = torch.tensor(self.test_sequence.reshape(-1, 1), dtype=torch.float32) # type: ignore

        self.train_sequence = self.scaler.transform(train_tensor).numpy().flatten()
        self.val_sequence = self.scaler.transform(val_tensor).numpy().flatten()
        self.test_sequence = self.scaler.transform(test_tensor).numpy().flatten()

        self._is_normalized = True
        if verbose >= 2:
            _LOGGER.info("Sequence data normalized using DragonScaler.")

    def _generate_windows(self, verbose: int = 3) -> None:
        """
        Generates overlapping windows for features and labels.
        """
        if not self._is_split:
            _LOGGER.error("Cannot generate windows before splitting data.")
            raise RuntimeError()
        
        if not self._is_normalized:
            _LOGGER.error("Cannot generate windows before normalizing data.")
            raise RuntimeError()
        
        if self._are_windows_generated:
            if verbose >= 1:
                _LOGGER.warning("Windows have already been generated.")
            return

        self._train_dataset = self._create_windowed_dataset(self.train_sequence, verbose=verbose) # type: ignore
        self._val_dataset = self._create_windowed_dataset(self.val_sequence, verbose=verbose) # type: ignore
        self._test_dataset = self._create_windowed_dataset(self.test_sequence, verbose=verbose) # type: ignore
        
        # attach feature scaler and target scaler to datasets
        if self.scaler is not None:
            for ds in [self._train_dataset, self._val_dataset, self._test_dataset]:
                if ds is not None:
                    ds._feature_scaler = self.scaler # type: ignore
                    ds._target_scaler = self.scaler # type: ignore
        
        self._are_windows_generated = True
        if verbose >= 2:
            _LOGGER.info("Feature and label windows generated for train, validation, and test sets.")

    def _create_windowed_dataset(self, data: numpy.ndarray, verbose: int = 3) -> Dataset:
        """Efficiently creates windowed features and labels using numpy."""
        if len(data) <= self.sequence_length:
            # Validation/Test sets of size 0 might be passed
            if verbose >= 1:
                _LOGGER.warning(f"Data length ({len(data)}) is not greater than sequence_length ({self.sequence_length}). Cannot create windows. Returning empty dataset.")
            return _PytorchDataset(numpy.array([]), numpy.array([]), labels_dtype=torch.float32)
        
        # Define a generic name for the univariate feature
        f_names = [SequenceDatasetKeys.FEATURE_NAME]
        t_names = [SequenceDatasetKeys.TARGET_NAME]
        
        if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
            # sequence-to-value
            features = data[:-1]
            labels = data[self.sequence_length:]
            
            n_windows = len(features) - self.sequence_length + 1
            bytes_per_item = features.strides[0]
            strided_features = numpy.lib.stride_tricks.as_strided(
                features, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item)
            )
            # Ensure labels align with the end of each feature window
            aligned_labels = labels[:n_windows]
            return _PytorchDataset(strided_features, aligned_labels, 
                                   labels_dtype=torch.float32,
                                   feature_names=f_names,
                                   target_names=t_names)
        
        else:
            # Sequence-to-sequence
            x_data = data[:-1]
            y_data = data[1:]
            
            n_windows = len(x_data) - self.sequence_length + 1
            bytes_per_item = x_data.strides[0]
            
            strided_x = numpy.lib.stride_tricks.as_strided(x_data, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item))
            strided_y = numpy.lib.stride_tricks.as_strided(y_data, shape=(n_windows, self.sequence_length), strides=(bytes_per_item, bytes_per_item))
            
            return _PytorchDataset(strided_x, strided_y, 
                                   labels_dtype=torch.float32,
                                   feature_names=f_names,
                                   target_names=t_names)

    def plot_splits(self, save_dir: Union[str, Path], verbose: int = 3) -> None:
        """Plots the training, validation and testing data."""
        if not self._is_split:
            _LOGGER.error("Cannot plot before splitting data.")
            raise RuntimeError()
        
        if self.scaler is None:
            _LOGGER.error("Cannot plot: data has not been normalized, or scaler is missing.")
            return
        
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = save_path / "SequenceSplits.svg"

        plt.figure(figsize=(15, 6))
        plt.title("Sequential Data")
        plt.grid(True)
        plt.xlabel("Sequence")
        plt.ylabel("Value")
        
        # Plot denormalized training data
        plt.plot(self.train_time_axis, self.scaler.inverse_transform(self.train_sequence.reshape(-1, 1)), label='Train Data') # type: ignore
        
        # Plot denormalized validation data
        # We must skip the overlapping 'sequence_length' part for plotting
        val_plot_data = self.val_sequence[self.sequence_length:] # type: ignore
        plt.plot(self.val_time_axis, self.scaler.inverse_transform(val_plot_data.reshape(-1, 1)), label='Validation Data', c='orange') # type: ignore

        # Plot denormalized test data
        # We must skip the overlapping 'sequence_length' part for plotting
        test_plot_data = self.test_sequence[self.sequence_length:] # type: ignore
        plt.plot(self.test_time_axis, self.scaler.inverse_transform(test_plot_data.reshape(-1, 1)), label='Test Data', c='green') # type: ignore

        plt.legend()
        
        plt.tight_layout()
        plt.savefig(full_path)
        if verbose >= 2:
            _LOGGER.info(f"📈 Sequence data splits saved as '{full_path.name}'.")
        plt.close()

    def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        """Returns the final train, validation, and test datasets."""
        if not self._are_windows_generated:
            _LOGGER.error("Windows have not been generated. Call .generate_windows() first.")
            raise RuntimeError()
        return self._train_dataset, self._val_dataset, self._test_dataset # type: ignore
    
    def save_scaler(self, directory: Union[str, Path], verbose: bool=True) -> None:
        """
        Saves the fitted DragonScaler's state to a .pth file using the Unified
        dictionary format.
        
        Since this is univariate data, features and targets share the same
        scaling statistics.
        
        Args:
            directory (str | Path): The directory where the scaler will be saved.
        """
        if not self.scaler: 
            _LOGGER.error("No scaler was fitted or provided.")
            raise RuntimeError()

        save_path = make_fullpath(directory, make=True, enforce="directory")
        filename = f"{DatasetKeys.SCALER_PREFIX}{self.prediction_mode}.pth"
        filepath = save_path / filename

        # Unified Scaler Dictionary Format
        # For univariate sequences, features and targets share the same scaling statistics.
        scaler_state = self.scaler._get_state()
        combined_state = {
            ScalerKeys.FEATURE_SCALER: scaler_state,
            ScalerKeys.TARGET_SCALER: scaler_state
        }

        torch.save(combined_state, filepath)
        
        if verbose:
            _LOGGER.info(f"Unified Scaler saved as '{filepath.name}'.")
    
    def get_last_training_sequence(self) -> numpy.ndarray:
        """
        Returns the final, un-scaled sequence from the training data.
        """
        if not self._is_split:
            _LOGGER.error("Data has not been split. Cannot get last training sequence.")
            raise RuntimeError()
        
        # The length of train_time_axis is our validation split index
        val_split_idx = len(self.train_time_axis) # type: ignore
        
        if val_split_idx < self.sequence_length:
            _LOGGER.error(f"Training data length ({val_split_idx}) is less than sequence_length ({self.sequence_length}).")
            raise ValueError()

        # Get the slice from the *original* sequence
        start_idx = val_split_idx - self.sequence_length
        end_idx = val_split_idx
        
        return self.sequence[start_idx:end_idx] # type: ignore
    
    @property
    def feature_names(self):
        return [SequenceDatasetKeys.FEATURE_NAME]
    
    @property
    def target_names(self):
        return [SequenceDatasetKeys.TARGET_NAME]
    
    @property
    def train_dataset(self) -> Dataset:
        if self._train_dataset is None: 
            _LOGGER.error("Train Dataset not created.")
            raise RuntimeError()
        return self._train_dataset
    
    @property
    def validation_dataset(self) -> Dataset:
        if self._val_dataset is None: 
            _LOGGER.error("Validation Dataset not yet created.")
            raise RuntimeError()
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        if self._test_dataset is None: 
            _LOGGER.error("Test Dataset not yet created.")
            raise RuntimeError()
        return self._test_dataset

    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}>:\n"
        s += f"  Prediction Mode: {self.prediction_mode}\n"
        s += f"  Sequence Length (Window): {self.sequence_length}\n"
        s += f"  Total Data Points: {len(self.sequence)}\n"
        s += "  --- Status ---\n"
        s += f"  Split: {self._is_split}\n"
        s += f"  Normalized: {self._is_normalized}\n"
        s += f"  Windows Generated: {self._are_windows_generated}\n"
        
        if self._are_windows_generated:
            train_len = len(self._train_dataset) if self._train_dataset else 0 # type: ignore
            val_len = len(self._val_dataset) if self._val_dataset else 0 # type: ignore
            test_len = len(self._test_dataset) if self._test_dataset else 0 # type: ignore
            s += f"  Datasets (Train | Validation | Test): {train_len} | {val_len} | {test_len} windows\n"
            
        return s

