import torch
import pandas
import numpy
from typing import Literal, Union, Optional

from ..ML_scaler import DragonScaler
from ..schema import FeatureSchema

from ..keys._keys import MLTaskKeys
from .._core import get_logger

from ._base_sequence_dataset import _BaseSequenceDataset, _PytorchSequenceDataset 


_LOGGER = get_logger("Sequence Dataset")


__all__ = [
    "DragonDatasetSequenceAutoregressive",
    "DragonDatasetSequenceExogenous"
]


class DragonDatasetSequenceAutoregressive(_BaseSequenceDataset):
    """
    Creates windowed PyTorch datasets for autoregressive sequence tasks.
    Targets are extracted directly from the feature columns, allowing columns 
    to serve simultaneously as historical inputs and future outputs.
    """
    def __init__(self, 
                pandas_df: pandas.DataFrame,
                targets: list[str],
                schema: FeatureSchema,
                prediction_mode: Literal["autoregressive-sequence-to-value",
                                        "autoregressive-sequence-to-sequence"],
                sequence_length: int,
                feature_scaler: Union[Literal["fit"], Literal["none"], DragonScaler] = "fit",
                validation_size: Optional[float] = 0.2,
                test_size: Optional[float] = 0.1,
                verbose: int = 2):
        """
        Initializes the DragonDatasetSequenceAutoregressive with chronological splitting, scaling, and windowing.
        
        Args:
            pandas_df (pandas.DataFrame): The input DataFrame containing features and targets.
            targets (list[str]): List of column names to be used as targets.
            schema (FeatureSchema): Schema defining the expected features and their types.
            prediction_mode (str): The prediction mode for the dataset. This determines how windows are created:
                - autoregressive tasks include the targets in the input windows for feedback.
                - sequence-to-sequence predicts a full sequence of outputs.
                - sequence-to-value predicts a single value for the final time step.
            sequence_length (int): The length of the input sequences (window size).
            feature_scaler (str | DragonScaler): "fit" to fit a new scaler, "none" for no scaling, or an existing DragonScaler instance.
            validation_size (float | None): Fraction of data to use for validation.
            test_size (float | None): Fraction of data to use for testing.
            verbose (int): Verbosity level for logging (0: silent, 1: warnings, 2: info, 3: detailed).
        """
        if prediction_mode not in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE]:
            _LOGGER.error(f"Invalid prediction_mode '{prediction_mode}' for autoregressive dataset.")
            raise ValueError()
        
        super().__init__(pandas_df=pandas_df,
                         targets=targets,
                         schema=schema,
                         prediction_mode=prediction_mode,
                         sequence_length=sequence_length,
                         feature_scaler=feature_scaler,
                         validation_size=validation_size,
                         test_size=test_size,
                         verbose=verbose)
    
    def _create_windowed_dataset(self, data: numpy.ndarray, verbose: int = 3) -> _PytorchSequenceDataset:
        if len(data) <= self.sequence_length:
            if verbose >= 1 and len(data) > 0:
                _LOGGER.warning(f"Data length ({len(data)}) not greater than sequence_length ({self.sequence_length}). Returning empty dataset.")
            
            num_features = data.shape[1]
            num_targets = len(self._target_indices)
            
            empty_features = numpy.empty((0, self.sequence_length, num_features), dtype=numpy.float32)
            
            if self.prediction_mode == MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE:
                empty_labels = numpy.empty((0, num_targets), dtype=numpy.float32)
            else: # AUTOREGRESSIVE_SEQUENCE_SEQUENCE
                empty_labels = numpy.empty((0, self.sequence_length, num_targets), dtype=numpy.float32)
                
            return _PytorchSequenceDataset(empty_features, empty_labels, 
                               labels_dtype=torch.float32,
                               feature_names=self._feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        n_windows = len(data) - self.sequence_length + 1
        row_stride, col_stride = data.strides
        
        strided_data = numpy.lib.stride_tricks.as_strided(
            data, 
            shape=(n_windows, self.sequence_length, data.shape[1]), 
            strides=(row_stride, row_stride, col_stride)
        )
        
        if self.prediction_mode == MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE:
            # Drop the last window for inputs because there is no "next step" to predict
            features = strided_data[:-1]
            # Targets are the vectors at sequence_length offset, selecting only target columns
            labels = data[self.sequence_length:, self._target_indices]
            
        else: # AUTOREGRESSIVE_SEQUENCE_SEQUENCE
            # Drop the last window for inputs
            features = strided_data[:-1]
            # Targets are the shifted windows, selecting only target columns
            labels = strided_data[1:, :, self._target_indices]
        
        subset_dataset = _PytorchSequenceDataset(features, labels,
                               labels_dtype=torch.float32,
                               feature_names=self._feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        subset_dataset._class_map = self.class_map
        
        return subset_dataset


class DragonDatasetSequenceExogenous(_BaseSequenceDataset):
    """
    Creates windowed PyTorch datasets for exogenous sequence tasks.
    Targets are explicitly excluded from the input feature windows, 
    meaning the model only relies on independent (exogenous) variables to predict the targets.
    """
    def __init__(self, 
                 pandas_df: pandas.DataFrame,
                 targets: list[str],
                 schema: FeatureSchema,
                 prediction_mode: Literal["exogenous-sequence-to-value", 
                                          "exogenous-sequence-to-sequence"],
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
            prediction_mode (str): The prediction mode for the dataset. This determines how windows are created:
                - exogenous tasks exclude the targets from training.
                - sequence-to-sequence predicts a full sequence of outputs.
                - sequence-to-value predicts a single value for the final time step.
            sequence_length (int): The length of the input sequences (window size).
            feature_scaler (str | DragonScaler): "fit" to fit a new scaler, "none" for no scaling, or an existing DragonScaler instance.
            validation_size (float | None): Fraction of data to use for validation.
            test_size (float | None): Fraction of data to use for testing.
            verbose (int): Verbosity level for logging (0: silent, 1: warnings, 2: info, 3: detailed).
        """
        if prediction_mode not in [MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE, MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE]:
            _LOGGER.error(f"Invalid prediction_mode '{prediction_mode}' for exogenous dataset.")
            raise ValueError()
        
        super().__init__(pandas_df=pandas_df,
                         targets=targets,
                         schema=schema,
                         prediction_mode=prediction_mode,
                         sequence_length=sequence_length,
                         feature_scaler=feature_scaler,
                         validation_size=validation_size,
                         test_size=test_size,
                         verbose=verbose)
    
    @property
    def feature_names(self) -> list[str]:
        """Return the exogenous feature names."""
        feature_indices = [i for i, name in enumerate(self._feature_names) if name not in self._target_names]
        return [self._feature_names[i] for i in feature_indices]
        
    @property
    def number_of_features(self) -> int:
        """Return the number of exogenous features."""
        return len(self.feature_names)

    def _create_windowed_dataset(self, data: numpy.ndarray, verbose: int = 3) -> _PytorchSequenceDataset:
        # Determine features to keep (exclude targets)
        feature_indices = [i for i in range(data.shape[1]) if i not in self._target_indices]
        exogenous_feature_names = self.feature_names
        
        if len(data) <= self.sequence_length:
            if verbose >= 1 and len(data) > 0:
                _LOGGER.warning(f"Data length ({len(data)}) not greater than sequence_length ({self.sequence_length}). Returning empty dataset.")
            
            num_features = len(feature_indices)
            num_targets = len(self._target_indices)
            
            empty_features = numpy.empty((0, self.sequence_length, num_features), dtype=numpy.float32)
            
            if self.prediction_mode == MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE:
                empty_labels = numpy.empty((0, num_targets), dtype=numpy.float32)
            else: # exogenous-sequence-to-sequence
                empty_labels = numpy.empty((0, self.sequence_length, num_targets), dtype=numpy.float32)
                
            return _PytorchSequenceDataset(empty_features, empty_labels, 
                               labels_dtype=torch.float32,
                               feature_names=exogenous_feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        n_windows = len(data) - self.sequence_length + 1
        row_stride, col_stride = data.strides
        
        strided_data = numpy.lib.stride_tricks.as_strided(
            data, 
            shape=(n_windows, self.sequence_length, data.shape[1]), 
            strides=(row_stride, row_stride, col_stride)
        )
        
        # Drop the last window for inputs and isolate ONLY exogenous features
        features = strided_data[:-1][..., feature_indices]
        
        if self.prediction_mode == MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE:
            labels = data[self.sequence_length:, self._target_indices]
        else: # exogenous-sequence-to-sequence
            labels = strided_data[1:, :, self._target_indices]
        
        subset_dataset = _PytorchSequenceDataset(features, labels,
                               labels_dtype=torch.float32,
                               feature_names=exogenous_feature_names,
                               target_names=self._target_names,
                               target_types=self._target_types)
        
        subset_dataset._class_map = self.class_map
        
        return subset_dataset
