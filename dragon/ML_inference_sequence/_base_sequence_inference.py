import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from ..schema import FeatureSchema
from ..ML_inference._base_inference import _CoreInferenceHandler, _ScalerMixin

from .._core import get_logger
from ..path_manager import make_fullpath, sanitize_filename
from ..keys._keys import MLTaskKeys, PyTorchCheckpointKeys, DatasetKeys


_LOGGER = get_logger("Sequence Inference")


__all__ = [
    "_BaseSequenceInferenceHandler"
]


class _BaseSequenceInferenceHandler(_CoreInferenceHandler, _ScalerMixin, ABC):
    """
    Abstract base class for PyTorch sequence inference handlers.
    
    Handles loading the model state, schema validation, automatic scaling, 
    and standard prediction methods. Subclasses must implement the specific 
    forecasting logic (e.g., autoregressive vs. exogenous).
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 schema: Union[str, Path, FeatureSchema],
                 scaler: Optional[Union[str, Path]],
                 target_types: Optional[dict[str, str]] = None,
                 task: Optional[str] = None,
                 device: str = 'cpu'):
        
        # 1. Initialize Universal Core
        _CoreInferenceHandler.__init__(self, model=model, state_dict=state_dict, device=device, task=task)
        
        # 2. Initialize Scaler Mixin and load scalers
        _ScalerMixin.__init__(self)

        # this safely returns None if no scaler is provided, or loads the scaler if a path is given
        self._load_scalers(scaler)
        
        # --- Load Schema ---
        if isinstance(schema, (str, Path)):
            self.schema = FeatureSchema.from_json(schema, verbose=False)
            _LOGGER.info("FeatureSchema loaded from file.")
        elif isinstance(schema, FeatureSchema):
            self.schema = schema
            _LOGGER.info("FeatureSchema object successfully bound to inference handler.")
        else:
            _LOGGER.error("The 'schema' argument must be a FeatureSchema object or a valid path.")
            raise TypeError()

        # --- Load Sequence Metadata ---
        seq_meta = self._file_handler.parse_sequence_metadata()

        # --- Load Target Types and Resolve Indices ---
        if target_types is not None and isinstance(target_types, dict):
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in target_types.items()):
                _LOGGER.error("All keys and values in 'target_types' must be strings.")
                raise TypeError()
            elif not all(_val in [DatasetKeys.TARGET_CATEGORICAL, DatasetKeys.TARGET_CONTINUOUS] for _val in target_types.values()):
                _LOGGER.error(f"All values in 'target_types' must be either '{DatasetKeys.TARGET_CATEGORICAL}' or '{DatasetKeys.TARGET_CONTINUOUS}'.")
                raise TypeError()
            self.target_types = target_types

            _LOGGER.info(f"Target types provided directly: {self.target_types}")
        else:
            if seq_meta.target_types is not None:
                self.target_types = seq_meta.target_types
                _LOGGER.info(f"Target types loaded from model file: {self.target_types}")
            else:
                _LOGGER.error("Target types not found in the FinalizedFile bundle. This is required for sequence reconstruction.")
                raise ValueError()
            
        # all targets must be present in the schema
        missing_targets = [t for t in self.target_types.keys() if t not in self.schema.feature_names]
        if missing_targets:
            _LOGGER.error(f"The following target(s) are not present in the provided FeatureSchema: {missing_targets}")
            raise ValueError()
        
        self.target_names = list(self.target_types.keys())
        
        # Dynamically resolve indices using the definitive FeatureSchema
        self.target_indices = [self.schema.feature_names.index(t) for t in self.target_names]
        
        ### Validate scaler
        # If all features are categorical, feature_scaler can be None. If any features are continuous, feature_scaler must be provided.
        _true_features = {f for f in self.schema.feature_names if f not in self.target_names}
        _has_continuous_features = any(_true_feature in self.schema.continuous_feature_names for _true_feature in _true_features)
        if _has_continuous_features and self.feature_scaler is None:
            _LOGGER.error("A feature scaler is required for continuous features, but none was provided or loaded.")
            raise ValueError()
        
        # If all targets are categorical, target_scaler can be None. If any targets are continuous, target_scaler must be provided.
        _has_continuous_targets = any(t in self.schema.continuous_feature_names for t in self.target_names)
        if _has_continuous_targets and self.target_scaler is None:
            _LOGGER.error("A target scaler is required for continuous targets, but none was provided or loaded.")
            raise ValueError()
        
        ### Precompute exogenous features for efficiency
        self._num_features = self.schema.number_of_features
        self._exog_indices = [i for i in range(self._num_features) if i not in self.target_indices]
        self._exog_names = [self.schema.feature_names[i] for i in self._exog_indices]
        self._num_exog = len(self._exog_indices)
        
        if self.feature_scaler is None and self.target_scaler is None:
            _LOGGER.error("A scaler is required to scale inputs and de-scale predictions.")
            raise ValueError()
        
        # --- Load Sequence Dimensions ---
        self.sequence_length = seq_meta.sequence_length
        self.initial_sequence = seq_meta.initial_sequence
        
        if self.sequence_length is not None:
            _LOGGER.info(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' found and set to {self.sequence_length}")
            
        if self.initial_sequence is not None:
            _LOGGER.info(f"'{PyTorchCheckpointKeys.INITIAL_SEQUENCE}' for forecasting loaded from model file.")

    @property
    def _is_seq_to_val(self) -> bool:
        """Helper property to abstract shape checks for sequence-to-value tasks."""
        return self.task in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE, MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE]

    @property
    def _is_seq_to_seq(self) -> bool:
        """Helper property to abstract shape checks for sequence-to-sequence tasks."""
        return self.task in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE]

    def _preprocess_input(self, features: torch.Tensor) -> torch.Tensor:
        """
        Converts input sequence to a torch.Tensor, applies FEATURE scaling, 
        and moves it to the correct device.
        """
        features_tensor = features.float()
        
        if self.feature_scaler:
            batch_size, seq_len, num_features = features_tensor.shape
            features_flat = features_tensor.reshape(-1, num_features)
            
            scaled_flat = self.feature_scaler.transform(features_flat)
            
            scaled_features = scaled_flat.reshape(batch_size, seq_len, num_features)
        else:
            scaled_features = features_tensor

        return scaled_features.to(self.device)

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Core batch prediction method for sequences."""
        if features.ndim != 3:
            _LOGGER.error("Input for batch prediction must be a 3D array/tensor (batch_size, sequence_length, num_features).")
            raise ValueError()
        
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        input_tensor = self._preprocess_input(features_tensor) 

        with torch.no_grad():
            output_dict = self.model(input_tensor)

        combined_preds = []
        for target in self.target_names:
            pred = output_dict[target]
            
            if self.target_types.get(target) == DatasetKeys.TARGET_CATEGORICAL:
                pred = torch.argmax(pred, dim=-1).float()
                
            combined_preds.append(pred)
            
        combined_tensor = torch.stack(combined_preds, dim=-1)

        scaler_to_use = self.target_scaler if self.target_scaler else None
        
        if scaler_to_use:
            if self._is_seq_to_val:
                descaled_output = scaler_to_use.inverse_transform(combined_tensor)
                
            elif self._is_seq_to_seq:
                batch_size, seq_len, num_targets = combined_tensor.shape
                output_flat = combined_tensor.reshape(-1, num_targets)
                descaled_flat = scaler_to_use.inverse_transform(output_flat)
                descaled_output = descaled_flat.reshape(batch_size, seq_len, num_targets)
            else:
                 _LOGGER.error(f"Invalid prediction mode: {self.task}")
                 raise RuntimeError()
        else:
            descaled_output = combined_tensor

        final_predictions = {}
        for i, target in enumerate(self.target_names):
            final_predictions[target] = descaled_output[..., i]

        return final_predictions

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Core single-sample prediction method for sequences.
        
        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of input features, shape (sequence_length, num_features); or a 3D array/tensor of shape (1, sequence_length, num_features).
        
        Returns:
            A dictionary containing the de-scaled prediction as a torch.Tensor.
        """
        if features.ndim == 2:
            if isinstance(features, torch.Tensor):
                features = features.unsqueeze(0)
            else:
                features = np.expand_dims(features, axis=0)
        
        if features.shape[0] != 1 or features.ndim != 3:
            _LOGGER.error("predict() is for a single sequence (2D). Use predict_batch() for multiple (3D).")
            raise ValueError()

        batch_results = self.predict_batch(features)
        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results
    
    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        
        Args:
            features (np.ndarray | torch.Tensor): A 3D array/tensor of 
                input sequences, shape (batch_size, sequence_length, num_features).
        
        Returns:
            A dictionary containing the de-scaled prediction as a NumPy array.
        """
        tensor_results = self.predict_batch(features)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        Convenience wrapper for predict that returns NumPy arrays.
        
        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of 
                input features, shape (sequence_length, num_features); or a 3D array/tensor of shape (1, sequence_length, num_features).
        
        Returns:
            A dictionary containing the de-scaled prediction as a NumPy array
        """
        tensor_results = self.predict(features)
        return {key: value.cpu().numpy() for key, value in tensor_results.items()}

    def decode_predictions(self, predictions: Union[dict[str, torch.Tensor], dict[str, np.ndarray], pd.DataFrame]) -> pd.DataFrame:
        """
        Converts numerical predictions into a pandas DataFrame and remaps categorical 
        integer indices back to their original string labels.
        
        Args:
            predictions: The output from predict(), predict_numpy(), predict_batch(), predict_batch_numpy(), or forecast().
        
        Returns:
            pd.DataFrame: A formatted DataFrame with strings for categorical targets and floats for continuous targets.
        """
        if isinstance(predictions, pd.DataFrame):
            df = predictions.copy()
        elif isinstance(predictions, dict):
            df_dict = {}
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                
                if val.ndim > 1:
                     val = list(val) 
                
                df_dict[key] = val
            df = pd.DataFrame(df_dict)
        else:
            _LOGGER.error("Input must be a dictionary of Tensors/NumPy arrays or a pandas DataFrame.")
            raise TypeError()
        
        missing_targets = [t for t in self.target_names if t not in df.columns]
        if missing_targets:
            _LOGGER.warning(f"The following target(s) are missing from the predictions and will be skipped: {missing_targets}")
            
        unknown_cols = [col for col in df.columns if col not in self.target_names]
        if unknown_cols:
            _LOGGER.warning(f"The following columns in the predictions are not recognized as targets and will be ignored: {unknown_cols}")

        # remap categorical targets
        if self.schema.categorical_mappings:
            for target in self.schema.categorical_feature_names:
                if target in df.columns and target in self.schema.categorical_mappings:
                    
                    forward_map = self.schema.categorical_mappings[target]
                    reverse_map = {v: k for k, v in forward_map.items()}
                    
                    def remap_value(x):
                        if isinstance(x, (list, np.ndarray)):
                            return [reverse_map.get(int(i), int(i)) for i in x]
                        if pd.notna(x):
                            return reverse_map.get(int(x), int(x))
                        return x
                        
                    df[target] = df[target].apply(remap_value)
                    
        return df

    @abstractmethod
    def forecast(self, *args, **kwargs) -> pd.DataFrame:
        """Abstract method for autoregressive or exogenous forecasting loops."""
        pass

    def plot_forecast(self,  
                      n_steps: int, 
                      save_dir: Union[str, Path], 
                      future_exogenous: Optional[pd.DataFrame],
                      filename: str = "forecast_plot.svg",
                      initial_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Runs a forecast and saves a visualization of the results.
        
        This method automatically handles plotting single or multiple target variables 
        on the same graph. It clearly distinguishes between the historical input sequence 
        and the forecasted future steps, separating them with a vertical boundary line. 
        The output is saved as an SVG file to ensure high-quality vector scalability.
        
        Args:
            n_steps (int): The number of future time steps to predict.
            save_dir (Union[str, Path]): The directory where the plot file will be saved.
                Parent directories will be created automatically if they do not exist.
            future_exogenous (pd.DataFrame | None): A pandas DataFrame containing 
                unscaled future values for any exogenous features required by the model. 
                Must contain at least `n_steps` rows.
            filename (str, optional): The name of the output plot file. If the provided 
                name does not end with the '.svg' extension, it will be automatically 
                enforced.
            initial_sequence (Union[np.ndarray, torch.Tensor], optional): The unscaled 
                historical sequence used to prime the model. If None, it will attempt 
                to use the default `initial_sequence` loaded from the model bundle.
        """
        forecast_df = self.forecast(n_steps=n_steps, 
                                    future_exogenous=future_exogenous,
                                    initial_sequence=initial_sequence)
        
        if initial_sequence is None:
            plot_initial_sequence = self.initial_sequence
            if plot_initial_sequence is None:
                 _LOGGER.error("Cannot plot: No 'initial_sequence' provided and no default found.")
                 return
        elif isinstance(initial_sequence, torch.Tensor):
            plot_initial_sequence = initial_sequence.cpu().numpy()
        else: 
            plot_initial_sequence = initial_sequence
        
        history_targets = plot_initial_sequence[:, self.target_indices]
            
        seq_len = len(plot_initial_sequence)
        history_x = np.arange(0, seq_len)
        forecast_x = np.arange(seq_len, seq_len + n_steps)

        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(12, 6))

        color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for i, target_name in enumerate(self.target_names):
            color = next(color_cycle)

            plt.plot(
                history_x,
                history_targets[:, i],
                label=f"History: {target_name}",
                color=color,
            )

            plt.plot(
                forecast_x,
                forecast_df[target_name],
                label=f"Forecast: {target_name}",
                linestyle="--",
                color=color,
            )

        plt.axvline(x=history_x[-1], color='red', linestyle=':', label='Forecast Start')

        plt.title(f"{n_steps}-Step Forecast")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if not filename.lower().endswith(".svg"):
            filepath = Path(filename).stem + ".svg"
        else:
            filepath = filename
        
        dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = dir_path / sanitize_filename(filepath)
        
        try:
            plt.savefig(full_path, bbox_inches='tight')
            _LOGGER.info(f"📈 Forecast plot saved to '{full_path.name}'.")
        except Exception as e:
            _LOGGER.error(f"Failed to save plot:\n{e}")
        finally:
            plt.close()

    def __repr__(self) -> str:
        has_feature_scaler = self.feature_scaler is not None
        has_target_scaler = self.target_scaler is not None
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  task='{self.task}',\n"
            f"  device={self.device},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  targets={self.target_names},\n"
            f"  target_types={list(self.target_types.values())},\n"
            f"  num_features={self._num_features},\n"
            f"  num_exogenous={self._num_exog},\n"
            f"  feature_scaler={has_feature_scaler},\n"
            f"  target_scaler={has_target_scaler}\n"
            f")"
        )
