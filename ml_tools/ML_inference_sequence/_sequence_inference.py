import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from .._core import get_logger
from ..path_manager import make_fullpath, sanitize_filename
from ..keys._keys import MLTaskKeys, PyTorchCheckpointKeys, ScalerKeys
from ..ML_scaler import DragonScaler
from ..schema import FeatureSchema

from ..ML_inference._base_inference import _CoreInferenceHandler


_LOGGER = get_logger("Sequence Inference")


__all__ = [
    "DragonSequenceInferenceHandler"
]


class DragonSequenceInferenceHandler(_CoreInferenceHandler):
    """
    Handles loading a PyTorch sequence model's state and performing inference
    for univariate and multivariate sequence tasks.
    
    This handler automatically scales inputs and de-scales outputs.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 scaler: Union[str, Path],
                 schema: Union[str, Path, FeatureSchema],
                 target_names: Optional[list[str]] = None,
                 task: Optional[Literal["sequence-to-sequence", "sequence-to-value"]]=None,
                 device: str = 'cpu'):
        """
        Initializes the handler for sequence tasks.

        Args:
            model (nn.Module): A trained model architecture that must output a dictionary mapping target names to their respective prediction tensors.
            state_dict (str | Path): Path to the saved .pth model state_dict file or a FinalizedFile format.
            scaler (str | Path): File path to a saved DragonScaler state. This is required to correctly scale inputs and de-scale predictions.
            schema (str | Path | FeatureSchema): The feature schema or a path to a directory containing the schema.
            target_names (list[str], optional): A list of target variable names. If None, will attempt to load from the FinalizedFile.
            task (str, optional): The type of sequence task. If None, detected from file if a FinalizedFile is provided.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps'). 
        """
        # 1. Initialize Universal Core
        super().__init__(model=model, state_dict=state_dict, device=device, task=task)
        
        self.sequence_length: Optional[int] = None
        self.initial_sequence: Optional[np.ndarray] = None
        
        valid_tasks = [
            MLTaskKeys.SEQUENCE_SEQUENCE, 
            MLTaskKeys.SEQUENCE_VALUE
        ]

        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
            raise ValueError()
        
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

        # --- Load Target Names and Resolve Indices ---
        if target_names is not None:
            self.target_names = target_names
            _LOGGER.info(f"Target names provided directly: {self.target_names}")
        elif getattr(self._file_handler, PyTorchCheckpointKeys.TARGET_NAMES, None) is not None:
            self.target_names = self._file_handler.target_names
            _LOGGER.info(f"Target names loaded from model file: {self.target_names}")
        else:
            _LOGGER.error("Target names not found in the FinalizedFile bundle. This is required for sequence reconstruction.")
            raise ValueError()
        
        try:
            # Dynamically resolve indices using the definitive FeatureSchema
            self.target_indices = [self.schema.feature_names.index(t) for t in self.target_names]
        except ValueError as e:
            _LOGGER.error(f"A target name from the model bundle was not found in the FeatureSchema.")
            raise ValueError() from e
        
        # --- Load Scalers ---
        self.feature_scaler: Optional[DragonScaler] = None
        self.target_scaler: Optional[DragonScaler] = None

        if scaler is not None:
            if isinstance(scaler, (str, Path)):
                path_obj = make_fullpath(scaler, enforce="file")
                loaded_scaler_data = torch.load(path_obj)
                
                if isinstance(loaded_scaler_data, dict) and (ScalerKeys.FEATURE_SCALER in loaded_scaler_data or ScalerKeys.TARGET_SCALER in loaded_scaler_data):
                    if ScalerKeys.FEATURE_SCALER in loaded_scaler_data:
                        self.feature_scaler = DragonScaler.load(loaded_scaler_data[ScalerKeys.FEATURE_SCALER], verbose=False)
                        _LOGGER.info("Loaded DragonScaler state for feature scaling.")
                    if ScalerKeys.TARGET_SCALER in loaded_scaler_data:
                        self.target_scaler = DragonScaler.load(loaded_scaler_data[ScalerKeys.TARGET_SCALER], verbose=False)
                        _LOGGER.info("Loaded DragonScaler state for target scaling.")
                else:
                    _LOGGER.warning("Loaded scaler file does not contain separate feature/target scalers. Assuming it is a feature scaler (legacy format).")
                    self.feature_scaler = DragonScaler.load(loaded_scaler_data)
            else:
                _LOGGER.error("Scaler must be a file path (str or Path) to a saved DragonScaler state file.")
                raise ValueError()
        
        if self.feature_scaler is None and self.target_scaler is None:
            _LOGGER.error("A scaler is required to scale inputs and de-scale predictions.")
            raise ValueError()
        
        # Load sequence length from the FinalizedFileHandler
        if self._file_handler.sequence_length is not None:
            self.sequence_length = self._file_handler.sequence_length
            _LOGGER.info(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' found and set to {self.sequence_length}")
        else:
            _LOGGER.warning(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' not found in model file. Forecasting validation will be skipped.")
            
        # Load initial sequence from FinalizedFileHandler
        if self._file_handler.initial_sequence is not None:
            self.initial_sequence = self._file_handler.initial_sequence
            _LOGGER.info(f"Default 'initial_sequence' for forecasting loaded from model file.")
            if self.sequence_length and len(self.initial_sequence) != self.sequence_length: # type: ignore
                _LOGGER.warning(f"Loaded 'initial_sequence' length ({len(self.initial_sequence)}) mismatches 'sequence_length' ({self.sequence_length}).") # type: ignore
        else:
            _LOGGER.info("No default 'initial_sequence' found in model file. Must be provided for forecasting.")

    def _preprocess_input(self, features: torch.Tensor) -> torch.Tensor:
        """
        Converts input sequence to a torch.Tensor, applies FEATURE scaling, 
        and moves it to the correct device.
        """
        features_tensor = features.float()
        
        if self.feature_scaler:
            # Handle multivariate: (batch, seq_len, num_features) -> (batch * seq_len, num_features)
            batch_size, seq_len, num_features = features_tensor.shape
            features_flat = features_tensor.reshape(-1, num_features)
            
            scaled_flat = self.feature_scaler.transform(features_flat)
            
            scaled_features = scaled_flat.reshape(batch_size, seq_len, num_features)
        else:
            scaled_features = features_tensor

        return scaled_features.to(self.device)

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Core batch prediction method for sequences.
        """
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

        # 1. Recombine the multi-head dictionary into a single tensor
        combined_preds = []
        for target in self.target_names:
            pred = output_dict[target]
            
            # Convert categorical logits back to numerical class indices
            if target in self.schema.categorical_feature_names:
                pred = torch.argmax(pred, dim=-1).float()
                
            combined_preds.append(pred)
            
        # Stack to shape: (batch, num_targets) or (batch, seq_len, num_targets)
        combined_tensor = torch.stack(combined_preds, dim=-1)

        # 2. De-scale the combined predictions
        scaler_to_use = self.target_scaler if self.target_scaler else None
        
        if scaler_to_use:
            if self.task == MLTaskKeys.SEQUENCE_VALUE:
                descaled_output = scaler_to_use.inverse_transform(combined_tensor)
                
            elif self.task == MLTaskKeys.SEQUENCE_SEQUENCE:
                batch_size, seq_len, num_targets = combined_tensor.shape
                output_flat = combined_tensor.reshape(-1, num_targets)
                descaled_flat = scaler_to_use.inverse_transform(output_flat)
                descaled_output = descaled_flat.reshape(batch_size, seq_len, num_targets)
            else:
                 _LOGGER.error(f"Invalid prediction mode: {self.task}")
                 raise RuntimeError()
        else:
            descaled_output = combined_tensor

        # 3. Split back into a dictionary mapping targets to their descaled tensors
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
            # Reshape (seq_len, num_features) to (1, seq_len, num_features)
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
    
    # --- NumPy Convenience Wrappers (on CPU) ---

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
        # Returns a NumPy array directly; handles both seq-to-val vectors and seq-to-seq matrices
        tensor_results = self.predict(features)
        return {key: value.cpu().numpy() for key, value in tensor_results.items()}
    
    def forecast(self, 
                 n_steps: int,
                 initial_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 future_exogenous: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Autoregressively forecasts 'n_steps' into the future.
        
        If the model requires exogenous features (features that are not targets),
        'future_exogenous' must be provided as a pandas DataFrame containing at least 'n_steps' of future data.

        Args:
            n_steps (int): The number of future time steps to predict.
            initial_sequence (np.ndarray | torch.Tensor, optional): The historical sequence (un-scaled).
            future_exogenous (pd.DataFrame, optional): Future known variables (un-scaled).

        Returns:
            pd.DataFrame: A DataFrame containing the forecasts.
        """
        if n_steps <= 0:
            _LOGGER.error(f"'n_steps' must be a positive integer greater than 0. Received: {n_steps}")
            raise ValueError()
        
        # --- 1. Resolve Initial Sequence ---
        if initial_sequence is None:
            if self.initial_sequence is None:
                _LOGGER.error("No 'initial_sequence' provided/loaded. Cannot forecast.")
                raise ValueError()
            initial_sequence_tensor = torch.from_numpy(self.initial_sequence).float()
        elif isinstance(initial_sequence, np.ndarray):
            initial_sequence_tensor = torch.from_numpy(initial_sequence).float()
        else:
            initial_sequence_tensor = initial_sequence.float()

        if initial_sequence_tensor.ndim != 2:
             _LOGGER.error(f"initial_sequence must be 2D (seq_len, num_features). Got {initial_sequence_tensor.ndim}D.")
             raise ValueError()
             
        seq_len, num_features = initial_sequence_tensor.shape
        
        # Calculate exogenous requirements based on resolved target indices
        exog_indices = [i for i in range(num_features) if i not in self.target_indices]
        exog_names = [self.schema.feature_names[i] for i in exog_indices]
        num_exog = len(exog_indices)

        # --- 2. Scale Initial Sequence ---
        if self.feature_scaler is None:
             current_scaled_sequence = initial_sequence_tensor.to(self.device)
        else:
            scaled_sequence_flat = self.feature_scaler.transform(initial_sequence_tensor)
            current_scaled_sequence = scaled_sequence_flat.reshape(seq_len, num_features).to(self.device)

        # --- 3. Resolve & Scale Future Exogenous Data ---
        scaled_exog_tensor = None
        if num_exog > 0:
            if future_exogenous is None:
                _LOGGER.error(f"Model requires {num_exog} exogenous features, but 'future_exogenous' was not provided.")
                raise ValueError()
            
            if not isinstance(future_exogenous, pd.DataFrame):
                _LOGGER.error("'future_exogenous' must be a pandas DataFrame to ensure safe column alignment.")
                raise TypeError()
                
            if len(future_exogenous) < n_steps:
                _LOGGER.error(f"'future_exogenous' must have at least {n_steps} rows for a {n_steps}-step forecast.")
                raise ValueError()
                
            missing_cols = [col for col in exog_names if col not in future_exogenous.columns]
            if missing_cols:
                _LOGGER.error(f"'future_exogenous' is missing required exogenous columns: {missing_cols}")
                raise ValueError()
                
            exog_np = future_exogenous[exog_names].iloc[:n_steps].to_numpy()
            exog_tensor = torch.from_numpy(exog_np).float()
            
            if self.feature_scaler:
                dummy_full = torch.zeros((n_steps, num_features))
                dummy_full[:, exog_indices] = exog_tensor
                dummy_scaled = self.feature_scaler.transform(dummy_full)
                scaled_exog_tensor = dummy_scaled[:, exog_indices].to(self.device)
            else:
                scaled_exog_tensor = exog_tensor.to(self.device)

        # --- 4. Autoregressive Loop ---
        descaled_predictions = []
        with torch.no_grad():
            for t in range(n_steps):
                input_tensor = current_scaled_sequence.unsqueeze(0)
                model_output_dict = self.model(input_tensor) 
                
                next_step_preds = []
                for target in self.target_names:
                    # Remove batch dimension -> (seq_len, ...) or (...)
                    pred = model_output_dict[target].squeeze(0)
                    
                    if self.task == MLTaskKeys.SEQUENCE_VALUE:
                        target_pred = pred
                    else: 
                        target_pred = pred[-1] 
                        
                    # Resolve categorical logits to class integers
                    if target in self.schema.categorical_feature_names:
                        target_pred = torch.argmax(target_pred, dim=-1).float()
                        
                    next_step_preds.append(target_pred)
                
                # Combine individual target predictions back to shape (num_targets,)
                scaled_prediction = torch.stack(next_step_preds)
                
                if self.target_scaler:
                    descaled_val = self.target_scaler.inverse_transform(scaled_prediction.unsqueeze(0)).squeeze(0).cpu().numpy()
                else:
                    descaled_val = scaled_prediction.cpu().numpy()
                    
                descaled_predictions.append(descaled_val)
                
                # Reconstruct the complete feature vector
                next_step = torch.zeros(num_features, device=self.device)
                next_step[self.target_indices] = scaled_prediction
                if num_exog > 0 and scaled_exog_tensor is not None:
                    next_step[exog_indices] = scaled_exog_tensor[t]
                    
                current_scaled_sequence = torch.cat((current_scaled_sequence[1:], next_step.unsqueeze(0)))
                
        # --- 5. Return Results ---
        forecast_df = pd.DataFrame(descaled_predictions, columns=[self.schema.feature_names[i] for i in self.target_indices])
        
        return forecast_df

    def plot_forecast(self,  
                      n_steps: int, 
                      save_dir: Union[str, Path], 
                      filename: str = "forecast_plot.svg",
                      initial_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      future_exogenous: Optional[pd.DataFrame] = None):
        """
        Runs an autoregressive forecast and saves a visualization of the results.
        
        This method automatically handles plotting single or multiple target variables 
        on the same graph. It clearly distinguishes between the historical input sequence 
        and the forecasted future steps, separating them with a vertical boundary line. 
        The output is saved as an SVG file to ensure high-quality vector scalability.

        Args:
            n_steps (int): The number of future time steps to predict.
            save_dir (Union[str, Path]): The directory where the plot file will be saved.
                Parent directories will be created automatically if they do not exist.
            filename (str, optional): The name of the output plot file. If the provided 
                name does not end with the '.svg' extension, it will be automatically 
                enforced.
            initial_sequence (Union[np.ndarray, torch.Tensor], optional): The unscaled 
                historical sequence used to prime the model. If None, it will attempt 
                to use the default `initial_sequence` loaded from the model bundle.
            future_exogenous (pd.DataFrame, optional): A pandas DataFrame containing 
                unscaled future values for any exogenous features required by the model. 
                Must contain at least `n_steps` rows.
        """
        # --- 1. Get Forecast Data ---
        forecast_df = self.forecast(n_steps=n_steps, 
                                    initial_sequence=initial_sequence,
                                    future_exogenous=future_exogenous)
        
        # --- 2. Determine Initial Sequence ---
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
            
        # --- 3. Create X-axis indices ---
        seq_len = len(plot_initial_sequence)
        history_x = np.arange(0, seq_len)
        forecast_x = np.arange(seq_len, seq_len + n_steps)

        # --- 4. Plot ---
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

        plt.title(f"{n_steps}-Step Multivariate Forecast")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # --- 5. Save Plot ---
        # ensure the extension is svg
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
            
    def decode_predictions(self, predictions: Union[dict[str, torch.Tensor], dict[str, np.ndarray], pd.DataFrame]) -> pd.DataFrame:
        """
        Converts numerical predictions into a pandas DataFrame and remaps categorical 
        integer indices back to their original string labels.
        
        Args:
            predictions: The output from predict(), predict_numpy(), predict_batch(), predict_batch_numpy(), or forecast().
            
        Returns:
            pd.DataFrame: A formatted DataFrame with strings for categorical targets and floats for continuous targets.
        """
        # --- 1. Standardize Input to DataFrame ---
        if isinstance(predictions, pd.DataFrame):
            df = predictions.copy()
        elif isinstance(predictions, dict):
            df_dict = {}
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                
                # If the output is from a batched sequence (batch_size, seq_len), 
                # a standard 1D DataFrame column can't hold it natively without nesting.
                # convert 2D/3D arrays into lists of arrays per row.
                if val.ndim > 1:
                     val = list(val) 
                
                df_dict[key] = val
            df = pd.DataFrame(df_dict)
        else:
            _LOGGER.error("Input must be a dictionary of Tensors/NumPy arrays or a pandas DataFrame.")
            raise TypeError()
        
        # check that keys in df match the target names
        missing_targets = [t for t in self.target_names if t not in df.columns]
        if missing_targets:
            _LOGGER.warning(f"The following target(s) are missing from the predictions and will be skipped: {missing_targets}")
            
        # unknown columns in df are ignored, but a warning is logged
        unknown_cols = [col for col in df.columns if col not in self.target_names]
        if unknown_cols:
            _LOGGER.warning(f"The following columns in the predictions are not recognized as targets and will be ignored: {unknown_cols}")

        # --- 2. Remap Categorical Targets ---
        # Ensure the schema actually has mappings stored
        if self.schema.categorical_mappings:
            for target in self.schema.categorical_feature_names:
                # Check if the target is in the predictions and has a mapping
                if target in df.columns and target in self.schema.categorical_mappings:
                    
                    # Reverse the mapping from {string: int} to {int: string}
                    forward_map = self.schema.categorical_mappings[target]
                    reverse_map = {v: k for k, v in forward_map.items()}
                    
                    # Define a mapping function that handles both scalars and nested lists/arrays
                    def remap_value(x):
                        if isinstance(x, (list, np.ndarray)):
                            return [reverse_map.get(int(i), int(i)) for i in x]
                        # Handle standard scalar values (like from forecast or single predict)
                        if pd.notna(x):
                            return reverse_map.get(int(x), int(x))
                        return x
                        
                    # Apply the mapping to the column
                    df[target] = df[target].apply(remap_value)
                    
        return df
