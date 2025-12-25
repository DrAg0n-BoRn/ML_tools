import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from .._core import get_logger
from ..path_manager import make_fullpath, sanitize_filename
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys, PyTorchCheckpointKeys

from ..ML_inference._base_inference import _BaseInferenceHandler


_LOGGER = get_logger("DragonSequenceInference")


__all__ = [
    "DragonSequenceInferenceHandler"
]


class DragonSequenceInferenceHandler(_BaseInferenceHandler):
    """
    Handles loading a PyTorch sequence model's state and performing inference
    for univariate sequence tasks.
    
    This handler automatically scales inputs and de-scales outputs.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 scaler: Union[str, Path],
                 task: Optional[Literal["sequence-to-sequence", "sequence-to-value"]]=None,
                 device: str = 'cpu'):
        """
        Initializes the handler for sequence tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model architecture.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            task (str, optional): The type of sequence task. If None, detected from file.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (str | Path): File path to a saved DragonScaler state. This is required to correctly scale inputs and de-scale predictions.
        """
        # Call the parent constructor to handle model loading and device
        super().__init__(model=model, 
                         state_dict=state_dict, 
                         device=device, 
                         scaler=scaler, 
                         task=task)
        
        self.sequence_length: Optional[int] = None
        self.initial_sequence: Optional[np.ndarray] = None
        
        valid_tasks = [
            MLTaskKeys.SEQUENCE_SEQUENCE, 
            MLTaskKeys.SEQUENCE_VALUE
        ]

        if self.task not in valid_tasks:
            _LOGGER.error(f"'task' recognized as '{self.task}', but this handler only supports: {valid_tasks}.")
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
            # Optional: Validate shape
            if self.sequence_length and len(self.initial_sequence) != self.sequence_length: # type: ignore
                _LOGGER.warning(f"Loaded 'initial_sequence' length ({len(self.initial_sequence)}) mismatches 'sequence_length' ({self.sequence_length}).") # type: ignore
        else:
            _LOGGER.info("No default 'initial_sequence' found in model file. Must be provided for forecasting.")

    def _preprocess_input(self, features: torch.Tensor) -> torch.Tensor:
        """
        Converts input sequence to a torch.Tensor, applies FEATURE scaling, 
        and moves it to the correct device.

        Overrides _BaseInferenceHandler._preprocess_input.
        """
        features_tensor = features.float()
        
        if self.feature_scaler:
            # Scale the sequence values
            # Assumption: Univariate sequence (batch, seq_len) -> (batch * seq_len, 1) for scaling
            batch_size, seq_len = features_tensor.shape
            features_flat = features_tensor.reshape(-1, 1)
            
            scaled_flat = self.feature_scaler.transform(features_flat)
            
            # (batch * seq_len, 1) -> (batch, seq_len)
            scaled_features = scaled_flat.reshape(batch_size, seq_len)
        else:
            scaled_features = features_tensor

        return scaled_features.to(self.device)

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Core batch prediction method for sequences.
        Runs a batch of sequences through the model, de-scales the output,
        and returns the predictions.

        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of input sequences, shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the de-scaled prediction tensors.
        """
        if features.ndim != 2:
            _LOGGER.error("Input for batch prediction must be a 2D array/tensor (batch_size, sequence_length).")
            raise ValueError()
        
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        # _preprocess_input scales data using feature_scaler
        input_tensor = self._preprocess_input(features_tensor) 

        with torch.no_grad():
            output = self.model(input_tensor)

        # De-scale the output using the TARGET scaler
        # If no target scaler exists (unlikely for regression), return raw output
        scaler_to_use = self.target_scaler if self.target_scaler else None
        
        if scaler_to_use:
            if self.task == MLTaskKeys.SEQUENCE_VALUE:
                # output is (batch) -> reshape to (batch, 1) for scaler
                output_reshaped = output.reshape(-1, 1)
                descaled_output = scaler_to_use.inverse_transform(output_reshaped)
                descaled_output = descaled_output.squeeze(-1) # (batch)
            
            elif self.task == MLTaskKeys.SEQUENCE_SEQUENCE:
                # output is (batch, seq_len)
                batch_size, seq_len = output.shape
                output_flat = output.reshape(-1, 1)
                descaled_flat = scaler_to_use.inverse_transform(output_flat)
                descaled_output = descaled_flat.reshape(batch_size, seq_len)
            else:
                 _LOGGER.error(f"Invalid prediction mode: {self.task}")
                 raise RuntimeError()
        else:
            descaled_output = output

        return {PyTorchInferenceKeys.PREDICTIONS: descaled_output}

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Core single-sample prediction method for sequences.
        Runs a single sequence through the model, de-scales the output,
        and returns the prediction.

        Args:
            features (np.ndarray | torch.Tensor): A 1D array/tensor of 
                input features, shape (sequence_length).

        Returns:
            A dictionary containing the de-scaled prediction tensor.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1) # Reshape (seq_len) to (1, seq_len)
        
        if features.shape[0] != 1 or features.ndim != 2:
            _LOGGER.error("predict() is for a single sequence (1D). Use predict_batch() for multiple (2D).")
            raise ValueError()

        batch_results = self.predict_batch(features)
        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results
    
    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        Convenience wrapper for predict_batch that returns NumPy arrays.
        
        Args:
            features (np.ndarray | torch.Tensor): A 2D array/tensor of 
                input sequences, shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the de-scaled prediction as a NumPy array.
        """
        tensor_results = self.predict_batch(features)
        numpy_results = {key: value.cpu().numpy() for key, value in tensor_results.items()}
        return numpy_results

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
        """
        Convenience wrapper for predict that returns NumPy arrays or scalars.
        
        Args:
            features (np.ndarray | torch.Tensor): A 1D array/tensor of 
                input features, shape (sequence_length).
                
        Returns:
            A dictionary containing the de-scaled prediction.
            - For 'sequence-to-value', the value is a Python scalar.
            - For 'sequence-to-sequence', the value is a 1D NumPy array.
        """
        tensor_results = self.predict(features)
        
        if self.task == MLTaskKeys.SEQUENCE_VALUE:
             # Prediction is a 0-dim tensor, .item() gets the scalar
             return {PyTorchInferenceKeys.PREDICTIONS: tensor_results[PyTorchInferenceKeys.PREDICTIONS].item()}
        else: # sequence-to-sequence
             # Prediction is a 1D tensor
             return {PyTorchInferenceKeys.PREDICTIONS: tensor_results[PyTorchInferenceKeys.PREDICTIONS].cpu().numpy()}
        
    def forecast(self, 
                 n_steps: int,
                 initial_sequence: Optional[Union[np.ndarray, torch.Tensor]]=None) -> np.ndarray:
        """
        Autoregressively forecasts 'n_steps' into the future.

        This method works for both 'sequence-to-value' and 
        'sequence-to-sequence' models.
        
        If 'initial_sequence' is not provided, this method will use the
        default sequence that was saved with the model (if available).

        Args:
            initial_sequence (np.ndarray | torch.Tensor): The sequence
                to start forecasting from. If None, uses the loaded default.
                This should be a 1D array of *un-scaled* data.
            n_steps (int): The number of future time steps to predict.

        Returns:
            np.ndarray: A 1D array containing the 'n_steps' forecasted values.
        """
        # --- Validation ---
        if initial_sequence is None:
            if self.initial_sequence is None:
                _LOGGER.error("No 'initial_sequence' provided/loaded. Cannot forecast.")
                raise ValueError()
            _LOGGER.info("Using default 'initial_sequence' loaded from model file.")
            initial_sequence_tensor = torch.from_numpy(self.initial_sequence).float()
        elif isinstance(initial_sequence, np.ndarray):
            initial_sequence_tensor = torch.from_numpy(initial_sequence).float()
        else:
            initial_sequence_tensor = initial_sequence.float()

        if initial_sequence_tensor.ndim != 1:
             _LOGGER.error(f"initial_sequence must be 1D. Got {initial_sequence_tensor.ndim}D.")
             raise ValueError()
        
        # --- Scaling ---
        # For autoregression (Forecasting Y), the input is history of Y.
        # We must use target_scaler.
        scaler_to_use = self.target_scaler if self.target_scaler else self.feature_scaler
        
        if scaler_to_use is None:
             _LOGGER.warning("No scaler found for forecasting. Using raw data.")
             current_scaled_sequence = initial_sequence_tensor.to(self.device)
        else:
            scaled_sequence_flat = scaler_to_use.transform(initial_sequence_tensor.reshape(-1, 1))
            current_scaled_sequence = scaled_sequence_flat.squeeze(-1).to(self.device)
        
        descaled_predictions = []

        # --- Autoregressive Loop ---
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                # (seq_len) -> (1, seq_len)
                input_tensor = current_scaled_sequence.reshape(1, -1)
                
                # Run model directly (bypassing _preprocess_input to keep control of scaling)
                model_output = self.model(input_tensor).squeeze() 
                
                # Extract the single new prediction
                if self.task == MLTaskKeys.SEQUENCE_VALUE:
                    scaled_prediction = model_output
                else: 
                    scaled_prediction = model_output[-1]
                
                # De-scale
                if scaler_to_use:
                    descaled_val = scaler_to_use.inverse_transform(scaled_prediction.reshape(1, 1)).item()
                else:
                    descaled_val = scaled_prediction.item()
                    
                descaled_predictions.append(descaled_val)
                
                # Update sequence (Roll window)
                current_scaled_sequence = torch.cat((current_scaled_sequence[1:], scaled_prediction.unsqueeze(0)))
                
        return np.array(descaled_predictions)
    
    def plot_forecast(self,  
                      n_steps: int, 
                      save_dir: Union[str, Path], 
                      filename: str = "forecast_plot.svg",
                      initial_sequence: Optional[Union[np.ndarray, torch.Tensor]]=None):
        """
        Runs a forecast and saves a plot of the results.

        Args:
            n_steps (int): The number of future time steps to predict.
            save_dir (str | Path): Directory to save the plot.
            filename (str, optional): Name for the saved plot file.
            initial_sequence (np.ndarray | torch.Tensor | None): The sequence
                to start forecasting from. If None, uses the loaded default.
        """
        # --- 1. Get Forecast Data ---
        predictions = self.forecast(n_steps=n_steps, 
                                    initial_sequence=initial_sequence)
        
        # --- 2. Determine which initial sequence was used for plotting ---
        if initial_sequence is None:
            plot_initial_sequence = self.initial_sequence
            if plot_initial_sequence is None: # Should be caught by forecast() but good to check
                 _LOGGER.error("Cannot plot: No 'initial_sequence' provided and no default found.")
                 return
        elif isinstance(initial_sequence, torch.Tensor):
            plot_initial_sequence = initial_sequence.cpu().numpy()
        else: # Is numpy array
            plot_initial_sequence = initial_sequence
            
        # --- 3. Create X-axis indices ---
        # The x-axis will be integer time steps
        seq_len = len(plot_initial_sequence)
        history_x = np.arange(0, seq_len)
        forecast_x = np.arange(seq_len, seq_len + n_steps)

        # --- 4. Plot ---
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(12, 6))

        # Plot the historical data
        plt.plot(history_x, plot_initial_sequence, label="Historical Data")
        
        # Plot the forecasted data
        plt.plot(forecast_x, predictions, label="Forecasted Data", linestyle="--")
        
        # Add a vertical line to mark the start of the forecast
        plt.axvline(x=history_x[-1], color='red', linestyle=':', label='Forecast Start')

        plt.title(f"{n_steps}-Step Forecast")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        # --- 5. Save Plot ---
        dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = dir_path / sanitize_filename(filename)
        
        try:
            plt.savefig(full_path)
            _LOGGER.info(f"📈 Forecast plot saved to '{full_path.name}'.")
        except Exception as e:
            _LOGGER.error(f"Failed to save plot:\n{e}")
        finally:
            plt.close()
    
