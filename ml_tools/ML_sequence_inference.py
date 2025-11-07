import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Union, Literal, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from .ML_scaler import DragonScaler
from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath, sanitize_filename
from ._keys import PyTorchInferenceKeys, MLTaskKeys, PyTorchCheckpointKeys
from .ML_inference import _BaseInferenceHandler


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
                 prediction_mode: Literal["sequence-to-sequence", "sequence-to-value"],
                 scaler: Union[DragonScaler, str, Path],
                 device: str = 'cpu'):
        """
        Initializes the handler for sequence tasks.

        Args:
            model (nn.Module): An instantiated PyTorch model architecture.
            state_dict (str | Path): Path to the saved .pth model state_dict file.
            prediction_mode (str): The type of sequence task.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps').
            scaler (DragonScaler | str | Path): A DragonScaler instance or 
                the file path to a saved DragonScaler state. This is required
                to correctly scale inputs and de-scale predictions.
        """
        # Call the parent constructor to handle model loading and device
        super().__init__(model, state_dict, device, scaler)
        
        self.sequence_length: Optional[int] = None
        self.initial_sequence: Optional[np.ndarray] = None

        if prediction_mode not in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            _LOGGER.error(f"'prediction_mode' not recognized: '{prediction_mode}'.")
            raise ValueError()
        self.prediction_mode = prediction_mode
        
        if self.scaler is None:
            _LOGGER.error("A 'scaler' is required for DragonSequenceInferenceHandler to scale inputs and de-scale predictions.")
            raise ValueError()
        
        # Load sequence length from the loaded dict (populated by _BaseInferenceHandler)
        if PyTorchCheckpointKeys.SEQUENCE_LENGTH in self._loaded_data_dict:
            try:
                self.sequence_length = int(self._loaded_data_dict[PyTorchCheckpointKeys.SEQUENCE_LENGTH])
                _LOGGER.info(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' found and set to {self.sequence_length}")
            except Exception as e_int:
                _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' but an error occurred when retrieving it:\n{e_int}")
        else:
            _LOGGER.warning(f"'{PyTorchCheckpointKeys.SEQUENCE_LENGTH}' not found in model file. Forecasting validation will be skipped.")
            
        # Load initial sequence
        if PyTorchCheckpointKeys.INITIAL_SEQUENCE in self._loaded_data_dict:
            try:
                self.initial_sequence = self._loaded_data_dict[PyTorchCheckpointKeys.INITIAL_SEQUENCE]
                _LOGGER.info(f"Default 'initial_sequence' for forecasting loaded from model file.")
                # Optional: Validate shape
                if self.sequence_length and len(self.initial_sequence) != self.sequence_length: # type: ignore
                    _LOGGER.warning(f"Loaded 'initial_sequence' length ({len(self.initial_sequence)}) mismatches 'sequence_length' ({self.sequence_length}).") # type: ignore
            except Exception as e_seq:
                _LOGGER.warning(f"State Dictionary has the key '{PyTorchCheckpointKeys.INITIAL_SEQUENCE}' but an error occurred when retrieving it:\n{e_seq}")
        else:
            _LOGGER.info("No default 'initial_sequence' found in model file. Must be provided for forecasting.")

    def _preprocess_input(self, features: torch.Tensor) -> torch.Tensor:
        """
        Converts input sequence to a torch.Tensor, applies scaling, and moves it to the correct device.

        Overrides _BaseInferenceHandler._preprocess_input.

        Args:
            features (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Scaled tensor on the correct device.
        """
        if self.scaler is None:
            # This check is redundant due to __init__ check, but good for safety.
            _LOGGER.error("Scaler is not available for preprocessing.")
            raise RuntimeError()
            
        features_tensor = features.float()
        
        # Scale the sequence values
        # (batch, seq_len) -> (batch * seq_len, 1)
        batch_size, seq_len = features_tensor.shape
        features_flat = features_tensor.reshape(-1, 1)
        
        scaled_flat = self.scaler.transform(features_flat)
        
        # (batch * seq_len, 1) -> (batch, seq_len)
        scaled_features = scaled_flat.reshape(batch_size, seq_len)

        return scaled_features.to(self.device)

    def predict_batch(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            _LOGGER.error("Input for batch prediction must be a 2D array or tensor (batch_size, sequence_length).")
            raise ValueError()
        
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        # _preprocess_input scales the data and moves it to the correct device
        input_tensor = self._preprocess_input(features_tensor) 

        with torch.no_grad():
            scaled_output = self.model(input_tensor)

        # De-scale the output using the scaler
        if self.scaler is None: # Should be impossible due to __init__
             raise RuntimeError("Scaler not found for de-scaling.")

        if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
            # scaled_output is (batch)
            # Reshape to (batch, 1) for scaler
            scaled_output_reshaped = scaled_output.reshape(-1, 1)
            descaled_output = self.scaler.inverse_transform(scaled_output_reshaped)
            descaled_output = descaled_output.squeeze(-1) # (batch)
        
        elif self.prediction_mode == MLTaskKeys.SEQUENCE_SEQUENCE:
            # scaled_output is (batch, seq_len)
            batch_size, seq_len = scaled_output.shape
            scaled_flat = scaled_output.reshape(-1, 1)
            descaled_flat = self.scaler.inverse_transform(scaled_flat)
            descaled_output = descaled_flat.reshape(batch_size, seq_len)
        
        else:
             # Should not happen
            _LOGGER.error(f"Invalid prediction mode: {self.prediction_mode}")
            raise RuntimeError()

        return {PyTorchInferenceKeys.PREDICTIONS: descaled_output}

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            _LOGGER.error("The 'predict()' method is for a single sequence (1D tensor). Use 'predict_batch()' for multiple sequences (2D tensor).")
            raise ValueError()

        batch_results = self.predict_batch(features)

        # Extract the first (and only) result from the batch output
        # For seq-to-value, result is shape ()
        # For seq-to-seq, result is shape (seq_len)
        single_results = {key: value[0] for key, value in batch_results.items()}
        return single_results
    
    # --- NumPy Convenience Wrappers (on CPU) ---

    def predict_batch_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
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

    def predict_numpy(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
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
        
        if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
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
                _LOGGER.error("No 'initial_sequence' provided and no default sequence was loaded. Cannot forecast.")
                raise ValueError()
            _LOGGER.info("Using default 'initial_sequence' loaded from model file for forecast.")
            initial_sequence_tensor = torch.from_numpy(self.initial_sequence).float()
        elif isinstance(initial_sequence, np.ndarray):
            initial_sequence_tensor = torch.from_numpy(initial_sequence).float()
        else:
            initial_sequence_tensor = initial_sequence.float()

        if initial_sequence_tensor.ndim != 1:
             _LOGGER.error(f"initial_sequence must be a 1D array. Got {initial_sequence_tensor.ndim} dimensions.")
             raise ValueError()
        
        if self.sequence_length is not None:
            if len(initial_sequence_tensor) != self.sequence_length:
                _LOGGER.error(f"Input sequence length ({len(initial_sequence_tensor)}) does not match model's required sequence_length ({self.sequence_length}).")
                raise ValueError()
        else:
            _LOGGER.warning("Model's 'sequence_length' is unknown. Cannot validate input sequence length. Assuming it is correct.")
        
        # --- Pre-processing ---
        # 1. Scale the entire initial sequence
        # We need to use the scaler: (seq_len) -> (seq_len, 1)
        if self.scaler is None: # Should be impossible due to __init__
            raise RuntimeError("Scaler not found for forecasting.")
            
        scaled_sequence_flat = self.scaler.transform(initial_sequence_tensor.reshape(-1, 1))
        # (seq_len, 1) -> (seq_len)
        current_scaled_sequence = scaled_sequence_flat.squeeze(-1).to(self.device)
        
        descaled_predictions = []

        # --- Autoregressive Loop ---
        self.model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            for _ in range(n_steps):
                # (seq_len) -> (1, seq_len)
                input_tensor = current_scaled_sequence.reshape(1, -1)
                
                # Run the model
                # input_tensor is (1, seq_len)
                model_output = self.model(input_tensor).squeeze() # remove batch dim
                
                # Extract the single new prediction
                if self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
                    # Output is shape (), a single scalar tensor
                    scaled_prediction = model_output
                else: # MLTaskKeys.SEQUENCE_SEQUENCE
                    # Output is shape (seq_len), we need the last value
                    scaled_prediction = model_output[-1]
                
                # De-scale the prediction for storage
                # scaler input (1, 1)
                descaled_prediction = self.scaler.inverse_transform(scaled_prediction.reshape(1, 1)).item()
                descaled_predictions.append(descaled_prediction)
                
                # Create the new input sequence for the next loop
                # "autoregression": roll the window by dropping the first value and appending the new scaled prediction.
                # .unsqueeze(0) is needed to make the 0-dim tensor 1-dim for cat
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
    

def info():
    _script_info(__all__)
