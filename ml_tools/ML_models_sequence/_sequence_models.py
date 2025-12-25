import torch
from torch import nn
from typing import Literal

from .._core import get_logger
from ..keys._keys import MLTaskKeys

from ..ML_models._base_save_load import _ArchitectureHandlerMixin


_LOGGER = get_logger("DragonSequenceLSTM")


__all__ = [
    "DragonSequenceLSTM"
]


class DragonSequenceLSTM(nn.Module, _ArchitectureHandlerMixin):
    """
    An LSTM-based network for single-feature (univariate) sequence prediction tasks.
    It can be configured for:
    1. 'sequence-to-sequence': Predicts a full sequence.
    2. 'sequence-to-value': Predicts a single value from the last time step.
    """
    def __init__(self, 
                 prediction_mode: Literal["sequence-to-sequence", "sequence-to-value"],
                 hidden_size: int = 100,
                 recurrent_layers: int = 1,
                 dropout: float = 0.1):
        """
        Args:
            hidden_size (int): The number of features in the LSTM's hidden state.
            recurrent_layers (int): The number of recurrent LSTM layers.
            prediction_mode (str): Determines the model's output behavior.
                - 'sequence-to-sequence': Returns a full sequence.
                - 'sequence-to-value': Returns a single prediction based on the last time step.
            dropout (float): The dropout probability for all but the last LSTM layer.
        """
        super().__init__()

        # --- Validation ---
        if not prediction_mode in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            _LOGGER.error(f"Unrecognized prediction mode: '{prediction_mode}'.")
            raise ValueError()
        else:
            self.prediction_mode = prediction_mode
        
        if not isinstance(hidden_size, int) or hidden_size < 1:
            _LOGGER.error("hidden_size must be a positive integer.")
            raise ValueError()
        if not isinstance(recurrent_layers, int) or recurrent_layers < 1:
            _LOGGER.error("recurrent_layers must be a positive integer.")
            raise ValueError()
        if not (0.0 <= dropout < 1.0):
            _LOGGER.error("dropout must be a float between 0.0 and 1.0.")
            raise ValueError()
        
        # --- Save configuration ---
        self.features = 1 # Univariate
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.dropout = dropout
        
        # Build model
        self.lstm = nn.LSTM(
            input_size=self.features,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            dropout=dropout,
            batch_first=True  # This is crucial for (batch, seq, feature) input
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): The input tensor. Can be 2D (batch_size, sequence_length)
                              or 3D (batch_size, sequence_length, features).
                              The model will automatically handle 2D inputs
                              by assuming a feature size of 1.

        Returns:
            torch.Tensor: The output tensor.
                - (batch_size, sequence_length, features) if 'sequence-to-sequence'
                - (batch_size, features) if 'sequence-to-value'
        """
        # --- Handle Input Shape ---
        if x.ndim == 2:
            # Check if this 2D input is compatible with the model's expected features
            if self.features != 1:
                _LOGGER.error(f"Received 2D input (shape {x.shape}), but model was initialized with features={self.features}.")
                raise ValueError()
            
            # Add the feature dimension: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)
        
        # x is guaranteed to be 3D: (batch_size, seq_len, features)
        # The LSTM returns the full output sequence and the final hidden/cell states
        lstm_out, _ = self.lstm(x)
        
        # --- Handle Output Shape based on mode ---
        if self.prediction_mode == MLTaskKeys.SEQUENCE_SEQUENCE:
            # Use the full sequence
            # output shape: (batch_size, seq_len, 1)
            predictions = self.linear(lstm_out)
            # Squeeze to (batch_size, seq_len) to match target
            predictions = predictions.squeeze(-1)
        
        elif self.prediction_mode == MLTaskKeys.SEQUENCE_VALUE:
            # Isolate only the last time step's output
            # last_step shape: (batch_size, hidden_size)
            last_step = lstm_out[:, -1, :]
            predictions = self.linear(last_step)
        
            # Squeeze the 'features' dim to match label shape
            predictions = predictions.squeeze(-1)
        
        return predictions
    
    def get_architecture_config(self) -> dict:
        """Returns the configuration of the model."""
        return {
            'hidden_size': self.hidden_size,
            'recurrent_layers': self.recurrent_layers,
            'prediction_mode': self.prediction_mode,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        return (
            f"DragonSequenceLSTM(features={self.lstm.input_size}, "
            f"hidden_size={self.lstm.hidden_size}, "
            f"recurrent_layers={self.lstm.num_layers}), "
            f"mode='{self.prediction_mode}')")

