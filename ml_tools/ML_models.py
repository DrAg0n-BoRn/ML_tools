import torch
from torch import nn
from ._script_info import _script_info
from typing import List


__all__ = [
    "MultilayerPerceptron",
    "SequencePredictorLSTM"
]


class MultilayerPerceptron(nn.Module):
    """
    Creates a versatile Multilayer Perceptron (MLP) for regression or classification tasks.

    This model generates raw output values (logits) suitable for use with loss
    functions like `nn.CrossEntropyLoss` (for classification) or `nn.MSELoss`
    (for regression).

    Args:
        in_features (int): The number of input features (e.g., columns in your data).
        out_targets (int): The number of output targets. For regression, this is
            typically 1. For classification, it's the number of classes.
        hidden_layers (list[int]): A list where each integer represents the
            number of neurons in a hidden layer. Defaults to [40, 80, 40].
        drop_out (float): The dropout probability for neurons in each hidden
            layer. Must be between 0.0 and 1.0. Defaults to 0.2.
            
    ### Rules of thumb:
    - Choose a number of hidden neurons between the size of the input layer and the size of the output layer. 
    - The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. 
    - The number of hidden neurons should be less than twice the size of the input layer.
    """
    def __init__(self, in_features: int, out_targets: int,
                 hidden_layers: List[int] = [40, 80, 40], drop_out: float = 0.2) -> None:
        super().__init__()

        # --- Validation ---
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError("in_features must be a positive integer.")
        if not isinstance(out_targets, int) or out_targets < 1:
            raise ValueError("out_targets must be a positive integer.")
        if not isinstance(hidden_layers, list) or not all(isinstance(n, int) for n in hidden_layers):
            raise TypeError("hidden_layers must be a list of integers.")
        if not (0.0 <= drop_out < 1.0):
            raise ValueError("drop_out must be a float between 0.0 and 1.0.")

        # --- Build network layers ---
        layers = []
        current_features = in_features
        for neurons in hidden_layers:
            layers.extend([
                nn.Linear(current_features, neurons),
                nn.BatchNorm1d(neurons),
                nn.ReLU(),
                nn.Dropout(p=drop_out)
            ])
            current_features = neurons

        # Add the final output layer
        layers.append(nn.Linear(current_features, out_targets))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self._layers(x)


class SequencePredictorLSTM(nn.Module):
    """
    A simple LSTM-based network for sequence-to-sequence prediction tasks.

    This model is designed for datasets where each input sequence maps to an
    output sequence of the same length. It's suitable for forecasting problems
    prepared by the `SequenceMaker` class.

    The expected input shape is `(batch_size, sequence_length, features)`.

    Args:
        features (int): The number of features in the input sequence. Defaults to 1.
        hidden_size (int): The number of features in the LSTM's hidden state.
                           Defaults to 100.
        recurrent_layers (int): The number of recurrent LSTM layers. Defaults to 1.
        dropout (float): The dropout probability for all but the last LSTM layer.
                         Defaults to 0.
    """
    def __init__(self, features: int = 1, hidden_size: int = 100,
                 recurrent_layers: int = 1, dropout: float = 0):
        super().__init__()

        # --- Validation ---
        if not isinstance(features, int) or features < 1:
            raise ValueError("features must be a positive integer.")
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(recurrent_layers, int) or recurrent_layers < 1:
            raise ValueError("recurrent_layers must be a positive integer.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be a float between 0.0 and 1.0.")

        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            dropout=dropout,
            batch_first=True  # This is crucial for (batch, seq, feature) input
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): The input tensor with shape
                              (batch_size, sequence_length, features).

        Returns:
            torch.Tensor: The output tensor with shape
                          (batch_size, sequence_length, features).
        """
        # The LSTM returns the full output sequence and the final hidden/cell states
        lstm_out, _ = self.lstm(x)
        
        # Pass the LSTM's output sequence to the linear layer
        predictions = self.linear(lstm_out)
        
        return predictions


def info():
    _script_info(__all__)
