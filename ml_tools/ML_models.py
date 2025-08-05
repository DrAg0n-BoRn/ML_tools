import torch
from torch import nn
from ._script_info import _script_info
from typing import List, Union
from pathlib import Path
import json
from ._logger import _LOGGER
from .path_manager import make_fullpath


__all__ = [
    "MultilayerPerceptron",
    "SequencePredictorLSTM",
    "save_architecture",
    "load_architecture"
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
        
        # --- Save configuration ---
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out

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
    
    def get_config(self) -> dict:
        """Returns the configuration of the model."""
        return {
            'in_features': self.in_features,
            'out_targets': self.out_targets,
            'hidden_layers': self.hidden_layers,
            'drop_out': self.drop_out
        }
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        # Extracts the number of neurons from each nn.Linear layer
        layer_sizes = [layer.in_features for layer in self._layers if isinstance(layer, nn.Linear)]
        
        # Get the last layer and check its type before accessing the attribute
        last_layer = self._layers[-1]
        if isinstance(last_layer, nn.Linear):
            layer_sizes.append(last_layer.out_features)
        
        # Creates a string like: 10 -> 40 -> 80 -> 40 -> 2
        arch_str = ' -> '.join(map(str, layer_sizes))
        
        return f"MultilayerPerceptron(arch: {arch_str})"


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
        
        # --- Save configuration ---
        self.features = features
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.dropout = dropout
        
        # Build model
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
    
    def get_config(self) -> dict:
        """Returns the configuration of the model."""
        return {
            'features': self.features,
            'hidden_size': self.hidden_size,
            'recurrent_layers': self.recurrent_layers,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        return (
            f"SequencePredictorLSTM(features={self.lstm.input_size}, "
            f"hidden_size={self.lstm.hidden_size}, "
            f"recurrent_layers={self.lstm.num_layers})"
        )


def save_architecture(model: nn.Module, directory: Union[str, Path], verbose: bool=True):
    """
    Saves a model's architecture to a 'architecture.json' file.

    This function relies on the model having a `get_config()` method that
    returns a dictionary of the arguments needed to initialize it.

    Args:
        model (nn.Module): The PyTorch model instance to save.
        directory (str | Path): The directory to save the JSON file.

    Raises:
        AttributeError: If the model does not have a `get_config()` method.
    """
    if not hasattr(model, 'get_config'):
        raise AttributeError(
            f"Model '{model.__class__.__name__}' does not have a 'get_config()' method. "
            "Please implement it to return the model's constructor arguments."
        ) 

    # Ensure the target directory exists
    path_dir = make_fullpath(directory, make=True, enforce="directory")
    full_path = path_dir / "architecture.json"

    config = {
        'model_class': model.__class__.__name__,
        'config': model.get_config() # type: ignore
    }

    with open(full_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    if verbose:
        _LOGGER.info(f"✅ Architecture for '{model.__class__.__name__}' saved to '{path_dir}'")


def load_architecture(filepath: Union[str, Path], expected_model_class: type, verbose: bool=True) -> nn.Module:
    """
    Loads a model architecture from a JSON file.

    This function instantiates a model by providing an explicit class to use
    and checking that it matches the class name specified in the file.

    Args:
        filepath (Union[str, Path]): The path of the JSON architecture file.
        expected_model_class (type): The model class expected to load (e.g., MultilayerPerceptron).

    Returns:
        nn.Module: An instance of the model with a freshly initialized state.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If the class name in the file does not match the `expected_model_class`.
    """
    path_obj = make_fullpath(filepath, enforce="file")

    with open(path_obj, 'r') as f:
        saved_data = json.load(f)

    saved_class_name = saved_data['model_class']
    config = saved_data['config']

    if saved_class_name != expected_model_class.__name__:
        raise ValueError(
            f"Model class mismatch. File specifies '{saved_class_name}', "
            f"but you expected '{expected_model_class.__name__}'."
        )

    # Create an instance of the model using the provided class and config
    model = expected_model_class(**config)
    if verbose:
        _LOGGER.info(f"✅ Successfully loaded architecture for '{saved_class_name}'")
    return model


def info():
    _script_info(__all__)
