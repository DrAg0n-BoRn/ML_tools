import torch
from torch import nn
from typing import List, Union, Tuple, Dict, Any
from pathlib import Path
import json
from ._logger import _LOGGER
from .path_manager import make_fullpath
from ._script_info import _script_info

__all__ = [
    "MultilayerPerceptron",
    "AttentionMLP",
    "MultiHeadAttentionMLP",
    "TabularTransformer",
    "SequencePredictorLSTM",
    "save_architecture",
    "load_architecture"
]


class _BaseMLP(nn.Module):
    """
    A base class for Multilayer Perceptrons.
    
    Handles validation, configuration, and the creation of the core MLP layers,
    allowing subclasses to define their own pre-processing and forward pass.
    """
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: List[int], 
                 drop_out: float) -> None:
        super().__init__()

        # --- Validation ---
        if not isinstance(in_features, int) or in_features < 1:
            _LOGGER.error("'in_features' must be a positive integer.")
            raise ValueError()
        if not isinstance(out_targets, int) or out_targets < 1:
            _LOGGER.error("'out_targets' must be a positive integer.")
            raise ValueError()
        if not isinstance(hidden_layers, list) or not all(isinstance(n, int) for n in hidden_layers):
            _LOGGER.error("'hidden_layers' must be a list of integers.")
            raise TypeError()
        if not (0.0 <= drop_out < 1.0):
            _LOGGER.error("'drop_out' must be a float between 0.0 and 1.0.")
            raise ValueError()
        
        # --- Save configuration ---
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out

        # --- Build the core MLP network ---
        mlp_layers = []
        current_features = in_features
        for neurons in hidden_layers:
            mlp_layers.extend([
                nn.Linear(current_features, neurons),
                nn.BatchNorm1d(neurons),
                nn.ReLU(),
                nn.Dropout(p=drop_out)
            ])
            current_features = neurons
        
        self.mlp = nn.Sequential(*mlp_layers)
        # Set a customizable Prediction Head for flexibility, specially in transfer learning and fine-tuning
        self.output_layer = nn.Linear(current_features, out_targets)

    def get_config(self) -> Dict[str, Any]:
        """Returns the base configuration of the model."""
        return {
            'in_features': self.in_features,
            'out_targets': self.out_targets,
            'hidden_layers': self.hidden_layers,
            'drop_out': self.drop_out
        }
        
    def _repr_helper(self, name: str, mlp_layers: list[str]):
        last_layer = self.output_layer
        if isinstance(last_layer, nn.Linear):
            mlp_layers.append(str(last_layer.out_features))
        else:
            mlp_layers.append("Custom Prediction Head")
        
        # Creates a string like: 10 -> 40 -> 80 -> 40 -> 2
        arch_str = ' -> '.join(mlp_layers)
        
        return f"{name}(arch: {arch_str})"


class MultilayerPerceptron(_BaseMLP):
    """
    Creates a versatile Multilayer Perceptron (MLP) for regression or classification tasks.
    """
    def __init__(self, in_features: int, out_targets: int,
                 hidden_layers: List[int] = [256, 128], drop_out: float = 0.2) -> None:
        """
        Args:
            in_features (int): The number of input features (e.g., columns in your data).
            out_targets (int): The number of output targets. For regression, this is
                typically 1. For classification, it's the number of classes.
            hidden_layers (list[int]): A list where each integer represents the
                number of neurons in a hidden layer.
            drop_out (float): The dropout probability for neurons in each hidden
                layer. Must be between 0.0 and 1.0.
                
        ### Rules of thumb:
        - Choose a number of hidden neurons between the size of the input layer and the size of the output layer. 
        - The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. 
        - The number of hidden neurons should be less than twice the size of the input layer.
        """
        super().__init__(in_features, out_targets, hidden_layers, drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        x = self.mlp(x)
        logits = self.output_layer(x)
        return logits
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        # Extracts the number of neurons from each nn.Linear layer
        layer_sizes = [str(layer.in_features) for layer in self.mlp if isinstance(layer, nn.Linear)]
        
        return self._repr_helper(name="MultilayerPerceptron", mlp_layers=layer_sizes)


class AttentionMLP(_BaseMLP):
    """
    A Multilayer Perceptron (MLP) that incorporates an Attention layer to dynamically weigh input features.
    
    In inference mode use `forward_attention()` to get a tuple with `(output, attention_weights)`
    """
    def __init__(self, in_features: int, out_targets: int,
                 hidden_layers: List[int] = [256, 128], drop_out: float = 0.2) -> None:
        """
        Args:
            in_features (int): The number of input features (e.g., columns in your data).
            out_targets (int): The number of output targets. For regression, this is
                typically 1. For classification, it's the number of classes.
            hidden_layers (list[int]): A list where each integer represents the
                number of neurons in a hidden layer.
            drop_out (float): The dropout probability for neurons in each hidden
                layer. Must be between 0.0 and 1.0.
        """
        super().__init__(in_features, out_targets, hidden_layers, drop_out)
        # Attention
        self.attention = _AttentionLayer(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the standard forward pass.
        """
        logits, _attention_weights = self.forward_attention(x)
        return logits
    
    def forward_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns logits and attention weights
        """
        # The attention layer returns the processed x and the weights
        x, attention_weights = self.attention(x)
        
        # Pass the attention-modified tensor through the MLP
        logits = self.mlp(x)
        
        return logits, attention_weights
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        # Start with the input features and the attention marker
        arch = [str(self.in_features), "[Attention]"]

        # Find all other linear layers in the MLP 
        for layer in self.mlp[1:]:
            if isinstance(layer, nn.Linear):
                arch.append(str(layer.in_features))
        
        return self._repr_helper(name="AttentionMLP", mlp_layers=arch)


class MultiHeadAttentionMLP(_BaseMLP):
    """
    An MLP that incorporates a standard `nn.MultiheadAttention` layer to process
    the input features.

    In inference mode use `forward_attention()` to get a tuple with `(output, attention_weights)`.
    """
    def __init__(self, in_features: int, out_targets: int,
                 hidden_layers: List[int] = [256, 128], drop_out: float = 0.2,
                 num_heads: int = 4, attention_dropout: float = 0.1) -> None:
        """
        Args:
            in_features (int): The number of input features.
            out_targets (int): The number of output targets.
            hidden_layers (list[int]): A list of neuron counts for each hidden layer.
            drop_out (float): The dropout probability for the MLP layers.
            num_heads (int): The number of attention heads.
            attention_dropout (float): Dropout probability in the attention layer.
        """
        super().__init__(in_features, out_targets, hidden_layers, drop_out)
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        
        self.attention = _MultiHeadAttentionLayer(
            num_features=in_features,
            num_heads=num_heads,
            dropout=attention_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the standard forward pass of the model."""
        logits, _attention_weights = self.forward_attention(x)
        return logits
    
    def forward_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns logits and attention weights.
        """
        # The attention layer returns the processed x and the weights
        x, attention_weights = self.attention(x)
        
        # Pass the attention-modified tensor through the MLP and prediction head
        x = self.mlp(x)
        logits = self.output_layer(x)
        
        return logits, attention_weights

    def get_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['attention_dropout'] = self.attention_dropout
        return config
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        mlp_part = " -> ".join(
            [str(self.in_features)] + 
            [str(h) for h in self.hidden_layers] + 
            [str(self.out_targets)]
        )
        arch_str = f"{self.in_features} -> [MultiHead(h={self.num_heads})] -> {mlp_part}"
        
        return f"MultiHeadAttentionMLP(arch: {arch_str})"


class TabularTransformer(nn.Module):
    """
    A Transformer-based model for tabular data tasks.
    
    This model uses a Feature Tokenizer to convert all input features into a sequence of embeddings, prepends a [CLS] token, and processes the
    sequence with a standard Transformer Encoder.
    """
    def __init__(self, *,
                 out_targets: int,
                 numerical_indices: List[int],
                 categorical_map: Dict[int, int],
                 embedding_dim: int = 32,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """
        Args:
            out_targets (int): Number of output targets (1 for regression).
            numerical_indices (List[int]): Column indices for numerical features.
            categorical_map (Dict[int, int]): Maps categorical column index to its cardinality (number of unique categories).
            embedding_dim (int): The dimension for all feature embeddings. Must be divisible by num_heads.
            num_heads (int): The number of heads in the multi-head attention mechanism.
            num_layers (int): The number of sub-encoder-layers in the transformer encoder.
            dropout (float): The dropout value.
            
        Note: 
        - All arguments are keyword-only to promote clarity.
        - Column indices start at 0.
        
        ### Data Preparation
        The model requires a specific input format. All columns in the input DataFrame must be numerical, but they are treated differently based on the 
        provided index lists.

        **Nominal Categorical Features** (e.g., 'City', 'Color'): Should **NOT** be one-hot encoded. 
        Instead, convert them to integer codes (label encoding). You must then provide a dictionary mapping their column indices to 
        their cardinality (the number of unique categories) via the `categorical_map` parameter.

        **Ordinal & Binary Features** (e.g., 'Low/Medium/High', 'True/False'): Should be treated as **numerical**. Map them to numbers that 
        represent their state (e.g., `{'Low': 0, 'Medium': 1}` or `{False: 0, True: 1}`). Their column indices should be included in the 
        `numerical_indices` list.

        **Standard Numerical Features** (e.g., 'Age', 'Price'): Should be included in the `numerical_indices` list. It is highly recommended to 
        scale them before training.
        """
        super().__init__()

        # --- Save configuration ---
        self.out_targets = out_targets
        self.numerical_indices = numerical_indices
        self.categorical_map = categorical_map
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # --- 1. Feature Tokenizer ---
        self.tokenizer = _FeatureTokenizer(
            numerical_indices=numerical_indices,
            categorical_map=categorical_map,
            embedding_dim=embedding_dim
        )
        
        # --- 2. CLS Token ---
        # A learnable token that will be prepended to the sequence.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # --- 3. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True # Crucial for (batch, seq, feature) input
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # --- 4. Prediction Head ---
        self.output_layer = nn.Linear(embedding_dim, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        # Get the batch size for later use
        batch_size = x.shape[0]
        
        # 1. Get feature tokens from the tokenizer
        # -> tokens shape: (batch_size, num_features, embedding_dim)
        tokens = self.tokenizer(x)
        
        # 2. Prepend the [CLS] token to the sequence
        # -> cls_tokens shape: (batch_size, 1, embedding_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # -> full_sequence shape: (batch_size, num_features + 1, embedding_dim)
        full_sequence = torch.cat([cls_tokens, tokens], dim=1)

        # 3. Pass the full sequence through the Transformer Encoder
        # -> transformer_out shape: (batch_size, num_features + 1, embedding_dim)
        transformer_out = self.transformer_encoder(full_sequence)
        
        # 4. Isolate the output of the [CLS] token (it's the first one)
        # -> cls_output shape: (batch_size, embedding_dim)
        cls_output = transformer_out[:, 0]
        
        # 5. Pass the [CLS] token's output through the prediction head
        # -> logits shape: (batch_size, out_targets)
        logits = self.output_layer(cls_output)
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
        return {
            'out_targets': self.out_targets,
            'numerical_indices': self.numerical_indices,
            'categorical_map': self.categorical_map,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        num_features = len(self.numerical_indices) + len(self.categorical_map)

        # Build the architecture string part-by-part
        parts = [
            f"Tokenizer(features={num_features}, dim={self.embedding_dim})",
            "[CLS]",
            f"TransformerEncoder(layers={self.num_layers}, heads={self.num_heads})",
            f"PredictionHead(outputs={self.out_targets})"
        ]
        
        arch_str = " -> ".join(parts)
        
        return f"TabularTransformer(arch: {arch_str})"


class _FeatureTokenizer(nn.Module):
    """
    Transforms raw numerical and categorical features from any column order into a sequence of embeddings.
    """
    def __init__(self,
                 numerical_indices: List[int],
                 categorical_map: Dict[int, int],
                 embedding_dim: int):
        """
        Args:
            numerical_indices (List[int]): A list of column indices for the numerical features.
            categorical_map (Dict[int, int]): A dictionary mapping each categorical column index to its cardinality (number of unique categories).
            embedding_dim (int): The dimension for all feature embeddings.
        """
        super().__init__()
        
        # Unpack the dictionary into separate lists for indices and cardinalities
        self.categorical_indices = list(categorical_map.keys())
        cardinalities = list(categorical_map.values())
        
        self.numerical_indices = numerical_indices
        self.embedding_dim = embedding_dim
        
        # A learnable embedding for each numerical feature
        self.numerical_embeddings = nn.Parameter(torch.randn(len(numerical_indices), embedding_dim))
        
        # A standard embedding layer for each categorical feature
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=c, embedding_dim=embedding_dim) for c in cardinalities]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes features from a single input tensor and concatenates them
        into a sequence of tokens.
        """
        # Select the correct columns for each type using the stored indices
        x_numerical = x[:, self.numerical_indices].float()
        x_categorical = x[:, self.categorical_indices].long()

        # Process numerical features
        numerical_tokens = x_numerical.unsqueeze(-1) * self.numerical_embeddings
        
        # Process categorical features
        categorical_tokens = []
        for i, embed_layer in enumerate(self.categorical_embeddings):
            token = embed_layer(x_categorical[:, i]).unsqueeze(1)
            categorical_tokens.append(token)
        
        # Concatenate all tokens into a single sequence
        if not self.categorical_indices:
             all_tokens = numerical_tokens
        elif not self.numerical_indices:
             all_tokens = torch.cat(categorical_tokens, dim=1)
        else:
             all_categorical_tokens = torch.cat(categorical_tokens, dim=1)
             all_tokens = torch.cat([numerical_tokens, all_categorical_tokens], dim=1)
        
        return all_tokens


class _AttentionLayer(nn.Module):
    """
    Calculates attention weights and applies them to the input features, incorporating a residual connection for improved stability and performance.
    
    Returns both the final output and the weights for interpretability.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # The hidden layer size is a hyperparameter
        hidden_size = max(16, num_features // 4)
        
        # Learn to produce attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_features) # Output one score per feature
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, num_features)
        
        # Get one raw "importance" score per feature
        attention_scores = self.attention_net(x)
        
        # Apply the softmax module to get weights that sum to 1
        attention_weights = self.softmax(attention_scores)
        
        # Weighted features (attention mechanism's output)
        weighted_features = x * attention_weights
        
        # Residual connection
        residual_connection = x + weighted_features
        
        return residual_connection, attention_weights


class _MultiHeadAttentionLayer(nn.Module):
    """
    A wrapper for the standard `torch.nn.MultiheadAttention` layer.

    This layer treats the entire input feature vector as a single item in a
    sequence and applies self-attention to it. It is followed by a residual
    connection and layer normalization, which is a standard block in
    Transformer-style models.
    """
    def __init__(self, num_features: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Crucial for (batch, seq, feature) input
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, num_features)

        # nn.MultiheadAttention expects a sequence dimension.
        # We add a sequence dimension of length 1.
        # x_reshaped shape: (batch_size, 1, num_features)
        x_reshaped = x.unsqueeze(1)

        # Apply self-attention. query, key, and value are all the same.
        # attn_output shape: (batch_size, 1, num_features)
        # attn_weights shape: (batch_size, 1, 1)
        attn_output, attn_weights = self.attention(
            query=x_reshaped,
            key=x_reshaped,
            value=x_reshaped,
            need_weights=True,
            average_attn_weights=True # Average weights across heads
        )

        # Add residual connection and apply layer normalization (Post-LN)
        out = self.layer_norm(x + attn_output.squeeze(1))

        # Squeeze weights for a consistent output shape
        return out, attn_weights.squeeze()


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
        _LOGGER.error(f"Model '{model.__class__.__name__}' does not have a 'get_config()' method.")
        raise AttributeError() 

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
        _LOGGER.info(f"Architecture for '{model.__class__.__name__}' saved to '{path_dir.name}'")


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
        _LOGGER.error(f"Model class mismatch. File specifies '{saved_class_name}', but '{expected_model_class.__name__}' was expected.")
        raise ValueError()

    # Create an instance of the model using the provided class and config
    model = expected_model_class(**config)
    if verbose:
        _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
    return model


def info():
    _script_info(__all__)
