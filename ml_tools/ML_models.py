import torch
from torch import nn
from typing import List, Union, Tuple, Dict, Any
from pathlib import Path
import json

from ._logger import _LOGGER
from .path_manager import make_fullpath
from ._script_info import _script_info
from ._keys import PytorchModelArchitectureKeys
from ._schema import FeatureSchema


__all__ = [
    "DragonMLP",
    "DragonAttentionMLP",
    "DragonMultiHeadAttentionNet",
    "DragonTabularTransformer"
]


class _ArchitectureHandlerMixin:
    """
    A mixin class to provide save and load functionality for model architectures.
    """
    def save(self: nn.Module, directory: Union[str, Path], verbose: bool = True): # type: ignore
        """Saves the model's architecture to a JSON file."""
        if not hasattr(self, 'get_architecture_config'):
            _LOGGER.error(f"Model '{self.__class__.__name__}' must have a 'get_architecture_config()' method to use this functionality.")
            raise AttributeError()

        path_dir = make_fullpath(directory, make=True, enforce="directory")
        
        json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
        
        full_path = path_dir / json_filename

        config = {
            PytorchModelArchitectureKeys.MODEL: self.__class__.__name__,
            PytorchModelArchitectureKeys.CONFIG: self.get_architecture_config() # type: ignore
        }

        with open(full_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        if verbose:
            _LOGGER.info(f"Architecture for '{self.__class__.__name__}' saved as '{full_path.name}'")

    @classmethod
    def load(cls: type, file_or_dir: Union[str, Path], verbose: bool = True) -> nn.Module:
        """Loads a model architecture from a JSON file. If a directory is provided, the function will attempt to load a JSON file inside."""
        user_path = make_fullpath(file_or_dir)
        
        if user_path.is_dir():
            json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
            target_path = make_fullpath(user_path / json_filename, enforce="file")
        elif user_path.is_file():
            target_path = user_path
        else:
            _LOGGER.error(f"Invalid path: '{file_or_dir}'")
            raise IOError()

        with open(target_path, 'r') as f:
            saved_data = json.load(f)

        saved_class_name = saved_data[PytorchModelArchitectureKeys.MODEL]
        config = saved_data[PytorchModelArchitectureKeys.CONFIG]

        if saved_class_name != cls.__name__:
            _LOGGER.error(f"Model class mismatch. File specifies '{saved_class_name}', but '{cls.__name__}' was expected.")
            raise ValueError()

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model


class _BaseMLP(nn.Module, _ArchitectureHandlerMixin):
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

    def get_architecture_config(self) -> Dict[str, Any]:
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


class _BaseAttention(_BaseMLP):
    """
    Abstract base class for MLP models that incorporate an attention mechanism
    before the main MLP layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # By default, models inheriting this do not have the flag.
        self.attention = None
        self.has_interpretable_attention = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the standard forward pass."""
        logits, _attention_weights = self.forward_attention(x)
        return logits

    def forward_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and attention weights."""
        # This logic is now shared and defined in one place
        x, attention_weights = self.attention(x) # type: ignore
        x = self.mlp(x)
        logits = self.output_layer(x)
        return logits, attention_weights


class DragonMLP(_BaseMLP):
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
        
        return self._repr_helper(name="DragonMLP", mlp_layers=layer_sizes)


class DragonAttentionMLP(_BaseAttention):
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
        self.has_interpretable_attention = True
    
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        # Start with the input features and the attention marker
        arch = [str(self.in_features), "[Attention]"]

        # Find all other linear layers in the MLP 
        for layer in self.mlp[1:]:
            if isinstance(layer, nn.Linear):
                arch.append(str(layer.in_features))
        
        return self._repr_helper(name="DragonAttentionMLP", mlp_layers=arch)


class DragonMultiHeadAttentionNet(_BaseAttention):
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

    def get_architecture_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
        config = super().get_architecture_config()
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
        
        return f"DragonMultiHeadAttentionNet(arch: {arch_str})"


class DragonTabularTransformer(nn.Module, _ArchitectureHandlerMixin):
    """
    A Transformer-based model for tabular data tasks.
    
    This model uses a Feature Tokenizer to convert all input features into a
    sequence of embeddings, prepends a [CLS] token, and processes the
    sequence with a standard Transformer Encoder.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.2):
        """
        Args:
            schema (FeatureSchema): 
                The definitive schema object created by `data_exploration.finalize_feature_schema()`.
            out_targets (int): 
                Number of output targets (1 for regression).
            embedding_dim (int): 
                The dimension for all feature embeddings. Must be divisible by num_heads. Common values: (64, 128, 192, 256, etc.)
            num_heads (int): 
                The number of heads in the multi-head attention mechanism. Common values: (4, 8, 16)
            num_layers (int): 
                The number of sub-encoder-layers in the transformer encoder. Common values: (4, 8, 12)
            dropout (float): 
                The dropout value.
                
        ## Note:
        
        **Embedding Dimension:** "Width" of the model. It's the N-dimension vector that will be used to represent each one of the features.
            - Each continuous feature gets its own learnable N-dimension vector.
            - Each categorical feature gets an embedding table that maps every category (e.g., "color=red", "color=blue") to a unique N-dimension vector.
            
        **Attention Heads:** Controls the "Multi-Head Attention" mechanism. Instead of looking at all the feature interactions at once, the model splits its attention into N parallel heads.
            - Embedding Dimensions get divided by the number of Attention Heads, resulting in the dimensions assigned per head.

        **Number of Layers:** "Depth" of the model. Number of identical `TransformerEncoderLayer` blocks that are stacked on top of each other.
            - Layer 1: The attention heads find simple, direct interactions between the features.
            - Layer 2: Takes the output of Layer 1 and finds interactions between those interactions and so on.
            - Trade-off: More layers are more powerful but are slower to train and more prone to overfitting. If the training loss goes down but the validation loss goes up, you might have too many layers (or need more dropout).
            
        """
        super().__init__()
        
         # --- Get info from schema ---
        in_features = len(schema.feature_names)
        categorical_index_map = schema.categorical_index_map

         # --- Validation ---
        if categorical_index_map and (max(categorical_index_map.keys()) >= in_features):
            _LOGGER.error(f"A categorical index ({max(categorical_index_map.keys())}) is out of bounds for the provided input features ({in_features}).")
            raise ValueError()
        
        # --- Save configuration ---
        self.schema = schema # <-- Save the whole schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # --- 1. Feature Tokenizer (now takes the schema) ---
        self.tokenizer = _FeatureTokenizer(
            schema=schema,
            embedding_dim=embedding_dim
        )
        
        # --- 2. CLS Token ---
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
    
    def get_architecture_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
        # Deconstruct schema into a JSON-friendly dict
        # Tuples are saved as lists
        schema_dict = {
            'feature_names': self.schema.feature_names,
            'continuous_feature_names': self.schema.continuous_feature_names,
            'categorical_feature_names': self.schema.categorical_feature_names,
            'categorical_index_map': self.schema.categorical_index_map,
            'categorical_mappings': self.schema.categorical_mappings
        }

        return {
            'schema_dict': schema_dict,
            'out_targets': self.out_targets,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
    
    @classmethod
    def load(cls: type, file_or_dir: Union[str, Path], verbose: bool = True) -> nn.Module:
        """Loads a model architecture from a JSON file."""
        user_path = make_fullpath(file_or_dir)
        
        if user_path.is_dir():
            json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
            target_path = make_fullpath(user_path / json_filename, enforce="file")
        elif user_path.is_file():
            target_path = user_path
        else:
            _LOGGER.error(f"Invalid path: '{file_or_dir}'")
            raise IOError()

        with open(target_path, 'r') as f:
            saved_data = json.load(f)

        saved_class_name = saved_data[PytorchModelArchitectureKeys.MODEL]
        config = saved_data[PytorchModelArchitectureKeys.CONFIG]

        if saved_class_name != cls.__name__:
            _LOGGER.error(f"Model class mismatch. File specifies '{saved_class_name}', but '{cls.__name__}' was expected.")
            raise ValueError()

        # --- RECONSTRUCTION LOGIC ---
        if 'schema_dict' not in config:
            _LOGGER.error("Invalid architecture file: missing 'schema_dict'. This file may be from an older version.")
            raise ValueError("Missing 'schema_dict' in config.")
            
        schema_data = config.pop('schema_dict')
        
        # Re-hydrate the categorical_index_map
        # JSON saves all dict keys as strings, so we must convert them back to int.
        raw_index_map = schema_data['categorical_index_map']
        if raw_index_map is not None:
            rehydrated_index_map = {int(k): v for k, v in raw_index_map.items()}
        else:
            rehydrated_index_map = None

        # Re-hydrate the FeatureSchema object
        # JSON deserializes tuples as lists, so we must convert them back.
        schema = FeatureSchema(
            feature_names=tuple(schema_data['feature_names']),
            continuous_feature_names=tuple(schema_data['continuous_feature_names']),
            categorical_feature_names=tuple(schema_data['categorical_feature_names']),
            categorical_index_map=rehydrated_index_map,
            categorical_mappings=schema_data['categorical_mappings']
        )
        
        config['schema'] = schema
        # --- End Reconstruction ---

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model
        
    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        # Build the architecture string part-by-part
        parts = [
            f"Tokenizer(features={len(self.schema.feature_names)}, dim={self.embedding_dim})",
            "[CLS]",
            f"TransformerEncoder(layers={self.num_layers}, heads={self.num_heads})",
            f"PredictionHead(outputs={self.out_targets})"
        ]
        
        arch_str = " -> ".join(parts)
        
        return f"DragonTabularTransformer(arch: {arch_str})"


class _FeatureTokenizer(nn.Module):
    """
    Transforms raw numerical and categorical features from any column order 
    into a sequence of embeddings.
    """
    def __init__(self,
                 schema: FeatureSchema,
                 embedding_dim: int):
        """
        Args:
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            embedding_dim (int): 
                The dimension for all feature embeddings.
        """
        super().__init__()
        
        # --- Get info from schema ---
        categorical_map = schema.categorical_index_map
        
        if categorical_map:
            # Unpack the dictionary into separate lists
            self.categorical_indices = list(categorical_map.keys())
            cardinalities = list(categorical_map.values())
        else:
            self.categorical_indices = []
            cardinalities = []
        
        # Derive numerical indices by finding what's not categorical
        all_indices = set(range(len(schema.feature_names)))
        categorical_indices_set = set(self.categorical_indices)
        self.numerical_indices = sorted(list(all_indices - categorical_indices_set))
        
        self.embedding_dim = embedding_dim
        
        # A learnable embedding for each numerical feature
        self.numerical_embeddings = nn.Parameter(torch.randn(len(self.numerical_indices), embedding_dim))
        
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
            # x_categorical[:, i] selects the i-th categorical column
            # (e.g., all values for the 'color' feature)
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


def info():
    _script_info(__all__)
