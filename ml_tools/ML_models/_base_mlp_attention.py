import torch
from torch import nn
from typing import Any

from .._core import get_logger

from ._base_save_load import _ArchitectureHandlerMixin


_LOGGER = get_logger("DragonModel: MLP")


__all__ = [
    "_BaseMLP",
    "_BaseAttention",
    "_AttentionLayer",
    "_MultiHeadAttentionLayer",
]


class _BaseMLP(nn.Module, _ArchitectureHandlerMixin):
    """
    A base class for Multilayer Perceptrons.
    
    Handles validation, configuration, and the creation of the core MLP layers,
    allowing subclasses to define their own pre-processing and forward pass.
    """
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: list[int], 
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

    def get_architecture_config(self) -> dict[str, Any]:
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

    def forward_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and attention weights."""
        # This logic is now shared and defined in one place
        x, attention_weights = self.attention(x) # type: ignore
        x = self.mlp(x)
        logits = self.output_layer(x)
        return logits, attention_weights


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

