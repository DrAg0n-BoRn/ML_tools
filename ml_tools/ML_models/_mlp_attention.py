import torch
from torch import nn
from typing import Any

from ._base_mlp_attention import _BaseMLP, _BaseAttention, _AttentionLayer, _MultiHeadAttentionLayer


__all__ = [
    "DragonMLP",
    "DragonAttentionMLP",
    "DragonMultiHeadAttentionNet",
]


class DragonMLP(_BaseMLP):
    """
    Creates a versatile Multilayer Perceptron (MLP) for regression or classification tasks.
    """
    def __init__(self, in_features: int, out_targets: int,
                 hidden_layers: list[int] = [256, 128], drop_out: float = 0.2) -> None:
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
                 hidden_layers: list[int] = [256, 128], drop_out: float = 0.2) -> None:
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
        for layer in self.mlp[1:]: # type: ignore
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
                 hidden_layers: list[int] = [256, 128], drop_out: float = 0.2,
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

    def get_architecture_config(self) -> dict[str, Any]:
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

