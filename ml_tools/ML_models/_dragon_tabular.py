import torch
from torch import nn
from typing import Any

from ..schema import FeatureSchema

from .._core import get_logger
from ..keys._keys import SchemaKeys

from ._base_save_load import _ArchitectureBuilder


_LOGGER = get_logger("DragonTabularTransformer")


__all__ = [
    "DragonTabularTransformer"
]


class DragonTabularTransformer(_ArchitectureBuilder):
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
                The definitive FeatureSchema object.
            out_targets (int): 
                Number of output targets.
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
        # _ArchitectureBuilder init sets up self.model_hparams
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
    
    def get_architecture_config(self) -> dict[str, Any]:
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
            SchemaKeys.SCHEMA_DICT: schema_dict,
            'out_targets': self.out_targets,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        
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

