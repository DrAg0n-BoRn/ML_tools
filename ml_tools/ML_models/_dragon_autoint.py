import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from ..schema import FeatureSchema
from .._core import get_logger
from ..keys._keys import SchemaKeys

from ._base_save_load import _ArchitectureBuilder
from ._models_advanced_helpers import (
    Embedding2dLayer,
)


_LOGGER = get_logger("DragonAutoInt")


__all__ = [
    "DragonAutoInt",
]

# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


class DragonAutoInt(_ArchitectureBuilder):
    """
    Native implementation of AutoInt (Automatic Feature Interaction Learning).
    
    Maps categorical and continuous features into a shared embedding space,
    then uses Multi-Head Self-Attention to learn high-order feature interactions.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 attn_embed_dim: int = 32,
                 num_heads: int = 2,
                 num_attn_blocks: int = 3,
                 attn_dropout: float = 0.1,
                 has_residuals: bool = True,
                 attention_pooling: bool = True,
                 deep_layers: bool = True,
                 layers: str = "128-64-32",
                 activation: str = "ReLU",
                 embedding_dropout: float = 0.0,
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema (FeatureSchema): 
                Schema object containing feature names and types.
            out_targets (int): 
                Number of output targets.
            embedding_dim (int, optional): 
                Initial embedding dimension for features. 
                Suggested: 16 to 64.
            attn_embed_dim (int, optional): 
                Projection dimension for the attention mechanism.
                Suggested: 16 to 64.
            num_heads (int, optional): 
                Number of attention heads. 
                Suggested: 2 to 8.
            num_attn_blocks (int, optional): 
                Number of self-attention layers (depth of interaction learning).
                Suggested: 2 to 5.
            attn_dropout (float, optional): 
                Dropout rate within the attention blocks.
                Suggested: 0.0 to 0.2.
            has_residuals (bool, optional): 
                If True, adds residual connections (ResNet style) to attention blocks.
            attention_pooling (bool, optional): 
                If True, concatenates outputs of all attention blocks (DenseNet style).
                If False, uses only the output of the last block.
            deep_layers (bool, optional): 
                If True, adds a standard MLP (Deep Layers) before the attention mechanism
                to process features initially.
            layers (str, optional): 
                Hyphen-separated string for MLP layer sizes if deep_layers is True.
            activation (str, optional): 
                Activation function for the MLP layers.
            embedding_dropout (float, optional): 
                Dropout applied to the initial feature embeddings.
            batch_norm_continuous (bool, optional): 
                If True, applies Batch Normalization to continuous features.
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'attn_embed_dim': attn_embed_dim,
            'num_heads': num_heads,
            'num_attn_blocks': num_attn_blocks,
            'attn_dropout': attn_dropout,
            'has_residuals': has_residuals,
            'attention_pooling': attention_pooling,
            'deep_layers': deep_layers,
            'layers': layers,
            'activation': activation,
            'embedding_dropout': embedding_dropout,
            'batch_norm_continuous': batch_norm_continuous
        }
        
        # -- 1. Setup Embeddings --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        n_continuous = len(self.numerical_indices)
        
        self.embedding_layer = Embedding2dLayer(
            continuous_dim=n_continuous,
            categorical_cardinality=self.cardinalities,
            embedding_dim=embedding_dim,
            embedding_dropout=embedding_dropout,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        # -- 2. Deep Layers (Optional MLP) --
        curr_units = embedding_dim
        self.deep_layers_mod = None
        
        if deep_layers:
            layers_list = []
            layer_sizes = [int(x) for x in layers.split("-")]
            activation_fn = getattr(nn, activation, nn.ReLU)
            
            for units in layer_sizes:
                layers_list.append(nn.Linear(curr_units, units))
                
                # Changed BatchNorm1d to LayerNorm to handle (Batch, Tokens, Embed) shape correctly
                layers_list.append(nn.LayerNorm(units)) 

                layers_list.append(activation_fn())
                layers_list.append(nn.Dropout(embedding_dropout))
                curr_units = units
            
            self.deep_layers_mod = nn.Sequential(*layers_list)
            
        # -- 3. Attention Backbone --
        self.attn_proj = nn.Linear(curr_units, attn_embed_dim)
        
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=attn_embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout
            )
            for _ in range(num_attn_blocks)
        ])
        
        # Residuals
        self.has_residuals = has_residuals
        self.attention_pooling = attention_pooling
        
        if has_residuals:
            # If pooling, we project input to match the concatenated output size
            # If not pooling, we project input to match the single block output size
            res_dim = attn_embed_dim * num_attn_blocks if attention_pooling else attn_embed_dim
            self.V_res_embedding = nn.Linear(curr_units, res_dim)
            
        # -- 4. Output Dimension Calculation --
        num_features = n_continuous + len(self.cardinalities)
        
        # Output is flattened: (Num_Features * Attn_Dim)
        final_dim = num_features * attn_embed_dim
        if attention_pooling:
            final_dim = final_dim * num_attn_blocks
            
        self.output_dim = final_dim
        self.head = nn.Linear(final_dim, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embed -> (Batch, Num_Features, Embed_Dim)
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. Deep Layers
        if self.deep_layers_mod:
            x = self.deep_layers_mod(x)
            
        # 3. Attention Projection -> (Batch, Num_Features, Attn_Dim)
        cross_term = self.attn_proj(x)
        
        # Transpose for MultiheadAttention (Seq, Batch, Embed)
        cross_term = cross_term.transpose(0, 1)
        
        attention_ops = []
        for self_attn in self.self_attns:
            # Self Attention: Query=Key=Value=cross_term
            # Output: (Seq, Batch, Embed)
            out, _ = self_attn(cross_term, cross_term, cross_term)
            cross_term = out # Sequential connection
            if self.attention_pooling:
                attention_ops.append(out)
                
        if self.attention_pooling:
            # Concatenate all attention outputs along the embedding dimension
            cross_term = torch.cat(attention_ops, dim=-1)
            
        # Transpose back -> (Batch, Num_Features, Final_Attn_Dim)
        cross_term = cross_term.transpose(0, 1)
        
        # 4. Residual Connection
        if self.has_residuals:
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
            
        # 5. Flatten and Head
        # ReLU before flattening as per original implementation
        cross_term = F.relu(cross_term)
        cross_term = cross_term.reshape(cross_term.size(0), -1)
        
        return self.head(cross_term)
    
    def data_aware_initialization(self, train_dataset, num_samples: int = 2000, verbose: int = 3):
        """
        Performs data-aware initialization for the final head bias.
        """
        # 1. Prepare Data
        if verbose >= 2:
            _LOGGER.info(f"Performing AutoInt data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device

        # 2. Extract Targets
        if hasattr(train_dataset, "labels") and isinstance(train_dataset.labels, torch.Tensor):
             limit = min(len(train_dataset.labels), num_samples)
             targets = train_dataset.labels[:limit]
        else:
            indices = range(min(len(train_dataset), num_samples))
            y_accum = []
            for i in indices:
                sample = train_dataset[i]
                # Handle tuple (X, y) or dict
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    y_val = sample[1]
                elif isinstance(sample, dict):
                    y_val = sample.get('target', sample.get('y', None))
                else:
                    y_val = None
                
                if y_val is not None:
                    if not isinstance(y_val, torch.Tensor):
                        y_val = torch.tensor(y_val)
                    y_accum.append(y_val)

            if not y_accum:
                if verbose >= 1:
                    _LOGGER.warning("Could not extract targets for AutoInt initialization. Skipping.")
                return
            
            targets = torch.stack(y_accum)

        targets = targets.to(device).float()
        
        # 3. Initialize Head Bias
        with torch.no_grad():
            mean_target = torch.mean(targets, dim=0)
            if hasattr(self.head, 'bias') and self.head.bias is not None:
                if self.head.bias.shape == mean_target.shape:
                    self.head.bias.data = mean_target
                    if verbose >= 2:
                        _LOGGER.info("AutoInt Initialization Complete. Ready to train.")
                    _LOGGER.debug(f"Initialized AutoInt head bias to {mean_target.cpu().numpy()}")
                elif self.head.bias.numel() == 1 and mean_target.numel() == 1:
                    self.head.bias.data = mean_target.view(self.head.bias.shape)
                    if verbose >= 2:
                        _LOGGER.info("AutoInt Initialization Complete. Ready to train.")
                    _LOGGER.debug(f"Initialized AutoInt head bias to {mean_target.item()}")
            else:
                if verbose >= 1:
                    _LOGGER.warning("AutoInt Head does not have a bias parameter. Skipping initialization.")
    
    def get_architecture_config(self) -> dict[str, Any]:
        """Returns the full configuration of the model."""
        schema_dict = {
            'feature_names': self.schema.feature_names,
            'continuous_feature_names': self.schema.continuous_feature_names,
            'categorical_feature_names': self.schema.categorical_feature_names,
            'categorical_index_map': self.schema.categorical_index_map,
            'categorical_mappings': self.schema.categorical_mappings
        }
        
        config = {
            SchemaKeys.SCHEMA_DICT: schema_dict,
            'out_targets': self.out_targets,
            **self.model_hparams
        }
        return config

