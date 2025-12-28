import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Literal

from ..schema import FeatureSchema
from .._core import get_logger
from ..keys._keys import SchemaKeys

from ._base_save_load import _ArchitectureBuilder
from ._models_advanced_helpers import (
    Embedding1dLayer,
    entmax15,
    entmoid15,
    sparsemax,
    sparsemoid,
    DenseODSTBlock,
)


_LOGGER = get_logger("DragonNodeModel")


__all__ = [
    "DragonNodeModel",
]

# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


class DragonNodeModel(_ArchitectureBuilder):
    """
    Native implementation of Neural Oblivious Decision Ensembles (NODE).
    
    The 'Dense' architecture concatenates the outputs of previous layers to the 
    features of subsequent layers, allowing for deep feature interaction learning.
    """
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": F.softmax,
    }
    
    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": torch.sigmoid,
    }

    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 24,
                 num_trees: int = 1024,
                 num_layers: int = 2,
                 tree_depth: int = 6,
                 additional_tree_output_dim: int = 3,
                 max_features: Optional[int] = None,
                 input_dropout: float = 0.0,
                 embedding_dropout: float = 0.0,
                 choice_function: Literal['entmax', 'sparsemax', 'softmax'] = 'entmax',
                 bin_function: Literal['entmoid', 'sparsemoid', 'sigmoid'] = 'entmoid',
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema (FeatureSchema): 
                Schema object containing feature names and types.
            out_targets (int): 
                Number of output targets.
            embedding_dim (int, optional): 
                Embedding dimension for categorical features.
                Suggested: 16 to 64.
            num_trees (int, optional): 
                Number of Oblivious Decision Trees per layer. NODE relies on a large number 
                of trees (wider layers) compared to standard forests.
                Suggested: 512 to 2048.
            num_layers (int, optional): 
                Number of DenseODST layers. Since layers are densely connected, deeper 
                networks increase memory usage significantly.
                Suggested: 2 to 5.
            tree_depth (int, optional): 
                Depth of the oblivious trees. Oblivious trees are symmetric, so 
                parameters scale with 2^depth.
                Suggested: 4 to 8.
            additional_tree_output_dim (int, optional): 
                Extra output channels per tree. These are used for internal representation 
                in deeper layers but discarded for the final prediction.
                Suggested: 1 to 5.
            max_features (int, optional): 
                Max features to keep in the dense connection to prevent explosion in 
                feature dimension for deeper layers. If None, keeps all.
            input_dropout (float, optional): 
                Dropout applied to the input of the Dense Block.
                Suggested: 0.0 to 0.2.
            embedding_dropout (float, optional): 
                Dropout applied specifically to embeddings.
                Suggested: 0.0 to 0.2.
            choice_function (str, optional): 
                Activation for feature selection. 'entmax' allows sparse feature selection.
                Options: 'entmax', 'sparsemax', 'softmax'. 
            bin_function (str, optional): 
                Activation for the soft binning steps.
                Options: 'entmoid', 'sparsemoid', 'sigmoid'.
            batch_norm_continuous (bool, optional): 
                If True, applies Batch Normalization to continuous features.
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # -- Configuration for saving --
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'num_trees': num_trees,
            'num_layers': num_layers,
            'tree_depth': tree_depth,
            'additional_tree_output_dim': additional_tree_output_dim,
            'max_features': max_features,
            'input_dropout': input_dropout,
            'embedding_dropout': embedding_dropout,
            'choice_function': choice_function,
            'bin_function': bin_function,
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
        
        embedding_dims = [(c, embedding_dim) for c in self.cardinalities]
        n_continuous = len(self.numerical_indices)
        
        self.embedding_layer = Embedding1dLayer(
            continuous_dim=n_continuous,
            categorical_embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        total_embedded_dim = n_continuous + sum([d for _, d in embedding_dims])
        
        # -- 2. Backbone (Dense ODST) --
        # The tree output dim includes the target dim + auxiliary dims for deep learning
        self.tree_dim = out_targets + additional_tree_output_dim
        
        self.backbone = DenseODSTBlock(
            input_dim=total_embedded_dim,
            num_trees=num_trees,
            num_layers=num_layers,
            tree_output_dim=self.tree_dim,
            max_features=max_features,
            input_dropout=input_dropout,
            flatten_output=False, # We want (Batch, Num_Layers * Num_Trees, Tree_Dim)
            depth=tree_depth,
            # Activations
            choice_function=self.ACTIVATION_MAP[choice_function],
            bin_function=self.BINARY_ACTIVATION_MAP[bin_function],
            # Init strategies (defaults)
            initialize_response_=nn.init.normal_,
            initialize_selection_logits_=nn.init.uniform_,
        )
        
        # Note: NODE has a fixed Head (averaging) which is defined in forward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embeddings
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. Backbone
        # Output shape: (Batch, Total_Trees, Tree_Dim)
        x = self.backbone(x)
        
        # 3. Head (Averaging)
        # We take the first 'out_targets' channels and average them across all trees
        # subset: x[..., :out_targets]
        # mean: .mean(dim=-2) -> average over Total_Trees dimension
        return x[..., :self.out_targets].mean(dim=-2)

    def data_aware_initialization(self, train_dataset, num_samples: int = 2000, verbose: int = 3):
        """
        Performs data-aware initialization for the ODST trees using a dataset.
        Crucial for NODE convergence.
        """
        # 1. Prepare Data
        if verbose >= 2:
            _LOGGER.info(f"Performing NODE data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device
            
        # 2. Extract Features
        # Fast path: If the dataset exposes the full feature tensor (like _PytorchDataset)
        if hasattr(train_dataset, "features") and isinstance(train_dataset.features, torch.Tensor):
             # Slice directly
             limit = min(len(train_dataset.features), num_samples)
             x_input = train_dataset.features[:limit]
        else:
            # Slow path: Iterate and stack (Generic Dataset)
            indices = range(min(len(train_dataset), num_samples))
            x_accum = []
            for i in indices:
                # Expecting (features, targets) tuple from standard datasets
                sample = train_dataset[i]
                if isinstance(sample, (tuple, list)):
                    x_accum.append(sample[0])
                elif isinstance(sample, dict) and 'features' in sample:
                    x_accum.append(sample['features'])
                elif isinstance(sample, dict) and 'x' in sample:
                    x_accum.append(sample['x'])
                else:
                    # Fallback: assume the sample itself is the feature
                    x_accum.append(sample)
            
            if not x_accum:
                if verbose >= 1:
                    _LOGGER.warning("Dataset empty or format unrecognized. Skipping NODE initialization.")
                return
                
            x_input = torch.stack(x_accum)
            
        x_input = x_input.to(device).float()
        
        # 3. Process features (Split -> Embed)
        x_cont = x_input[:, self.numerical_indices].float()
        x_cat = x_input[:, self.categorical_indices].long()
        
        with torch.no_grad():
            x_embedded = self.embedding_layer(x_cont, x_cat)
            
            # 4. Initialize Backbone
            if hasattr(self.backbone, 'initialize'):
                self.backbone.initialize(x_embedded)
                if verbose >= 2:
                    _LOGGER.info("NODE Initialization Complete. Ready to train.")
            else:
                if verbose >= 1:
                    _LOGGER.warning("NODE Backbone does not have an 'initialize' method. Skipping.")
            
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

