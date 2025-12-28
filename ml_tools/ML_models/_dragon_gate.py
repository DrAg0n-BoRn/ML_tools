import torch
import torch.nn as nn
from typing import Any, Literal

from ..schema import FeatureSchema
from .._core import get_logger
from ..keys._keys import SchemaKeys

from ._base_save_load import _ArchitectureBuilder
from ._models_advanced_helpers import (
    Embedding1dLayer,
    GatedFeatureLearningUnit,
    NeuralDecisionTree,
    entmax15,
    entmoid15,
    sparsemax,
    sparsemoid,
    t_softmax,
    SimpleLinearHead,
    _GateHead
)


_LOGGER = get_logger("DragonGateModel")


__all__ = [
    "DragonGateModel",
    ]

# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


class DragonGateModel(_ArchitectureBuilder):
    """
    Native implementation of the Gated Additive Tree Ensemble (GATE).
    
    Combines Gated Feature Learning Units (GFLU) for feature interaction learning
    with Differentiable Decision Trees for prediction.
    """
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": lambda x: nn.functional.softmax(x, dim=-1),
        "t-softmax": t_softmax,
    }
    
    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": torch.sigmoid,
    }

    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 16,
                 gflu_stages: int = 6,
                 gflu_dropout: float = 0.1,
                 num_trees: int = 20,
                 tree_depth: int = 4,
                 tree_dropout: float = 0.1,
                 chain_trees: bool = False,
                 tree_wise_attention: bool = True,
                 tree_wise_attention_dropout: float = 0.1,
                 binning_activation: Literal['entmoid', 'sparsemoid', 'sigmoid'] = "entmoid",
                 feature_mask_function: Literal['entmax', 'sparsemax', 'softmax', 't-softmax'] = "entmax",
                 share_head_weights: bool = True,
                 batch_norm_continuous: bool = True):
        """
        Args:
            schema (FeatureSchema): 
                Schema object containing feature names and types.
            out_targets (int): 
                Number of output targets (e.g., 1 for regression/binary, N for multi-class).
            embedding_dim (int, optional): 
                Embedding dimension for categorical features. 
                Suggested: 8 to 64.
            gflu_stages (int, optional): 
                Number of Gated Feature Learning Unit stages in the backbone.
                Higher values allow learning deeper feature interactions.
                Suggested: 2 to 10.
            gflu_dropout (float, optional): 
                Dropout rate applied within GFLU stages. 
                Suggested: 0.0 to 0.3.
            num_trees (int, optional): 
                Number of Neural Decision Trees to use in the ensemble.
                Suggested: 10 to 50.
            tree_depth (int, optional): 
                Depth of the decision trees. Deeper trees capture more complex logic 
                but may overfit. 
                Suggested: 3 to 6.
            tree_dropout (float, optional): 
                Dropout rate applied to the tree leaves. 
                Suggested: 0.1 to 0.3.
            chain_trees (bool, optional): 
                If True, feeds the output of tree T-1 into tree T (Boosting-style). 
                If False, trees run in parallel (Bagging-style). 
            tree_wise_attention (bool, optional): 
                If True, applies Self-Attention across the outputs of the trees 
                to weigh their contributions dynamically. 
            tree_wise_attention_dropout (float, optional): 
                Dropout rate for the tree-wise attention mechanism.
                Suggested: 0.1.
            binning_activation (str, optional): 
                Activation function for the soft binning in trees. 
                Options: 'entmoid' (sparse), 'sparsemoid', 'sigmoid'. 
            feature_mask_function (str, optional): 
                Activation function for feature selection/masking.
                Options: 'entmax' (sparse), 'sparsemax', 'softmax', 't-softmax'.
            share_head_weights (bool, optional): 
                If True, all trees share the same linear projection head weights.
            batch_norm_continuous (bool, optional): 
                If True, applies Batch Normalization to continuous features before embedding.
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # -- Configuration for saving --
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'gflu_stages': gflu_stages,
            'gflu_dropout': gflu_dropout,
            'num_trees': num_trees,
            'tree_depth': tree_depth,
            'tree_dropout': tree_dropout,
            'chain_trees': chain_trees,
            'tree_wise_attention': tree_wise_attention,
            'tree_wise_attention_dropout': tree_wise_attention_dropout,
            'binning_activation': binning_activation,
            'feature_mask_function': feature_mask_function,
            'share_head_weights': share_head_weights,
            'batch_norm_continuous': batch_norm_continuous
        }

        # -- 1. Setup Data Processing --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        embedding_dims = [(c, embedding_dim) for c in self.cardinalities]
        n_continuous = len(self.numerical_indices)
        
        # -- 2. Embedding Layer --
        self.embedding_layer = Embedding1dLayer(
            continuous_dim=n_continuous,
            categorical_embedding_dims=embedding_dims,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        # Calculate total feature dimension after embedding
        total_embedded_cat_dim = sum([d for _, d in embedding_dims])
        self.n_features = n_continuous + total_embedded_cat_dim
        
        # -- 3. GFLU Backbone --
        self.gflu_stages = gflu_stages
        if gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=gflu_stages,
                feature_mask_function=self.ACTIVATION_MAP[feature_mask_function],
                dropout=gflu_dropout,
                feature_sparsity=0.3, # Standard default
                learnable_sparsity=True
            )
            
        # -- 4. Neural Decision Trees --
        self.num_trees = num_trees
        self.chain_trees = chain_trees
        self.tree_depth = tree_depth
        
        if num_trees > 0:
            # Calculate input dim for trees (chaining adds to input)
            tree_input_dim = self.n_features
            
            self.trees = nn.ModuleList()
            for _ in range(num_trees):
                tree = NeuralDecisionTree(
                    depth=tree_depth,
                    n_features=tree_input_dim,
                    dropout=tree_dropout,
                    binning_activation=self.BINARY_ACTIVATION_MAP[binning_activation],
                    feature_mask_function=self.ACTIVATION_MAP[feature_mask_function],
                )
                self.trees.append(tree)
                if chain_trees:
                    # Next tree sees original features + output of this tree (2^depth leaves)
                    tree_input_dim += 2**tree_depth

            self.tree_output_dim = 2**tree_depth
            
            # Optional: Tree-wise Attention
            self.tree_wise_attention = tree_wise_attention
            if tree_wise_attention:
                self.tree_attention = nn.MultiheadAttention(
                    embed_dim=self.tree_output_dim,
                    num_heads=1,
                    batch_first=False, # (Seq, Batch, Feature) standard for PyTorch Attn
                    dropout=tree_wise_attention_dropout
                )
        else:
            self.tree_output_dim = self.n_features

        # -- 5. Prediction Head --
        if num_trees > 0:
            self.head = _GateHead(
                input_dim=self.tree_output_dim,
                output_dim=out_targets,
                num_trees=num_trees,
                share_head_weights=share_head_weights
            )
        else:
            # Fallback if no trees (just GFLU -> Linear)
            self.head = SimpleLinearHead(self.n_features, out_targets)
            # Add T0 manually for consistency if needed, but SimpleLinear covers bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embeddings
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. GFLU
        if self.gflu_stages > 0:
            x = self.gflus(x)
            
        # 3. Trees
        if self.num_trees > 0:
            tree_outputs = []
            tree_input = x
            
            for i in range(self.num_trees):
                # Tree returns (leaf_nodes, feature_masks)
                # leaf_nodes shape: (Batch, 2^depth)
                leaf_nodes, _ = self.trees[i](tree_input)
                
                tree_outputs.append(leaf_nodes.unsqueeze(-1))
                
                if self.chain_trees:
                    tree_input = torch.cat([tree_input, leaf_nodes], dim=1)
            
            # Stack: (Batch, Output_Dim_Tree, Num_Trees)
            tree_outputs = torch.cat(tree_outputs, dim=-1)
            
            # 4. Attention
            if self.tree_wise_attention:
                # Permute for MultiheadAttention: (Num_Trees, Batch, Output_Dim_Tree)
                # Treating 'Trees' as the sequence length
                attn_input = tree_outputs.permute(2, 0, 1)
                
                attn_output, _ = self.tree_attention(attn_input, attn_input, attn_input)
                
                # Permute back: (Batch, Output_Dim_Tree, Num_Trees)
                tree_outputs = attn_output.permute(1, 2, 0)
                
            # 5. Head
            return self.head(tree_outputs)
            
        else:
            # No trees, just linear on top of GFLU
            return self.head(x)

    def data_aware_initialization(self, train_dataset, num_samples: int = 2000, verbose: int = 3):
        """
        Performs data-aware initialization for the global bias T0.
        This often speeds up convergence significantly.
        """
        # 1. Prepare Data
        if verbose >= 2:
            _LOGGER.info(f"Performing GATE data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device
            
        # 2. Extract Targets
        # Fast path: direct tensor access (Works with DragonDataset/_PytorchDataset)
        if hasattr(train_dataset, "labels") and isinstance(train_dataset.labels, torch.Tensor):
            limit = min(len(train_dataset.labels), num_samples)
            targets = train_dataset.labels[:limit]
        else:
            # Slow path: Iterate
            indices = range(min(len(train_dataset), num_samples))
            y_accum = []
            for i in indices:
                # Expecting (features, targets) tuple
                sample = train_dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    # Standard (X, y) tuple
                    y_val = sample[1]
                elif isinstance(sample, dict):
                    # Try common keys
                    y_val = sample.get('target', sample.get('y', None))
                else:
                    y_val = None
                
                if y_val is not None:
                    # Ensure it's a tensor
                    if not isinstance(y_val, torch.Tensor):
                        y_val = torch.tensor(y_val)
                    y_accum.append(y_val)

            if not y_accum:
                if verbose >= 1:
                    _LOGGER.warning("Could not extract targets for T0 initialization. Skipping.")
                return
            
            targets = torch.stack(y_accum)

        targets = targets.to(device).float()
        
        with torch.no_grad():
            if self.num_trees > 0:
                # Initialize T0 to mean of targets
                mean_target = torch.mean(targets, dim=0)
                
                # Check shapes to avoid broadcasting errors
                if self.head.T0.shape == mean_target.shape:
                    self.head.T0.data = mean_target
                    if verbose >= 2:
                        _LOGGER.info(f"GATE Initialization Complete. Ready to train.")
                elif self.head.T0.numel() == 1 and mean_target.numel() == 1: # type: ignore
                    # scalar case
                    self.head.T0.data = mean_target.view(self.head.T0.shape) # type: ignore
                    if verbose >= 2:
                        _LOGGER.info("GATE Initialization Complete. Ready to train.")
                else:
                    _LOGGER.debug(f"Target shape mismatch for T0 init. Model: {self.head.T0.shape}, Data: {mean_target.shape}")
                    if verbose >= 1:
                        _LOGGER.warning(f"GATE initialization skipped due to shape mismatch:\n    Model: {self.head.T0.shape}\n    Data: {mean_target.shape}")

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
    
