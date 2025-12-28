import torch
import torch.nn as nn
from typing import Any, Literal

from ..schema import FeatureSchema
from .._core import get_logger
from ..keys._keys import SchemaKeys

from ._base_save_load import _ArchitectureBuilder
from ._models_advanced_helpers import (
    FeatTransformer, 
    AttentiveTransformer, 
    initialize_non_glu,
)


_LOGGER = get_logger("DragonTabNet")


__all__ = [
    "DragonTabNet"
]

# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


class DragonTabNet(_ArchitectureBuilder):
    """
    Native implementation of TabNet (Attentive Interpretable Tabular Learning).
    
    Includes the Initial Splitter, Ghost Batch Norm, and GLU scaling.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 virtual_batch_size: int = 128,
                 momentum: float = 0.02,
                 mask_type: Literal['sparsemax', 'entmax', 'softmax'] = 'sparsemax',
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema (FeatureSchema): 
                Schema object containing feature names and types.
            out_targets (int): 
                Number of output targets.
            n_d (int, optional): 
                Dimension of the prediction layer (decision step).
                Suggested: 8 to 64.
            n_a (int, optional): 
                Dimension of the attention layer (masking step).
                Suggested: 8 to 64.
            n_steps (int, optional): 
                Number of sequential attention steps (architecture depth).
                Suggested: 3 to 10.
            gamma (float, optional): 
                Relaxation parameter for sparsity in the mask.
                Suggested: 1.0 to 2.0.
            n_independent (int, optional): 
                Number of independent Gated Linear Unit (GLU) layers in each block.
                Suggested: 1 to 5.
            n_shared (int, optional): 
                Number of shared GLU layers across all blocks.
                Suggested: 1 to 5.
            virtual_batch_size (int, optional): 
                Batch size for Ghost Batch Normalization.
                Suggested: 128 to 1024.
            momentum (float, optional): 
                Momentum for Batch Normalization.
                Suggested: 0.01 to 0.4.
            mask_type (str, optional): 
                Masking function to use. 'sparsemax' enforces sparsity.
                Options: 'sparsemax', 'entmax', 'softmax'.
            batch_norm_continuous (bool, optional): 
                If True, applies Batch Normalization to continuous features before processing.
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # Save config
        self.model_hparams = {
            'n_d': n_d,
            'n_a': n_a,
            'n_steps': n_steps,
            'gamma': gamma,
            'n_independent': n_independent,
            'n_shared': n_shared,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum,
            'mask_type': mask_type,
            'batch_norm_continuous': batch_norm_continuous
        }
        
        # -- 1. Setup Input Features --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        # Standard TabNet Embeddings:
        # We use a simple embedding for each categorical feature and concat with continuous.
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, 1) for card in self.cardinalities
        ])
        
        self.n_continuous = len(self.numerical_indices)
        self.input_dim = self.n_continuous + len(self.cardinalities)
        
        # -- 2. TabNet Backbone Components --
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = 1e-15
        
        # Initial BN
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        # Shared GLU Layers
        if n_shared > 0:
            self.shared_feat_transform = nn.ModuleList()
            for i in range(n_shared):
                if i == 0:
                    self.shared_feat_transform.append(
                        nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    self.shared_feat_transform.append(
                        nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )
        else:
            self.shared_feat_transform = None

        # Initial Splitter
        # This processes the input BEFORE the first step to generate the initial attention vector 'a'
        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            self.shared_feat_transform,
            n_glu_independent=n_independent,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        # Steps
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                self.shared_feat_transform,
                n_glu_independent=n_independent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.input_dim, # We assume group_dim = input_dim (no grouping)
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
                mask_type=mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

        # -- 3. Final Mapping Head --
        self.final_mapping = nn.Linear(n_d, out_targets, bias=False)
        initialize_non_glu(self.final_mapping, n_d, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -- Preprocessing --
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        cat_list = []
        for i, embed in enumerate(self.cat_embeddings):
            cat_list.append(embed(x_cat[:, i])) # (B, 1)
        
        if cat_list:
            x_in = torch.cat([x_cont, *cat_list], dim=1)
        else:
            x_in = x_cont
            
        
        # -- TabNet Encoder Pass --
        x_bn = self.initial_bn(x_in)
        # Initial Split
        # The splitter produces [d, a]. We only need 'a' to start the loop.
        att = self.initial_splitter(x_bn)[:, self.n_d :]
        priors = torch.ones(x_bn.shape, device=x.device)
        out_accumulated = 0
        self.regularization_loss = 0

        for step in range(self.n_steps):
            # 1. Attention
            mask = self.att_transformers[step](priors, att)
            # 2. Accumulate sparsity loss matching original implementation
            loss = torch.sum(torch.mul(mask, torch.log(mask + self.epsilon)), dim=1)
            self.regularization_loss += torch.mean(loss)
            # 3. Update Prior
            priors = torch.mul(self.gamma - mask, priors)
            # 4. Masking
            masked_x = torch.mul(mask, x_bn)
            # 5. Feature Transformer
            out = self.feat_transformers[step](masked_x)
            # 6. Split Output
            d = nn.ReLU()(out[:, :self.n_d])
            att = out[:, self.n_d:]
            # 7. Accumulate Decision
            out_accumulated = out_accumulated + d

        self.regularization_loss /= self.n_steps
        return self.final_mapping(out_accumulated)
    
    def data_aware_initialization(self, train_dataset, num_samples: int = 2000, verbose: int = 3):
        """  
        TabNet does not require data-aware initialization. Method Implemented for compatibility.
        """
        if verbose >= 2:
            _LOGGER.info("TabNet does not require data-aware initialization. Skipping.")

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

