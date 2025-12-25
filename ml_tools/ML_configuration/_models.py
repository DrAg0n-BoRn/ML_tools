from typing import Optional, Literal

from ..schema import FeatureSchema

from ._base_model_config import _BaseModelParams


__all__ = [    
    # --- Model Parameter Configs ---
    "DragonMLPParams",
    "DragonAttentionMLPParams",
    "DragonMultiHeadAttentionNetParams",
    "DragonTabularTransformerParams",
    "DragonGateParams",
    "DragonNodeParams",
    "DragonTabNetParams",
    "DragonAutoIntParams",
]


# ----------------------------
# Model Parameters Configurations
# ----------------------------

# --- Standard Models ---

class DragonMLPParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: list[int], 
                 drop_out: float = 0.2) -> None:
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out


class DragonAttentionMLPParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: list[int], 
                 drop_out: float = 0.2) -> None:
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out


class DragonMultiHeadAttentionNetParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: list[int], 
                 drop_out: float = 0.2,
                 num_heads: int = 4, 
                 attention_dropout: float = 0.1) -> None:
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout


class DragonTabularTransformerParams(_BaseModelParams):
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.2) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

# --- Advanced Models ---

class DragonGateParams(_BaseModelParams):
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
                 batch_norm_continuous: bool = True) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.tree_dropout = tree_dropout
        self.chain_trees = chain_trees
        self.tree_wise_attention = tree_wise_attention
        self.tree_wise_attention_dropout = tree_wise_attention_dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self.share_head_weights = share_head_weights
        self.batch_norm_continuous = batch_norm_continuous


class DragonNodeParams(_BaseModelParams):
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
                 batch_norm_continuous: bool = False) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.num_trees = num_trees
        self.num_layers = num_layers
        self.tree_depth = tree_depth
        self.additional_tree_output_dim = additional_tree_output_dim
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.embedding_dropout = embedding_dropout
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.batch_norm_continuous = batch_norm_continuous


class DragonAutoIntParams(_BaseModelParams):
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
                 batch_norm_continuous: bool = False) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.attn_embed_dim = attn_embed_dim
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.has_residuals = has_residuals
        self.attention_pooling = attention_pooling
        self.deep_layers = deep_layers
        self.layers = layers
        self.activation = activation
        self.embedding_dropout = embedding_dropout
        self.batch_norm_continuous = batch_norm_continuous


class DragonTabNetParams(_BaseModelParams):
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
                 batch_norm_continuous: bool = False) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.batch_norm_continuous = batch_norm_continuous

