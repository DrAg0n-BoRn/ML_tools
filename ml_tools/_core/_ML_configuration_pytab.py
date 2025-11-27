from typing import Literal

from ._ML_configuration import _BaseModelParams
from ._schema import FeatureSchema
from ._script_info import _script_info


__all__ = [
    "PyTabGateParams",
    "PyTabNodeParams",
    "PyTabTabNetParams",
    "PyTabAutoIntParams",
]


class PyTabGateParams(_BaseModelParams):
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 gflu_stages: int = 4,
                 num_trees: int = 20,
                 tree_depth: int = 4,
                 dropout: float = 0.1) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.gflu_stages = gflu_stages
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.dropout = dropout


class PyTabNodeParams(_BaseModelParams):
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 num_trees: int = 1024,
                 num_layers: int = 2,
                 tree_depth: int = 6,
                 dropout: float = 0.1,
                 backend_function: Literal['softmax', 'entmax15'] = 'softmax'
                 ) -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.num_trees = num_trees
        self.num_layers = num_layers
        self.tree_depth = tree_depth
        self.dropout = dropout
        self.backend_function = backend_function


class PyTabTabNetParams(_BaseModelParams):
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
                 mask_type: Literal['sparsemax', 'entmax', 'softmax'] = 'sparsemax') -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type


class PyTabAutoIntParams(_BaseModelParams):
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 num_heads: int = 2,
                 num_attn_blocks: int = 3,
                 attn_dropout: float = 0.1,
                 has_residuals: bool = True,
                 deep_layers: bool = True,
                 layers: str = "128-64-32") -> None:
        self.schema = schema
        self.out_targets = out_targets
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.has_residuals = has_residuals
        self.deep_layers = deep_layers
        self.layers = layers


def info():
    _script_info(__all__)
