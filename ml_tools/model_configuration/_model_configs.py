from typing import Optional, Literal, Union

from ..schema import FeatureSchema

from ..ML_configuration._base_model_config import _BaseModelParams


__all__ = [    
    # Standard Models
    "DragonMLPParams",
    "DragonAttentionMLPParams",
    "DragonMultiHeadAttentionNetParams",
    "DragonTabularTransformerParams",
    "DragonGateParams",
    "DragonNodeParams",
    "DragonTabNetParams",
    "DragonAutoIntParams",
    # Tabular Autoencoder
    "DragonAutoencoderParams",
    "DragonAutoencoderV2Params",
    # DiT guided and standard
    "DragonDiTParams",
    "DragonDiTV2Params",
    # Sequence Models
    "DragonSequenceLSTMParams",
    "DragonSequenceTransformerParams",
    # Image Classification Models
    "DragonResNetParams",
    "DragonEfficientNetParams",
    "DragonVGGParams",
    # Image Segmentation Models
    "DragonFCNParams",
    "DragonDeepLabv3Params",
    # Object Detection Models
    "DragonFastRCNNParams",
]


# ----------------------------
# Model Parameters Configurations
# ----------------------------

# --- Standard Models ---

class DragonMLPParams(_BaseModelParams):
    """Parameters for a standard Multi-Layer Perceptron (MLP) model."""
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
    """Parameters for an Attention-based Multi-Layer Perceptron (MLP) model."""
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
    """Parameters for a Multi-Head Attention Network model."""
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
    """Parameters for a Dragon Tabular Transformer model."""
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
    """Parameters for a Dragon Gate model."""
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
    """Parameters for a Dragon Node model."""
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 24,
                 num_trees: int = 256,
                 num_layers: int = 1,
                 tree_depth: int = 4,
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
    """Parameters for a Dragon AutoInt model."""
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
    """Parameters for a Dragon TabNet model."""
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

# Tabular Autoencoder

class DragonAutoencoderParams(_BaseModelParams):
    """Parameters for a Dragon Autoencoder model."""
    def __init__(self, *,
                 schema: FeatureSchema,
                 embedding_dim: int = 64,
                 fourier_sigma: float = 1.0) -> None:
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.fourier_sigma = fourier_sigma


class DragonAutoencoderV2Params(_BaseModelParams):
    """Parameters for a Dragon Autoencoder V2 model."""
    def __init__(self, *,
                 schema: FeatureSchema,
                 embedding_dim: int = 64,
                 numerical_embedding_type: Literal['fourier', 'ple'] = 'ple', 
                 fourier_sigma: float = 1.0,
                 ple_bins: int = 100,
                 transformer_depth: int = 2,
                 transformer_heads: int = 4) -> None:
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.numerical_embedding_type = numerical_embedding_type
        self.fourier_sigma = fourier_sigma
        self.ple_bins = ple_bins
        self.transformer_depth = transformer_depth
        self.transformer_heads = transformer_heads
        

# DiT guided and standard

class DragonDiTParams(_BaseModelParams):
    """
    Parameters for a Guided and Unguided Dragon DiT model.
    """
    def __init__(self, *,
                 embed_dim: int, 
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2) -> None:
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.depth = depth
        
class DragonDiTV2Params(_BaseModelParams):
    """
    Parameters for a Guided and Unguided Dragon DiT V2 model with Optimal Transport Flow Matching.
    """
    def __init__(self, *,
                 embed_dim: int, 
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2) -> None:
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.depth = depth


# Sequence Models

class DragonSequenceLSTMParams(_BaseModelParams):
    """Parameters for a Dragon Sequence LSTM model."""
    def __init__(self, *,
                schema: FeatureSchema,
                targets: list[str],
                prediction_mode: Union[Literal["autoregressive-sequence-to-sequence", 
                                         "autoregressive-sequence-to-value", 
                                         "exogenous-sequence-to-sequence", 
                                         "exogenous-sequence-to-value"], str],
                sequence_length: int,
                hidden_size: int = 100,
                recurrent_layers: int = 2,
                dropout: float = 0.1):
        self.schema = schema
        self.targets = targets
        self.prediction_mode = prediction_mode
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.dropout = dropout
        
class DragonSequenceTransformerParams(_BaseModelParams):
    """Parameters for a Dragon Sequence Transformer model."""
    def __init__(self, *,
                schema: FeatureSchema,
                targets: list[str],
                prediction_mode: Union[Literal["autoregressive-sequence-to-sequence", 
                                         "autoregressive-sequence-to-value", 
                                         "exogenous-sequence-to-sequence", 
                                         "exogenous-sequence-to-value"], str],
                sequence_length: int,
                d_model: int = 128,
                nhead: int = 4,
                num_layers: int = 3,
                dim_feedforward: int = 512,
                dropout: float = 0.1):
        self.schema = schema
        self.targets = targets
        self.prediction_mode = prediction_mode
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        
#############
# Computer vision models
##############

# Image classification

class DragonResNetParams(_BaseModelParams):
    """Parameters for a Dragon ResNet model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = "resnet152",
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained


class DragonEfficientNetParams(_BaseModelParams):
    """Parameters for a Dragon EfficientNet model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = 'efficientnet_b7',
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained


class DragonVGGParams(_BaseModelParams):
    """Parameters for a Dragon VGG model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = "vgg19_bn",
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained


# Image segmentation

class DragonFCNParams(_BaseModelParams):
    """Parameters for a Dragon FCN model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = 'fcn_resnet101',
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained


class DragonDeepLabv3Params(_BaseModelParams):
    """Parameters for a Dragon DeepLabv3 model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = 'deeplabv3_resnet101',
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained


# Object detection

class DragonFastRCNNParams(_BaseModelParams):
    """Parameters for a Dragon Fast R-CNN model."""
    def __init__(self, *,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = 'fasterrcnn_resnet50_fpn_v2',
                 init_with_pretrained: bool = False) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_with_pretrained = init_with_pretrained

