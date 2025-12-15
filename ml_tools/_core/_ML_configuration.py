from typing import Union, Optional, List, Any, Dict, Literal
from collections.abc import Mapping
import numpy as np

from ._schema import FeatureSchema
from ._script_info import _script_info
from ._logger import get_logger
from ._path_manager import sanitize_filename
from ._keys import MLTaskKeys


_LOGGER = get_logger("Configuration")


__all__ = [
    # --- Metrics Formats ---
    "RegressionMetricsFormat",
    "MultiTargetRegressionMetricsFormat",
    "BinaryClassificationMetricsFormat",
    "MultiClassClassificationMetricsFormat",
    "BinaryImageClassificationMetricsFormat",
    "MultiClassImageClassificationMetricsFormat",
    "MultiLabelBinaryClassificationMetricsFormat",
    "BinarySegmentationMetricsFormat",
    "MultiClassSegmentationMetricsFormat",
    "SequenceValueMetricsFormat",
    "SequenceSequenceMetricsFormat",
    
    # --- Finalize Configs ---
    "FinalizeBinaryClassification",
    "FinalizeBinarySegmentation",
    "FinalizeBinaryImageClassification",
    "FinalizeMultiClassClassification",
    "FinalizeMultiClassImageClassification",
    "FinalizeMultiClassSegmentation",
    "FinalizeMultiLabelBinaryClassification",
    "FinalizeMultiTargetRegression",
    "FinalizeRegression",
    "FinalizeObjectDetection",
    "FinalizeSequenceSequencePrediction",
    "FinalizeSequenceValuePrediction",
    
    # --- Model Parameter Configs ---
    "DragonMLPParams",
    "DragonAttentionMLPParams",
    "DragonMultiHeadAttentionNetParams",
    "DragonTabularTransformerParams",
    "DragonGateParams",
    "DragonNodeParams",
    "DragonTabNetParams",
    "DragonAutoIntParams",
    
    # --- Training Config ---
    "DragonTrainingConfig"
]


# --- Private base classes ---

class _BaseClassificationFormat:
    """
    [PRIVATE] Base configuration for single-label classification metrics.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 xtick_size: int=22,
                 ytick_size: int=22,
                 legend_size: int=26,
                 font_size: int=26,
                 cm_font_size: int=26) -> None:
        """
        Initializes the formatting configuration for single-label classification metrics.

        Args:
            cmap (str): The matplotlib colormap name for the confusion matrix
                and report heatmap.
                - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples'
                - Diverging options: 'coolwarm', 'viridis', 'plasma', 'inferno'
            
            ROC_PR_line (str): The color name or hex code for the line plotted
                on the ROC and Precision-Recall curves.
                - Common color names: 'darkorange', 'cornflowerblue', 'crimson', 'forestgreen'
                - Hex codes: '#FF6347', '#4682B4'
            
            calibration_bins (int): The number of bins to use when
                creating the calibration (reliability) plot.
            
            font_size (int): The base font size to apply to the plots.
            
            xtick_size (int): Font size for x-axis tick labels.
            
            ytick_size (int): Font size for y-axis tick labels.
            
            legend_size (int): Font size for plot legends.
            
            cm_font_size (int): Font size for the confusion matrix.
        
        <br>
        
        ### [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.calibration_bins = calibration_bins
        self.font_size = font_size
        self.xtick_size = xtick_size
        self.ytick_size = ytick_size
        self.legend_size = legend_size
        self.cm_font_size = cm_font_size
        
    def __repr__(self) -> str:
        parts = [
            f"cmap='{self.cmap}'",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"calibration_bins={self.calibration_bins}",
            f"font_size={self.font_size}",
            f"xtick_size={self.xtick_size}",
            f"ytick_size={self.ytick_size}",
            f"legend_size={self.legend_size}",
            f"cm_font_size={self.cm_font_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseMultiLabelFormat:
    """
    [PRIVATE] Base configuration for multi-label binary classification metrics.
    """
    def __init__(self,
                 cmap: str = "BuGn",
                 ROC_PR_line: str='darkorange',
                 font_size: int = 25,
                 xtick_size: int=20,
                 ytick_size: int=20,
                 legend_size: int=23) -> None:
        """
        Initializes the formatting configuration for multi-label classification metrics.

        Args:
            cmap (str): The matplotlib colormap name for the per-label
                    confusion matrices.
                    - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples'
                    - Diverging options: 'coolwarm', 'viridis', 'plasma', 'inferno'
        
            ROC_PR_line (str): The color name or hex code for the line plotted
                on the ROC and Precision-Recall curves (one for each label). 
                - Common color names: 'darkorange', 'cornflowerblue', 'crimson', 'forestgreen'
                - Hex codes: '#FF6347', '#4682B4'
            
            font_size (int): The base font size to apply to the plots.
            
            xtick_size (int): Font size for x-axis tick labels.
            
            ytick_size (int): Font size for y-axis tick labels.
            
            legend_size (int): Font size for plot legends.
            
        <br>
        
        ### [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.font_size = font_size
        self.xtick_size = xtick_size
        self.ytick_size = ytick_size
        self.legend_size = legend_size
        
    def __repr__(self) -> str:
        parts = [
            f"cmap='{self.cmap}'",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"font_size={self.font_size}",
            f"xtick_size={self.xtick_size}",
            f"ytick_size={self.ytick_size}",
            f"legend_size={self.legend_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseRegressionFormat:
    """
    [PRIVATE] Base configuration for regression metrics.
    """
    def __init__(self, 
                 font_size: int=26,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto',
                 xtick_size: int=22,
                 ytick_size: int=22) -> None:
        """
        Initializes the formatting configuration for regression metrics.

        Args:
            font_size (int): The base font size to apply to the plots.
            scatter_color (str): Matplotlib color for the scatter plot points.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            scatter_alpha (float): Alpha transparency for scatter plot points.
            ideal_line_color (str): Matplotlib color for the 'ideal' y=x line in the 
                True vs. Predicted plot.
                - Common color names: 'k', 'red', 'darkgrey', '#FF6347'
            residual_line_color (str): Matplotlib color for the y=0 line in the 
                Residual plot.
                - Common color names: 'red', 'blue', 'k', '#4682B4'
            hist_bins (int | str): The number of bins for the residuals histogram. 
                Defaults to 'auto' to use seaborn's automatic bin selection.
                - Options: 'auto', 'sqrt', 10, 20
            xtick_size (int): Font size for x-axis tick labels.
            ytick_size (int): Font size for y-axis tick labels.
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.font_size = font_size
        self.scatter_color = scatter_color
        self.scatter_alpha = scatter_alpha
        self.ideal_line_color = ideal_line_color
        self.residual_line_color = residual_line_color
        self.hist_bins = hist_bins
        self.xtick_size = xtick_size
        self.ytick_size = ytick_size
        
    def __repr__(self) -> str:
        parts = [
            f"font_size={self.font_size}",
            f"scatter_color='{self.scatter_color}'",
            f"scatter_alpha={self.scatter_alpha}",
            f"ideal_line_color='{self.ideal_line_color}'",
            f"residual_line_color='{self.residual_line_color}'",
            f"hist_bins='{self.hist_bins}'",
            f"xtick_size={self.xtick_size}",
            f"ytick_size={self.ytick_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseSegmentationFormat:
    """
    [PRIVATE] Base configuration for segmentation metrics.
    """
    def __init__(self,
                 heatmap_cmap: str = "BuGn",
                 cm_cmap: str = "Purples",
                 font_size: int = 16) -> None:
        """
        Initializes the formatting configuration for segmentation metrics.

        Args:
            heatmap_cmap (str): The matplotlib colormap name for the per-class
                metrics heatmap.
                - Sequential options: 'viridis', 'plasma', 'inferno', 'cividis'
                - Diverging options: 'coolwarm', 'bwr', 'seismic'
            cm_cmap (str): The matplotlib colormap name for the pixel-level
                confusion matrix.
                - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges'
            font_size (int): The base font size to apply to the plots.
        
        <br>
        
        ### [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        """
        self.heatmap_cmap = heatmap_cmap
        self.cm_cmap = cm_cmap
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"heatmap_cmap='{self.heatmap_cmap}'",
            f"cm_cmap='{self.cm_cmap}'",
            f"font_size={self.font_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseSequenceValueFormat:
    """
    [PRIVATE] Base configuration for sequence to value metrics.
    """
    def __init__(self, 
                 font_size: int=25,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto') -> None:
        """
        Initializes the formatting configuration for sequence to value metrics.

        Args:
            font_size (int): The base font size to apply to the plots.
            scatter_color (str): Matplotlib color for the scatter plot points.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            scatter_alpha (float): Alpha transparency for scatter plot points.
            ideal_line_color (str): Matplotlib color for the 'ideal' y=x line in the 
                True vs. Predicted plot.
                - Common color names: 'k', 'red', 'darkgrey', '#FF6347'
            residual_line_color (str): Matplotlib color for the y=0 line in the 
                Residual plot.
                - Common color names: 'red', 'blue', 'k', '#4682B4'
            hist_bins (int | str): The number of bins for the residuals histogram. 
                Defaults to 'auto' to use seaborn's automatic bin selection.
                - Options: 'auto', 'sqrt', 10, 20

        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.font_size = font_size
        self.scatter_color = scatter_color
        self.scatter_alpha = scatter_alpha
        self.ideal_line_color = ideal_line_color
        self.residual_line_color = residual_line_color
        self.hist_bins = hist_bins
        
    def __repr__(self) -> str:
        parts = [
            f"font_size={self.font_size}",
            f"scatter_color='{self.scatter_color}'",
            f"scatter_alpha={self.scatter_alpha}",
            f"ideal_line_color='{self.ideal_line_color}'",
            f"residual_line_color='{self.residual_line_color}'",
            f"hist_bins='{self.hist_bins}'"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseSequenceSequenceFormat:
    """
    [PRIVATE] Base configuration for sequence-to-sequence metrics.
    """
    def __init__(self,
                 font_size: int = 25,
                 grid_style: str = '--',
                 rmse_color: str = 'tab:blue',
                 rmse_marker: str = 'o-',
                 mae_color: str = 'tab:orange',
                 mae_marker: str = 's--'):
        """
        Initializes the formatting configuration for seq-to-seq metrics.

        Args:
            font_size (int): The base font size to apply to the plots.
            grid_style (str): Matplotlib linestyle for the plot grid.
                - Options: '--' (dashed), ':' (dotted), '-.' (dash-dot), '-' (solid)
            rmse_color (str): Matplotlib color for the RMSE line.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            rmse_marker (str): Matplotlib marker style for the RMSE line.
                - Options: 'o-' (circle), 's--' (square), '^:' (triangle), 'x' (x marker)
            mae_color (str): Matplotlib color for the MAE line.
                - Common color names: 'tab:orange', 'purple', 'black', '#FF6347'
            mae_marker (str): Matplotlib marker style for the MAE line.
                - Options: 's--', 'o-', 'v:', '+' (plus marker)
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        
        <br>
        
        ### [Matplotlib Linestyles](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
        
        <br>
        
        ### [Matplotlib Markers](https://matplotlib.org/stable/api/markers_api.html)
        """
        self.font_size = font_size
        self.grid_style = grid_style
        self.rmse_color = rmse_color
        self.rmse_marker = rmse_marker
        self.mae_color = mae_color
        self.mae_marker = mae_marker

    def __repr__(self) -> str:
        parts = [
            f"font_size={self.font_size}",
            f"grid_style='{self.grid_style}'",
            f"rmse_color='{self.rmse_color}'",
            f"mae_color='{self.mae_color}'"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseModelParams(Mapping):
    """
    [PRIVATE] Base class for model parameter configs.
    
    Inherits from Mapping to behave like a dictionary, enabling
    `**params` unpacking directly into model constructors.
    """
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __or__(self, other) -> Dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(self) | dict(other)
        return NotImplemented
    
    def __ror__(self, other) -> Dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(other) | dict(self)
        return NotImplemented

    def __repr__(self) -> str:
        """Returns a formatted multi-line string representation."""
        class_name = self.__class__.__name__
        # Format parameters for clean logging
        params = []
        for k, v in self.__dict__.items():
            # If value is huge (like FeatureSchema), use its own repr
            val_str = repr(v)
            params.append(f"  {k}={val_str}")
            
        params_str = ",\n".join(params)
        return f"{class_name}(\n{params_str}\n)"
    
    def to_log(self) -> Dict[str, Any]:
        """        
        Safely converts complex types (like FeatureSchema) to their string 
        representation for cleaner JSON logging.
        """
        clean_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FeatureSchema):
                # Force the repr() string, otherwise json.dump treats it as a list
                clean_dict[k] = repr(v)
            else:
                clean_dict[k] = v
        return clean_dict


# --- Public API classes ---

# ----------------------------
# Model Parameters Configurations
# ----------------------------

# --- Standard Models ---

class DragonMLPParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: List[int], 
                 drop_out: float = 0.2) -> None:
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out


class DragonAttentionMLPParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: List[int], 
                 drop_out: float = 0.2) -> None:
        self.in_features = in_features
        self.out_targets = out_targets
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out


class DragonMultiHeadAttentionNetParams(_BaseModelParams):
    def __init__(self, 
                 in_features: int, 
                 out_targets: int,
                 hidden_layers: List[int], 
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


# --- Training Configuration ---

class DragonTrainingConfig(_BaseModelParams):
    """
    Configuration object for the training process.
    
    Can be unpacked as a dictionary for logging or accessed as an object.
    """
    def __init__(self,
                 validation_size: float,
                 test_size: float,
                 initial_learning_rate: float,
                 batch_size: int,
                 random_state: int = 101,
                 early_stop_patience: Optional[int] = None,
                 scheduler_patience: Optional[int] = None,
                 scheduler_lr_factor: Optional[float] = None,
                 scheduler_threshold: Optional[float] = None) -> None:
        self.validation_size = validation_size
        self.test_size = test_size
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.early_stop_patience = early_stop_patience
        self.scheduler_patience = scheduler_patience
        self.scheduler_lr_factor = scheduler_lr_factor
        self.scheduler_threshold = scheduler_threshold


# ----------------------------
# Metrics Configurations
# ----------------------------

# Regression
class RegressionMetricsFormat(_BaseRegressionFormat):
    """
    Configuration for single-target regression.
    """
    def __init__(self, 
                 font_size: int=26,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto',
                 xtick_size: int=22,
                 ytick_size: int=22) -> None:
        super().__init__(font_size=font_size, 
                         scatter_color=scatter_color, 
                         scatter_alpha=scatter_alpha, 
                         ideal_line_color=ideal_line_color, 
                         residual_line_color=residual_line_color, 
                         hist_bins=hist_bins,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size)


# Multitarget regression
class MultiTargetRegressionMetricsFormat(_BaseRegressionFormat):
    """
    Configuration for multi-target regression.
    """
    def __init__(self, 
                 font_size: int=26,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto',
                 xtick_size: int=22,
                 ytick_size: int=22) -> None:
        super().__init__(font_size=font_size, 
                         scatter_color=scatter_color, 
                         scatter_alpha=scatter_alpha, 
                         ideal_line_color=ideal_line_color, 
                         residual_line_color=residual_line_color, 
                         hist_bins=hist_bins,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size)


# Classification
class BinaryClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for binary classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=26,
                 xtick_size: int=22,
                 ytick_size: int=22,
                 legend_size: int=26,
                 cm_font_size: int=26
                 ) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size,
                         legend_size=legend_size,
                         cm_font_size=cm_font_size)


class MultiClassClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for multi-class classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=26,
                 xtick_size: int=22,
                 ytick_size: int=22,
                 legend_size: int=26,
                 cm_font_size: int=26
                 ) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size,
                         legend_size=legend_size,
                         cm_font_size=cm_font_size)

class BinaryImageClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for binary image classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=26,
                 xtick_size: int=22,
                 ytick_size: int=22,
                 legend_size: int=26,
                 cm_font_size: int=26
                 ) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size,
                         legend_size=legend_size,
                         cm_font_size=cm_font_size)

class MultiClassImageClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for multi-class image classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=26,
                 xtick_size: int=22,
                 ytick_size: int=22,
                 legend_size: int=26,
                 cm_font_size: int=26
                 ) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size,
                         legend_size=legend_size,
                         cm_font_size=cm_font_size)

# Multi-Label classification
class MultiLabelBinaryClassificationMetricsFormat(_BaseMultiLabelFormat):
    """
    Configuration for multi-label binary classification.
    """
    def __init__(self,
                 cmap: str = "BuGn",
                 ROC_PR_line: str='darkorange',
                 font_size: int = 25,
                 xtick_size: int=20,
                 ytick_size: int=20,
                 legend_size: int=23
                 ) -> None:
        super().__init__(cmap=cmap,
                         ROC_PR_line=ROC_PR_line, 
                         font_size=font_size,
                         xtick_size=xtick_size,
                         ytick_size=ytick_size,
                         legend_size=legend_size)

# Segmentation
class BinarySegmentationMetricsFormat(_BaseSegmentationFormat):
    """
    Configuration for binary segmentation.
    """
    def __init__(self,
                 heatmap_cmap: str = "BuGn",
                 cm_cmap: str = "Purples",
                 font_size: int = 16) -> None:
        super().__init__(heatmap_cmap=heatmap_cmap, 
                         cm_cmap=cm_cmap, 
                         font_size=font_size)


class MultiClassSegmentationMetricsFormat(_BaseSegmentationFormat):
    """
    Configuration for multi-class segmentation.
    """
    def __init__(self,
                 heatmap_cmap: str = "BuGn",
                 cm_cmap: str = "Purples",
                 font_size: int = 16) -> None:
        super().__init__(heatmap_cmap=heatmap_cmap, 
                         cm_cmap=cm_cmap, 
                         font_size=font_size)


# Sequence 
class SequenceValueMetricsFormat(_BaseSequenceValueFormat):
    """
    Configuration for sequence-to-value prediction.
    """
    def __init__(self, 
                 font_size: int=25,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto') -> None:
        super().__init__(font_size=font_size, 
                         scatter_color=scatter_color, 
                         scatter_alpha=scatter_alpha, 
                         ideal_line_color=ideal_line_color, 
                         residual_line_color=residual_line_color, 
                         hist_bins=hist_bins)


class SequenceSequenceMetricsFormat(_BaseSequenceSequenceFormat):
    """
    Configuration for sequence-to-sequence prediction.
    """
    def __init__(self,
                 font_size: int = 25,
                 grid_style: str = '--',
                 rmse_color: str = 'tab:blue',
                 rmse_marker: str = 'o-',
                 mae_color: str = 'tab:orange',
                 mae_marker: str = 's--'):
        super().__init__(font_size=font_size, 
                         grid_style=grid_style, 
                         rmse_color=rmse_color, 
                         rmse_marker=rmse_marker, 
                         mae_color=mae_color, 
                         mae_marker=mae_marker)


# -------- Finalize classes --------
class _FinalizeModelTraining:
    """
    Base class for finalizing model training.

    This class is not intended to be instantiated directly. Instead, use one of its specific subclasses.
    """
    def __init__(self,
                 filename: str,
                 ) -> None:
        self.filename = _validate_string(string=filename, attribute_name="filename", extension=".pth")
        self.target_name: Optional[str] = None
        self.target_names: Optional[list[str]] = None
        self.classification_threshold: Optional[float] = None
        self.class_map: Optional[dict[str,int]] = None
        self.initial_sequence: Optional[np.ndarray] = None
        self.sequence_length: Optional[int] = None
        self.task: str = 'UNKNOWN'


class FinalizeRegression(_FinalizeModelTraining):
    """Parameters for finalizing a single-target regression model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.task = MLTaskKeys.REGRESSION
    
    
class FinalizeMultiTargetRegression(_FinalizeModelTraining):
    """Parameters for finalizing a multi-target regression model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
        """
        super().__init__(filename=filename)
        safe_names = [_validate_string(string=target_name, attribute_name="All target names") for target_name in target_names]
        self.target_names = safe_names
        self.task = MLTaskKeys.MULTITARGET_REGRESSION


class FinalizeBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 classification_threshold: float,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_CLASSIFICATION


class FinalizeMultiClassClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class classification model."""
    def __init__(self,
                 filename: str,
                 target_name: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_name (str): The name of the target variable.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.target_name = _validate_string(string=target_name, attribute_name="Target name")
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_CLASSIFICATION
    
    
class FinalizeBinaryImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a binary image classification model."""
    def __init__(self,
                 filename: str,
                 classification_threshold: float,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            classification_threshold (float): The cutoff threshold for
                classifying as the positive class.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_IMAGE_CLASSIFICATION


class FinalizeMultiClassImageClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class image classification model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            class_map (dict[str,int]): A dictionary mapping class names (str)
                to their integer representations (e.g., {'cat': 0, 'dog': 1}).
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
    
    
class FinalizeMultiLabelBinaryClassification(_FinalizeModelTraining):
    """Parameters for finalizing a multi-label binary classification model."""
    def __init__(self,
                 filename: str,
                 target_names: list[str],
                 classification_threshold: float,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            target_names (list[str]): A list of names for the target variables.
            classification_threshold (float): The cutoff threshold for classifying as the positive class.
        """
        super().__init__(filename=filename)
        safe_names = [_validate_string(string=target_name, attribute_name="All target names") for target_name in target_names]
        self.target_names = safe_names
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.task = MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION


class FinalizeBinarySegmentation(_FinalizeModelTraining):
    """Parameters for finalizing a binary segmentation model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int],
                 classification_threshold: float,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            classification_threshold (float): The cutoff threshold for classifying as the positive class (mask).
        """
        super().__init__(filename=filename)
        self.classification_threshold = _validate_threshold(classification_threshold)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.BINARY_SEGMENTATION
    
    
class FinalizeMultiClassSegmentation(_FinalizeModelTraining):
    """Parameters for finalizing a multi-class segmentation model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.MULTICLASS_SEGMENTATION


class FinalizeObjectDetection(_FinalizeModelTraining):
    """Parameters for finalizing an object detection model."""
    def __init__(self,
                 filename: str,
                 class_map: dict[str,int]
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
        """
        super().__init__(filename=filename)
        self.class_map = _validate_class_map(class_map)
        self.task = MLTaskKeys.OBJECT_DETECTION


class FinalizeSequenceSequencePrediction(_FinalizeModelTraining):
    """Parameters for finalizing a sequence-to-sequence prediction model."""
    def __init__(self,
                 filename: str,
                 last_training_sequence: np.ndarray,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            last_training_sequence (np.ndarray): The last sequence from the training data, needed to start predictions.
        """
        super().__init__(filename=filename)
        
        if not isinstance(last_training_sequence, np.ndarray):
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got {type(last_training_sequence)}.")
            raise TypeError()
        
        if last_training_sequence.ndim == 1:
            # It's already 1D, (N,). This is valid.
            self.initial_sequence = last_training_sequence
        elif last_training_sequence.ndim == 2:
            # Handle both (1, N) and (N, 1)
            if last_training_sequence.shape[0] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            elif last_training_sequence.shape[1] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            else:
                _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
                raise ValueError()
        else:
            # It's 3D or more, which is not supported
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
            raise ValueError()
        
        # Save the length of the validated 1D sequence
        self.sequence_length = len(self.initial_sequence) # type: ignore
        self.task = MLTaskKeys.SEQUENCE_SEQUENCE


class FinalizeSequenceValuePrediction(_FinalizeModelTraining):
    """Parameters for finalizing a sequence-to-value prediction model."""
    def __init__(self,
                 filename: str,
                 last_training_sequence: np.ndarray,
                 ) -> None:
        """Initializes the finalization parameters.

        Args:
            filename (str): The name of the file to be saved.
            last_training_sequence (np.ndarray): The last sequence from the training data, needed to start predictions.
        """
        super().__init__(filename=filename)
        
        if not isinstance(last_training_sequence, np.ndarray):
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got {type(last_training_sequence)}.")
            raise TypeError()
        
        if last_training_sequence.ndim == 1:
            # It's already 1D, (N,). This is valid.
            self.initial_sequence = last_training_sequence
        elif last_training_sequence.ndim == 2:
            # Handle both (1, N) and (N, 1)
            if last_training_sequence.shape[0] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            elif last_training_sequence.shape[1] == 1:
                self.initial_sequence = last_training_sequence.flatten()
            else:
                _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
                raise ValueError()
        else:
            # It's 3D or more, which is not supported
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
            raise ValueError()
        
        # Save the length of the validated 1D sequence
        self.sequence_length = len(self.initial_sequence) # type: ignore
        self.task = MLTaskKeys.SEQUENCE_VALUE


def _validate_string(string: str, attribute_name: str, extension: Optional[str]=None) -> str:
    """Helper for finalize classes"""
    if not isinstance(string, str):
        _LOGGER.error(f"{attribute_name} must be a string.")
        raise TypeError()

    if extension:
        safe_name = sanitize_filename(string)
        
        if not safe_name.endswith(extension):
            safe_name += extension
    else:
        safe_name = string
            
    return safe_name

def _validate_threshold(threshold: float):
    """Helper for finalize classes"""
    if not isinstance(threshold, float):
        _LOGGER.error(f"Classification threshold must be a float.")
        raise TypeError()
    elif threshold < 0.1 or threshold > 0.9:
        _LOGGER.error(f"Classification threshold must be in the range [0.1, 0.9]")
        raise ValueError()
    
    return threshold

def _validate_class_map(map_dict: dict[str, int]):
    """Helper for finalize classes"""
    if not isinstance(map_dict, dict):
        _LOGGER.error(f"Class map must be a dictionary, but got {type(map_dict)}.")
        raise TypeError()
    
    if not map_dict:
        _LOGGER.error("Class map dictionary cannot be empty.")
        raise ValueError()

    for key, val in map_dict.items():
        if not isinstance(key, str):
            _LOGGER.error(f"All keys in the class map must be strings, but found key: {key} ({type(key)}).")
            raise TypeError()
        if not isinstance(val, int):
            _LOGGER.error(f"All values in the class map must be integers, but for key '{key}' found value: {val} ({type(val)}).")
            raise TypeError()
            
    return map_dict

def info():
    _script_info(__all__)
