from typing import Union, Optional
import numpy as np

from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import sanitize_filename


__all__ = [
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
    "FinalizeSequencePrediction"
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
                 font_size: int=16) -> None:
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
        
        <br>
        
        ### [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.calibration_bins = calibration_bins
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"cmap='{self.cmap}'",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"calibration_bins={self.calibration_bins}",
            f"font_size={self.font_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseMultiLabelFormat:
    """
    [PRIVATE] Base configuration for multi-label binary classification metrics.
    """
    def __init__(self,
                 cmap: str = "BuGn",
                 ROC_PR_line: str='darkorange',
                 font_size: int = 16) -> None:
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
            
        <br>
        
        ### [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        
        <br>
        
        ### [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
        """
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"cmap='{self.cmap}'",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"font_size={self.font_size}"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class _BaseRegressionFormat:
    """
    [PRIVATE] Base configuration for regression metrics.
    """
    def __init__(self, 
                 font_size: int=16,
                 scatter_color: str='tab:blue',
                 scatter_alpha: float=0.6,
                 ideal_line_color: str='k',
                 residual_line_color: str='red',
                 hist_bins: Union[int, str] = 'auto') -> None:
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
                 font_size: int=16,
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
                 font_size: int = 16,
                 plot_figsize: tuple[int, int] = (10, 6),
                 grid_style: str = '--',
                 rmse_color: str = 'tab:blue',
                 rmse_marker: str = 'o-',
                 mae_color: str = 'tab:orange',
                 mae_marker: str = 's--'):
        """
        Initializes the formatting configuration for seq-to-seq metrics.

        Args:
            font_size (int): The base font size to apply to the plots.
            plot_figsize (Tuple[int, int]): Figure size for the plot.
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
        self.plot_figsize = plot_figsize
        self.grid_style = grid_style
        self.rmse_color = rmse_color
        self.rmse_marker = rmse_marker
        self.mae_color = mae_color
        self.mae_marker = mae_marker

    def __repr__(self) -> str:
        parts = [
            f"font_size={self.font_size}",
            f"plot_figsize={self.plot_figsize}",
            f"grid_style='{self.grid_style}'",
            f"rmse_color='{self.rmse_color}'",
            f"mae_color='{self.mae_color}'"
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"

# --- Public API classes ---

# Regression
class RegressionMetricsFormat(_BaseRegressionFormat):
    """
    Configuration for single-target regression.
    """
    def __init__(self, 
                 font_size: int=16,
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


# Multitarget regression
class MultiTargetRegressionMetricsFormat(_BaseRegressionFormat):
    """
    Configuration for multi-target regression.
    """
    def __init__(self, 
                 font_size: int=16,
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


# Classification
class BinaryClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for binary classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=16) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size)


class MultiClassClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for multi-class classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=16) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size)


class BinaryImageClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for binary image classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=16) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size)


class MultiClassImageClassificationMetricsFormat(_BaseClassificationFormat):
    """
    Configuration for multi-class image classification.
    """
    def __init__(self, 
                 cmap: str="BuGn",
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=16) -> None:
        super().__init__(cmap=cmap, 
                         ROC_PR_line=ROC_PR_line, 
                         calibration_bins=calibration_bins, 
                         font_size=font_size)


# Multi-Label classification
class MultiLabelBinaryClassificationMetricsFormat(_BaseMultiLabelFormat):
    """
    Configuration for multi-label binary classification.
    """
    def __init__(self,
                 cmap: str = "BuGn",
                 ROC_PR_line: str='darkorange',
                 font_size: int = 16) -> None:
        super().__init__(cmap=cmap,
                         ROC_PR_line=ROC_PR_line, 
                         font_size=font_size)


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
                 font_size: int=16,
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
                 font_size: int = 16,
                 plot_figsize: tuple[int, int] = (10, 6),
                 grid_style: str = '--',
                 rmse_color: str = 'tab:blue',
                 rmse_marker: str = 'o-',
                 mae_color: str = 'tab:orange',
                 mae_marker: str = 's--'):
        super().__init__(font_size=font_size, 
                         plot_figsize=plot_figsize, 
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


class FinalizeSequencePrediction(_FinalizeModelTraining):
    """Parameters for finalizing a sequence prediction model."""
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
            # It's 2D, check for shape (1, N)
            if last_training_sequence.shape[0] == 1:
                # Shape is (1, N), flatten to (N,)
                self.initial_sequence = last_training_sequence.flatten()
            else:
                # Shape is (N, 1) or (N, M), which is invalid
                _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
                raise ValueError()
        else:
            # It's 3D or more, which is not supported
            _LOGGER.error(f"The last training sequence must be a 1D numpy array, got shape {last_training_sequence.shape}.")
            raise ValueError()
        
        # Save the length of the validated 1D sequence
        self.sequence_length = len(self.initial_sequence)


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
