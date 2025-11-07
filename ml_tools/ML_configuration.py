from typing import Optional, Union
from ._script_info import _script_info


__all__ = [
    "ClassificationMetricsFormat",
    "MultiClassificationMetricsFormat",
    "RegressionMetricsFormat",
    "SegmentationMetricsFormat",
    "SequenceValueMetricsFormat",
    "SequenceSequenceMetricsFormat"
]


class ClassificationMetricsFormat:
    """
    Optional configuration for classification tasks.
    """
    def __init__(self, 
                 cmap: str="Blues",
                 class_map: Optional[dict[str,int]]=None, 
                 ROC_PR_line: str='darkorange',
                 calibration_bins: int=15, 
                 font_size: int=16) -> None:
        """
        Initializes the formatting configuration for single-label classification metrics.

        Args:
            cmap (str): The matplotlib colormap name for the confusion matrix
                and report heatmap. Defaults to "Blues".
                - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples'
                - Diverging options: 'coolwarm', 'viridis', 'plasma', 'inferno'
            
            class_map (dict[str,int] | None): A dictionary mapping 
                class string names to their integer indices (e.g., {'cat': 0, 'dog': 1}). 
                This is used to label the axes of the confusion matrix and classification 
                report correctly. Defaults to None.
            
            ROC_PR_line (str): The color name or hex code for the line plotted
                on the ROC and Precision-Recall curves. Defaults to 'darkorange'.
                - Common color names: 'darkorange', 'cornflowerblue', 'crimson', 'forestgreen'
                - Hex codes: '#FF6347', '#4682B4'
            
            calibration_bins (int): The number of bins to use when
                creating the calibration (reliability) plot. Defaults to 15.
            
            font_size (int): The base font size to apply to the plots. Defaults to 16.
        
        <br>
        
        ## [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        """
        self.cmap = cmap
        self.class_map = class_map
        self.ROC_PR_line = ROC_PR_line
        self.calibration_bins = calibration_bins
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"cmap='{self.cmap}'",
            f"class_map={self.class_map}",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"calibration_bins={self.calibration_bins}",
            f"font_size={self.font_size}"
        ]
        return f"ClassificationMetricsFormat({', '.join(parts)})"


class MultiClassificationMetricsFormat:
    """
    Optional configuration for multi-label classification tasks.
    """
    def __init__(self,
                 ROC_PR_line: str='darkorange',
                 cmap: str = "Blues",
                 font_size: int = 16) -> None:
        """
        Initializes the formatting configuration for multi-label classification metrics.

        Args:
            ROC_PR_line (str): The color name or hex code for the line plotted
                on the ROC and Precision-Recall curves (one for each label). 
                Defaults to 'darkorange'.
                - Common color names: 'darkorange', 'cornflowerblue', 'crimson', 'forestgreen'
                - Hex codes: '#FF6347', '#4682B4'
            
            cmap (str): The matplotlib colormap name for the per-label
                confusion matrices. Defaults to "Blues".
                - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples'
                - Diverging options: 'coolwarm', 'viridis', 'plasma', 'inferno'
            
            font_size (int): The base font size to apply to the plots. Defaults to 16.
            
        <br>
        
        ## [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)    
        """
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"cmap='{self.cmap}'",
            f"font_size={self.font_size}"
        ]
        return f"MultiClassificationMetricsFormat({', '.join(parts)})"


class RegressionMetricsFormat:
    """
    Optional configuration for single-target regression and multi-target regression tasks.
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
            font_size (int): The base font size to apply to the plots. Defaults to 16.
            scatter_color (str): Matplotlib color for the scatter plot points. Defaults to 'tab:blue'.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            scatter_alpha (float): Alpha transparency for scatter plot points. Defaults to 0.6.
            ideal_line_color (str): Matplotlib color for the 'ideal' y=x line in the 
                True vs. Predicted plot. Defaults to 'k' (black).
                - Common color names: 'k', 'red', 'darkgrey', '#FF6347'
            residual_line_color (str): Matplotlib color for the y=0 line in the 
                Residual plot. Defaults to 'red'.
                - Common color names: 'red', 'blue', 'k', '#4682B4'
            hist_bins (int | str): The number of bins for the residuals histogram. 
                Defaults to 'auto' to use seaborn's automatic bin selection.
                - Options: 'auto', 'sqrt', 10, 20
        
        <br>
        
        ## [Matplotlib Colors](https://matplotlib.org/stable/users/explain/colors/colors.html)
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
        return f"RegressionMetricsFormat({', '.join(parts)})"


class SegmentationMetricsFormat:
    """
    Optional configuration for segmentation tasks.
    """
    def __init__(self,
                 heatmap_cmap: str = 'viridis',
                 cm_cmap: str = "Blues",
                 font_size: int = 16) -> None:
        """
        Initializes the formatting configuration for segmentation metrics.

        Args:
            heatmap_cmap (str): The matplotlib colormap name for the per-class
                metrics heatmap. Defaults to "viridis".
                - Sequential options: 'viridis', 'plasma', 'inferno', 'cividis'
                - Diverging options: 'coolwarm', 'bwr', 'seismic'
            cm_cmap (str): The matplotlib colormap name for the pixel-level
                confusion matrix. Defaults to "Blues".
                - Sequential options: 'Blues', 'Greens', 'Reds', 'Oranges'
            font_size (int): The base font size to apply to the plots. Defaults to 16.
        
        <br>
        
        ## [Matplotlib Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
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
        return f"SegmentationMetricsFormat({', '.join(parts)})"


# Similar to regression configuration
class SequenceValueMetricsFormat:
    """
    Optional configuration for sequence to value prediction tasks.
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
            font_size (int): The base font size to apply to the plots. Defaults to 16.
            scatter_color (str): Matplotlib color for the scatter plot points. Defaults to 'tab:blue'.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            scatter_alpha (float): Alpha transparency for scatter plot points. Defaults to 0.6.
            ideal_line_color (str): Matplotlib color for the 'ideal' y=x line in the 
                True vs. Predicted plot. Defaults to 'k' (black).
                - Common color names: 'k', 'red', 'darkgrey', '#FF6347'
            residual_line_color (str): Matplotlib color for the y=0 line in the 
                Residual plot. Defaults to 'red'.
                - Common color names: 'red', 'blue', 'k', '#4682B4'
            hist_bins (int | str): The number of bins for the residuals histogram. 
                Defaults to 'auto' to use seaborn's automatic bin selection.
                - Options: 'auto', 'sqrt', 10, 20
        
        <br>
        
        ## [Matplotlib Colors](https://matplotlib.org/stable/users/explain/colors/colors.html)
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
        return f"SequenceValueMetricsFormat({', '.join(parts)})"


class SequenceSequenceMetricsFormat:
    """
    Optional configuration for sequence-to-sequence evaluation plots.
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
            font_size (int): The base font size to apply to the plots. Defaults to 16.
            plot_figsize (Tuple[int, int]): Figure size for the plot. Defaults to (10, 6).
            grid_style (str): Matplotlib linestyle for the plot grid. Defaults to '--'.
                - Options: '--' (dashed), ':' (dotted), '-.' (dash-dot), '-' (solid)
            rmse_color (str): Matplotlib color for the RMSE line. Defaults to 'tab:blue'.
                - Common color names: 'tab:blue', 'crimson', 'forestgreen', '#4682B4'
            rmse_marker (str): Matplotlib marker style for the RMSE line. Defaults to 'o-'.
                - Options: 'o-' (circle), 's--' (square), '^:' (triangle), 'x' (x marker)
            mae_color (str): Matplotlib color for the MAE line. Defaults to 'tab:orange'.
                - Common color names: 'tab:orange', 'purple', 'black', '#FF6347'
            mae_marker (str): Matplotlib marker style for the MAE line. Defaults to 's--'.
                - Options: 's--', 'o-', 'v:', '+' (plus marker)
        
        <br>
        
        ## [Matplotlib Colors](https://matplotlib.org/stable/users/explain/colors/colors.html)
        ## [Matplotlib Linestyles](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
        ## [Matplotlib Markers](https://matplotlib.org/stable/api/markers_api.html)
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
        return f"SequenceMetricsFormat({', '.join(parts)})"

def info():
    _script_info(__all__)
