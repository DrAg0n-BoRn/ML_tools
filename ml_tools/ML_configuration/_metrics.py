from typing import Union


__all__ = [
    # --- Metrics Formats ---
    "FormatRegressionMetrics",
    "FormatMultiTargetRegressionMetrics",
    "FormatBinaryClassificationMetrics",
    "FormatMultiClassClassificationMetrics",
    "FormatBinaryImageClassificationMetrics",
    "FormatMultiClassImageClassificationMetrics",
    "FormatMultiLabelBinaryClassificationMetrics",
    "FormatBinarySegmentationMetrics",
    "FormatMultiClassSegmentationMetrics",
    "FormatSequenceValueMetrics",
    "FormatSequenceSequenceMetrics",
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


# ----------------------------
# Metrics Configurations
# ----------------------------

# Regression
class FormatRegressionMetrics(_BaseRegressionFormat):
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
class FormatMultiTargetRegressionMetrics(_BaseRegressionFormat):
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
class FormatBinaryClassificationMetrics(_BaseClassificationFormat):
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


class FormatMultiClassClassificationMetrics(_BaseClassificationFormat):
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


class FormatBinaryImageClassificationMetrics(_BaseClassificationFormat):
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


class FormatMultiClassImageClassificationMetrics(_BaseClassificationFormat):
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
class FormatMultiLabelBinaryClassificationMetrics(_BaseMultiLabelFormat):
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
class FormatBinarySegmentationMetrics(_BaseSegmentationFormat):
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


class FormatMultiClassSegmentationMetrics(_BaseSegmentationFormat):
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
class FormatSequenceValueMetrics(_BaseSequenceValueFormat):
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


class FormatSequenceSequenceMetrics(_BaseSequenceSequenceFormat):
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

