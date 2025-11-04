from typing import Optional
from ._script_info import _script_info


__all__ = [
    "ClassificationMetricsFormat",
    "MultiClassificationMetricsFormat"
]


class ClassificationMetricsFormat:
    """
    Optional configuration for classification tasks, use in the '.evaluate()' method of the MLTrainer.
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
    Optional configuration for multi-label classification tasks, use in the '.evaluate()' method of the MLTrainer.
    """
    def __init__(self,
                 threshold: float=0.5,
                 ROC_PR_line: str='darkorange',
                 cmap: str = "Blues",
                 font_size: int = 16) -> None:
        """
        Initializes the formatting configuration for multi-label classification metrics.

        Args:
            threshold (float): The probability threshold (0.0 to 1.0) used
                to convert sigmoid outputs into binary (0 or 1) predictions for
                calculating the confusion matrix and overall metrics. Defaults to 0.5.
            
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
        self.threshold = threshold
        self.cmap = cmap
        self.ROC_PR_line = ROC_PR_line
        self.font_size = font_size
        
    def __repr__(self) -> str:
        parts = [
            f"threshold={self.threshold}",
            f"ROC_PR_line='{self.ROC_PR_line}'",
            f"cmap='{self.cmap}'",
            f"font_size={self.font_size}"
        ]
        return f"MultiClassificationMetricsFormat({', '.join(parts)})"


def info():
    _script_info(__all__)
