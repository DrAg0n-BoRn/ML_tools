from typing import Union, Any, Literal
import torch
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from ..ML_models_diffusion import (DragonAutoencoder, 
                                   DragonAutoencoderV2, 
                                   DragonDiT, DragonDiTV2, 
                                   DragonDiTGuided, 
                                   DragonDiTGuidedV2)
from ..ML_utilities import validate_torch_device
from ..data_exploration import plot_value_distributions, plot_numeric_overview_boxplot_macro

from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("DiT Generator")


__all__ = [
    "_BaseDiffusionGenerator",
]


class _BaseDiffusionGenerator(ABC):
    def __init__(self,
                 diffusion_model: Union[DragonDiT, DragonDiTV2, DragonDiTGuided, DragonDiTGuidedV2],
                 encoder: Union[DragonAutoencoder, DragonAutoencoderV2],
                 device: Union[torch.device, str]):
        """Base class for diffusion generators."""
        
        self.diffusion_model = diffusion_model
        self.encoder = encoder

        if isinstance(device, str):
            self.device = validate_torch_device(device)
        else:
            self.device = device
        
        # Move models to the specified device
        self.diffusion_model.to(self.device)
        self.encoder.to(self.device)
        # Set models to evaluation mode
        self.diffusion_model.eval()
        self.encoder.eval()
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        pass
    
    def plot_metrics(self, 
                     df_generated: pd.DataFrame, 
                     save_dir: Union[Path, str],
                     plot_title: str = "Generated Data Distributions",
                     handle_zero_variance: Literal["constant", "drop"] = "constant",
                     show_means: bool = True,
                     font_scaling: float = 1.5) -> None:
        """
        Plots value distributions and numeric overview boxplots for the generated DataFrame.
        
        Args:
            df_generated (pd.DataFrame): The generated DataFrame for which to plot metrics.
            save_dir (Path | str): The directory where the plots will be saved.
            plot_title (str): The title for the boxplots.
            handle_zero_variance (Literal["constant", "drop"]): How to handle columns with zero variance.
            show_means (bool): Whether to display means on the plots.
            font_scaling (float): The scaling factor for font sizes on the plots.
        """
        # check if df_generated is empty
        if df_generated.empty:
            _LOGGER.warning("The provided DataFrame for plotting is empty. No plots will be generated.")
            return
        
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        
        plot_value_distributions(df=df_generated, 
                                 save_dir=save_path,
                                 font_scaling=font_scaling,)

        plot_numeric_overview_boxplot_macro(df=df_generated, 
                                            save_dir=save_path, 
                                            plot_title=plot_title,
                                            handle_zero_variance=handle_zero_variance,
                                            show_means=show_means,
                                            font_scaling=font_scaling)
