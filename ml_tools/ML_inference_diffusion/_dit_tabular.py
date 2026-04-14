from typing import Union, Literal
import torch
from pathlib import Path
import pandas as pd

from ..ML_models_diffusion import DragonAutoencoder, DragonDiT
from ..math_utilities import handle_negative_values, round_float_values
from ..data_exploration import plot_value_distributions, plot_numeric_overview_boxplot_macro
from ..utilities import save_dataframe_filename

from .._core import get_logger

from ._base_generator import _BaseDiffusionGenerator


_LOGGER = get_logger("DragonDiTGenerator")


__all__ = [
    "DragonDiTGenerator",
]


class DragonDiTGenerator(_BaseDiffusionGenerator):
    """
    A DataFrame generator for creating synthetic tabular data using a diffusion model.
    
    This generator takes a trained diffusion model and an autoencoder to generate synthetic tabular data, and plots relevant metrics to evaluate the generated data.
    """
    def __init__(self,
                 save_dir: Union[Path, str],
                 diffusion_model: DragonDiT,
                 encoder: DragonAutoencoder,
                 device: Union[torch.device, str]):
        """
        Initializes the DragonDiTGenerator with the specified parameters.
        
        Args:
            save_dir (Path | str): The root directory where generated data and plots will be saved.
            diffusion_model (DragonDiT): The trained diffusion model to use for generating synthetic data.
            encoder (DragonAutoencoder): The autoencoder used to decode the generated embeddings back to tabular format.
            device (torch.device | str): The device to run the model on (e.g., "cpu" or "cuda"). The models will be moved to this device for generation.
        """
        super().__init__(save_dir, diffusion_model, encoder, device)
    
    def generate(self, 
                 batch_size: int,
                 ode_steps: int=20,
                 positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                 round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                 float_rounding_precision: int = 3,
                 autosave: bool = True) -> pd.DataFrame:
        """
        Generates synthetic tabular data using the diffusion model.
        
        Args:
            batch_size (int): The number of synthetic samples to generate.
            ode_steps (int): The number of ODE steps to use during sampling. More steps might improve quality but will increase generation time.
            positive_columns (list[str] | "all" | "none"): Which columns should be forced to have only positive values or 0. 
                - If "all", all columns will be processed to ensure positivity (identifies numeric columns automatically).
                - If "none", no columns will be modified.
            round_float_columns (list[str] | "all" | "none"): Which columns should have their float values rounded. 
                - If "all", all columns will be processed to round float values (identifies numeric columns automatically).
                - If "none", no columns will be modified for rounding.
            float_rounding_precision (int): The number of decimal places to round float values to if `round_float_columns` is not "none".
            autosave (bool): Whether to automatically save the generated DataFrame to a CSV file in the provided save directory.
            
        Returns:
            pd.DataFrame: The generated synthetic tabular data as a DataFrame.
        """
        self.diffusion_model: DragonDiT #type hint
        
        # Generate synthetic data using the diffusion model
        generated_batch = self.diffusion_model.generate_sequence(batch_size=batch_size, num_steps=ode_steps)
        
        # Decode the generated embeddings back to tabular format using the autoencoder's decoder
        decoded_data = self.encoder.approximate_decode(generated_batch)
        
        if positive_columns != "none":
            _positive_code = None if positive_columns == "all" else positive_columns
            
            decoded_data = handle_negative_values(df=decoded_data, columns=_positive_code)
            
        if round_float_columns != "none":
            _rounding_code = None if round_float_columns == "all" else round_float_columns
            
            decoded_data = round_float_values(df=decoded_data, columns=_rounding_code, n=float_rounding_precision)
            
        if autosave:
            batch_info = f"Generated_{batch_size}_samples"
            
            save_dataframe_filename(df=decoded_data, save_dir=self.save_root_dir, filename=batch_info, verbose=1)
            
            _LOGGER.info(f"Generated data saved to {self.save_root_dir} as '{batch_info}.csv'.")
        else:
            _LOGGER.info(f"Generated {batch_size} samples.")
        
        return decoded_data
    
    def plot_metrics(self, 
                     df_generated: pd.DataFrame, 
                     base_plot_title: str = "Generated Data Distributions",
                     handle_zero_variance: Literal["constant", "drop"] = "constant") -> None:
        """
        Plots value distributions and numeric overview boxplots for the generated DataFrame.
        
        Args:
            df_generated (pd.DataFrame): The generated DataFrame for which to plot metrics.
            base_plot_title (str): The base title for the plots.
            handle_zero_variance (Literal["constant", "drop"]): How to handle columns with zero variance.
        """
        # check if df_generated is empty
        if df_generated.empty:
            _LOGGER.warning("The provided DataFrame for plotting is empty. No plots will be generated.")
            return
        
        plot_value_distributions(df=df_generated, save_dir=self.save_root_dir)

        plot_numeric_overview_boxplot_macro(df=df_generated, 
                                            save_dir=self.save_root_dir, 
                                            plot_title=base_plot_title,
                                            handle_zero_variance=handle_zero_variance)
