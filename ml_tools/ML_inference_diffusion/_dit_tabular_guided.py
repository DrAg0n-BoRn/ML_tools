from typing import Union, Literal, Optional
import torch
from pathlib import Path
import pandas as pd

from ..ML_models_diffusion import DragonAutoencoder, DragonDiTGuided
from ..math_utilities import handle_negative_values, round_float_values
from ..data_exploration import plot_value_distributions, plot_numeric_overview_boxplot
from ..utilities import save_dataframe_filename

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

from ._base_generator import _BaseDiffusionGenerator


_LOGGER = get_logger("DragonDiTGuidedGenerator")


__all__ = [
    "DragonDiTGuidedGenerator",
]


class DragonDiTGuidedGenerator(_BaseDiffusionGenerator):
    """
    A DataFrame generator for creating synthetic tabular data using a guided diffusion model.
    
    This generator takes a trained guided diffusion model and an autoencoder to generate synthetic tabular data conditioned on specific target values, and plots relevant metrics to evaluate the generated data.
    """
    def __init__(self,
                 save_dir: Union[Path, str],
                 diffusion_model: DragonDiTGuided,
                 encoder: DragonAutoencoder,
                 device: Union[torch.device, str]):
        """
        Initializes the DragonDiTGuidedGenerator with the specified parameters.
        
        Args:
            save_dir (Path | str): The root directory where generated data and plots will be saved.
            diffusion_model (DragonDiTGuided): The trained guided diffusion model to use for generating synthetic data.
            encoder (DragonAutoencoder): The autoencoder used to decode the generated embeddings back to tabular format.
            device (torch.device | str): The device to run the model on (e.g., "cpu" or "cuda"). The models will be moved to this device for generation.
        """

        super().__init__(save_dir, diffusion_model, encoder, device)
    
    def generate(self, 
                 batch_size: int,
                 target_value: float,
                 guidance_scale: float = 3.0,
                 ode_steps: int = 20,
                 positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                 round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                 float_rounding_precision: int = 3,
                 autosave: bool = True) -> pd.DataFrame:
        """
        Generates synthetic tabular data conditioned on a specific target value.
        
        Args:
            batch_size (int): The number of synthetic samples to generate.
            target_value (float): The specific target value to condition the generation on.
            guidance_scale (float): The strength of the guidance during generation.
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
        self.diffusion_model: DragonDiTGuided
        
        # Generate synthetic latent embeddings conditioned on the target
        generated_batch = self.diffusion_model.generate_sequence(
            batch_size=batch_size, 
            target_value=target_value,
            num_steps=ode_steps,
            guidance_scale=guidance_scale
        )
        
        # Decode embeddings back to tabular format
        decoded_data = self.encoder.approximate_decode(generated_batch)
        
        if positive_columns != "none":
            _positive_code = None if positive_columns == "all" else positive_columns
            decoded_data = handle_negative_values(df=decoded_data, columns=_positive_code)
            
        if round_float_columns != "none":
            _rounding_code = None if round_float_columns == "all" else round_float_columns
            decoded_data = round_float_values(df=decoded_data, columns=_rounding_code, n=float_rounding_precision)
            
        if autosave:
            batch_info = f"Generated-{batch_size}-Target-{target_value}-Guidance-{guidance_scale}".replace(".", "_")
            save_dataframe_filename(df=decoded_data, save_dir=self.save_root_dir, filename=batch_info, verbose=1)
            _LOGGER.info(f"Generated data saved to {self.save_root_dir} as '{batch_info}.csv'.")
        else:
            _LOGGER.info(f"Generated {batch_size} samples for target {target_value}.")
        
        return decoded_data
    
    def plot_metrics(self,
                     df_generated: pd.DataFrame, 
                     target_value: float,
                     base_plot_title: str = "Generated Data Distributions",
                     add_strategy_title: bool = True,
                     handle_zero_variance: Literal["constant", "drop"] = "constant",
                     subdirectory: Optional[str] = None) -> None:
        """
        Plots value distributions and numeric overview boxplots.
        
        Args:
            df_generated (pd.DataFrame): The generated DataFrame for which to plot metrics.
            target_value (float): The target value used for generation, included in plot titles for clarity.
            base_plot_title (str): The base title for the plots.
            add_strategy_title (bool): Whether to include the strategy name in the plot titles for clarity.
            handle_zero_variance (Literal["constant", "drop"]): How to handle columns with zero variance when plotting boxplots.
            subdirectory (str | None): Optional subdirectory within the save directory to save the plots. If None, saves in the root save directory.
        """
        if df_generated.empty:
            _LOGGER.warning("The provided DataFrame for plotting is empty. No plots will be generated.")
            return
            
        target_title = f"{base_plot_title} (Target: {target_value})"
        
        if isinstance(subdirectory, str):
            subdirectory = sanitize_filename(subdirectory)
            
            target_dir = make_fullpath(self.save_root_dir / subdirectory, make=True, enforce="directory")
        else:
            target_dir = self.save_root_dir
        
        plot_value_distributions(df=df_generated, save_dir=target_dir)
        
        strategies: tuple[Literal["value", "scale", "log"], ...] = ("value", "scale", "log")
        
        for _strategy in strategies:
            final_plot_title = f"{target_title} ({_strategy.capitalize()})" if add_strategy_title else target_title

            plot_numeric_overview_boxplot(
                df=df_generated, 
                strategy=_strategy,
                save_dir=target_dir, 
                plot_title=final_plot_title,
                handle_zero_variance=handle_zero_variance
            )

    def generate_plot_multi(self,
                            targets: list[float],
                            batch_size: int,
                            guidance_scale: float = 3.0,
                            ode_steps: int = 20,
                            positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                            round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                            float_rounding_precision: int = 3,
                            handle_zero_variance: Literal["constant", "drop"] = "constant") -> None:
        """
        Iterates over a list of targets, generating and plotting data for each, saving outputs in isolated subdirectories.
        
        Args:
            targets (list[float]): A list of target values to condition the generation on.
            batch_size (int): The number of synthetic samples to generate for each target.
            guidance_scale (float): The strength of the guidance during generation.
            ode_steps (int): The number of ODE steps to use during sampling. More steps might improve quality but will increase generation time.
            positive_columns (list[str] | "all" | "none"): Which columns should be forced to have only positive values or 0. 
                - If "all", all columns will be processed to ensure positivity (identifies numeric columns automatically).
                - If "none", no columns will be modified.
            round_float_columns (list[str] | "all" | "none"): Which columns should have their float values rounded. 
                - If "all", all columns will be processed to round float values (identifies numeric columns automatically).
                - If "none", no columns will be modified for rounding.
            float_rounding_precision (int): The number of decimal places to round float values to if `round_float_columns` is not "none".
            handle_zero_variance (Literal["constant", "drop"]): How to handle columns with zero variance when plotting boxplots.
        """
        for target in targets:
            # Create a dedicated directory for this specific target
            basic_info = f"Target-{target}-Guidance-{guidance_scale}".replace(".", "_")
            
            target_dir = self.save_root_dir / basic_info
            
            df_generated = self.generate(
                batch_size=batch_size,
                target_value=target,
                guidance_scale=guidance_scale,
                ode_steps=ode_steps,
                positive_columns=positive_columns,
                round_float_columns=round_float_columns,
                float_rounding_precision=float_rounding_precision,
                autosave=False
            )
            
            self.plot_metrics(
                df_generated=df_generated,
                target_value=target,
                subdirectory=basic_info,
                handle_zero_variance=handle_zero_variance
            )
            
            # save the generated DataFrame for this target
            save_dataframe_filename(df=df_generated, save_dir=target_dir, filename=f"Generated-{batch_size}-samples", verbose=1)
 
        _LOGGER.info("Multi-target generation and plotting completed.")
  