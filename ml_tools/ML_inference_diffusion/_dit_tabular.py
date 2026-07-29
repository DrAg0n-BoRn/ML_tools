from typing import Union, Literal
import torch
import pandas as pd

from ..ML_models_diffusion import DragonAutoencoder, DragonAutoencoderV2, DragonDiT, DragonDiTV2
from ..math_utilities import handle_negative_values, round_float_values

from .._core import get_logger

from ._base_generator import _BaseDiffusionGenerator


_LOGGER = get_logger("DiT Generator")


__all__ = [
    "DragonDiTGenerator",
]


class DragonDiTGenerator(_BaseDiffusionGenerator):
    """
    A DataFrame generator for creating synthetic tabular data using a diffusion model.
    
    This generator takes a trained diffusion model and an autoencoder to generate synthetic tabular data, and plots relevant metrics to evaluate the generated data.
    """
    def __init__(self,
                 diffusion_model: Union[DragonDiT, DragonDiTV2],
                 encoder: Union[DragonAutoencoder, DragonAutoencoderV2],
                 device: Union[torch.device, str]):
        """
        Initializes the DragonDiTGenerator with the specified parameters.
        
        Args:
            diffusion_model (DragonDiT | DragonDiTV2): The trained diffusion model to use for generating synthetic data.
            encoder (DragonAutoencoder | DragonAutoencoderV2): The autoencoder used to decode the generated embeddings back to tabular format.
            device (torch.device | str): The device to run the model on (e.g., "cpu" or "cuda"). The models will be moved to this device for generation.
        """
        super().__init__(diffusion_model, encoder, device)
    
    def generate(self, 
                 batch_size: int,
                 ode_steps: int=20,
                 positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                 round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                 float_rounding_precision: int = 3) -> pd.DataFrame:
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
            
        _LOGGER.info(f"Generated {batch_size} samples.")
        
        return decoded_data
