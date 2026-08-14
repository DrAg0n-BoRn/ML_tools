from typing import Union, Literal, Optional
import torch
import pandas as pd

from ..ML_models_diffusion import DragonAutoencoder, DragonAutoencoderV2, DragonDiTGuided, DragonDiTGuidedV2
from ..math_utilities import handle_negative_values, round_float_values

from .._core import get_logger

from ._base_generator import _BaseDiffusionGenerator


_LOGGER = get_logger("DiT Guided Generator")


__all__ = [
    "DragonDiTGuidedGenerator",
]


class DragonDiTGuidedGenerator(_BaseDiffusionGenerator):
    """
    A DataFrame generator for creating synthetic tabular data using a guided diffusion model.
    
    This generator takes a trained guided diffusion model and an autoencoder to generate synthetic tabular data conditioned on specific target values, and plots relevant metrics to evaluate the generated data.
    """
    def __init__(self,
                 diffusion_model: Union[DragonDiTGuided, DragonDiTGuidedV2],
                 encoder: Union[DragonAutoencoder, DragonAutoencoderV2],
                 device: Union[torch.device, str]):
        """
        Initializes the DragonDiTGuidedGenerator with the specified parameters.
        
        Args:
            diffusion_model (DragonDiTGuided | DragonDiTGuidedV2): The trained guided diffusion model to use for generating synthetic data.
            encoder (DragonAutoencoder | DragonAutoencoderV2): The autoencoder used to decode the generated embeddings back to tabular format.
            device (torch.device | str): The device to run the model on (e.g., "cpu" or "cuda"). The models will be moved to this device for generation.
        """

        super().__init__(diffusion_model, encoder, device)
    
    def generate(self, 
                 batch_size: int,
                 target_value: float,
                 target_name: Optional[str] = None,
                 guidance_scale: float = 3.0,
                 cfg_rescale: float = 0.0,
                 ode_steps: int = 20,
                 positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                 round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                 float_rounding_precision: int = 3) -> pd.DataFrame:
        """
        Generates synthetic tabular data conditioned on a specific target value.
        
        Args:
            batch_size (int): The number of synthetic samples to generate.
            target_value (float): The specific target value to condition the generation on.
            target_name (str | None): Optional column name to append the target value to the resulting DataFrame.
            guidance_scale (float): The strength of the guidance during generation.
            cfg_rescale (float): The rescaling factor for classifier-free guidance. 
                - Min: `0.0` Rescaling is completely turned off. This yields standard CFG behavior.
                - Max: `1.0` The guided prediction's variance is strictly forced to match the original conditional variance. 
                - Recommended 0.5 to 0.7 if guidance_scale is high (5.0 or more).
            ode_steps (int): The number of ODE steps to use during sampling. More steps might improve quality but will increase generation time.
                - For V2 models, 5 to 10 steps are often sufficient for good quality thanks to OT-CFM.
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
        self.diffusion_model: DragonDiTGuided
        
        # Generate synthetic latent embeddings conditioned on the target
        generated_batch = self.diffusion_model.generate_sequence(
            batch_size=batch_size, 
            target_value=target_value,
            num_steps=ode_steps,
            guidance_scale=guidance_scale,
            cfg_rescale=cfg_rescale
        )
        
        # Decode embeddings back to tabular format
        decoded_data = self.encoder.approximate_decode(generated_batch)
        
        if positive_columns != "none":
            _positive_code = None if positive_columns == "all" else positive_columns
            decoded_data = handle_negative_values(df=decoded_data, columns=_positive_code)
            
        if round_float_columns != "none":
            _rounding_code = None if round_float_columns == "all" else round_float_columns
            decoded_data = round_float_values(df=decoded_data, columns=_rounding_code, n=float_rounding_precision)
            
        if target_name is not None:
            decoded_data[target_name] = target_value
            
        _LOGGER.info(f"Generated {batch_size} samples for target {target_value}.")
        
        return decoded_data

    def generate_multi(self,
                       target_range: tuple[float, float, float],
                       batch_per_step: int,
                       target_name: str,
                       guidance_scale: float = 3.0,
                       cfg_rescale: float = 0.0,
                       ode_steps: int = 20,
                       positive_columns: Union[list[str], Literal["all"], Literal["none"]] = "none",
                       round_float_columns: Union[list[str], Literal["all"], Literal["none"]] = "all",
                       float_rounding_precision: int = 3) -> pd.DataFrame:
        """
        Iterates over a calculated range of targets, generating data for each step, and combines 
        all results into a single DataFrame.
        
        Args:
            target_range (tuple[float, float, float]): The range of target values to condition on, 
                formatted as `START(Inclusive), END(Exclusive), STEP`.
            batch_per_step (int): The number of synthetic samples to generate per step.
            target_name (str): The name of the column to append to the DataFrame to record the conditioning target.
            guidance_scale (float): The strength of the guidance during generation.
            cfg_rescale (float): The rescaling factor for classifier-free guidance. 
                - Min: `0.0` Rescaling is completely turned off. This yields standard CFG behavior.
                - Max: `1.0` The guided prediction's variance is strictly forced to match the original conditional variance. 
                - Recommended 0.5 to 0.7 if guidance_scale is high (5.0 or more).
            ode_steps (int): The number of ODE steps to use during sampling.
            positive_columns (list[str] | "all" | "none"): Which columns should be forced to have only positive values.
            round_float_columns (list[str] | "all" | "none"): Which columns should have float values rounded.
            float_rounding_precision (int): Decimal places to round to if `round_float_columns` is not "none".
            
        Returns:
            pd.DataFrame: A single consolidated DataFrame containing all generated samples across the target range.
        """
        start, end, step = target_range
        
        if step == 0:
            _LOGGER.error("Step in target_range cannot be zero.")
            raise ValueError()
        elif (step > 0 and start >= end) or (step < 0 and start <= end):
            _LOGGER.error("Invalid target_range: Ensure that the step direction aligns with the start and end values.")
            raise ValueError()
            
        # 1. Calculate float-compatible targets
        targets = []
        current = start
        if step > 0:
            while current < end:
                targets.append(round(current, 6))  # Rounding helps avoid float arithmetic issues
                current += step
        else:
            while current > end:
                targets.append(round(current, 6))
                current += step

        generated_dfs = []
        
        # 2. Generate samples for each target
        for target in targets:
            df_generated = self.generate(
                batch_size=batch_per_step,
                target_value=target,
                target_name=target_name,
                guidance_scale=guidance_scale,
                cfg_rescale=cfg_rescale,
                ode_steps=ode_steps,
                positive_columns=positive_columns,
                round_float_columns=round_float_columns,
                float_rounding_precision=float_rounding_precision
            )
            
            generated_dfs.append(df_generated)
            
        # 3. Consolidate into a single DataFrame
        final_df = pd.concat(generated_dfs, ignore_index=True)
 
        _LOGGER.info(f"Multi-target generation completed. {len(final_df)} total samples generated and combined.")
        
        return final_df
