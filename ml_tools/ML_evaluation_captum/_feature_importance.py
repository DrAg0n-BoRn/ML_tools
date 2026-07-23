from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from captum.attr import IntegratedGradients

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger, wrap_text
from ..keys._keys import CaptumKeys, _EvaluationConfig

_LOGGER = get_logger("Captum")

__all__ = [
    "captum_feature_importance"
]

def captum_feature_importance(model: nn.Module,
                              input_data: torch.Tensor,
                              feature_names: Optional[list[str]],
                              save_dir: Union[str, Path],
                              target_names: Optional[list[str]] = None,
                              n_steps: int = 50,
                              device: Union[str, torch.device] = 'cpu',
                              verbose: int = 0):
    """
    Calculates feature importance using Captum's Integrated Gradients.

    This function iterates over every target specified in `target_names` (or inferred from the model)
    and saves a separate CSV report and plot for each.

    Args:
        model (nn.Module): The PyTorch model.
        input_data (torch.Tensor): The input tensor to explain (batch of samples).
        feature_names (list[str] | None): Names of features.
        save_dir (str | Path): Output directory.
        target_names (List[str] | None): 
            A list of names corresponding to the model's outputs.
            - For **Single Target** (Binary/Regression): Provide a list with 1 name.
            - For **Multi-Target/Multi-Class**: Provide a list of names matching the number of outputs.
            - If `None`, generic names (e.g., "Output_0") will be generated based on model output shape.
        n_steps (int): Number of steps for the integral approximation. Higher means more accurate but slower.
        device (str | torch.device): Torch device.
        verbose (int): Verbosity level.
    <br>
        
    ### NOTE: 
    The internal batch size used by Captum will be set to the number of samples. If you run into OOM errors, consider reducing `n_samples`.
    """
    # Ensure output directory exists
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # Move model and data to device
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device_obj)
    
    # Create a clone to avoid modifying the original tensor in-place
    inputs = input_data.clone().detach().to(device_obj)
    inputs.requires_grad = True
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs).to(device_obj)

    # --- 1. Infer Targets if not provided ---
    # We perform a dummy forward pass to check the output dimension/shape
    with torch.no_grad():
        dummy_out = model(inputs[0:1])
    
    num_outputs = 1
    output_is_1d = False
    
    if dummy_out.ndim == 1:
        # Output is shape (Batch,), implied single target
        num_outputs = 1
        output_is_1d = True
    elif dummy_out.ndim == 2:
        # Output is shape (Batch, Num_Targets)
        num_outputs = dummy_out.shape[1]
    else:
        # Fallback for complex shapes (e.g. sequence outputs), default to treating as 1 block for safety
        _LOGGER.warning(f"Model output has shape {dummy_out.shape}. Captum wrapper defaults to single-target interpretation for dimensions > 2.")
        num_outputs = 1

    # Generate names if missing
    if target_names is None:
        if num_outputs == 1:
            target_names = ["Output"]
        else:
            target_names = [f"Output_{i}" for i in range(num_outputs)]
        _LOGGER.info(f"No 'target_names' provided. Generated generic names: {target_names}")
    
    # Validate names
    if len(target_names) != num_outputs:
        _LOGGER.error(f"Name mismatch: Provided {len(target_names)} target names, but model appears to have {num_outputs} outputs.")
        raise ValueError()

    # --- 2. Iterate and Explain ---
    _LOGGER.info(f"売 Calculating Captum importance for {len(target_names)} target(s)")
    
    for i, name in enumerate(target_names):
        # Sanitize name for file saving
        clean_name = sanitize_filename(name)
        
        # Determine correct index for Captum
        # If model output is 1D (N,), target must be None.
        # If model output is 2D (N, C), target is the index i.
        idx_to_explain = None if output_is_1d else i
        
        # _LOGGER.info(f"   > Processing target: '{name}'")
        
        _process_single_target(
            ig=ig,
            inputs=inputs,
            baseline=baseline,
            target_index=idx_to_explain,
            feature_names=feature_names,
            save_dir=save_dir_path,
            n_steps=n_steps,
            file_suffix=f"_{clean_name}",
            target_name=name,  # Pass original name for plotting
            verbose=verbose
        )


def _process_single_target(ig: 'IntegratedGradients', # type: ignore
                           inputs: torch.Tensor,
                           baseline: torch.Tensor,
                           target_index: Union[int, None],
                           feature_names: Optional[list[str]],
                           save_dir: Path,
                           n_steps: int,
                           file_suffix: str,
                           target_name: str = "",
                           verbose: int = 0):
    """
    Private helper to run the attribution, aggregation, and saving for a single context.
    """
    try:
        # attribute() returns values of the same shape as input_data
        attributions, delta = ig.attribute(inputs, 
                                           baselines=baseline, 
                                           target=target_index,
                                           n_steps=n_steps,
                                           internal_batch_size=inputs.shape[0],
                                           return_convergence_delta=True)
        # Check convergence quality
        mean_delta = torch.mean(torch.abs(delta)).item()
        if mean_delta > 0.1 and verbose > 0:
            _LOGGER.warning(f"Captum Convergence Delta is high ({mean_delta:.4f}). Consider increasing 'n_steps'.")
        
    except Exception as e:
        _LOGGER.error(f"Captum attribution failed for target '{target_index}': {e}")
        return

    # --- Aggregate Feature Importance ---
    # take the mean of the absolute attribution values across the batch
    attributions_np = attributions.detach().cpu().numpy()
    
    # Logic to collapse dimensions based on input shape vs feature names
    # Case A: Tabular (Batch, Features)
    if attributions_np.ndim == 2:
        mean_abs_attr = np.mean(np.abs(attributions_np), axis=0) #Global average importance
        
    # Case B: Sequence (Batch, Sequence_Length, Features)
    elif attributions_np.ndim == 3:
        # Mean across sequence length (axis 1), then mean across batch (axis 0)
        mean_abs_attr = np.mean(np.mean(np.abs(attributions_np), axis=1), axis=0)
        
    # Case C: Image (Batch, Channels, Height, Width)
    elif attributions_np.ndim == 4:
        num_channels = attributions_np.shape[1]
        # Check if feature names correspond to channels
        if feature_names is not None and len(feature_names) == num_channels:
            mean_abs_attr = np.mean(np.sum(np.abs(attributions_np), axis=(2, 3)), axis=0)
        else:
            # Default to channel aggregation
            mean_abs_attr = np.mean(np.sum(np.abs(attributions_np), axis=(2, 3)), axis=0)
            if feature_names is None or len(feature_names) != num_channels:
                feature_names = [f"Channel_{i}" for i in range(num_channels)]

    else:
        # Fallback
        mean_abs_attr = np.mean(np.abs(attributions_np), axis=0).flatten()

    # Verify shape matches feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_attr))]

    if len(mean_abs_attr) != len(feature_names):
        min_len = min(len(mean_abs_attr), len(feature_names))
        mean_abs_attr = mean_abs_attr[:min_len]
        feature_names = feature_names[:min_len]
    
    # Calculate percentages (Before Min-Max scaling to preserve relative importance)
    total_attr_sum = np.sum(mean_abs_attr)
    if total_attr_sum > 0:
        attr_percentages = (mean_abs_attr / total_attr_sum) * 100.0
    else:
        attr_percentages = np.zeros_like(mean_abs_attr)
    
    # Min-Max Scaling
    target_min = 0.01
    target_max = 1.0
    
    _min = np.min(mean_abs_attr)
    _max = np.max(mean_abs_attr)
    
    if _max > _min:
        # 1. Normalize to [0, 1]
        mean_abs_attr = (mean_abs_attr - _min) / (_max - _min)
        # 2. Scale to [target_min, target_max]
        mean_abs_attr = mean_abs_attr * (target_max - target_min) + target_min
    else:
        # Fallback: if all values are identical (e.g. all 0.0), set to target_min
        fill_val = target_min if _max == 0 else target_max
        mean_abs_attr = np.full_like(mean_abs_attr, fill_val)

    # --- Save Data to CSV ---
    summary_df = pd.DataFrame({
        CaptumKeys.FEATURE_COLUMN: feature_names,
        CaptumKeys.IMPORTANCE_COLUMN: mean_abs_attr,
        CaptumKeys.PERCENT_COLUMN: attr_percentages
    }).sort_values(CaptumKeys.IMPORTANCE_COLUMN, ascending=False)
    
    csv_name = f"{CaptumKeys.SAVENAME}{file_suffix}.csv"
    csv_path = save_dir / csv_name
    summary_df.to_csv(csv_path, index=False)

    # --- Generate Plot ---
    plot_df = summary_df.head(20).sort_values(CaptumKeys.PERCENT_COLUMN, ascending=True)
    
    # Increase wrap width to prevent multi-line breaks for most feature names
    plot_df[CaptumKeys.FEATURE_COLUMN] = plot_df[CaptumKeys.FEATURE_COLUMN].apply(lambda x: wrap_text(x, width=35))
    
    # Calculate a dynamic height to ensure large fonts do not overlap
    dynamic_height = max(_EvaluationConfig.CAPTUM_PLOT_SIZE[1], len(plot_df) * 0.8)
    
    plt.figure(figsize=(_EvaluationConfig.CAPTUM_PLOT_SIZE[0], dynamic_height), dpi=_EvaluationConfig.DPI)
    plt.barh(plot_df[CaptumKeys.FEATURE_COLUMN], plot_df[CaptumKeys.PERCENT_COLUMN], color='mediumpurple')
    # plt.xlim(0, 1.05) # standardized scale # Removed to reflect actual percentages
    plt.xlim(left=0) # start at 0
    # plt.xlabel("Scaled Mean Absolute Attribution")
    plt.xlabel("Relative Importance (%)", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    
    title = "Feature Importance"
    
    # Use the original target name if provided, otherwise fallback to suffix logic
    if target_name:
        title += f" '{target_name}'"
    elif file_suffix:
        # Remove the leading underscore for the title
        clean_suffix = file_suffix.lstrip("_").replace("_", " ")
        title += f" '{clean_suffix}'"
        
    plt.title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE + 2)
    plt.xticks(fontsize=_EvaluationConfig.CAPTUM_X_TICK_SIZE)
    plt.yticks(fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_name = f"{CaptumKeys.PLOT_NAME}{file_suffix}.svg"
    plot_path = save_dir / plot_name
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # Use target_name for logging if available, otherwise fallback to cleaning the suffix
    log_name = target_name if target_name else file_suffix.lstrip("_").replace("_", " ")
    _LOGGER.info(f"🔬 Captum explanation for target '{log_name}' saved to '{save_dir.name}'")
