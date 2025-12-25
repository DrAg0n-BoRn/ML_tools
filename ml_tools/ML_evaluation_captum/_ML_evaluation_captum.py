from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import CaptumKeys


_LOGGER = get_logger("Captum")


__all__ = [
    "captum_feature_importance", 
    "captum_image_heatmap",
    "captum_segmentation_heatmap"
    ]



def captum_feature_importance(model: nn.Module,
                              input_data: torch.Tensor,
                              feature_names: Optional[list[str]],
                              save_dir: Union[str, Path],
                              target_names: Optional[list[str]] = None,
                              n_steps: int = 50,
                              device: Union[str, torch.device] = 'cpu'):
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
    _LOGGER.info(f"🔄 Calculating Captum importance for {len(target_names)} target(s)")
    
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
            file_suffix=f"_{clean_name}"
        )


def _process_single_target(ig: 'IntegratedGradients', # type: ignore
                           inputs: torch.Tensor,
                           baseline: torch.Tensor,
                           target_index: Union[int, None],
                           feature_names: Optional[list[str]],
                           save_dir: Path,
                           n_steps: int,
                           file_suffix: str):
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
        if mean_delta > 0.1:
            _LOGGER.warning(f"Captum Convergence Delta is high ({mean_delta:.4f}). The attribution approximation may be inaccurate. Consider increasing 'n_steps'.")
        
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

    # --- Save Data to CSV ---
    summary_df = pd.DataFrame({
        CaptumKeys.FEATURE_COLUMN: feature_names,
        CaptumKeys.IMPORTANCE_COLUMN: mean_abs_attr
    }).sort_values(CaptumKeys.IMPORTANCE_COLUMN, ascending=False)
    
    csv_name = f"{CaptumKeys.SAVENAME}{file_suffix}.csv"
    csv_path = save_dir / csv_name
    summary_df.to_csv(csv_path, index=False)

    # --- Generate Plot ---
    plot_df = summary_df.head(20).sort_values(CaptumKeys.IMPORTANCE_COLUMN, ascending=True)
    plt.figure(figsize=(10, 8), dpi=200)
    plt.barh(plot_df[CaptumKeys.FEATURE_COLUMN], plot_df[CaptumKeys.IMPORTANCE_COLUMN], color='mediumpurple')
    plt.xlabel("Mean Absolute Attribution")
    
    title = "Feature Importance"
    if file_suffix:
        # Remove the leading underscore for the title
        clean_suffix = file_suffix.lstrip("_").replace("_", " ")
        title += f" ({clean_suffix})"
        
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_name = f"{CaptumKeys.PLOT_NAME}{file_suffix}.svg"
    plot_path = save_dir / plot_name
    plt.savefig(plot_path)
    plt.close()
    
    _LOGGER.info(f"🔬 Captum explanation for target '{file_suffix.lstrip("_").replace("_", " ")}' saved to '{save_dir.name}'")


def captum_image_heatmap(model: nn.Module,
                         input_data: torch.Tensor,
                         save_dir: Union[str, Path],
                         target_names: Optional[list[str]] = None,
                         n_steps: int = 50,
                         device: Union[str, torch.device] = 'cpu'):
    """
    Generates Saliency Heatmaps for Image Classification models using Integrated Gradients.

    This function calculates the pixel-wise attribution for the predicted classes and 
    overlays it as a heatmap on the original image. It visualizes the first sample 
    in the input batch.

    Args:
        model (nn.Module): The PyTorch Image Classification model.
        input_data (torch.Tensor): A batch of input images to explain. Shape: (N, C, H, W).
        save_dir (str | Path): The directory where the heatmap images will be saved.
        target_names (List[str] | None): A list of class names corresponding to the model outputs.
                                         If None, generic names (e.g., "Class_0") are generated.
        n_steps (int): The number of steps used by the Integrated Gradients approximation. 
                       Higher values increase accuracy but require more memory/time.
        device (str | torch.device): The device to run the calculation on.
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    device_obj = torch.device(device) if isinstance(device, str) else device
    
    model.eval()
    model.to(device_obj)
    
    # We simply need the model to infer the number of classes once
    with torch.no_grad():
        # Check 1st sample for dimensions
        dummy_out = model(input_data[0:1].to(device_obj)) 
        num_classes = dummy_out.shape[1] if dummy_out.ndim > 1 else 1

    if target_names is None:
        target_names = [f"Class_{i}" for i in range(num_classes)]

    _LOGGER.info(f"🔄 Calculating Image Heatmaps for {len(target_names)} targets across {len(input_data)} samples...")

    ig = IntegratedGradients(model)

    # --- OUTER LOOP: Iterate over samples to save memory ---
    for sample_idx in range(len(input_data)):
        
        # Slice: (1, C, H, W) -> Process one image at a time to avoid OOM
        single_input = input_data[sample_idx:sample_idx+1].clone().detach().to(device_obj)
        single_input.requires_grad = True
        baseline = torch.zeros_like(single_input).to(device_obj)

        # --- INNER LOOP: Iterate over targets ---
        for class_idx, class_name in enumerate(target_names):
            clean_name = sanitize_filename(class_name)
            target_param = None if num_classes == 1 else class_idx

            try:
                attributions, _ = ig.attribute(single_input, 
                                               baselines=baseline, 
                                               target=target_param,
                                               n_steps=n_steps,
                                               return_convergence_delta=True)
                
                attr_tensor = attributions[0].cpu().detach()
                orig_tensor = single_input[0].cpu().detach()
                
                attr_np = attr_tensor.permute(1, 2, 0).numpy()
                orig_np = orig_tensor.permute(1, 2, 0).numpy()
                orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())

                # Create plot
                fig, _ = viz.visualize_image_attr(
                    attr_np,
                    orig_np,
                    method="heat_map",
                    sign="all",
                    show_colorbar=True,
                    title=f"Sample {sample_idx} - '{class_name}'",
                    use_pyplot=False
                )
                
                # Save with Sample ID
                save_path = save_dir_path / f"Saliency_Sample{sample_idx}_{clean_name}.png"
                fig.savefig(save_path)
                plt.close(fig)

            except Exception as e:
                _LOGGER.error(f"Failed to generate heatmap for Sample {sample_idx}, Class {class_name}: {e}")

    
def captum_segmentation_heatmap(model: nn.Module,
                                input_data: torch.Tensor,
                                save_dir: Union[str, Path],
                                target_names: Optional[list[str]],
                                n_steps: int = 30,
                                device: Union[str, torch.device] = 'cpu'):
    """
    Generates attribution heatmaps for Semantic Segmentation models.
    
    Since segmentation outputs are spatial (H, W), this function wraps the model
    to sum the logits for a specific class across the entire image, effectively
    answering: "Which pixels contributed to the total evidence for Class X?"

    Args:
        model (nn.Module): The segmentation model.
        input_data (torch.Tensor): Input batch. Should be small (e.g. 1-5 images) as this is expensive.
        save_dir (str | Path): Output directory.
        target_names (List[str]): List of class names corresponding to the model's output channels. If None, generic names will be generated based on output shape.
        n_steps (int): Integration steps. Kept lower by default (30) for performance on high-res images.
        device (str | torch.device): Torch device.
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device_obj)
    
    # --- Infer Classes if not provided ---
    with torch.no_grad():
        # Check first sample
        first_sample = input_data[0:1].to(device_obj)
        dummy_out = model(first_sample)
        # Handle dict output (common in torchvision models)
        if isinstance(dummy_out, dict) and 'out' in dummy_out:
            dummy_out: torch.Tensor = dummy_out['out'] # type: ignore
    
    # Shape should be (N, C, H, W)
    if dummy_out.ndim == 4:
        num_classes = dummy_out.shape[1]
    else:
        _LOGGER.warning(f"Unexpected segmentation output shape {dummy_out.shape}. Assuming 1 class.")
        num_classes = 1

    if target_names is None:
        target_names = [f"Class_{i}" for i in range(num_classes)]
        # _LOGGER.info(f"No 'target_names' provided for segmentation. Generated generics: {target_names}")

    if len(target_names) != num_classes:
        _LOGGER.error(f"Name mismatch: Provided {len(target_names)} names, but model has {num_classes} output channels.")
        raise ValueError()
    
    # Wrapper 
    def segmentation_wrapper(inp):
        out = model(inp)
        if isinstance(out, dict) and 'out' in out:
            out: torch.Tensor = out['out']  # type: ignore
        return out.sum(dim=(2, 3))
    
    ig = IntegratedGradients(segmentation_wrapper)
    
    _LOGGER.info(f"🔄 Calculating Segmentation Heatmaps for {len(target_names)} classes across {len(input_data)} samples...")

    # --- OUTER LOOP: Iterate over samples ---
    for sample_idx in range(len(input_data)):
        
        # Slice: (1, C, H, W)
        single_input = input_data[sample_idx:sample_idx+1].clone().detach().to(device_obj)
        single_input.requires_grad = True
        baseline = torch.zeros_like(single_input).to(device_obj)
        
        # --- INNER LOOP: Iterate over classes ---
        for class_idx, class_name in enumerate(target_names):
            clean_name = sanitize_filename(class_name)
            
            try:
                attributions, _ = ig.attribute(single_input, 
                                               baselines=baseline, 
                                               target=class_idx,
                                               n_steps=n_steps,
                                               return_convergence_delta=True)
                
                attr_tensor = attributions[0].cpu().detach()
                orig_tensor = single_input[0].cpu().detach()
                
                attr_np = attr_tensor.permute(1, 2, 0).numpy()
                orig_np = orig_tensor.permute(1, 2, 0).numpy()
                orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())
                
                fig, _ = viz.visualize_image_attr(
                    attr_np,
                    orig_np,
                    method="heat_map",
                    sign="all",
                    show_colorbar=True,
                    title=f"Sample {sample_idx} - '{class_name}'",
                    use_pyplot=False 
                )
                
                save_path = save_dir_path / f"Heatmap_Sample{sample_idx}_{clean_name}.png"
                fig.savefig(save_path)
                plt.close(fig)
                
            except Exception as e:
                _LOGGER.error(f"Failed to generate heatmap for Sample {sample_idx}, Class {class_name}: {e}")

