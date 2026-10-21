from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

_LOGGER = get_logger("Captum")

__all__ = [
    "captum_image_heatmap"
]

def captum_image_heatmap(model: nn.Module,
                         input_data: torch.Tensor,
                         save_dir: Union[str, Path],
                         target_names: Optional[list[str]] = None,
                         n_steps: int = 50,
                         device: Union[str, torch.device] = 'cpu',
                         verbose: int = 2):
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
        verbose (int): Verbosity level for logging.
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
    
    if verbose >= 3:
        _LOGGER.info(f"Calculating Image Heatmaps for {len(target_names)} targets across {len(input_data)} samples...")

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
                # Add epsilon to prevent division by zero for uniform images
                orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)

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
                save_path = save_dir_path / f"Saliency_Sample{sample_idx}_{clean_name}.svg"
                plt.tight_layout()
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                _LOGGER.error(f"Failed to generate heatmap for Sample {sample_idx}, Class {class_name}: {e}")
    
    if verbose >= 2:
        _LOGGER.info(f"🔬 Completed generating heatmaps for {len(input_data)} samples and {len(target_names)} targets. Saved to '{save_dir_path}'")
