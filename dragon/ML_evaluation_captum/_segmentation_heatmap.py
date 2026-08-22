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
    "captum_segmentation_heatmap"
]
    
def captum_segmentation_heatmap(model: nn.Module,
                                input_data: torch.Tensor,
                                save_dir: Union[str, Path],
                                target_names: Optional[list[str]],
                                n_steps: int = 30,
                                device: Union[str, torch.device] = 'cpu',
                                verbose: int = 2):
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
        verbose (int): Verbosity level for logging.
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
        if verbose >= 1:
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
    
    if verbose >= 3:
        _LOGGER.info(f"Calculating Segmentation Heatmaps for {len(target_names)} classes across {len(input_data)} samples...")

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
                # Add epsilon to prevent division by zero for uniform images
                orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
                
                fig, _ = viz.visualize_image_attr(
                    attr_np,
                    orig_np,
                    method="heat_map",
                    sign="all",
                    show_colorbar=True,
                    title=f"Sample {sample_idx} - '{class_name}'",
                    use_pyplot=False 
                )
                
                save_path = save_dir_path / f"Heatmap_Sample{sample_idx}_{clean_name}.svg"
                fig.tight_layout()
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                _LOGGER.error(f"Failed to generate heatmap for Sample {sample_idx}, Class {class_name}: {e}")
                
    if verbose >= 2:
        _LOGGER.info(f"Segmentation heatmaps saved to '{save_dir_path}'")
