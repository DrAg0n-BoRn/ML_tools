from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from captum.attr import IntegratedGradients

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from .._helpers import wrap_text
from ..keys._keys import CaptumKeys
from ..keys._config import _EvaluationConfig


_LOGGER = get_logger("Captum Sequence")

__all__ = [
    "captum_sequence_feature_importance"
]

def captum_sequence_feature_importance(model: nn.Module,
                                       input_data: torch.Tensor,
                                       feature_names: Optional[list[str]],
                                       save_dir: Union[str, Path],
                                       target_names: Optional[list[str]] = None,
                                       n_steps: int = 50,
                                       device: Union[str, torch.device] = 'cpu',
                                       verbose: int = 0):
    """
    Calculates temporal and global feature importance for Sequence models using Captum's Integrated Gradients.

    Generates three visualizations per target:
    1. Global Feature Ranking (Bar chart)
    2. Temporal Lag Importance (Line plot)
    3. Feature-Time Attribution Matrix (Heatmap)
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    device_obj = torch.device(device) if isinstance(device, str) else device
    
    model.eval()
    model.to(device_obj)
    
    inputs = input_data.clone().detach().to(device_obj)
    inputs.requires_grad = True
    
    if inputs.ndim != 3:
        _LOGGER.error(f"Sequence Captum requires 3D inputs (Batch, Seq_Len, Features). Got {inputs.ndim}D.")
        raise ValueError()
        
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs).to(device_obj)

    # --- 1. Infer Targets ---
    with torch.no_grad():
        dummy_out = model(inputs[0:1])
    
    num_outputs = 1
    output_is_1d = False
    
    if dummy_out.ndim == 1:
        num_outputs = 1
        output_is_1d = True
    elif dummy_out.ndim == 2:
        num_outputs = dummy_out.shape[1]
    else:
        # Note: If output is 3D (Batch, Seq_Len, Targets), it is expected that the wrapper 
        # (e.g., _CaptumDictWrapper) has already averaged the output sequence dimension.
        _LOGGER.warning(f"Model output has shape {dummy_out.shape}. Captum wrapper defaults to single-target interpretation.")
        num_outputs = 1

    if target_names is None:
        target_names = ["Output"] if num_outputs == 1 else [f"Output_{i}" for i in range(num_outputs)]
    
    if len(target_names) != num_outputs:
        _LOGGER.error(f"Provided {len(target_names)} target names, but model has {num_outputs} outputs.")
        raise ValueError()

    # --- 2. Iterate and Explain ---
    _LOGGER.info(f"⏳ Calculating Sequence Captum importance for {len(target_names)} target(s)...")
    
    for i, name in enumerate(target_names):
        clean_name = sanitize_filename(name)
        idx_to_explain = None if output_is_1d else i
        
        _process_single_sequence_target(
            ig=ig,
            inputs=inputs,
            baseline=baseline,
            target_index=idx_to_explain,
            feature_names=feature_names,
            save_dir=save_dir_path,
            n_steps=n_steps,
            file_suffix=f"_{clean_name}",
            target_name=name,
            verbose=verbose
        )

def _process_single_sequence_target(ig: 'IntegratedGradients', # type: ignore
                                    inputs: torch.Tensor,
                                    baseline: torch.Tensor,
                                    target_index: Union[int, None],
                                    feature_names: Optional[list[str]],
                                    save_dir: Path,
                                    n_steps: int,
                                    file_suffix: str,
                                    target_name: str,
                                    verbose: int):
    try:
        with torch.backends.cudnn.flags(enabled=False):
            attributions, delta = ig.attribute(inputs, 
                                            baselines=baseline, 
                                            target=target_index,
                                            n_steps=n_steps,
                                            internal_batch_size=inputs.shape[0],
                                            return_convergence_delta=True)
        
        mean_delta = torch.mean(torch.abs(delta)).item()
        if mean_delta > 0.1 and verbose > 1:
            _LOGGER.warning(f"Captum Convergence Delta is high ({mean_delta:.4f}). Consider increasing 'n_steps'.")
            
    except Exception as e:
        _LOGGER.error(f"Captum sequence attribution failed for target '{target_index}': {e}")
        return

    # Shape: (Batch, Seq_Len, Features)
    attributions_np = attributions.detach().cpu().numpy()
    seq_len = attributions_np.shape[1]
    num_features = attributions_np.shape[2]
    
    if feature_names is None or len(feature_names) != num_features:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    # --- Aggregations ---
    # 1. Heatmap (Average across batch) -> (Seq_Len, Features)
    heatmap_attr = np.mean(np.abs(attributions_np), axis=0)
    
    # 2. Temporal Lag (Average across batch and features) -> (Seq_Len,)
    temporal_attr = np.mean(heatmap_attr, axis=1)
    
    # 3. Global Feature (Average across batch and time) -> (Features,)
    global_attr = np.mean(heatmap_attr, axis=0)
    
    # Min-Max scale global attributes for the bar chart
    _min, _max = np.min(global_attr), np.max(global_attr)
    if _max > _min:
        global_attr_scaled = ((global_attr - _min) / (_max - _min)) * 0.99 + 0.01
    else:
        global_attr_scaled = np.full_like(global_attr, 0.01)
        
    total_attr_sum = np.sum(global_attr)
    attr_percentages = (global_attr / total_attr_sum * 100.0) if total_attr_sum > 0 else np.zeros_like(global_attr)

    # --- Save CSV (Global) ---
    summary_df = pd.DataFrame({
        CaptumKeys.FEATURE_COLUMN: feature_names,
        CaptumKeys.IMPORTANCE_COLUMN: global_attr_scaled,
        CaptumKeys.PERCENT_COLUMN: attr_percentages
    }).sort_values(CaptumKeys.IMPORTANCE_COLUMN, ascending=False)
    
    summary_df.to_csv(save_dir / f"{CaptumKeys.SAVENAME}{file_suffix}.csv", index=False)

    # ==========================================
    # PLOT 1: Feature-Time Heatmap
    # ==========================================
    fig_w = max(10, seq_len * 0.5)
    fig_h = max(6, num_features * 0.4)
    
    plt.figure(figsize=(fig_w, fig_h), dpi=_EvaluationConfig.DPI)
    sns.heatmap(
        heatmap_attr.T,
        cmap="rocket_r",
        yticklabels=feature_names,
        xticklabels=[str(i) for i in range(1, seq_len + 1)],
    )
                
    plt.title(f"Feature-Time Attribution Heatmap\n'{target_name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE + 2)
    plt.xlabel("Sequence Time Step (Lag -> Recent)", labelpad=_EvaluationConfig.LABEL_PADDING)
    plt.ylabel("Features", labelpad=_EvaluationConfig.LABEL_PADDING)
    plt.yticks(fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(save_dir / f"feature_time_heatmap{file_suffix}.svg", bbox_inches='tight')
    plt.close()

    # ==========================================
    # PLOT 2: Temporal Lag Line Plot
    # ==========================================
    plt.figure(figsize=(_EvaluationConfig.CAPTUM_PLOT_SIZE[0], 6), dpi=_EvaluationConfig.DPI)
    plt.plot(range(1, seq_len + 1), temporal_attr, marker='o', color='mediumpurple', linewidth=2, markersize=8)
    
    plt.title(f"Temporal Lag Importance\n'{target_name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE + 2)
    plt.xlabel("Sequence Time Step (Lag -> Recent)", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.ylabel("Mean Absolute Attribution", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.xticks(range(1, seq_len + 1), fontsize=_EvaluationConfig.CAPTUM_X_TICK_SIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"temporal_lag_importance{file_suffix}.svg", bbox_inches='tight')
    plt.close()

    # ==========================================
    # PLOT 3: Global Feature Bar Chart
    # ==========================================
    plot_df = summary_df.head(20).sort_values(CaptumKeys.PERCENT_COLUMN, ascending=True)
    plot_df[CaptumKeys.FEATURE_COLUMN] = plot_df[CaptumKeys.FEATURE_COLUMN].apply(lambda x: wrap_text(x, width=20))
    
    dynamic_height = max(_EvaluationConfig.CAPTUM_PLOT_SIZE[1], len(plot_df) * 0.8)
    
    plt.figure(figsize=(_EvaluationConfig.CAPTUM_PLOT_SIZE[0], dynamic_height), dpi=_EvaluationConfig.DPI)
    plt.barh(plot_df[CaptumKeys.FEATURE_COLUMN], plot_df[CaptumKeys.PERCENT_COLUMN], color='mediumpurple')
    plt.xlim(left=0)
    plt.xlabel("Relative Importance (%)", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.title(f"Feature Importance '{target_name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE + 2)
    plt.xticks(fontsize=_EvaluationConfig.CAPTUM_X_TICK_SIZE)
    plt.yticks(fontsize=_EvaluationConfig.CAPTUM_FONT_SIZE)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_dir / f"{CaptumKeys.PLOT_NAME}{file_suffix}.svg", bbox_inches='tight')
    plt.close()

    log_name = target_name if target_name else file_suffix.lstrip("_").replace("_", " ")
    _LOGGER.info(f"🔬 Sequence Captum explanations for target '{log_name}' saved to '{save_dir.name}'")
