import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from pathlib import Path
from typing import Union, Optional

from ..ML_configuration._metrics import FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics, _BaseSegmentationFormat

from ..path_manager import make_fullpath
from .._core import get_logger, wrap_text
from ..keys._keys import VisionKeys, _EvaluationConfig

from ._radar_plots import save_radar_chart, mpl_to_plotly_rgba, calculate_smart_margin_left_right

_LOGGER = get_logger("Segmentation Metrics")
DPI_value = _EvaluationConfig.DPI

__all__ = ["segmentation_metrics"]


def _calculate_global_metrics(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, labels: np.ndarray) -> dict:
    """Calculates global pixel accuracy, and average/per-class Dice and IoU scores."""
    return {
        'pix_acc': accuracy_score(y_true_flat, y_pred_flat),
        'dice_micro': f1_score(y_true_flat, y_pred_flat, average='micro', labels=labels),
        'iou_micro': jaccard_score(y_true_flat, y_pred_flat, average='micro', labels=labels),
        'dice_macro': f1_score(y_true_flat, y_pred_flat, average='macro', labels=labels, zero_division=0),
        'iou_macro': jaccard_score(y_true_flat, y_pred_flat, average='macro', labels=labels, zero_division=0),
        'dice_weighted': f1_score(y_true_flat, y_pred_flat, average='weighted', labels=labels, zero_division=0),
        'iou_weighted': jaccard_score(y_true_flat, y_pred_flat, average='weighted', labels=labels, zero_division=0),
        'dice_per_class': f1_score(y_true_flat, y_pred_flat, average=None, labels=labels, zero_division=0),
        'iou_per_class': jaccard_score(y_true_flat, y_pred_flat, average=None, labels=labels, zero_division=0)
    }


def _calculate_per_image_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, display_names: list[str]) -> pd.DataFrame:
    """Calculates Dice and IoU scores per image to capture distribution variance."""
    # Ensure 3D shape [N, H, W] for iteration
    if y_true.ndim == 2:
        y_true = np.expand_dims(y_true, axis=0)
        y_pred = np.expand_dims(y_pred, axis=0)
        
    records = []
    for i in range(y_true.shape[0]):
        yt = y_true[i].ravel()
        yp = y_pred[i].ravel()
        
        # Use 0 for zero_division to avoid NaNs in per-image metrics when a class is missing in either GT or predictions
        dice = np.asarray(f1_score(yt, yp, average=None, labels=labels, zero_division=0))
        iou = np.asarray(jaccard_score(yt, yp, average=None, labels=labels, zero_division=0))
        
        for j, _ in enumerate(labels):
            # Only record if the class was actually present in ground truth or predictions
            if not np.isnan(dice[j]):
                records.append({'Image': i, 'Class': display_names[j], 'Metric': 'Dice', 'Score': dice[j]})
                records.append({'Image': i, 'Class': display_names[j], 'Metric': 'IoU', 'Score': iou[j]})
                
    return pd.DataFrame(records)


def _generate_text_report(metrics: dict, per_class_df: pd.DataFrame, save_dir_path: Path) -> None:
    """Formats global metrics into a textual report and saves it."""
    report_lines = [
        "--- Segmentation Report ---",
        f"\nOverall Pixel Accuracy: {metrics['pix_acc']:.4f}\n",
        "--- Averaged Metrics ---",
        f"{'Average':<10} | {'Dice (F1)':<12} | {'IoU (Jaccard)':<12}",
        "-"*41,
        f"{'Micro':<10} | {metrics['dice_micro']:<12.4f} | {metrics['iou_micro']:<12.4f}",
        f"{'Macro':<10} | {metrics['dice_macro']:<12.4f} | {metrics['iou_macro']:<12.4f}",
        f"{'Weighted':<10} | {metrics['dice_weighted']:<12.4f} | {metrics['iou_weighted']:<12.4f}",
        "\n--- Per-Class Metrics ---",
        per_class_df.to_string(index=False, float_format="%.4f")
    ]
    
    report_string = "\n".join(report_lines)
    report_path = save_dir_path / f"{VisionKeys.SEGMENTATION_REPORT}.txt"
    report_path.write_text(report_string, encoding="utf-8")
    _LOGGER.info(f"📝 Segmentation report saved as '{report_path.name}'")


def _plot_metrics_heatmap(per_class_df: pd.DataFrame, format_config, save_dir_path: Path) -> None:
    """Generates and saves a seaborn heatmap for per-class global metrics."""
    try:
        # Increased base figure size to accommodate larger fonts
        plt.figure(figsize=(max(10, len(per_class_df) * 1.0), 8), dpi=DPI_value)
        
        # Smart dynamic annotation size scaling
        dynamic_annot_size = max(8, format_config.font_size - max(0, (len(per_class_df) // 2)))
        
        ax = sns.heatmap(
            per_class_df.set_index('Class').T, 
            annot=True, 
            cmap=format_config.heatmap_cmap, 
            fmt='.3f',
            linewidths=0.5,
            annot_kws={"size": dynamic_annot_size},
            cbar=False,
            vmin=0.0,
            vmax=1.0
        )
        
        xtick_size = getattr(format_config, 'xtick_size', format_config.font_size - 4)
        ytick_size = getattr(format_config, 'ytick_size', format_config.font_size - 4)
        
        # Remove the "Class" x-label
        ax.set_xlabel("")
        
        plt.xticks(fontsize=xtick_size, rotation=45, ha='right', rotation_mode='anchor')
        plt.yticks(fontsize=ytick_size)
        
        plt.title("Per-Class Segmentation Metrics", pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
        plt.tight_layout()
        heatmap_path = save_dir_path / f"{VisionKeys.SEGMENTATION_HEATMAP}.svg"
        plt.savefig(heatmap_path, bbox_inches='tight')
        _LOGGER.info(f"📊 Metrics heatmap saved as '{heatmap_path.name}'")
        plt.close()
    except Exception as e:
        _LOGGER.error(f"Could not generate segmentation metrics heatmap: {e}")


def _plot_confusion_matrix(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, labels: np.ndarray, 
                           display_names: list[str], format_config, save_dir_path: Path) -> None:
    """Calculates and plots a pixel-level confusion matrix."""
    try:
        # Normalize to scale values from 0.0 to 1.0 across true labels (rows)
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels, normalize='true')
        
        # Increased base figure size
        fig_cm, ax_cm = plt.subplots(figsize=(max(10, len(labels) * 1.0), max(10, len(labels) * 1.0)), dpi=DPI_value)
        
        xtick_size = getattr(format_config, 'xtick_size', format_config.font_size - 4)
        ytick_size = getattr(format_config, 'ytick_size', format_config.font_size - 4)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        
        # Disable the default colorbar, format the numbers, and strictly lock the color scale to 0.0 - 1.0
        disp.plot(cmap=format_config.cm_cmap, 
                  ax=ax_cm, 
                  colorbar=False, 
                  values_format='.2f', 
                  im_kw={'vmin': 0.0, 'vmax': 1.0}) 
        
        # Create a smaller colorbar and scale down its ticks
        cbar = fig_cm.colorbar(disp.im_, ax=ax_cm, shrink=0.75)
        cbar.ax.tick_params(labelsize=xtick_size)
        
        # Smart dynamic annotation size for numbers inside the matrix
        dynamic_cm_size = max(8, format_config.font_size - max(0, len(labels) // 2))
        if disp.text_ is not None:
            for text in disp.text_.flatten():
                text.set_fontsize(dynamic_cm_size)
        
        ax_cm.set_xlabel(ax_cm.get_xlabel(), fontsize=format_config.font_size)
        ax_cm.set_ylabel(ax_cm.get_ylabel(), fontsize=format_config.font_size)
        ax_cm.tick_params(axis='x', labelsize=xtick_size)
        ax_cm.tick_params(axis='y', labelsize=ytick_size)
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        ax_cm.set_title("Pixel-Level Confusion Matrix", pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
        plt.tight_layout()
        cm_path = save_dir_path / f"{VisionKeys.SEGMENTATION_CONFUSION_MATRIX}.svg"
        plt.savefig(cm_path, bbox_inches='tight')
        _LOGGER.info(f"❇️ Pixel-level confusion matrix saved as '{cm_path.name}'")
        plt.close(fig_cm)
    except Exception as e:
        _LOGGER.error(f"Could not generate confusion matrix: {e}")


def _plot_distribution_boxplots(image_metrics_df: pd.DataFrame, format_config, save_dir_path: Path) -> None:
    """Generates and saves Image-Level Distribution Boxplots."""
    if image_metrics_df.empty:
        return
        
    try:
        xtick_size = getattr(format_config, 'xtick_size', format_config.font_size - 4)
        ytick_size = getattr(format_config, 'ytick_size', format_config.font_size - 4)
        
        for metric in ['Dice', 'IoU']:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI_value)
            
            metric_df = image_metrics_df[image_metrics_df['Metric'] == metric]
            
            sns.boxplot(data=metric_df, x='Class', y='Score', ax=ax, hue='Class', palette='husl', legend=False, showfliers=False)
            sns.stripplot(data=metric_df, x='Class', y='Score', ax=ax, color="black", alpha=0.3, size=3, jitter=True)
            
            # Remove top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_title(f"Per-Image {metric} Distribution", pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            ax.set_ylim((-0.05, 1.05))
            
            # Remove X-label "Class" and scale Y-label
            ax.set_xlabel("")
            ax.set_ylabel("Score", fontsize=format_config.font_size)
            
            # Apply tick sizes and 45-degree rotation
            ax.tick_params(axis='x', labelsize=xtick_size)
            ax.tick_params(axis='y', labelsize=ytick_size)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')

            plt.tight_layout()
            dist_path = save_dir_path / f"{VisionKeys.SEGMENTATION_DISTRIBUTION_PLOT}_{metric}.svg"
            plt.savefig(dist_path, bbox_inches='tight')
            _LOGGER.info(f"📈 {metric} distribution boxplot saved as '{dist_path.name}'")
            plt.close(fig)
    except Exception as e:
        _LOGGER.error(f"Could not generate distribution boxplots: {e}")


def _plot_radar_charts(per_class_df: pd.DataFrame, display_names: list[str], format_config, save_dir_path: Path) -> None:
    """Generates and saves interactive Radar Charts using Plotly."""
    if len(display_names) < 3:
        _LOGGER.info("🕸️ Skipping radar charts: At least 3 classes are required to form a polygon.")
        return
    
    try:
        max_length = max([len(n) for n in display_names])
        margin_lr = calculate_smart_margin_left_right(max_length)
        fill_rgba = mpl_to_plotly_rgba(format_config.radar_line_color, format_config.radar_fill_alpha)
        plotly_line_color = mcolors.to_hex(format_config.radar_line_color)
        
        for metric in ['Dice', 'IoU']:
            scores = per_class_df[metric].tolist()
            save_base = save_dir_path / f"{VisionKeys.SEGMENTATION_RADAR_PLOT}_{metric}"
            
            save_radar_chart(
                scores=scores,
                target_names=display_names,
                line_color=plotly_line_color,
                fill_rgba=fill_rgba,
                title=f"Per-Class {metric} Score",
                save_path_base=save_base,
                margin_lr=margin_lr,
                font_size=format_config.font_size
            )
        _LOGGER.info("🕸️ Radar charts saved successfully.")
    except Exception as e:
        _LOGGER.error(f"Could not generate radar charts: {e}")


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[list[str]] = None,
    config: Optional[Union[FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics]] = None
):
    """
    Main orchestrator function. Calculates and saves pixel-level metrics and visual reports for segmentation tasks.
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    format_config = config if config is not None else _BaseSegmentationFormat()

    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})
    
    labels = np.unique(np.concatenate((np.unique(y_true), np.unique(y_pred)))).astype(int)
    
    if class_names is None or len(class_names) != len(labels):
        if class_names is not None:
            _LOGGER.warning(f"Number of class_names ({len(class_names)}) does not match unique labels ({len(labels)}). Using default names.")
        display_names = [f"Class {i}" for i in labels]
    else:
        display_names = [wrap_text(_name) for _name in class_names]

    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # _LOGGER.info("--- Calculating Segmentation Metrics ---")

    # 1. Calculate Global Metrics
    metrics_dict = _calculate_global_metrics(y_true_flat, y_pred_flat, labels)
    
    per_class_df = pd.DataFrame({
        'Class': display_names,
        'Dice': metrics_dict['dice_per_class'],
        'IoU': metrics_dict['iou_per_class']
    })

    # 2. Calculate Image-Level Distributions
    image_metrics_df = _calculate_per_image_metrics(y_true, y_pred, labels, display_names)

    # 3. Generate Reports & Static Plots
    _generate_text_report(metrics_dict, per_class_df, save_dir_path)
    _plot_metrics_heatmap(per_class_df, format_config, save_dir_path)
    _plot_confusion_matrix(y_true_flat, y_pred_flat, labels, display_names, format_config, save_dir_path)
    _plot_distribution_boxplots(image_metrics_df, format_config, save_dir_path)
    
    # 4. Generate Interactive Plots
    _plot_radar_charts(per_class_df, display_names, format_config, save_dir_path)

    plt.rcParams.update(original_rc_params)
