import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from pathlib import Path
from typing import Union, Optional
import json
from torchmetrics.detection import MeanAveragePrecision

from ..ML_configuration._metrics import FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics, _BaseSegmentationFormat

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionKeys, _EvaluationConfig


_LOGGER = get_logger("Vision Metrics")


__all__ = [
    "segmentation_metrics",
    "object_detection_metrics"
]


DPI_value = _EvaluationConfig.DPI


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[list[str]] = None,
    config: Optional[Union[FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics]] = None
):
    """
    Calculates and saves pixel-level metrics for segmentation tasks.

    Metrics include Pixel Accuracy, Dice (F1-score), and IoU (Jaccard).
    It calculates 'micro', 'macro', and 'weighted' averages and saves
    a pixel-level confusion matrix and a metrics heatmap.
    
    Note: This function expects integer-based masks (e.g., shape [N, H, W] or [H, W]),
    not one-hot encoded masks.
    
    Args:
        y_true (np.ndarray): Ground truth masks (e.g., shape [N, H, W]).
        y_pred (np.ndarray): Predicted masks (e.g., shape [N, H, W]).
        save_dir (str | Path): Directory to save the metrics report and plots.
        class_names (List[str] | None): Names of the classes for the report.
        config (object): Formatting configuration object.
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # --- Parse Config or use defaults ---
    if config is None:
        format_config = _BaseSegmentationFormat()
    else:
        format_config = config

    # --- Set Matplotlib font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})
    
    # Get all unique class labels present in either true or pred
    labels = np.unique(np.concatenate((np.unique(y_true), np.unique(y_pred)))).astype(int)
    
    # --- Setup Class Names ---
    display_names = []
    if class_names is None:
        display_names = [f"Class {i}" for i in labels]
    else:
        if len(class_names) != len(labels):
            _LOGGER.warning(f"Number of class_names ({len(class_names)}) does not match number of unique labels ({len(labels)}). Using default names.")
            display_names = [f"Class {i}" for i in labels]
        else:
            display_names = class_names

    # Flatten masks for sklearn metrics
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    _LOGGER.info("--- Calculating Segmentation Metrics ---")

    # --- 1. Calculate Metrics ---
    pix_acc = accuracy_score(y_true_flat, y_pred_flat)
    
    # Calculate all average types
    dice_micro = f1_score(y_true_flat, y_pred_flat, average='micro', labels=labels)
    iou_micro = jaccard_score(y_true_flat, y_pred_flat, average='micro', labels=labels)
    
    dice_macro = f1_score(y_true_flat, y_pred_flat, average='macro', labels=labels, zero_division=0)
    iou_macro = jaccard_score(y_true_flat, y_pred_flat, average='macro', labels=labels, zero_division=0)
    
    dice_weighted = f1_score(y_true_flat, y_pred_flat, average='weighted', labels=labels, zero_division=0)
    iou_weighted = jaccard_score(y_true_flat, y_pred_flat, average='weighted', labels=labels, zero_division=0)
    
    # Per-class metrics
    dice_per_class = f1_score(y_true_flat, y_pred_flat, average=None, labels=labels, zero_division=0)
    iou_per_class = jaccard_score(y_true_flat, y_pred_flat, average=None, labels=labels, zero_division=0)
    
    # --- 2. Create and Save Report ---
    report_lines = [
        "--- Segmentation Report ---",
        f"\nOverall Pixel Accuracy: {pix_acc:.4f}\n",
        "--- Averaged Metrics ---",
        f"{'Average':<10} | {'Dice (F1)':<12} | {'IoU (Jaccard)':<12}",
        "-"*41,
        f"{'Micro':<10} | {dice_micro:<12.4f} | {iou_micro:<12.4f}",
        f"{'Macro':<10} | {dice_macro:<12.4f} | {iou_macro:<12.4f}",
        f"{'Weighted':<10} | {dice_weighted:<12.4f} | {iou_weighted:<12.4f}",
        "\n--- Per-Class Metrics ---",
    ]

    per_class_data = {
        'Class': display_names,
        'Dice': dice_per_class,
        'IoU': iou_per_class
    }
    per_class_df = pd.DataFrame(per_class_data)
    report_lines.append(per_class_df.to_string(index=False, float_format="%.4f"))

    report_string = "\n".join(report_lines)
    # print(report_string) # <-- I removed the print(report_string)
    
    # Save text report
    save_filename = VisionKeys.SEGMENTATION_REPORT + ".txt"
    report_path = save_dir_path / save_filename
    report_path.write_text(report_string, encoding="utf-8")
    _LOGGER.info(f"📝 Segmentation report saved as '{report_path.name}'")

    # --- 3. Save Per-Class Metrics Heatmap ---
    try:
        plt.figure(figsize=(max(8, len(labels) * 0.5), 6), dpi=DPI_value)
        sns.heatmap(
            per_class_df.set_index('Class').T, 
            annot=True, 
            cmap=format_config.heatmap_cmap, # Use config cmap
            fmt='.3f',
            linewidths=0.5
        )
        plt.title("Per-Class Segmentation Metrics")
        plt.tight_layout()
        heatmap_filename = VisionKeys.SEGMENTATION_HEATMAP + ".svg"
        heatmap_path = save_dir_path / heatmap_filename
        plt.savefig(heatmap_path)
        _LOGGER.info(f"📊 Metrics heatmap saved as '{heatmap_path.name}'")
        plt.close()
    except Exception as e:
        _LOGGER.error(f"Could not generate segmentation metrics heatmap: {e}")

    # --- 4. Save Pixel-level Confusion Matrix ---
    try:
        # Calculate CM
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
        
        # Plot
        fig_cm, ax_cm = plt.subplots(figsize=(max(8, len(labels) * 0.8), max(8, len(labels) * 0.8)), dpi=DPI_value)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_names
        )
        disp.plot(cmap=format_config.cm_cmap, ax=ax_cm, xticks_rotation=45) # Use config cmap
        
        # Manually update font size of cell texts
        for text in disp.text_.flatten(): # type: ignore
            text.set_fontsize(format_config.font_size)
        
        ax_cm.set_title("Pixel-Level Confusion Matrix")
        plt.tight_layout()
        segmentation_cm_filename = VisionKeys.SEGMENTATION_CONFUSION_MATRIX + ".svg"
        cm_path = save_dir_path / segmentation_cm_filename
        plt.savefig(cm_path)
        _LOGGER.info(f"❇️ Pixel-level confusion matrix saved as '{cm_path.name}'")
        plt.close(fig_cm)
    except Exception as e:
        _LOGGER.error(f"Could not generate confusion matrix: {e}")
        
    # --- Restore RC params ---
    plt.rcParams.update(original_rc_params)


def object_detection_metrics(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    save_dir: Union[str, Path],
    class_names: Optional[list[str]] = None,
    print_output: bool=False
):
    """
    Calculates and saves object detection metrics (mAP) using torchmetrics.

    This function expects predictions and targets in the standard
    torchvision format (list of dictionaries).

    Args:
        preds (List[Dict[str, torch.Tensor]]): A list of predictions.
            Each dict must contain:
            - 'boxes': [N, 4] (xmin, ymin, xmax, ymax)
            - 'scores': [N]
            - 'labels': [N]
        targets (List[Dict[str, torch.Tensor]]): A list of ground truths.
            Each dict must contain:
            - 'boxes': [M, 4]
            - 'labels': [M]
        save_dir (str | Path): Directory to save the metrics report (as JSON).
        class_names (List[str] | None): A list of class names, including 'background'
            at index 0. Used to label per-class metrics in the report.
        print_output (bool): If True, prints the JSON report to the console.
    """
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")

    _LOGGER.info("--- Calculating Object Detection Metrics (mAP) ---")

    try:
        # Initialize the metric with standard COCO settings
        metric = MeanAveragePrecision(box_format='xyxy')
        
        # Move preds and targets to the same device (e.g., CPU for metric calculation)
        # This avoids device mismatches if model was on GPU
        device = torch.device("cpu")
        preds_cpu = [{k: v.to(device) for k, v in p.items()} for p in preds]
        targets_cpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Update the metric
        metric.update(preds_cpu, targets_cpu)
        
        # Compute the final metrics
        results = metric.compute()
        
        # --- Handle class names for per-class metrics ---
        report_class_names = None
        if class_names:
            if class_names[0].lower() in ['background', "bg"]:
                report_class_names = class_names[1:] # Skip background (class 0)
            else:
                _LOGGER.warning("class_names provided to object_detection_metrics, but 'background' was not class 0. Using all provided names.")
                report_class_names = class_names
        
        # Convert all torch tensors in results to floats/lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    serializable_results[key] = value.item()
                # Check if it's a 1D tensor, we have class names, and it's a known per-class key
                elif value.ndim == 1 and report_class_names and key in ('map_per_class', 'mar_100_per_class', 'mar_1_per_class', 'mar_10_per_class'):
                    per_class_list = value.cpu().numpy().tolist()
                    # Map names to values
                    if len(per_class_list) == len(report_class_names):
                        serializable_results[key] = {name: val for name, val in zip(report_class_names, per_class_list)}
                    else:
                        _LOGGER.warning(f"Length mismatch for '{key}': {len(per_class_list)} values vs {len(report_class_names)} class names. Saving as raw list.")
                        serializable_results[key] = per_class_list
                else:
                    serializable_results[key] = value.cpu().numpy().tolist()
            else:
                serializable_results[key] = value
        
        # Pretty print to console
        if print_output:
            print(json.dumps(serializable_results, indent=4))

        # Save JSON report
        detection_report_filename = VisionKeys.OBJECT_DETECTION_REPORT + ".json"
        report_path = save_dir_path / detection_report_filename
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        _LOGGER.info(f"📊 Object detection (mAP) report saved as '{report_path.name}'")

    except Exception as e:
        _LOGGER.error(f"Failed to compute mAP: {e}")
        raise

