import torch
import pandas as pd

from pathlib import Path
from typing import Union, Optional
import json
from torchmetrics.detection import MeanAveragePrecision

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionKeys
from ..keys._config import _EvaluationConfig


_LOGGER = get_logger("Object Detection Metrics")


__all__ = [
    "object_detection_metrics"
]


DPI_value = _EvaluationConfig.DPI


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
        detection_report_filename_json = VisionKeys.OBJECT_DETECTION_REPORT + ".json"
        report_path_json = save_dir_path / detection_report_filename_json
        with open(report_path_json, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        # Save CSV report
        detection_report_filename_csv = VisionKeys.OBJECT_DETECTION_REPORT + ".csv"
        report_path_csv = save_dir_path / detection_report_filename_csv
        pd.DataFrame([serializable_results]).to_csv(report_path_csv, index=False)
        
        _LOGGER.info(f"📊 Object detection (mAP) reports saved as '{report_path_json.name}' and '{report_path_csv.name}'")

    except Exception as e:
        _LOGGER.error(f"Failed to compute mAP: {e}")
        raise

