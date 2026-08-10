import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from pathlib import Path
from typing import Union, Optional, Any

from ._eval_regression import regression_metrics
from ._eval_classification import classification_metrics

from ..path_manager import make_fullpath
from ..keys._config import _EvaluationConfig
from .._core import get_logger

_LOGGER = get_logger("Sequence Metrics")

__all__ = [
    "sequence_to_sequence_regression_metrics",
    "sequence_to_sequence_classification_metrics"
]

DPI_value = _EvaluationConfig.DPI
SEQUENCE_PLOT_SIZE = _EvaluationConfig.SEQUENCE_PLOT_SIZE


def sequence_to_sequence_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[Any] = None
):
    """
    Saves overall regression metrics and per-step plots for Sequence-to-Sequence data.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Invalid dimensions. Sequence metrics expect (n_samples, sequence_length).")

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # 1. Flatten arrays to calculate overall robust regression reports
    overall_dir = save_dir_path / "overall_metrics"
    # Extract the sub-config for the base regression metrics
    sub_config = config.regression_config if config else None 
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    max_points = _EvaluationConfig.SEQ_SEQ_MAX_POINTS
    if len(y_true_flat) > max_points:
        _LOGGER.info(f"Subsampling overall regression metrics to {max_points} points to prevent massive SVG generation.")
        indices = np.random.choice(len(y_true_flat), max_points, replace=False)
        y_true_flat = y_true_flat[indices]
        y_pred_flat = y_pred_flat[indices]

    regression_metrics(y_true=y_true_flat, y_pred=y_pred_flat, save_dir=overall_dir, config=sub_config)

    # 2. Per-Step Metrics Plotting
    sequence_length = y_true.shape[1]
    steps = list(range(1, sequence_length + 1))
    per_step_rmse, per_step_mae = [], []

    for i in range(sequence_length):
        y_true_step, y_pred_step = y_true[:, i], y_pred[:, i]
        per_step_rmse.append(np.sqrt(mean_squared_error(y_true_step, y_pred_step)))
        per_step_mae.append(mean_absolute_error(y_true_step, y_pred_step))

    fig, ax1 = plt.subplots(figsize=SEQUENCE_PLOT_SIZE, dpi=DPI_value)

    # Plot RMSE
    color_rmse = getattr(config, 'rmse_color', 'blue') if config else 'blue'
    ax1.set_xlabel('Prediction Step', labelpad=_EvaluationConfig.LABEL_PADDING)
    ax1.set_ylabel('RMSE', color=color_rmse, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax1.plot(steps, per_step_rmse, marker='o', color=color_rmse, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, linestyle='--')

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    color_mae = getattr(config, 'mae_color', 'red') if config else 'red'
    ax2.set_ylabel('MAE', color=color_mae, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax2.plot(steps, per_step_mae, marker='s', color=color_mae, label='MAE')
    ax2.tick_params(axis='y', labelcolor=color_mae)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.set_title('Per-Step Regression Metrics', pad=_EvaluationConfig.LABEL_PADDING)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout()
    plot_path = save_dir_path / "per_step_regression_plot.svg"
    plt.savefig(plot_path, bbox_inches='tight')
    _LOGGER.info(f"Seq-to-Seq per-step plot saved as '{plot_path.name}'")
    plt.close(fig)


def sequence_to_sequence_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[Any] = None
):
    """
    Saves overall classification metrics and per-step plots for Sequence-to-Sequence data.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Invalid dimensions. Sequence metrics expect (n_samples, sequence_length).")

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # 1. Flatten arrays to calculate overall robust classification reports/heatmaps
    overall_dir = save_dir_path / "overall_metrics"
    # Extract the sub-config for the base classification metrics
    sub_config = config.classification_config if config else None
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    max_points = _EvaluationConfig.SEQ_SEQ_MAX_POINTS
    if len(y_true_flat) > max_points:
        _LOGGER.info(f"Subsampling overall classification metrics to {max_points} points to prevent massive SVG generation.")
        indices = np.random.choice(len(y_true_flat), max_points, replace=False)
        y_true_flat = y_true_flat[indices]
        y_pred_flat = y_pred_flat[indices]

    classification_metrics(y_true=y_true_flat, y_pred=y_pred_flat, save_dir=overall_dir, config=sub_config)

    # 2. Per-Step Metrics Plotting
    sequence_length = y_true.shape[1]
    steps = list(range(1, sequence_length + 1))
    per_step_acc, per_step_f1 = [], []

    for i in range(sequence_length):
        y_true_step, y_pred_step = y_true[:, i], y_pred[:, i]
        per_step_acc.append(accuracy_score(y_true_step, y_pred_step))
        per_step_f1.append(f1_score(y_true_step, y_pred_step, average='weighted', zero_division=0))

    fig, ax1 = plt.subplots(figsize=SEQUENCE_PLOT_SIZE, dpi=DPI_value)

    # Plot Accuracy
    color_acc = getattr(config, 'acc_color', 'purple') if config else 'purple'
    ax1.set_xlabel('Prediction Step', labelpad=_EvaluationConfig.LABEL_PADDING)
    ax1.set_ylabel('Accuracy', color=color_acc, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax1.plot(steps, per_step_acc, marker='o', color=color_acc, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--')

    # Create a second y-axis for F1-Score
    ax2 = ax1.twinx()
    color_f1 = getattr(config, 'f1_color', 'orange') if config else 'orange'
    ax2.set_ylabel('Weighted F1-Score', color=color_f1, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax2.plot(steps, per_step_f1, marker='^', color=color_f1, label='F1-Score')
    ax2.tick_params(axis='y', labelcolor=color_f1)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.set_title('Per-Step Classification Metrics', pad=_EvaluationConfig.LABEL_PADDING)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout()
    plot_path = save_dir_path / "per_step_classification_plot.svg"
    plt.savefig(plot_path, bbox_inches='tight')
    _LOGGER.info(f"Seq-to-Seq per-step classification plot saved as '{plot_path.name}'")
    plt.close(fig)
