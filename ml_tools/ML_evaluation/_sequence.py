import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score, 
    median_absolute_error,
)
from pathlib import Path
from typing import Union, Optional

from ..ML_configuration._metrics import FormatSequenceValueMetrics, FormatSequenceSequenceMetrics, _BaseSequenceValueFormat, _BaseSequenceSequenceFormat

from ..keys._keys import _EvaluationConfig
from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("Sequence Metrics")


__all__ = [
    "sequence_to_value_metrics", 
    "sequence_to_sequence_metrics"
]


DPI_value = _EvaluationConfig.DPI
SEQUENCE_PLOT_SIZE = _EvaluationConfig.SEQUENCE_PLOT_SIZE


def sequence_to_value_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[FormatSequenceValueMetrics] = None
):
    """
    Saves regression metrics and plots for sequence-to-value (many-to-one) tasks.

    Args:
        y_true (np.ndarray): Ground truth values (1D array).
        y_pred (np.ndarray): Predicted values (1D array).
        save_dir (str | Path): Directory to save plots and report.
        config (object): Formatting configuration object.
    """
    
    # --- Ensure 1D input ---
    if y_true.ndim > 1: y_true = y_true.flatten()
    if y_pred.ndim > 1: y_pred = y_pred.flatten()

    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseSequenceValueFormat()
    else:
        format_config = config
        
    # --- Set Matplotlib font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})
    
    # --- Calculate Metrics ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    report_lines = [
        "--- Sequence-to-Value Regression Report ---",
        f"  Root Mean Squared Error (RMSE): {rmse:.4f}",
        f"  Mean Absolute Error (MAE):      {mae:.4f}",
        f"  Median Absolute Error (MedAE):  {medae:.4f}",
        f"  Coefficient of Determination (R²): {r2:.4f}"
    ]
    report_string = "\n".join(report_lines)

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # Save text report
    report_path = save_dir_path / "sequence_to_value_report.txt"
    report_path.write_text(report_string)
    _LOGGER.info(f"📝 Seq-to-Value report saved as '{report_path.name}'")

    # --- Save residual plot ---
    residuals = y_true - y_pred
    fig_res, ax_res = plt.subplots(figsize=SEQUENCE_PLOT_SIZE, dpi=DPI_value)
    ax_res.scatter(y_pred, residuals, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_res.axhline(0, color=format_config.residual_line_color, linestyle='--')
    ax_res.set_xlabel("Predicted Values")
    ax_res.set_ylabel("Residuals")
    ax_res.set_title("Sequence-to-Value Residual Plot")
    ax_res.grid(True)
    plt.tight_layout()
    res_path = save_dir_path / "sequence_to_value_residual_plot.svg"
    plt.savefig(res_path)
    _LOGGER.info(f"📈 Seq-to-Value residual plot saved as '{res_path.name}'")
    plt.close(fig_res)

    # --- Save true vs predicted plot ---
    fig_tvp, ax_tvp = plt.subplots(figsize=SEQUENCE_PLOT_SIZE, dpi=DPI_value)
    ax_tvp.scatter(y_true, y_pred, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_tvp.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                linestyle='--', 
                lw=2,
                color=format_config.ideal_line_color)
    ax_tvp.set_xlabel('True Values')
    ax_tvp.set_ylabel('Predictions')
    ax_tvp.set_title('Sequence-to-Value: True vs. Predicted')
    ax_tvp.grid(True)
    plt.tight_layout()
    tvp_path = save_dir_path / "sequence_to_value_true_vs_predicted_plot.svg"
    plt.savefig(tvp_path)
    _LOGGER.info(f"📉 Seq-to-Value True vs. Predicted plot saved as '{tvp_path.name}'")
    plt.close(fig_tvp)
    
    # --- Restore RC params ---
    plt.rcParams.update(original_rc_params)


def sequence_to_sequence_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[FormatSequenceSequenceMetrics] = None
):
    """
    Saves per-step regression metrics for sequence-to-sequence (many-to-many) tasks.

    Args:
        y_true (np.ndarray): Ground truth sequences (n_samples, sequence_length).
        y_pred (np.ndarray): Predicted sequences (n_samples, sequence_length).
        save_dir (str | Path): Directory to save plots and report.
        config (object): Formatting configuration object.
    """

    if y_true.ndim != 2 or y_pred.ndim != 2:
        _LOGGER.error(f"Input arrays must be 2D (n_samples, sequence_length). Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        raise ValueError("Invalid input dimensions for sequence-to-sequence metrics.")

    if y_true.shape != y_pred.shape:
        _LOGGER.error(f"Input shapes do not match. Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        raise ValueError("Mismatched input shapes.")

    # --- Parse Config or use defaults ---
    if config is None:
        format_config = _BaseSequenceSequenceFormat()
    else:
        format_config = config
        
    # --- Set Matplotlib font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})

    sequence_length = y_true.shape[1]
    steps = list(range(1, sequence_length + 1))
    per_step_rmse = []
    per_step_mae = []

    # --- Calculate metrics for each time step ---
    for i in range(sequence_length):
        y_true_step = y_true[:, i]
        y_pred_step = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true_step, y_pred_step))
        mae = mean_absolute_error(y_true_step, y_pred_step)
        
        per_step_rmse.append(rmse)
        per_step_mae.append(mae)

    # --- Create and save DataFrame ---
    report_df = pd.DataFrame({
        "step": steps,
        "rmse": per_step_rmse,
        "mae": per_step_mae
    })

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    report_path = save_dir_path / "sequence_to_sequence_report.csv"
    report_df.to_csv(report_path, index=False)
    _LOGGER.info(f"📝 Seq-to-Seq per-step report saved as '{report_path.name}'")

    # --- Create and save plot ---
    fig, ax1 = plt.subplots(figsize=SEQUENCE_PLOT_SIZE, dpi=DPI_value)

    # Plot RMSE
    color_rmse = format_config.rmse_color
    ax1.set_xlabel('Prediction Step')
    ax1.set_ylabel('RMSE', color=color_rmse)
    ax1.plot(steps, per_step_rmse, format_config.rmse_marker, color=color_rmse, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, linestyle=format_config.grid_style)

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    color_mae = format_config.mae_color
    ax2.set_ylabel('MAE', color=color_mae)
    ax2.plot(steps, per_step_mae, format_config.mae_marker, color=color_mae, label='MAE')
    ax2.tick_params(axis='y', labelcolor=color_mae)

    fig.suptitle('Sequence-to-Sequence Metrics (Per-Step)')
    
    # Add a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust for suptitle
    
    plot_path = save_dir_path / "sequence_to_sequence_metrics_plot.svg"
    plt.savefig(plot_path)
    _LOGGER.info(f"📈 Seq-to-Seq per-step metrics plot saved as '{plot_path.name}'")
    plt.close(fig)
    
    # --- Restore RC params ---
    plt.rcParams.update(original_rc_params)

