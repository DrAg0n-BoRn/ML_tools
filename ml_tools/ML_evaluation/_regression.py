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

from ..ML_configuration._metrics import (_BaseRegressionFormat,
                                        FormatRegressionMetrics,
                                        FormatMultiTargetRegressionMetrics)

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import _EvaluationConfig


_LOGGER = get_logger("Regression Metrics")


__all__ = [
    "regression_metrics",
    "multi_target_regression_metrics"
]


DPI_value = _EvaluationConfig.DPI
REGRESSION_PLOT_SIZE = _EvaluationConfig.REGRESSION_PLOT_SIZE


def regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[FormatRegressionMetrics] = None
):
    """
    Saves regression metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        save_dir (str | Path): Directory to save plots and report.
        config (RegressionMetricsFormat, optional): Formatting configuration object.
    """
    
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseRegressionFormat()
    else:
        format_config = config
    
    # --- Resolve Font Sizes ---
    xtick_size = format_config.xtick_size
    ytick_size = format_config.ytick_size
    base_font_size = format_config.font_size
    
    # --- Calculate Metrics ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    report_lines = [
        "--- Regression Report ---",
        f"  Root Mean Squared Error (RMSE): {rmse:.4f}",
        f"  Mean Absolute Error (MAE):      {mae:.4f}",
        f"  Median Absolute Error (MedAE):  {medae:.4f}",
        f"  Coefficient of Determination (R²): {r2:.4f}"
    ]
    report_string = "\n".join(report_lines)
    # print(report_string)

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # Save text report
    report_path = save_dir_path / "regression_report.txt"
    report_path.write_text(report_string)
    _LOGGER.info(f"📝 Regression report saved as '{report_path.name}'")

    # --- Save residual plot ---
    residuals = y_true - y_pred
    fig_res, ax_res = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    ax_res.scatter(y_pred, residuals, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_res.axhline(0, color=format_config.residual_line_color, linestyle='--')
    ax_res.set_xlabel("Predicted Values", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_res.set_ylabel("Residuals", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_res.set_title("Residual Plot", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
    
    # Apply Ticks  
    ax_res.tick_params(axis='x', labelsize=xtick_size)
    ax_res.tick_params(axis='y', labelsize=ytick_size)
    
    ax_res.grid(True)
    plt.tight_layout()
    res_path = save_dir_path / "residual_plot.svg"
    plt.savefig(res_path)
    _LOGGER.info(f"📈 Residual plot saved as '{res_path.name}'")
    plt.close(fig_res)

    # --- Save true vs predicted plot ---
    fig_tvp, ax_tvp = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    ax_tvp.scatter(y_true, y_pred, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_tvp.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                linestyle='--', 
                lw=2,
                color=format_config.ideal_line_color)
    ax_tvp.set_xlabel('True Values', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_tvp.set_ylabel('Predictions', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_tvp.set_title('True vs. Predicted Values', pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
    
    # Apply Ticks
    ax_tvp.tick_params(axis='x', labelsize=xtick_size)
    ax_tvp.tick_params(axis='y', labelsize=ytick_size)
    
    ax_tvp.grid(True)
    plt.tight_layout()
    tvp_path = save_dir_path / "true_vs_predicted_plot.svg"
    plt.savefig(tvp_path)
    _LOGGER.info(f"📉 True vs. Predicted plot saved as '{tvp_path.name}'")
    plt.close(fig_tvp)
    
    # --- Save Histogram of Residuals ---
    fig_hist, ax_hist = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    sns.histplot(residuals, kde=True, ax=ax_hist, 
                 bins=format_config.hist_bins, 
                 color=format_config.scatter_color)
    ax_hist.set_xlabel("Residual Value", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_hist.set_ylabel("Frequency", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
    ax_hist.set_title("Distribution of Residuals", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
    
    # Apply Ticks
    ax_hist.tick_params(axis='x', labelsize=xtick_size)
    ax_hist.tick_params(axis='y', labelsize=ytick_size)
    
    ax_hist.grid(True)
    plt.tight_layout()
    hist_path = save_dir_path / "residuals_histogram.svg"
    plt.savefig(hist_path)
    _LOGGER.info(f"📊 Residuals histogram saved as '{hist_path.name}'")
    plt.close(fig_hist)


def multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_dir: Union[str, Path],
    config: Optional[FormatMultiTargetRegressionMetrics] = None
):
    """
    Calculates and saves regression metrics for each target individually.

    For each target, this function saves a residual plot and a true vs. predicted plot.
    It also saves a single CSV file containing the key metrics (RMSE, MAE, R², MedAE)
    for all targets.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_targets).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_targets).
        target_names (List[str]): A list of names for the target variables.
        save_dir (str | Path): Directory to save plots and the report.
        config (object): Formatting configuration object.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        _LOGGER.error("y_true and y_pred must be 2D arrays for multi-target regression.")
        raise ValueError()
    if y_true.shape != y_pred.shape:
        _LOGGER.error("Shapes of y_true and y_pred must match.")
        raise ValueError()
    if y_true.shape[1] != len(target_names):
        _LOGGER.error("Number of target names must match the number of columns in y_true.")
        raise ValueError()

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    metrics_summary = []
    
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseRegressionFormat()
    else:
        format_config = config
    
    # ticks font sizes 
    xtick_size = format_config.xtick_size
    ytick_size = format_config.ytick_size
    base_font_size = format_config.font_size

    _LOGGER.debug("--- Multi-Target Regression Evaluation ---")

    for i, name in enumerate(target_names):
        # print(f"  -> Evaluating target: '{name}'")
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        sanitized_name = sanitize_filename(name)

        # --- Calculate Metrics ---
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        medae = median_absolute_error(true_i, pred_i)
        metrics_summary.append({
            'Target': name,
            'RMSE': rmse,
            'MAE': mae,
            'MedAE': medae,
            'R2-score': r2,
        })

        # --- Save Residual Plot ---
        residuals = true_i - pred_i
        fig_res, ax_res = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        ax_res.scatter(pred_i, residuals, 
                       alpha=format_config.scatter_alpha, 
                       edgecolors='k', 
                       s=50,
                       color=format_config.scatter_color) # Use config color
        ax_res.axhline(0, color=format_config.residual_line_color, linestyle='--') # Use config color
        ax_res.set_xlabel("Predicted Values", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_res.set_ylabel("Residuals", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_res.set_title(f"Residual Plot for '{name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        
        # Apply Ticks
        ax_res.tick_params(axis='x', labelsize=xtick_size)
        ax_res.tick_params(axis='y', labelsize=ytick_size)
        
        ax_res.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        res_path = save_dir_path / f"residual_plot_{sanitized_name}.svg"
        plt.savefig(res_path)
        plt.close(fig_res)

        # --- Save True vs. Predicted Plot ---
        fig_tvp, ax_tvp = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        ax_tvp.scatter(true_i, pred_i, 
                       alpha=format_config.scatter_alpha, 
                       edgecolors='k', 
                       s=50,
                       color=format_config.scatter_color) # Use config color
        ax_tvp.plot([true_i.min(), true_i.max()], [true_i.min(), true_i.max()], 
                    linestyle='--', 
                    lw=2,
                    color=format_config.ideal_line_color) # Use config color
        ax_tvp.set_xlabel('True Values', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_tvp.set_ylabel('Predicted Values', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_tvp.set_title(f"True vs. Predicted for '{name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        
        # Apply Ticks
        ax_tvp.tick_params(axis='x', labelsize=xtick_size)
        ax_tvp.tick_params(axis='y', labelsize=ytick_size)
        
        ax_tvp.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        tvp_path = save_dir_path / f"true_vs_predicted_plot_{sanitized_name}.svg"
        plt.savefig(tvp_path)
        plt.close(fig_tvp)

    # --- Save Summary Report ---
    summary_df = pd.DataFrame(metrics_summary)
    report_path = save_dir_path / "regression_report_multi.csv"
    summary_df.to_csv(report_path, index=False)
    _LOGGER.info(f"Full regression report saved to '{report_path.name}'")

