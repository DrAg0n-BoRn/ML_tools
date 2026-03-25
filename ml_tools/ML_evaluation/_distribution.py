import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path
from typing import Union, Optional

from ..ML_configuration._metrics import (FormatRegressionMetrics,
                                         FormatMultiTargetRegressionMetrics,
                                         _BaseRegressionFormat)

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import _EvaluationConfig

from ._helpers import check_and_abbreviate_name


_LOGGER = get_logger("Distribution Metrics")


__all__ = [
    "distribution_metrics",
    "multi_target_distribution_metrics"
]


DPI_value = _EvaluationConfig.DPI
REGRESSION_PLOT_SIZE = _EvaluationConfig.REGRESSION_PLOT_SIZE


def _compute_distribution_stats(y_true: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray) -> dict:
    """Computes distribution-specific metrics."""
    std_pred = np.sqrt(var_pred)
    
    # Gaussian Negative Log-Likelihood
    nll = 0.5 * np.log(2 * np.pi * var_pred) + ((y_true - mean_pred)**2) / (2 * var_pred)
    mean_nll = np.mean(nll)
    
    # 95% Prediction Interval bounds (1.96 standard deviations)
    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred
    
    # Prediction Interval Coverage Probability (PICP)
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Mean Prediction Interval Width (MPIW)
    interval_width = np.mean(upper_bound - lower_bound)
    
    return {
        "NLL": mean_nll,
        "PICP_95": coverage,
        "MPIW_95": interval_width
    }

def _plot_prediction_intervals(y_true: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray, 
                               ax: Axes, format_config: _BaseRegressionFormat, title: str):
    """Plots sorted prediction intervals."""
    std_pred = np.sqrt(var_pred)
    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred
    
    # Sort by true values for a readable plot
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    mean_sorted = mean_pred[sort_idx]
    lower_sorted = lower_bound[sort_idx]
    upper_sorted = upper_bound[sort_idx]
    x_axis = np.arange(len(y_true))
    
    ax.fill_between(x_axis, lower_sorted, upper_sorted, color=format_config.scatter_color, alpha=0.3, label='95% Interval')
    ax.plot(x_axis, mean_sorted, color=format_config.ideal_line_color, linewidth=2, label='Predicted Mean')
    ax.scatter(x_axis, y_true_sorted, color='red', s=10, alpha=0.7, label='True Value')
    
    ax.set_xlabel("Samples (Sorted by True Value)", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Target Value", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    ax.tick_params(axis='both', labelsize=format_config.xtick_size)
    ax.legend(fontsize=format_config.font_size - 6)
    ax.grid(True, linestyle='--', alpha=0.6)

def _plot_error_vs_uncertainty(y_true: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray, 
                               ax: Axes, format_config: _BaseRegressionFormat, title: str):
    """Plots absolute error vs predicted standard deviation."""
    std_pred = np.sqrt(var_pred)
    abs_error = np.abs(y_true - mean_pred)
    
    ax.scatter(std_pred, abs_error, alpha=format_config.scatter_alpha, color=format_config.scatter_color)
    
    # Add a trendline to see if uncertainty correlates with error
    if len(std_pred) > 1:
        z = np.polyfit(std_pred, abs_error, 1)
        p = np.poly1d(z)
        ax.plot(std_pred, p(std_pred), color=format_config.ideal_line_color, linestyle="--", linewidth=2)
        
    ax.set_xlabel("Predicted Standard Deviation", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Absolute Error", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    ax.tick_params(axis='both', labelsize=format_config.xtick_size)
    ax.grid(True, linestyle='--', alpha=0.6)


def distribution_metrics(
    y_true: np.ndarray, 
    mean_pred: np.ndarray, 
    var_pred: np.ndarray,
    save_dir: Union[str, Path],
    config: Optional[FormatRegressionMetrics] = None
):
    """
    Calculates and saves probabilistic distribution metrics and plots for a single-target model.

    This function evaluates the quality of the model's uncertainty estimates by comparing 
    the predicted distributions (parameterized by mean and variance) against the true targets.
    It saves a text report and two SVG plots: a sorted 95% Prediction Interval plot and an 
    Error vs. Uncertainty scatter plot.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples,).
        mean_pred (np.ndarray): Predicted mean values, shape (n_samples,).
        var_pred (np.ndarray): Predicted variance values, shape (n_samples,).
        save_dir (str | Path): Directory to save the text report and plots.
        config (FormatRegressionMetrics | None): Optional formatting configuration object to customize plot aesthetics.

    Note:
        - **Gaussian NLL (Negative Log-Likelihood):** Measures the goodness of fit of the 
          predicted Gaussian distribution. Lower is better.
        - **PICP (Prediction Interval Coverage Probability):** The percentage of true values 
          that fall within the predicted 95% confidence interval. Ideally close to 95%.
        - **MPIW (Mean Prediction Interval Width):** The average size of the 95% intervals. 
          Smaller is better, provided the PICP remains near 95%.
    """
    format_config = config if config is not None else _BaseRegressionFormat()
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # --- Metrics ---
    stats = _compute_distribution_stats(y_true, mean_pred, var_pred)
    
    report_lines = [
        "--- Distribution Report ---",
        f"  Gaussian NLL:                     {stats['NLL']:.4f}",
        f"  95% Interval Coverage (PICP):     {stats['PICP_95'] * 100:.2f}%",
        f"  95% Mean Interval Width (MPIW):   {stats['MPIW_95']:.4f}"
    ]
    report_string = "\n".join(report_lines)
    
    report_path = save_dir_path / "distribution_report.txt"
    report_path.write_text(report_string)
    _LOGGER.info(f"📝 Distribution report saved as '{report_path.name}'")

    # --- Plot 1: Prediction Intervals ---
    fig_pi, ax_pi = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    _plot_prediction_intervals(y_true, mean_pred, var_pred, ax_pi, format_config, "Prediction Intervals")
    plt.tight_layout()
    pi_path = save_dir_path / "prediction_intervals.svg"
    plt.savefig(pi_path)
    _LOGGER.info(f"📊 Prediction Intervals plot saved as '{pi_path.name}'")
    plt.close(fig_pi)

    # --- Plot 2: Error vs Uncertainty ---
    fig_eu, ax_eu = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    _plot_error_vs_uncertainty(y_true, mean_pred, var_pred, ax_eu, format_config, "Error vs. Uncertainty")
    plt.tight_layout()
    eu_path = save_dir_path / "error_vs_uncertainty.svg"
    plt.savefig(eu_path)
    _LOGGER.info(f"📊 Error vs Uncertainty plot saved as '{eu_path.name}'")
    plt.close(fig_eu)


def multi_target_distribution_metrics(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    target_names: list[str],
    save_dir: Union[str, Path],
    config: Optional[FormatMultiTargetRegressionMetrics] = None
):
    """
    Calculates and saves probabilistic distribution metrics for each target individually.

    For each target, this function saves a 95% Prediction Interval plot and an Error vs. 
    Uncertainty plot. It also consolidates the distribution metrics (NLL, PICP, MPIW) for 
    all targets into a single CSV summary report.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_targets).
        mean_pred (np.ndarray): Predicted mean values, shape (n_samples, n_targets).
        var_pred (np.ndarray): Predicted variance values, shape (n_samples, n_targets).
        target_names (list[str]): A list of string names corresponding to each target column.
        save_dir (str | Path): Directory to save the per-target plots and the CSV report.
        config (FormatMultiTargetRegressionMetrics | None): Optional formatting configuration object to customize plot aesthetics.

    Note:
        - **Gaussian NLL:** Measures overall probabilistic fit. Lower is better.
        - **PICP (Coverage):** Percentage of targets falling in the 95% prediction interval.
        - **MPIW (Width):** Average width of those intervals. Sharpness/certainty measure.
    """
    if y_true.ndim != 2 or mean_pred.ndim != 2 or var_pred.ndim != 2:
        _LOGGER.error("Arrays must be 2D for multi-target distribution metrics.")
        raise ValueError()
    
    target_names = [check_and_abbreviate_name(name) for name in target_names]
    format_config = config if config is not None else _BaseRegressionFormat()
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    metrics_summary = []
    
    _LOGGER.debug("--- Multi-Target Distribution Evaluation ---")

    for i, name in enumerate(target_names):
        true_i = y_true[:, i]
        mean_i = mean_pred[:, i]
        var_i = var_pred[:, i]
        sanitized_name = sanitize_filename(name)

        # --- Metrics ---
        stats = _compute_distribution_stats(true_i, mean_i, var_i)
        metrics_summary.append({
            'Target': name,
            'Gaussian NLL': stats['NLL'],
            'PICP 95%': stats['PICP_95'],
            'MPIW 95%': stats['MPIW_95']
        })

        # --- Plots ---
        fig_pi, ax_pi = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        _plot_prediction_intervals(true_i, mean_i, var_i, ax_pi, format_config, f"Prediction Intervals '{name}'")
        plt.tight_layout()
        plt.savefig(save_dir_path / f"prediction_intervals_{sanitized_name}.svg")
        plt.close(fig_pi)

        fig_eu, ax_eu = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        _plot_error_vs_uncertainty(true_i, mean_i, var_i, ax_eu, format_config, f"Error vs. Uncertainty '{name}'")
        plt.tight_layout()
        plt.savefig(save_dir_path / f"error_vs_uncertainty_{sanitized_name}.svg")
        plt.close(fig_eu)

    # --- Summary Report ---
    summary_df = pd.DataFrame(metrics_summary)
    report_path = save_dir_path / "distribution_report_multi.csv"
    summary_df.to_csv(report_path, index=False)
    _LOGGER.info(f"📊 Multi-target distribution report and plots saved to '{save_dir_path.name}'")
