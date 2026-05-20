import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score, 
    median_absolute_error,
)
from pathlib import Path
from typing import Union, Optional
import warnings

from ..ML_configuration._metrics import (_BaseRegressionFormat,
                                        FormatRegressionMetrics,
                                        FormatMultiTargetRegressionMetrics)

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import _EvaluationConfig

from ._helpers import check_and_abbreviate_name
from ._radar_plots import (
    mpl_to_plotly_rgba,
    calculate_smart_font_size,
    calculate_smart_margin_left_right,
    save_radar_chart
)



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
    
    # remove top and right spines for cleaner look
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False)
    
    ax_res.grid(True)
    plt.tight_layout()
    res_path = save_dir_path / "residual_plot.svg"
    plt.savefig(res_path, bbox_inches='tight')
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
    
    # remove top and right spines for cleaner look
    ax_tvp.spines['top'].set_visible(False)
    ax_tvp.spines['right'].set_visible(False)
    
    ax_tvp.grid(True)
    plt.tight_layout()
    tvp_path = save_dir_path / "true_vs_predicted_plot.svg"
    plt.savefig(tvp_path, bbox_inches='tight')
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
    
    # remove top and right spines for cleaner look
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    
    ax_hist.grid(True)
    plt.tight_layout()
    hist_path = save_dir_path / "residuals_histogram.svg"
    plt.savefig(hist_path, bbox_inches='tight')
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
    
    # Initialize lists to store metrics for the radar charts
    rmse_scores = []
    mae_scores = []
    medae_scores = []
    r2_scores = []
    
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
        
        # abbreviate name for plotting if needed
        name_abbreviated = check_and_abbreviate_name(name)
        
        sanitized_name = sanitize_filename(name)

        # --- Calculate Metrics ---
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        medae = median_absolute_error(true_i, pred_i)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = r2_score(true_i, pred_i)
        
        metrics_summary.append({
            'Target': name,
            'RMSE': rmse,
            'MAE': mae,
            'MedAE': medae,
            'R2-score': r2,
        })
        
        # Store rounded metrics for radar charts (clip R2 at 0 so the radar doesn't break on negative values)
        rmse_scores.append(0.0 if np.isnan(rmse) else round(rmse, 4))
        mae_scores.append(0.0 if np.isnan(mae) else round(mae, 4))
        medae_scores.append(0.0 if np.isnan(medae) else round(medae, 4))
        
        safe_r2 = 0.0 if np.isnan(r2) else r2
        r2_scores.append(max(0.0, round(safe_r2, 4)))

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
        ax_res.set_title(f"Residual Plot '{name_abbreviated}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        
        # Apply Ticks
        ax_res.tick_params(axis='x', labelsize=xtick_size)
        ax_res.tick_params(axis='y', labelsize=ytick_size)
        
        # remove top and right spines for cleaner look
        ax_res.spines['top'].set_visible(False)
        ax_res.spines['right'].set_visible(False)
        
        ax_res.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        res_path = save_dir_path / f"{sanitized_name}_residual_plot.svg"
        plt.savefig(res_path, bbox_inches='tight')
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
        ax_tvp.set_title(name, pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        
        # Apply Ticks
        ax_tvp.tick_params(axis='x', labelsize=xtick_size)
        ax_tvp.tick_params(axis='y', labelsize=ytick_size)
        
        # remove top and right spines for cleaner look
        ax_tvp.spines['top'].set_visible(False)
        ax_tvp.spines['right'].set_visible(False)
        
        ax_tvp.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        tvp_path = save_dir_path / f"{sanitized_name}_true_vs_predicted.svg"
        plt.savefig(tvp_path, bbox_inches='tight')
        plt.close(fig_tvp)
        
        # --- Save Histogram of Residuals ---
        fig_hist, ax_hist = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        sns.histplot(residuals, kde=True, ax=ax_hist, 
                     bins=format_config.hist_bins, 
                     color=format_config.scatter_color)
        ax_hist.set_xlabel("Residual Value", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_hist.set_ylabel("Frequency", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_hist.set_title(f"Distribution of Residuals '{name_abbreviated}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        
        # Apply Ticks
        ax_hist.tick_params(axis='x', labelsize=xtick_size)
        ax_hist.tick_params(axis='y', labelsize=ytick_size)
        
        # remove top and right spines for cleaner look
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        
        ax_hist.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        hist_path = save_dir_path / f"{sanitized_name}_residuals_histogram.svg"
        plt.savefig(hist_path, bbox_inches='tight')
        plt.close(fig_hist)

    # --- Save Summary Report ---
    summary_df = pd.DataFrame(metrics_summary)
    report_path = save_dir_path / "regression_report_multi.csv"
    summary_df.to_csv(report_path, index=False)
    _LOGGER.info(f"Full regression report saved to '{report_path.name}'")

    # --- Save Radar Charts ---
    if len(target_names) > 2:
        radar_dir = save_dir_path / "radar_charts"
        radar_dir.mkdir(exist_ok=True)
        
        line_hex = mcolors.to_hex(format_config.scatter_color)
        fill_rgba = mpl_to_plotly_rgba(format_config.scatter_color, 0.15) 
        
        smart_font_size = calculate_smart_font_size(len(target_names), base_font_size)
        max_len = max([len(str(name)) for name in target_names])
        dynamic_margin = calculate_smart_margin_left_right(max_len)
        
        # RMSE Radar
        max_rmse = max(rmse_scores) if max(rmse_scores) > 0 else 0.1
        save_radar_chart(rmse_scores, 
                         target_names, 
                         line_hex, 
                         fill_rgba, 
                         "RMSE across Targets", 
                         radar_dir / "rmse_radar", 
                         dynamic_margin, 
                         smart_font_size, 
                         tick_range=[0, max_rmse],
                         tick_vals=[round(val, 2) for val in np.linspace(0, max_rmse, 6).tolist()])
        
        # MAE Radar
        max_mae = max(mae_scores) if max(mae_scores) > 0 else 0.1
        save_radar_chart(mae_scores, 
                         target_names, 
                         line_hex, 
                         fill_rgba, 
                         "MAE across Targets", 
                         radar_dir / "mae_radar", 
                         dynamic_margin, 
                         smart_font_size, 
                         tick_range=[0, max_mae],
                         tick_vals=[round(val, 2) for val in np.linspace(0, max_mae, 6).tolist()])
                         
        # MedAE Radar
        max_medae = max(medae_scores) if max(medae_scores) > 0 else 0.1
        save_radar_chart(medae_scores, 
                         target_names, 
                         line_hex, 
                         fill_rgba, 
                         "MedAE across Targets", 
                         radar_dir / "medae_radar", 
                         dynamic_margin, 
                         smart_font_size, 
                         tick_range=[0, max_medae],
                         tick_vals=[round(val, 2) for val in np.linspace(0, max_medae, 6).tolist()])
        
        # R2 Radar (Uses default 0.0 to 1.0 range)
        save_radar_chart(r2_scores, 
                         target_names, 
                         line_hex, 
                         fill_rgba, 
                         "R2-Score across Targets", 
                         radar_dir / "r2_radar", 
                         dynamic_margin, smart_font_size)
                         
        _LOGGER.info(f"🌀 Radar charts saved to '{radar_dir.name}'")
