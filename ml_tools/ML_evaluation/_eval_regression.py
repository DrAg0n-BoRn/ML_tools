import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score, 
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    explained_variance_score
)
from pathlib import Path
from typing import Union, Optional
import warnings

from ..ML_configuration._config_metrics import (_BaseRegressionFormat,
                                        FormatRegressionMetrics,
                                        FormatMultiTargetRegressionMetrics)


from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from .._helpers import _get_consistent_palette
from ..keys._keys import _EvaluationConfig

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


# =====================================================================
# PRIVATE HELPER FUNCTIONS
# =====================================================================

def _calculate_1d_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates all numeric regression metrics for a 1D array."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = r2_score(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MedAE': medae,
        'MAPE': mape,
        'Max Error': max_err,
        'R2-score': r2,
        'Explained Variance': ev
    }

def _save_residual_plot(y_pred: np.ndarray, residuals: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    
    # Remove edge colors for data points
    edge_color = 'none'
    ax.scatter(y_pred, residuals, alpha=format_config.scatter_alpha, 
               color=format_config.scatter_color, edgecolors=edge_color, s=50 if len(y_pred) < 1000 else 20)
    
    ax.axhline(0, color=format_config.residual_line_color, linestyle='--', linewidth=2)
    ax.set_xlabel("Predicted Values", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Residuals", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def _save_true_vs_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    
    # Smart Density Plotting: Hexbin for large data, Scatter for small data
    if len(y_true) > 2000:
        # Create a custom colormap based on the user's chosen scatter color
        custom_cmap = sns.light_palette(format_config.scatter_color, as_cmap=True)
        hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap=custom_cmap, mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=format_config.font_size - 2)
        cb.ax.tick_params(labelsize=format_config.ytick_size - 2)
    else:
        # Remove edge colors for data points
        edge_color = 'none'
        ax.scatter(y_true, y_pred, alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color, edgecolors=edge_color, s=50 if len(y_true) < 1000 else 20)
        
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', lw=2, color=format_config.ideal_line_color)
    
    ax.set_xlabel('True Values', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel('Predicted Values', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def _save_residual_histogram(residuals: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    
    sns.histplot(residuals, kde=True, ax=ax, bins=format_config.hist_bins, color=format_config.scatter_color)
    
    ax.set_xlabel("Residual Value", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Frequency", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax.set_yticks([]) # Hide Y-ticks to focus purely on distribution shape
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def _save_qq_plot(residuals: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    """Quantile-Quantile plot to assess if residuals follow a normal distribution."""
    fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    
    stats.probplot(residuals, dist="norm", plot=ax)
    
    # Customize SciPy's default plot colors to match the theme
    lines = ax.get_lines()
    lines[0].set_markerfacecolor(format_config.scatter_color)
    lines[0].set_markeredgecolor('none') # Remove edge color for better visibility
    lines[0].set_alpha(format_config.scatter_alpha)
    lines[1].set_color(format_config.ideal_line_color)
    lines[1].set_linewidth(2)
    lines[1].set_linestyle('--')
    
    ax.set_xlabel("Theoretical Quantiles", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Ordered Residuals", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def _save_ecdf_plot(abs_errors: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    """Empirical Cumulative Distribution Function (eCDF) plot for absolute errors."""
    fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    
    sns.ecdfplot(abs_errors, ax=ax, color=format_config.scatter_color, lw=2.5)
    
    ax.set_xlabel("Absolute Error", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Cumulative Probability", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def _save_error_boxplot(y_true: np.ndarray, abs_errors: np.ndarray, format_config: _BaseRegressionFormat, save_path: Path, title: str):
    """Boxplot of absolute errors binned by true target values."""
    fig, ax = plt.subplots(figsize=(max(10, REGRESSION_PLOT_SIZE[0]), REGRESSION_PLOT_SIZE[1]), dpi=DPI_value)
    
    df = pd.DataFrame({'True': y_true, 'AbsError': abs_errors})
    # Safely bin targets into deciles (or fallback to fewer bins if low variance)
    try:
        df['Bin'] = pd.qcut(df['True'], q=10, duplicates='drop')
    except ValueError:
        df['Bin'] = pd.cut(df['True'], bins=10)
        
    # get a consistent color palette for the boxplot based on the bins
    ordered_bins = df['Bin'].cat.categories.tolist()
    
    palette_dict = _get_consistent_palette(ordered_bins, palette_name=format_config.boxplot_palette)
        
    sns.boxplot(data=df, x='Bin', y='AbsError', ax=ax, palette=palette_dict, hue='Bin', legend=False, showfliers=False)
    
    ax.set_xlabel("Target Value Bins", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_ylabel("Absolute Error", labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    
    ax.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax.tick_params(axis='x', labelsize=max(8, format_config.xtick_size - 2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# MAIN FUNCTIONS
# =====================================================================

def regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[FormatRegressionMetrics] = None
):
    """
    Saves comprehensive regression metrics and plots for a single target.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        save_dir (str | Path): Directory to save plots and report.
        config (RegressionMetricsFormat, optional): Formatting configuration object.
    """
    format_config = config if config is not None else _BaseRegressionFormat()
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # Calculate Metrics
    metrics = _calculate_1d_metrics(y_true, y_pred)
    
    report_lines = [
        "--- Regression Report ---",
        f"  Root Mean Squared Error (RMSE):    {metrics['RMSE']:.4f}",
        f"  Mean Absolute Error (MAE):         {metrics['MAE']:.4f}",
        f"  Median Absolute Error (MedAE):     {metrics['MedAE']:.4f}",
        f"  Mean Absolute Percentage (MAPE):   {metrics['MAPE']:.4f}",
        f"  Max Error:                         {metrics['Max Error']:.4f}",
        f"  Explained Variance Score:          {metrics['Explained Variance']:.4f}",
        f"  Coefficient of Determination (R²): {metrics['R2-score']:.4f}"
    ]
    
    report_path = save_dir_path / "regression_report.txt"
    report_path.write_text("\n".join(report_lines))
    _LOGGER.info(f"📝 Regression report saved as '{report_path.name}'")

    # Generate Plots
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    
    _save_residual_plot(y_pred, residuals, format_config, save_dir_path / "residual_plot.svg", "Residual Plot")
    _save_true_vs_pred_plot(y_true, y_pred, format_config, save_dir_path / "true_vs_predicted_plot.svg", "True vs. Predicted Values")
    _save_residual_histogram(residuals, format_config, save_dir_path / "residuals_histogram.svg", "Distribution of Residuals")
    _save_qq_plot(residuals, format_config, save_dir_path / "qq_plot.svg", "Q-Q Plot of Residuals")
    _save_ecdf_plot(abs_errors, format_config, save_dir_path / "ecdf_errors.svg", "eCDF of Absolute Errors")
    _save_error_boxplot(y_true, abs_errors, format_config, save_dir_path / "error_vs_target_boxplot.svg", "Error by Target")
    
    _LOGGER.info(f"📊 All single-target regression plots generated successfully.")


def multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_dir: Union[str, Path],
    config: Optional[FormatMultiTargetRegressionMetrics] = None
):
    """
    Calculates and saves advanced regression metrics for each target individually.
    
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

    format_config = config if config is not None else _BaseRegressionFormat()
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    metrics_summary = []
    
    # Initialize lists for radar charts
    rmse_scores, mae_scores, medae_scores, mape_scores, r2_scores, ev_scores = [], [], [], [], [], []

    for i, name in enumerate(target_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        sanitized_name = sanitize_filename(name)

        # Calculate Metrics
        metrics = _calculate_1d_metrics(true_i, pred_i)
        metrics['Target'] = name
        metrics_summary.append(metrics)
        
        # Store rounded, bounded metrics for radar charts
        rmse_scores.append(max(0.0, round(metrics['RMSE'], 4)))
        mae_scores.append(max(0.0, round(metrics['MAE'], 4)))
        medae_scores.append(max(0.0, round(metrics['MedAE'], 4)))
        mape_scores.append(max(0.0, round(metrics['MAPE'], 4)))
        r2_scores.append(max(0.0, round(metrics['R2-score'], 4)))
        ev_scores.append(max(0.0, round(metrics['Explained Variance'], 4)))

        # Generate Plots
        residuals = true_i - pred_i
        abs_errors = np.abs(residuals)
        
        _save_residual_plot(pred_i, residuals, format_config, save_dir_path / f"{sanitized_name}_residual_plot.svg", f"Residual Plot\n'{name}'")
        _save_true_vs_pred_plot(true_i, pred_i, format_config, save_dir_path / f"{sanitized_name}_true_vs_predicted.svg", f"True vs Predicted\n'{name}'")
        _save_residual_histogram(residuals, format_config, save_dir_path / f"{sanitized_name}_residuals_histogram.svg", f"Residual Distribution\n'{name}'")
        _save_qq_plot(residuals, format_config, save_dir_path / f"{sanitized_name}_qq_plot.svg", f"Q-Q Plot\n'{name}'")
        _save_ecdf_plot(abs_errors, format_config, save_dir_path / f"{sanitized_name}_ecdf.svg", f"eCDF of Error\n'{name}'")
        _save_error_boxplot(true_i, abs_errors, format_config, save_dir_path / f"{sanitized_name}_error_boxplot.svg", f"Error by Target\n'{name}'")

    # Save Summary Report
    summary_df = pd.DataFrame(metrics_summary)
    
    # Reorder columns to ensure 'Target' is first
    cols = ['Target'] + [c for c in summary_df.columns if c != 'Target']
    summary_df = summary_df[cols]
    
    report_path = save_dir_path / "regression_report_multi.csv"
    summary_df.to_csv(report_path, index=False)
    _LOGGER.info(f"📝 Multi-target regression report saved to '{report_path.name}'")

    # Save Radar Charts
    if len(target_names) > 2:
        radar_dir = save_dir_path / "radar_charts"
        radar_dir.mkdir(exist_ok=True)
        
        line_hex = mcolors.to_hex(format_config.scatter_color)
        fill_rgba = mpl_to_plotly_rgba(format_config.scatter_color, 0.15) 
        
        smart_font_size = calculate_smart_font_size(len(target_names), format_config.font_size)
        max_len = max([len(str(name)) for name in target_names])
        dynamic_margin = calculate_smart_margin_left_right(max_len)
        
        # Helper config mapping
        radar_configs = [
            (rmse_scores, "RMSE", "rmse_radar"),
            (mae_scores, "MAE", "mae_radar"),
            (medae_scores, "MedAE", "medae_radar"),
            (mape_scores, "MAPE", "mape_radar"),
        ]
        
        for scores, title, filename in radar_configs:
            max_score = max(scores) if max(scores) > 0 else 0.1
            save_radar_chart(
                scores, target_names, line_hex, fill_rgba, f"{title} across Targets", 
                radar_dir / filename, dynamic_margin, smart_font_size, 
                tick_range=[0, max_score], tick_vals=[round(val, 2) for val in np.linspace(0, max_score, 6).tolist()]
            )
            
        # R2 and Explained Variance (default 0 to 1 scaling)
        save_radar_chart(r2_scores, target_names, line_hex, fill_rgba, "R² Score across Targets", radar_dir / "r2_radar", dynamic_margin, smart_font_size)
        save_radar_chart(ev_scores, target_names, line_hex, fill_rgba, "Explained Variance across Targets", radar_dir / "ev_radar", dynamic_margin, smart_font_size)
                         
        _LOGGER.info(f"🕸️ Radar charts saved to '{radar_dir.name}'")
