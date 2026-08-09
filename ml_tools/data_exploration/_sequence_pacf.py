import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Union, Optional
from pathlib import Path
from sklearn.metrics import mutual_info_score
from scipy.linalg import toeplitz
from scipy.stats import norm

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from .._helpers import get_valid_matplotlib_color


_LOGGER = get_logger("Data Exploration: Visualization")


__all__ = [
    "plot_target_temporal_analysis",
]


def _calculate_pacf_numpy(x: np.ndarray, n_lags: int, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the PACF and confidence intervals using Yule-Walker equations."""
    n = len(x)
    
    # 1. Calculate Autocorrelation Function (ACF)
    x_centered = x - np.mean(x)
    var = np.sum(x_centered**2)
    
    if var == 0:
        return np.zeros(n_lags + 1), np.zeros((n_lags + 1, 2))
        
    # Using correlate for efficiency, selecting the positive lags
    acf = np.correlate(x_centered, x_centered, mode='full')[n-1:] / var
    
    # 2. Calculate PACF using Yule-Walker equations
    pacf = np.zeros(n_lags + 1)
    pacf[0] = 1.0 # Lag 0 is always 1.0
    
    for k in range(1, n_lags + 1):
        if k == 1:
            pacf[1] = acf[1]
        else:
            R = toeplitz(acf[:k])
            r = acf[1:k+1]
            try:
                # Solve the Toeplitz system
                phi = np.linalg.solve(R, r)
                pacf[k] = phi[-1]
            except np.linalg.LinAlgError:
                pacf[k] = 0.0 # Fallback for singular matrix
                
    # 3. Calculate Confidence Interval
    # Standard error for PACF is 1/sqrt(n)
    z_score = norm.ppf(1 - alpha / 2)
    margin = z_score / np.sqrt(n)
    
    conf_int = np.column_stack((pacf - margin, pacf + margin))
    
    return pacf, conf_int


def plot_target_temporal_analysis(
    df: pd.DataFrame,
    continuous_targets: Union[list[str], None],
    categorical_targets: Union[list[str], None],
    save_dir: Union[str, Path],
    order_by: Optional[str] = None,
    max_lag: int = 50,
    confidence_interval: float = 0.95,
    verbose: int = 2,
    font_scaling: float = 1.5,
    color: str = "tab:purple"
) -> None:
    """
    Plots and saves the temporal memory for continuous and categorical targets to help determine
    the optimal `sequence_length` for sequence-based models. 
    
    This function applies the appropriate statistical measure based on the data type:
    - Continuous Targets: Evaluated using Partial Autocorrelation (PACF). Plots include a confidence 
      interval band. Lags outside this band are statistically significant.
    - Categorical Targets: Evaluated using Time-Lagged Mutual Information (MI). Plots show the 
      information gain (in nats) from historical lags. The optimal cut-off is where MI asymptotes to near-zero.
    
    Plots are saved as individual .svg files in a dedicated "Target_Temporal_Analysis" subdirectory.

    Args:
        df (pd.DataFrame): The input dataset containing the time series data.
        continuous_targets (list[str] | None): Column names of continuous numerical targets if any.
        categorical_targets (list[str] | None): Column names of categorical/object targets if any.
        save_dir (str | Path): Base directory where the analysis subdirectory will be created.
        order_by (str | None): If provided, sorts the DataFrame chronologically by this column 
            before analysis, and drops the column from the evaluated copy. If None, assumes 
            the DataFrame is already chronologically sorted.
        max_lag (int): The maximum number of historical time steps (lags) to compute.
        confidence_interval (float): The confidence level for continuous target PACF intervals.
        verbose (int): Verbosity level for logging warnings and progress.
        font_scaling (float): Multiplier for all text elements in the plots.
        color (str): A valid Matplotlib color string for the stem lines and markers.

    Notes:
        - NaNs in target columns are dropped prior to calculation, requiring mostly contiguous sequences.
        - To obtain the optimal `sequence_length`, identify the lag where PACF drops below the confidence interval for continuous targets, 
        and where MI approaches zero for categorical targets.
    """
    continuous_targets = continuous_targets or []
    categorical_targets = categorical_targets or []
    
    if not continuous_targets and not categorical_targets:
        if verbose >= 1:
            _LOGGER.warning("No continuous or categorical targets provided. Skipping PACF plotting.")
        return
    
    if not isinstance(df, pd.DataFrame):
        _LOGGER.error("Input 'df' must be a pandas DataFrame.")
        return
    
    if not isinstance(save_dir, (str, Path)):
        _LOGGER.error("Input 'save_dir' must be a string or Path object.")
        return
    
    if not isinstance(max_lag, int) or max_lag <= 0:
        _LOGGER.error("Input 'max_lag' must be a positive integer.")
        return
    
    if not (0 < confidence_interval < 1):
        _LOGGER.error("Input 'confidence_interval' must be a float between 0 and 1.")
        return
    
    # 1. Setup save directory
    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")
    target_save_dir = base_save_path / "Target_Temporal_Analysis"
    target_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Handle chronological sorting
    plot_df = df.copy()
    if order_by is not None:
        if order_by not in plot_df.columns:
            _LOGGER.error(f"Cannot sort by '{order_by}': Column not found in DataFrame.")
            return
        
        plot_df = plot_df.sort_values(by=order_by)
        plot_df = plot_df.drop(columns=[order_by])
        
        if verbose >= 3:
            _LOGGER.info(f"DataFrame sorted by '{order_by}' and column dropped for temporal analysis.")

    validated_color = get_valid_matplotlib_color(color)
    total_plots_saved = 0
    alpha = max(0.001, min(0.999, 1.0 - confidence_interval))  # Calculate significance level
    
    with sns.plotting_context("notebook", font_scale=font_scaling):
        
        # ==========================================
        # ENGINE 1: Continuous Targets (PACF)
        # ==========================================
        for target in continuous_targets:
            if target not in plot_df.columns:
                if verbose >= 1:
                    _LOGGER.warning(f"Continuous target '{target}' not found in DataFrame. Skipping.")
                continue
            
            target_series = plot_df[target].dropna()
            
            if len(target_series) <= 2:
                if verbose >= 1:
                    _LOGGER.warning(f"Target '{target}' does not have enough valid data points. Skipping.")
                continue
            
            # Yule-Walker requirement: max_lag must be less than 50% of the sample size
            effective_max_lag = min(max_lag, (len(target_series) // 2) - 1)
            
            try:
                pacf_vals, conf_int = _calculate_pacf_numpy(target_series.to_numpy(), n_lags=effective_max_lag, alpha=alpha)
            except Exception as e:
                _LOGGER.error(f"PACF calculation failed for '{target}': {e}")
                continue
            
            # Re-center conf_int to 0 for plotting as a band
            lower_bound = conf_int[:, 0] - pacf_vals
            upper_bound = conf_int[:, 1] - pacf_vals
            lags = np.arange(len(pacf_vals))
            
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            
            marker_line, stem_lines, _baseline = ax.stem(lags, pacf_vals, basefmt=" ")
            plt.setp(stem_lines, color=validated_color, alpha=0.7, linewidth=1.5)
            plt.setp(marker_line, color=validated_color, markersize=6)
            
            ax.fill_between(lags, lower_bound, upper_bound, color='gray', alpha=0.2, 
                            label=f'{int(confidence_interval*100)}% Confidence Interval')
            ax.axhline(0, color='black', linewidth=1, alpha=0.8)
            
            ax.set_title(f'Partial Autocorrelation Function\n{target}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Partial Autocorrelation')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='upper right')
            
            # despine top and right axes for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            plot_filename = f"{sanitize_filename(target)}_PACF_{max_lag}.svg"
            plot_path = target_save_dir / plot_filename
            
            try:
                plt.savefig(plot_path, bbox_inches="tight", format='svg')
                total_plots_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to save PACF plot for '{target}'. Error: {e}")
            plt.close()

        # ==========================================
        # ENGINE 2: Categorical Targets (Mutual Info)
        # ==========================================
        for target in categorical_targets:
            if target not in plot_df.columns:
                if verbose >= 1:
                    _LOGGER.warning(f"Categorical target '{target}' not found in DataFrame. Skipping.")
                continue
            
            target_series = plot_df[target].dropna()
            
            if len(target_series) <= 2:
                if verbose >= 1:
                    _LOGGER.warning(f"Target '{target}' does not have enough valid data points. Skipping.")
                continue
            
            effective_max_lag = min(max_lag, len(target_series) - 1)
            lags = np.arange(0, effective_max_lag + 1)
            mi_vals = []
            
            # Calculate Time-Lagged Mutual Information
            for k in lags:
                if k == 0:
                    # Lag 0 is the entropy of the variable with itself
                    mi = mutual_info_score(target_series, target_series)
                else:
                    mi = mutual_info_score(target_series.iloc[k:], target_series.iloc[:-k])
                mi_vals.append(mi)
                
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            
            marker_line, stem_lines, _baseline = ax.stem(lags, mi_vals, basefmt=" ")
            plt.setp(stem_lines, color=validated_color, alpha=0.7, linewidth=1.5)
            plt.setp(marker_line, color=validated_color, markersize=6)
            
            ax.axhline(0, color='black', linewidth=1, alpha=0.8)
            
            ax.set_title(f'Time-Lagged Mutual Information\n{target}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Information Gain')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # despine top and right axes for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            plot_filename = f"{sanitize_filename(target)}_MutualInfo_{max_lag}.svg"
            plot_path = target_save_dir / plot_filename
            
            try:
                plt.savefig(plot_path, bbox_inches="tight", format='svg')
                total_plots_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to save Mutual Information plot for '{target}'. Error: {e}")
            plt.close()

    if verbose >= 2:
        _LOGGER.info(f"Successfully saved {total_plots_saved} temporal memory plots to '{target_save_dir.name}'.")
