from typing import Union, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utilities import yield_dataframes_from_dir

from ..path_manager import sanitize_filename, make_fullpath, list_csv_paths
from .._core import get_logger


_LOGGER = get_logger("Optimization Plots")


__all__ = [
    "plot_optimal_feature_distributions",
    "plot_optimal_feature_distributions_from_dataframe",
]


def plot_optimal_feature_distributions(results_dir: Union[str, Path], 
                                       verbose: bool=False,
                                       target_columns: Optional[list[str]] = None):
    """
    Analyzes optimization results and plots the distribution of optimal values.

    This function is compatible with mixed-type CSVs (strings for
    categorical features, numbers for continuous). It automatically
    detects the data type for each feature and generates:
    
    - A Bar Plot for categorical (string) features.
    - A KDE Plot for continuous (numeric) features.
    
    Plots are saved in a subdirectory inside the source directory.

    Parameters
    ----------
    results_dir : str | Path
        The path to the directory containing the optimization result CSV files.
    target_columns (list[str] | None): 
        A list of target column names to explicitly exclude from plotting. If None, it defaults to excluding only the last column (assumed as the target).
    """
    # Check results_dir and create output path
    results_path = make_fullpath(results_dir, enforce="directory")
    output_path = make_fullpath(results_path / "DistributionPlots", make=True)
    
    # Check that the directory contains csv files
    list_csv_paths(results_path, verbose=False, raise_on_empty=True)

    # --- Data Loading and Preparation ---
    _LOGGER.debug(f"📁 Starting analysis from results in: '{results_dir}'")
    
    data_to_plot = []
    for df, df_name in yield_dataframes_from_dir(results_path, verbose=True):
        if df.shape[1] < 2:
            _LOGGER.warning(f"Skipping '{df_name}': must have at least 2 columns (feature + target).")
            continue
        
        # --- Column selection logic ---
        if target_columns:
            # 1. Explicitly drop known targets to isolate features
            existing_targets = [c for c in target_columns if c in df.columns]
            features_df = df.drop(columns=existing_targets)
            
            if features_df.empty:
                _LOGGER.warning(f"Skipping '{df_name}': All columns were dropped based on target_columns list.")
                continue
        else:
            # 2. Fallback: Assume the last column is the only target
            features_df = df.iloc[:, :-1]
        
        # 3. Melt the filtered dataframe
        melted_df = features_df.melt(var_name='feature', value_name='value')
        
        # Set target as the filename (or joined target names) to differentiate sources
        melted_df['target'] = '\n'.join(target_columns) if target_columns else df_name
        data_to_plot.append(melted_df)
    
    if not data_to_plot:
        _LOGGER.error("No valid data to plot after processing all CSVs.")
        return
        
    long_df = pd.concat(data_to_plot, ignore_index=True)

    # --- Delegate to Helper ---
    _generate_and_save_feature_plots(long_df, output_path, verbose)


def plot_optimal_feature_distributions_from_dataframe(dataframe: pd.DataFrame,
                                                      save_dir: Union[str, Path],
                                                      verbose: bool=False,
                                                      target_columns: Optional[list[str]] = None):
    """
    Analyzes a single dataframe of optimization results and plots the distribution of optimal values.

    This function is compatible with mixed-type data (strings for categorical features, 
    numbers for continuous). It automatically detects the data type for each feature 
    and generates:
    
    - A Bar Plot for categorical (string) features.
    - A KDE Plot for continuous (numeric) features.
    
    Plots are saved in a 'DistributionPlots' subdirectory inside the save_dir.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the optimization results (features + target/s).
    save_dir : str | Path
        The directory where the 'DistributionPlots' folder will be created.
    verbose : bool, optional
        If True, logs details about which plot type is chosen for each feature.
    target_columns : list[str] | None
        A list of target column names to explicitly exclude from plotting. 
        If None, it defaults to excluding only the last column (assumed as the target).
    """
    # Check results_dir and create output path
    root_path = make_fullpath(save_dir, make=True, enforce="directory")
    output_path = make_fullpath(root_path / "DistributionPlots", make=True, enforce="directory")
    
    _LOGGER.debug(f"📁 Starting analysis from provided DataFrame. Output: '{output_path}'")

    if dataframe.empty:
        _LOGGER.error("Provided dataframe is empty.")
        return

    if dataframe.shape[1] < 2:
        _LOGGER.warning("DataFrame has fewer than 2 columns. Expecting at least one feature and one target.")

    # --- Data Preparation ---
    if target_columns:
        # Explicitly drop known targets to isolate features
        existing_targets = [c for c in target_columns if c in dataframe.columns]
        features_df = dataframe.drop(columns=existing_targets)
        target_label = '\n'.join(target_columns)
    else:
        # Fallback: Assume the last column is the only target
        features_df = dataframe.iloc[:, :-1]
        target_label = "Optimization Result"

    if features_df.empty:
        _LOGGER.warning("Skipping plotting: All columns were dropped based on target_columns list.")
        return

    # Melt and assign static target label
    long_df = features_df.melt(var_name='feature', value_name='value')
    long_df['target'] = target_label

    # --- Delegate to Helper ---
    _generate_and_save_feature_plots(long_df, output_path, verbose)


def _generate_and_save_feature_plots(long_df: pd.DataFrame, output_path: Path, verbose: bool) -> None:
    """
    Private helper: iterates over a melted DataFrame (columns: feature, value, target)
    and generates/saves the appropriate plot (Bar or KDE) for each feature.
    """
    features = long_df['feature'].unique()
    unique_targets = long_df['target'].unique()
    
    _LOGGER.info(f"📊 Found data for {len(features)} features. Generating plots...")

    for feature_name in features:
        plt.figure(figsize=(12, 7))
        
        # .copy() to ensure we are working with a distinct object
        feature_df = long_df[long_df['feature'] == feature_name].copy()

        # --- Type-checking logic ---
        feature_df['numeric_value'] = pd.to_numeric(feature_df['value'], errors='coerce')
        
        # If *any* value failed conversion (is NaN), treat it as categorical.
        if feature_df['numeric_value'].isna().any():
            
            # --- PLOT 1: CATEGORICAL (String-based) ---
            if verbose:
                print(f"    Plotting '{feature_name}' as categorical (bar plot).")
            
            # Calculate percentages for a clean bar plot
            norm_df = (feature_df.groupby('target')['value']
                       .value_counts(normalize=True)
                       .mul(100)
                       .rename('percent')
                       .reset_index())
            
            ax = sns.barplot(data=norm_df, x='value', y='percent', hue='target')
            plt.ylabel("Frequency (%)", fontsize=12)
            ax.set_ylim(0, 100) 
            
            # always rotate x-ticks for categorical clarity
            plt.xticks(rotation=45, ha='right')

        else:
            # --- PLOT 2: CONTINUOUS (Numeric-based) ---
            if verbose:
                print(f"    Plotting '{feature_name}' as continuous (KDE plot).")
            
            ax = sns.kdeplot(data=feature_df, x='numeric_value', hue='target',
                             fill=True, alpha=0.1, warn_singular=False)
            
            plt.xlabel("Feature Value", fontsize=12)
            plt.ylabel("Density", fontsize=12)

        # --- Common settings for both plot types ---
        plt.title(f"Optimal Value Distribution for '{feature_name}'", fontsize=16)
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Target')

        sanitized_feature_name = sanitize_filename(feature_name)
        plot_filename = output_path / f"Distribution_{sanitized_feature_name}.svg"
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

    _LOGGER.info(f"All plots saved successfully to: '{output_path}'")
