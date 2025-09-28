import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Any, Literal, Optional
from pathlib import Path
import pandas as pd

from .path_manager import make_fullpath, list_csv_paths, sanitize_filename
from .utilities import yield_dataframes_from_dir
from ._logger import _LOGGER
from ._script_info import _script_info
from .SQL import DatabaseManager


__all__ = [
    "parse_lower_upper_bounds",
    "plot_optimal_feature_distributions"
]


def parse_lower_upper_bounds(source: dict[str,tuple[Any,Any]]):
    """
    Parse lower and upper boundaries, returning 2 lists:
    
    `lower_bounds`, `upper_bounds`
    """
    lower = [low[0] for low in source.values()]
    upper = [up[1] for up in source.values()]
    
    return lower, upper


def plot_optimal_feature_distributions(results_dir: Union[str, Path]):
    """
    Analyzes optimization results and plots the distribution of optimal values for each feature.

    For features with more than two unique values, this function generates a color-coded 
    Kernel Density Estimate (KDE) plot. For binary or constant features, it generates a bar plot
    showing relative frequency.
    
    Plots are saved in a subdirectory inside the source directory.

    Parameters
    ----------
    results_dir : str or Path
        The path to the directory containing the optimization result CSV files.
    """
    # Check results_dir and create output path
    results_path = make_fullpath(results_dir, enforce="directory")
    output_path = make_fullpath(results_path / "DistributionPlots", make=True)
    
    # Check that the directory contains csv files
    list_csv_paths(results_path, verbose=False)

    # --- Data Loading and Preparation ---
    _LOGGER.info(f"📁 Starting analysis from results in: '{results_dir}'")
    data_to_plot = []
    for df, df_name in yield_dataframes_from_dir(results_path):
        melted_df = df.iloc[:, :-1].melt(var_name='feature', value_name='value')
        melted_df['target'] = df_name.replace("Optimization_", "")
        data_to_plot.append(melted_df)
    
    long_df = pd.concat(data_to_plot, ignore_index=True)
    features = long_df['feature'].unique()
    _LOGGER.info(f"Found data for {len(features)} features across {len(long_df['target'].unique())} targets. Generating plots...")

    # --- Plotting Loop ---
    for feature_name in features:
        plt.figure(figsize=(12, 7))
        feature_df = long_df[long_df['feature'] == feature_name]

        # Check if the feature is binary or constant
        if feature_df['value'].nunique() <= 2:
            # PLOT 1: For discrete values, calculate percentages and use a true bar plot.
            # This ensures the X-axis is clean (e.g., just 0 and 1).
            norm_df = (feature_df.groupby('target')['value']
                       .value_counts(normalize=True)
                       .mul(100)
                       .rename('percent')
                       .reset_index())
            
            ax = sns.barplot(data=norm_df, x='value', y='percent', hue='target')
            
            plt.title(f"Optimal Value Distribution for '{feature_name}'", fontsize=16)
            plt.ylabel("Frequency (%)", fontsize=12)
            ax.set_ylim(0, 100) # Set Y-axis from 0 to 100

        else:
            # PLOT 2: KDE plot for continuous values.
            ax = sns.kdeplot(data=feature_df, x='value', hue='target',
                             fill=True, alpha=0.1, warn_singular=False)

            plt.title(f"Optimal Value Distribution for '{feature_name}'", fontsize=16)
            plt.ylabel("Density", fontsize=12) # Y-axis is "Density" for KDE plots

        # --- Common settings for both plot types ---
        plt.xlabel("Feature Value", fontsize=12)
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Target')

        sanitized_feature_name = sanitize_filename(feature_name)
        plot_filename = output_path / f"Distribution_{sanitized_feature_name}.svg"
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

    _LOGGER.info(f"All plots saved successfully to: '{output_path}'")
    

def _save_result(
        result_dict: dict,
        save_format: Literal['csv', 'sqlite', 'both'],
        csv_path: Path,
        db_manager: Optional[DatabaseManager] = None,
        db_table_name: Optional[str] = None
    ):
    """
    Private helper to handle saving a single result to CSV, SQLite, or both.
    """
    # Save to CSV
    if save_format in ['csv', 'both']:
        df_row = pd.DataFrame([result_dict])
        file_exists = csv_path.exists()
        df_row.to_csv(csv_path, mode='a', index=False, header=not file_exists)

    # Save to SQLite
    if save_format in ['sqlite', 'both']:
        if db_manager and db_table_name:
            db_manager.insert_row(db_table_name, result_dict)
        else:
            _LOGGER.warning("SQLite saving requested but db_manager or table_name not provided.")


def info():
    _script_info(__all__)
