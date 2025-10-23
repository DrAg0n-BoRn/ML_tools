import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Any, Literal, Optional, Dict, List, Tuple
from pathlib import Path
import pandas as pd

from .path_manager import make_fullpath, list_csv_paths, sanitize_filename
from .utilities import yield_dataframes_from_dir
from ._logger import _LOGGER
from ._script_info import _script_info
from .SQL import DatabaseManager


__all__ = [
    "create_optimization_bounds",
    "parse_lower_upper_bounds",
    "plot_optimal_feature_distributions",
]


def create_optimization_bounds(
    csv_path: Union[str, Path],
    continuous_bounds_map: Dict[str, Tuple[float, float]],
    categorical_map: Dict[int, int],
    target_column: Optional[str] = None,
    start_at_zero: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Generates the lower and upper bounds lists for the optimizer from a CSV header.

    This helper function automates the creation of unbiased bounds for
    categorical features and combines them with user-defined bounds for
    continuous features.

    It reads *only* the header of the provided CSV to determine the full
    list of feature columns and their order, excluding the specified target.
    This is memory-efficient as the full dataset is not loaded.

    Args:
        csv_path (Union[str, Path]):
            Path to the final, preprocessed CSV file. The column order in
            this file must match the order expected by the model.
        continuous_bounds_map (Dict[str, Tuple[float, float]]):
            A dictionary mapping the *name* of each **continuous** feature
            to its (min_bound, max_bound) tuple.
        categorical_map (Dict[int, int]):
            The map from the *index* of each **categorical** feature to its cardinality.
            (e.g., {2: 4} for a feature at index 2 with 4 categories).
        target_column (Optional[str], optional):
            The name of the target column to exclude. If None (default), the *last column* in the CSV is assumed to be the target.
        start_at_zero (bool):
            - If True, assumes categorical encoding is [0, 1, ..., k-1].
              Bounds will be set as [-0.5, k - 0.5].
            - If False, assumes encoding is [1, 2, ..., k].
              Bounds will be set as [0.5, k + 0.5].

    Returns:
        Tuple[List[float], List[float]]:
            A tuple containing two lists: (lower_bounds, upper_bounds).

    Raises:
        ValueError: If a feature is defined in both maps, is missing from
                    both maps, or if a name in `continuous_bounds_map`
                    or `target_column` is not found in the CSV columns.
    """
    # 1. Read header and determine feature names
    full_csv_path = make_fullpath(csv_path, enforce="file")
    try:
        df_header = pd.read_csv(full_csv_path, nrows=0, encoding="utf-8")
    except Exception as e:
        _LOGGER.error(f"Failed to read header from CSV: {e}")
        raise
        
    all_column_names = df_header.columns.to_list()
    feature_names: List[str] = []

    if target_column is None:
        feature_names = all_column_names[:-1]
        excluded_target = all_column_names[-1]
        _LOGGER.info(f"No target_column provided. Assuming last column '{excluded_target}' is the target.")
    else:
        if target_column not in all_column_names:
            _LOGGER.error(f"Target column '{target_column}' not found in CSV header.")
            raise ValueError()
        feature_names = [name for name in all_column_names if name != target_column]
        _LOGGER.info(f"Excluding target column '{target_column}'.")
    
    # 2. Initialize bound lists
    total_features = len(feature_names)
    if total_features <= 0:
        _LOGGER.error("No feature columns remain after excluding the target.")
        raise ValueError()

    lower_bounds: List[Optional[float]] = [None] * total_features
    upper_bounds: List[Optional[float]] = [None] * total_features
    
    _LOGGER.info(f"Generating bounds for {total_features} total features...")

    # 3. Populate categorical bounds (Index-based)
    # The indices in categorical_map (e.g., {2: 4}) directly correspond
    # to the indices in the `feature_names` list.
    for index, cardinality in categorical_map.items():
        if not (0 <= index < total_features):
            _LOGGER.error(f"Categorical index {index} is out of range for the {total_features} features.")
            raise ValueError()
            
        if start_at_zero:
            # Rule for [0, k-1]: bounds are [-0.5, k - 0.5]
            low = -0.5
            high = float(cardinality) - 0.5
        else:
            # Rule for [1, k]: bounds are [0.5, k + 0.5]
            low = 0.5
            high = float(cardinality) + 0.5
            
        lower_bounds[index] = low
        upper_bounds[index] = high
        
    _LOGGER.info(f"Automatically set bounds for {len(categorical_map)} categorical features.")

    # 4. Populate continuous bounds (Name-based)
    count_continuous = 0
    for name, (low, high) in continuous_bounds_map.items():
        try:
            # Map name to its index in the *feature-only* list
            index = feature_names.index(name)
        except ValueError:
            _LOGGER.warning(f"Feature name '{name}' from 'continuous_bounds_map' not found in the CSV's feature columns.")
            continue
            
        if lower_bounds[index] is not None:
            # This index was already set by the categorical map
            _LOGGER.error(f"Feature '{name}' (at index {index}) is defined in both 'categorical_map' and 'continuous_bounds_map'.")
            raise ValueError()
            
        lower_bounds[index] = float(low)
        upper_bounds[index] = float(high)
        count_continuous += 1
        
    _LOGGER.info(f"Manually set bounds for {count_continuous} continuous features.")

    # 5. Validation: Check for any remaining None values
    missing_indices = []
    for i in range(total_features):
        if lower_bounds[i] is None:
            missing_indices.append(i)
            
    if missing_indices:
        missing_names = [feature_names[i] for i in missing_indices]
        _LOGGER.error(f"Bounds not defined for all features. Missing: {missing_names}")
        raise ValueError()

    # _LOGGER.info("All bounds successfully created.")
    
    # Cast to float lists, as 'None' sentinels are gone
    return (
        [float(b) for b in lower_bounds],  # type: ignore
        [float(b) for b in upper_bounds] # type: ignore
    )


def parse_lower_upper_bounds(source: dict[str,tuple[Any,Any]]):
    """
    Parse lower and upper boundaries, returning 2 lists:
    
    `lower_bounds`, `upper_bounds`
    """
    lower = [low[0] for low in source.values()]
    upper = [up[1] for up in source.values()]
    
    return lower, upper


def plot_optimal_feature_distributions(results_dir: Union[str, Path], verbose: bool=False):
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
        if df.shape[1] < 2:
            _LOGGER.warning(f"Skipping '{df_name}': must have at least 2 columns (feature + target).")
            continue
        melted_df = df.iloc[:, :-1].melt(var_name='feature', value_name='value')
        melted_df['target'] = df_name
        data_to_plot.append(melted_df)
    
    if not data_to_plot:
        _LOGGER.error("No valid data to plot after processing all CSVs.")
        return
        
    long_df = pd.concat(data_to_plot, ignore_index=True)
    features = long_df['feature'].unique()
    _LOGGER.info(f"Found data for {len(features)} features across {len(long_df['target'].unique())} targets. Generating plots...")

    # --- Plotting Loop ---
    for feature_name in features:
        plt.figure(figsize=(12, 7))
        # Use .copy() to avoid SettingWithCopyWarning
        # feature_df = long_df[long_df['feature'] == feature_name].copy()
        feature_df = long_df[long_df['feature'] == feature_name]

        # --- Type-checking logic ---
        # Attempt to convert 'value' column to numeric.
        # errors='coerce' turns non-numeric strings (e.g., 'Category_A') into NaN
        feature_df['numeric_value'] = pd.to_numeric(feature_df['value'], errors='coerce')
        
        # If *any* value failed conversion (is NaN), treat it as categorical.
        if feature_df['numeric_value'].isna().any():
            
            # --- PLOT 1: CATEGORICAL (String-based) ---
            if verbose:
                _LOGGER.info(f"Plotting '{feature_name}' as categorical (bar plot).")
            
            # Calculate percentages for a clean bar plot
            norm_df = (feature_df.groupby('target')['value']
                       .value_counts(normalize=True)
                       .mul(100)
                       .rename('percent')
                       .reset_index())
            
            ax = sns.barplot(data=norm_df, x='value', y='percent', hue='target')
            plt.ylabel("Frequency (%)", fontsize=12)
            ax.set_ylim(0, 100) # Set Y-axis from 0 to 100
            
            # Rotate x-labels if there are many categories
            if norm_df['value'].nunique() > 10:
                plt.xticks(rotation=45, ha='right')

        else:
            # --- PLOT 2: CONTINUOUS (Numeric-based) ---
            # All values were successfully converted to numeric.
            if verbose:
                _LOGGER.info(f"Plotting '{feature_name}' as continuous (KDE plot).")
            
            # Use the 'numeric_value' column (which is float type) for the KDE
            ax = sns.kdeplot(data=feature_df, x='numeric_value', hue='target',
                             fill=True, alpha=0.1, warn_singular=False)
            
            # Set the x-axis label back to the original feature name
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


def _save_result(
        result_dict: dict,
        save_format: Literal['csv', 'sqlite', 'both'],
        csv_path: Path,
        db_manager: Optional[DatabaseManager] = None,
        db_table_name: Optional[str] = None,
        categorical_mappings: Optional[Dict[str, Dict[str, int]]] = None
    ):
    """
    Private helper to handle saving a single result to CSV, SQLite, or both.
    
    If `categorical_mappings` is provided, it will reverse-map integer values
    to their string representations before saving.
    """
    # --- Reverse Mapping Logic ---
    # Create a copy to hold the values to be saved
    save_dict = result_dict.copy()
    
    if categorical_mappings:
        for feature_name, mapping in categorical_mappings.items():
            if feature_name in save_dict:
                # Create a reverse map {0: 'Category_A', 1: 'Category_B'}
                reverse_map = {idx: name for name, idx in mapping.items()}
                
                # Get the integer value from the results (e.g., 0)
                int_value = save_dict[feature_name]
                
                # Find the corresponding string (e.g., 'Category_A')
                # Use .get() for safety, defaulting to the original value if not found
                string_value = reverse_map.get(int_value, int_value)
                
                # Update the dictionary that will be saved
                save_dict[feature_name] = string_value
    
    # Save to CSV
    if save_format in ['csv', 'both']:
        df_row = pd.DataFrame([save_dict])
        file_exists = csv_path.exists()
        df_row.to_csv(csv_path, mode='a', index=False, header=not file_exists)

    # Save to SQLite
    if save_format in ['sqlite', 'both']:
        if db_manager and db_table_name:
            db_manager.insert_row(db_table_name, save_dict)
        else:
            _LOGGER.warning("SQLite saving requested but db_manager or table_name not provided.")


def info():
    _script_info(__all__)
