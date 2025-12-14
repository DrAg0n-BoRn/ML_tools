import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Any, Literal, Optional, Dict, List, Tuple
from pathlib import Path
import pandas as pd

from ._path_manager import make_fullpath, list_csv_paths, sanitize_filename
from ._utilities import yield_dataframes_from_dir
from ._logger import get_logger
from ._script_info import _script_info
from ._SQL import DragonSQL
from ._IO_tools import save_json, load_json
from ._schema import FeatureSchema
from ._keys import OptimizationToolsKeys


_LOGGER = get_logger("Optimization Tools")


__all__ = [
    "make_continuous_bounds_template",
    "load_continuous_bounds_template",
    "create_optimization_bounds",
    "parse_lower_upper_bounds",
    "plot_optimal_feature_distributions",
    "plot_optimal_feature_distributions_from_dataframe",
]


def make_continuous_bounds_template(
    directory: Union[str, Path],
    feature_schema: FeatureSchema,
    default_bounds: Tuple[float, float] = (0, 1)
) -> None:
    """
    Creates a JSON template for manual entry of continuous feature optimization bounds.

    The resulting file maps each continuous feature name to a [min, max] list 
    populated with `default_bounds`. Edit the values in this file before using.

    Args:
        directory (str | Path): The directory where the template will be saved.
        feature_schema (FeatureSchema): The loaded schema containing feature definitions.
        default_bounds (Tuple[float, float]): Default (min, max) values to populate the template.
    """
    # validate directory path
    dir_path = make_fullpath(directory, make=True, enforce="directory")
    
    # 1. Check if continuous features exist
    if not feature_schema.continuous_feature_names:
        _LOGGER.warning("No continuous features found in FeatureSchema. Skipping bounds template generation.")
        return

    # 2. Construct the dictionary: {feature_name: [min, max]}
    bounds_map = {
        name: list(default_bounds)
        for name in feature_schema.continuous_feature_names
    }
    
    # use a fixed key for the filename
    filename = OptimizationToolsKeys.OPTIMIZATION_BOUNDS_FILENAME + ".json"

    # 3. Save to JSON using the IO tool
    save_json(
        data=bounds_map,
        directory=dir_path,
        filename=filename,
        verbose=False
    )
    
    _LOGGER.info(f"💾 Continuous bounds template saved to: '{dir_path.name}/{filename}'")
    

def load_continuous_bounds_template(directory: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Loads the continuous feature bounds template from JSON. Expected filename: `optimization_bounds.json`.

    Args:
        directory (str | Path): The directory where the template is located.

    Returns:
        Dictionary (Dict[str, List[float]]): A dictionary mapping feature names to [min, max] bounds.
    """
    dir_path = make_fullpath(directory, enforce="directory")
    full_path = dir_path / (OptimizationToolsKeys.OPTIMIZATION_BOUNDS_FILENAME + ".json")
    
    bounds_map = load_json(
        file_path=full_path,
        expected_type='dict',
        verbose=False
    )
    
    # validate loaded data
    if not all(
            isinstance(v, list) and     # Check type
            len(v) == 2 and     # Check length
            all(isinstance(i, (int, float)) for i in v) # Check contents are numbers
            for v in bounds_map.values()
        ):
        _LOGGER.error(f"Invalid format in bounds template at '{full_path}'. Each value must be a list of [min, max].")
        raise ValueError()
    
    _LOGGER.info(f"Continuous bounds template loaded from: '{dir_path.name}'")
    
    return bounds_map


def create_optimization_bounds(
    schema: FeatureSchema,
    continuous_bounds_map: Union[Dict[str, Tuple[float, float]], Dict[str, List[float]]],
    start_at_zero: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Generates the lower and upper bounds lists for the optimizer from a FeatureSchema.

    This helper function automates the creation of unbiased bounds for
    categorical features and combines them with user-defined bounds for
    continuous features, using the schema as the single source of truth
    for feature order and type.

    Args:
        schema (FeatureSchema):
            The definitive schema object created by 
            `data_exploration.finalize_feature_schema()`.
        continuous_bounds_map (Dict[str, Tuple[float, float]], Dict[str, List[float]]):
            A dictionary mapping the *name* of each **continuous** feature
            to its (min_bound, max_bound).
        start_at_zero (bool):
            - If True, assumes categorical encoding is [0, 1, ..., k-1].
              Bounds will be set as [-0.5, k - 0.5].
            - If False, assumes encoding is [1, 2, ..., k].
              Bounds will be set as [0.5, k + 0.5].

    Returns:
        Tuple[List[float], List[float]]:
            A tuple containing two lists: (lower_bounds, upper_bounds).

    Raises:
        ValueError: If a feature is missing from `continuous_bounds_map`
                    or if a feature name in the map is not a
                    continuous feature according to the schema.
    """
    # validate length in the continuous_bounds_map values
    for name, bounds in continuous_bounds_map.items():
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            _LOGGER.error(f"Bounds for feature '{name}' must be a list or tuple of length 2 (min, max). Found: {bounds}")
            raise ValueError()
    
    # 1. Get feature names and map from schema
    feature_names = schema.feature_names
    categorical_index_map = schema.categorical_index_map
    total_features = len(feature_names)

    if total_features <= 0:
        _LOGGER.error("Schema contains no features.")
        raise ValueError()
        
    _LOGGER.info(f"Generating bounds for {total_features} total features...")

    # 2. Initialize bound lists
    lower_bounds: List[Optional[float]] = [None] * total_features
    upper_bounds: List[Optional[float]] = [None] * total_features

    # 3. Populate categorical bounds (Index-based)
    if categorical_index_map:
        for index, cardinality in categorical_index_map.items():
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
        
        _LOGGER.info(f"Automatically set bounds for {len(categorical_index_map)} categorical features.")
    else:
        _LOGGER.info("No categorical features found in schema.")

    # 4. Populate continuous bounds (Name-based)
    # Use schema.continuous_feature_names for robust checking
    continuous_names_set = set(schema.continuous_feature_names)
    
    if continuous_names_set != set(continuous_bounds_map.keys()):
        missing_in_map = continuous_names_set - set(continuous_bounds_map.keys())
        if missing_in_map:
            _LOGGER.error(f"The following continuous features are missing from 'continuous_bounds_map': {list(missing_in_map)}")
        
        extra_in_map = set(continuous_bounds_map.keys()) - continuous_names_set
        if extra_in_map:
            _LOGGER.error(f"The following features in 'continuous_bounds_map' are not defined as continuous in the schema: {list(extra_in_map)}")
            
        raise ValueError("Mismatch between 'continuous_bounds_map' and schema's continuous features.")

    count_continuous = 0
    for name, (low, high) in continuous_bounds_map.items():
        # Map name to its index in the *feature-only* list
        # This is guaranteed to be correct by the schema
        index = feature_names.index(name)

        if lower_bounds[index] is not None:
            # This should be impossible if schema is correct, but good to check
            _LOGGER.error(f"Schema conflict: Feature '{name}' (at index {index}) is defined as both continuous and categorical.")
            raise ValueError()

        lower_bounds[index] = float(low)
        upper_bounds[index] = float(high)
        count_continuous += 1
        
    _LOGGER.info(f"Manually set bounds for {count_continuous} continuous features.")

    # 5. Final Validation (all Nones should be filled)
    if None in lower_bounds:
        missing_indices = [i for i, b in enumerate(lower_bounds) if b is None]
        missing_names = [feature_names[i] for i in missing_indices]
        _LOGGER.error(f"Failed to create all bounds. This indicates an internal logic error. Missing: {missing_names}")
        raise RuntimeError("Internal error: Not all bounds were populated.")
    
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


def plot_optimal_feature_distributions(results_dir: Union[str, Path], 
                                       verbose: bool=False,
                                       target_columns: Optional[List[str]] = None):
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
    list_csv_paths(results_path, verbose=False)

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
                                                      target_columns: Optional[List[str]] = None):
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


def _save_result(
        result_dict: dict,
        save_format: Literal['csv', 'sqlite', 'both'],
        csv_path: Path,
        db_manager: Optional[DragonSQL] = None,
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
