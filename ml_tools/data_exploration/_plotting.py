import pandas as pd
import numpy as np
from typing import Optional, Union, Literal
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_object_dtype

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Data Exploration: Plotting")


__all__ = [
    "plot_value_distributions",
    "plot_continuous_vs_target",
    "plot_categorical_vs_target",
    "plot_correlation_heatmap",
]


def plot_value_distributions(
    df: pd.DataFrame,
    save_dir: Union[str, Path],
    categorical_columns: Optional[list[str]] = None,
    max_categories: int = 100,
    fill_na_with: str = "MISSING DATA"
):
    """
    Plots and saves the value distributions for all columns in a DataFrame,
    using the best plot type for each column (histogram or count plot).

    Plots are saved as SVG files under two subdirectories in `save_dir`:
    - "Distribution_Continuous" for continuous numeric features (histograms).
    - "Distribution_Categorical" for categorical features (count plots).

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        save_dir (str | Path): Directory path to save the plots.
        categorical_columns (List[str] | None): If provided, these will be treated as categorical, and all other columns will be treated as continuous.
        max_categories (int): The maximum number of unique categories a categorical feature can have to be plotted. Features exceeding this limit will be skipped.
        fill_na_with (str): A string to replace NaN values in categorical columns. This allows plotting 'missingness' as its own category.

    Notes:
        - `seaborn.histplot` with KDE is used for continuous features.
        - `seaborn.countplot` is used for categorical features.
    """
    # 1. Setup save directories
    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")
    numeric_dir = base_save_path / "Distribution_Continuous"
    categorical_dir = base_save_path / "Distribution_Categorical"
    numeric_dir.mkdir(parents=True, exist_ok=True)
    categorical_dir.mkdir(parents=True, exist_ok=True)

    # 2. Filter columns to plot
    columns_to_plot = df.columns.to_list()

    # Setup for forced categorical logic
    categorical_set = set(categorical_columns) if categorical_columns is not None else None

    numeric_plots_saved = 0
    categorical_plots_saved = 0

    for col_name in columns_to_plot:
        try:
            is_numeric = is_numeric_dtype(df[col_name])
            n_unique = df[col_name].nunique()

            # --- 3. Determine Plot Type ---
            is_continuous = False
            if categorical_set is not None:
                # Use the explicit list
                if col_name not in categorical_set:
                    is_continuous = True
            else:
                # Use auto-detection
                if is_numeric:
                    is_continuous = True
            
            # --- Case 1: Continuous Numeric (Histogram) ---
            if is_continuous:
                plt.figure(figsize=(10, 6))
                # Drop NaNs for histogram, as they can't be plotted on a numeric axis
                sns.histplot(x=df[col_name].dropna(), kde=True, bins=30)
                plt.title(f"Distribution of '{col_name}' (Continuous)")
                plt.xlabel(col_name)
                plt.ylabel("Count")
                
                save_path = numeric_dir / f"{sanitize_filename(col_name)}.svg"
                numeric_plots_saved += 1

            # --- Case 2: Categorical (Count Plot) ---
            else:
                # Check max categories
                if n_unique > max_categories:
                    _LOGGER.warning(f"Skipping plot for '{col_name}': {n_unique} unique values > {max_categories} max_categories.")
                    continue

                # Adaptive figure size
                fig_width = max(10, n_unique * 0.5)
                plt.figure(figsize=(fig_width, 8))
                
                # Make a temporary copy for plotting to handle NaNs
                temp_series = df[col_name].copy()
                
                # Handle NaNs by replacing them with the specified string
                if temp_series.isnull().any():
                    # Convert to object type first to allow string replacement
                    temp_series = temp_series.astype(object).fillna(fill_na_with)
                
                # Convert all to string to be safe (handles low-card numeric)
                temp_series = temp_series.astype(str)
                
                # Get category order by frequency
                order = temp_series.value_counts().index
                sns.countplot(x=temp_series, order=order, palette="Oranges", hue=temp_series, legend=False)
                
                plt.title(f"Distribution of '{col_name}' (Categorical)")
                plt.xlabel(col_name)
                plt.ylabel("Count")
                
                # Smart tick rotation
                max_label_len = 0
                if n_unique > 0:
                    max_label_len = max(len(str(s)) for s in order)
                
                # Rotate if labels are long OR there are many categories
                if max_label_len > 10 or n_unique > 25:
                    plt.xticks(rotation=45, ha='right')
                
                save_path = categorical_dir / f"{sanitize_filename(col_name)}.svg"
                categorical_plots_saved += 1

            # --- 4. Save Plot ---
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()
            # Save as .svg
            plt.savefig(save_path, format='svg', bbox_inches="tight")
            plt.close()

        except Exception as e:
            _LOGGER.error(f"Failed to plot distribution for '{col_name}'. Error: {e}")
            plt.close()
    
    _LOGGER.info(f"Saved {numeric_plots_saved} continuous distribution plots to '{numeric_dir.name}'.")
    _LOGGER.info(f"Saved {categorical_plots_saved} categorical distribution plots to '{categorical_dir.name}'.")


def plot_continuous_vs_target(
    df_continuous: pd.DataFrame,
    df_targets: pd.DataFrame,
    save_dir: Union[str, Path],
    verbose: int = 1
):
    """
    Plots each continuous feature from df_continuous against each target in df_targets.

    This function creates a scatter plot for each feature-target pair, overlays a 
    simple linear regression line, and saves each plot as an individual .svg file.

    Plots are saved in a structured way, with a subdirectory created for
    each target variable.

    Args:
        df_continuous (pd.DataFrame): DataFrame containing continuous feature columns (x-axis).
        df_targets (pd.DataFrame): DataFrame containing target columns (y-axis).
        save_dir (str | Path): The base directory where plots will be saved.
        verbose (int): Verbosity level for logging warnings.

    Notes:
        - Only numeric features and numeric targets are processed.
        - Rows with NaN in either the feature or the target are dropped pairwise.
        - Assumes df_continuous and df_targets share the same index.
    """
    # 1. Validate the base save directory
    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")

    # 2. Validation helper
    def _get_valid_numeric_cols(df: pd.DataFrame, df_name: str) -> list[str]:
        valid_cols = []
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                if verbose > 0:
                    _LOGGER.warning(f"Column '{col}' in {df_name} is not numeric. Skipping.")
            else:
                valid_cols.append(col)
        return valid_cols

    # 3. Validate target columns
    valid_targets = _get_valid_numeric_cols(df_targets, "df_targets")
    if not valid_targets:
        _LOGGER.error("No valid numeric target columns provided in df_targets.")
        return
    
    # 4. Validate feature columns
    valid_features = _get_valid_numeric_cols(df_continuous, "df_continuous")
    if not valid_features:
        _LOGGER.error("No valid numeric feature columns provided in df_continuous.")
        return

    # 5. Main plotting loop
    total_plots_saved = 0
    
    for target_name in valid_targets:
        # Create a sanitized subdirectory for this target
        safe_target_dir_name = sanitize_filename(f"{target_name}_vs_Continuous")
        target_save_dir = base_save_path / safe_target_dir_name
        target_save_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose > 0:
            _LOGGER.info(f"Generating plots for target: '{target_name}' -> Saving to '{target_save_dir.name}'")

        for feature_name in valid_features:
            
            # Align data and drop NaNs pairwise - use concat to ensure we respect the index alignment between the two DFs
            temp_df = pd.concat([
                df_continuous[feature_name], 
                df_targets[target_name]
            ], axis=1).dropna()

            if temp_df.empty:
                if verbose > 1:
                    _LOGGER.warning(f"No non-null data for '{feature_name}' vs '{target_name}'. Skipping plot.")
                continue

            x = temp_df[feature_name]
            y = temp_df[target_name]

            # 6. Perform linear fit
            try:
                # Modern replacement for np.polyfit + np.poly1d
                p = np.polynomial.Polynomial.fit(x, y, deg=1)
                plot_regression_line = True
            except (np.linalg.LinAlgError, ValueError):
                if verbose > 0:
                    _LOGGER.warning(f"Linear regression failed for '{feature_name}' vs '{target_name}'. Plotting scatter only.")
                plot_regression_line = False

            # 7. Create the plot
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            
            # Plot the raw data points
            ax.plot(x, y, 'o', alpha=0.5, label='Data points', markersize=5)
            
            # Plot the regression line
            if plot_regression_line:
                ax.plot(x, p(x), "r--", label='Linear Fit') # type: ignore

            ax.set_title(f'{feature_name} vs {target_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel(target_name)
            ax.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # 8. Save the plot
            safe_feature_name = sanitize_filename(feature_name)
            plot_filename = f"{safe_feature_name}_vs_{safe_target_dir_name}.svg"
            plot_path = target_save_dir / plot_filename
            
            try:
                plt.savefig(plot_path, bbox_inches="tight", format='svg')
                total_plots_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to save plot: {plot_path}. Error: {e}")
            
            # Close the figure to free up memory
            plt.close()
    
    if verbose > 0:
        _LOGGER.info(f"Successfully saved {total_plots_saved} feature-vs-target plots to '{base_save_path}'.")


def plot_categorical_vs_target(
    df_categorical: pd.DataFrame,
    df_targets: pd.DataFrame,
    save_dir: Union[str, Path],
    max_categories: int = 50,
    fill_na_with: str = "MISSING DATA",
    drop_empty_targets: bool = True,
    verbose: int = 1
):
    """
    Plots each feature in df_categorical against each numeric target in df_targets using box plots.

    Automatically aligns the two DataFrames by index. If a numeric
    column is passed within df_categorical, it will be cast to object type to treat it as a category.

    Args:
        df_categorical (pd.DataFrame): DataFrame containing categorical feature columns (x-axis).
        df_targets (pd.DataFrame): DataFrame containing numeric target columns (y-axis).
        save_dir (str | Path): Base directory for saving plots.
        max_categories (int): The maximum number of unique categories a feature can have to be plotted.
        fill_na_with (str): String to replace NaN values in categorical columns.
        drop_empty_targets (bool): If True, drops rows where the target value is NaN before plotting.
        verbose (int): Verbosity level for logging warnings.

    Notes:
        - Assumes df_categorical and df_targets share the same index.
    """
    # 1. Validate the base save directory
    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")

    # 2. Validate target columns (must be numeric)
    valid_targets = []
    for col in df_targets.columns:
        if not is_numeric_dtype(df_targets[col]):
            if verbose > 0:
                _LOGGER.warning(f"Target column '{col}' in df_targets is not numeric. Skipping.")
        else:
            valid_targets.append(col)
    
    if not valid_targets:
        _LOGGER.error("No valid numeric target columns provided in df_targets.")
        return

    # 3. Validate feature columns (Flexible: Allow numeric but warn)
    valid_features = []
    for col in df_categorical.columns:
        # If numeric, warn but accept it (will be cast to object later)
        if is_numeric_dtype(df_categorical[col]):
            if verbose > 0:
                _LOGGER.warning(f"Feature '{col}' in df_categorical is numeric. It will be cast to 'object' and treated as categorical.")
            valid_features.append(col)
        else:
            # Assume it is already object/category
            valid_features.append(col)

    if not valid_features:
        _LOGGER.error("No valid feature columns provided in df_categorical.")
        return

    # 4. Main plotting loop
    total_plots_saved = 0
    
    for target_name in valid_targets:
        # Create a sanitized subdirectory for this target
        safe_target_dir_name = sanitize_filename(f"{target_name}_vs_Categorical")
        target_save_dir = base_save_path / safe_target_dir_name
        target_save_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose > 0:
            _LOGGER.info(f"Generating plots for target: '{target_name}' -> Saving to '{target_save_dir.name}'")
        
        for feature_name in valid_features:
            
            # Align data using concat to respect indices
            feature_series = df_categorical[feature_name]
            target_series = df_targets[target_name]

            # Create a temporary DataFrame for this pair
            temp_df = pd.concat([feature_series, target_series], axis=1)

            # Optional: Drop rows where the target is NaN
            if drop_empty_targets:
                temp_df = temp_df.dropna(subset=[target_name])
                if temp_df.empty:
                    if verbose > 1:
                        _LOGGER.warning(f"No valid data left for '{feature_name}' vs '{target_name}' after dropping empty targets. Skipping.")
                    continue

            # Force feature to object if it isn't already (handling the numeric flexibility)
            if not is_object_dtype(temp_df[feature_name]):
                temp_df[feature_name] = temp_df[feature_name].astype(object)

            # Handle NaNs in the feature column (treat as a category)
            if temp_df[feature_name].isnull().any():
                temp_df[feature_name] = temp_df[feature_name].fillna(fill_na_with)
            
            # Convert to string to ensure consistent plotting and cardinality check
            temp_df[feature_name] = temp_df[feature_name].astype(str)

            # Check cardinality
            n_unique = temp_df[feature_name].nunique()
            if n_unique > max_categories:
                if verbose > 1:
                    _LOGGER.warning(f"Skipping '{feature_name}': {n_unique} unique categories > {max_categories} max_categories.")
                continue

            # 5. Create the plot
            # Dynamic figure width based on number of categories
            plt.figure(figsize=(max(10, n_unique * 0.8), 10))
            
            sns.boxplot(x=feature_name, y=target_name, data=temp_df)

            plt.title(f'{target_name} vs {feature_name}')
            plt.xlabel(feature_name)
            plt.ylabel(target_name)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()

            # 6. Save the plot
            safe_feature_name = sanitize_filename(feature_name)
            plot_filename = f"{safe_feature_name}_vs_{safe_target_dir_name}.svg"
            plot_path = target_save_dir / plot_filename
            
            try:
                plt.savefig(plot_path, bbox_inches="tight", format='svg')
                total_plots_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to save plot: {plot_path}. Error: {e}")
            
            plt.close()
    
    if verbose > 0:
        _LOGGER.info(f"Successfully saved {total_plots_saved} categorical-vs-target plots to '{base_save_path}'.")



def plot_correlation_heatmap(df: pd.DataFrame,
                             plot_title: str,
                             save_dir: Union[str, Path, None] = None, 
                             method: Literal["pearson", "kendall", "spearman"]="pearson"):
    """
    Plots a heatmap of pairwise correlations between numeric features in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | Path | None): If provided, the heatmap will be saved to this directory as a svg file.
        plot_title: The suffix "`method` Correlation Heatmap" will be automatically appended.
        method (str): Correlation method to use. Must be one of:
            - 'pearson' (default): measures linear correlation (assumes normally distributed data),
            - 'kendall': rank correlation (non-parametric),
            - 'spearman': monotonic relationship (non-parametric).

    Notes:
        - Only numeric columns are included.
        - Annotations are disabled if there are more than 20 features.
        - Missing values are handled via pairwise complete observations.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        _LOGGER.warning("No numeric columns found. Heatmap not generated.")
        return
    if method not in ["pearson", "kendall", "spearman"]:
        _LOGGER.error(f"'method' must be pearson, kendall, or spearman.")
        raise ValueError()
    
    corr = numeric_df.corr(method=method)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot setup
    size = max(10, numeric_df.shape[1])
    plt.figure(figsize=(size, size * 0.8))

    annot_bool = numeric_df.shape[1] <= 20
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot_bool,
        cmap='coolwarm',
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        vmin=-1,  # Anchors minimum color to -1
        vmax=1,   # Anchors maximum color to 1
        center=0  # Ensures 0 corresponds to the neutral color (white)
    )
    
    # add suffix to title
    full_plot_title = f"{plot_title} - {method.title()} Correlation Heatmap"
    
    plt.title(full_plot_title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    
    if save_dir:
        save_path = make_fullpath(save_dir, make=True)
        # sanitize the plot title to save the file
        sanitized_plot_title = sanitize_filename(plot_title)
        plot_filename = sanitized_plot_title + ".svg"
        
        full_path = save_path / plot_filename
        
        plt.savefig(full_path, bbox_inches="tight", format='svg')
        _LOGGER.info(f"Saved correlation heatmap: '{plot_filename}'")
    
    plt.show()
    plt.close()

