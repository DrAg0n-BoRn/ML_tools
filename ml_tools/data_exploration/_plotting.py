import pandas as pd
import numpy as np
from typing import Optional, Union, Literal
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_object_dtype

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Data Exploration: Visualization")


__all__ = [
    "plot_value_distributions",
    "plot_value_distributions_multi",
    "plot_numeric_overview_boxplot",
    "plot_numeric_overview_boxplot_macro",
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


def plot_value_distributions_multi(
    named_dataframes: dict[str, pd.DataFrame],
    save_dir: Union[str, Path],
    max_categories: int = 100,
    fill_na_with: str = "MISSING DATA",
    font_scaling: float = 1.0
):
    """
    Plots and saves the value distributions for all columns across multiple DataFrames.
    Overlaps the data from each DataFrame for comparison.
    
    All DataFrames must have the same columns and data types for consistent plotting. 
    The function will automatically detect numeric vs categorical features and plot them accordingly.

    Plots are saved as SVG files under two subdirectories in `save_dir`:
    - "Distribution_Continuous" for continuous numeric features.
    - "Distribution_Categorical" for categorical features.

    Args:
        named_dataframes (dict[str, pd.DataFrame]): Dictionary mapping dataset names to DataFrames.
        save_dir (str | Path): Directory path to save the plots.
        max_categories (int): The maximum number of unique categories a categorical feature can have to be plotted.
        fill_na_with (str): A string to replace NaN values in categorical columns.
        font_scaling (float): Scaling factor for all fonts in the generated plots.
    """
    # 1. Setup save directories
    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")
    numeric_dir = base_save_path / "Distribution_Continuous"
    categorical_dir = base_save_path / "Distribution_Categorical"
    numeric_dir.mkdir(parents=True, exist_ok=True)
    categorical_dir.mkdir(parents=True, exist_ok=True)
    
    SECRET_COLUMN_NAME = "__SoUrCe_DaTaSeT__"

    # 2. Combine DataFrames for efficient plotting with Seaborn
    combined_dfs = []
    valid_data_types = None
    for name, df in named_dataframes.items():
        df_pd = df.copy()
        current_dtypes = df_pd.dtypes.sort_index()
        
        # Check for consistent columns and data types across DataFrames
        if valid_data_types is None:
            valid_data_types = current_dtypes
        elif not current_dtypes.equals(valid_data_types):
            _LOGGER.error(
                f"Schema mismatch in '{name}'. "
                f"Expected columns and types:\n{valid_data_types}\n"
                f"But found:\n{current_dtypes}"
            )
            raise ValueError()

        df_pd[SECRET_COLUMN_NAME] = name
        combined_dfs.append(df_pd)
        
    if not combined_dfs:
        _LOGGER.warning("No dataframes provided.")
        return
    
    # Concatenate over axis=0 to add rows
    plot_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Filter columns to plot (excluding the temporary label column)
    columns_to_plot = [col for col in plot_df.columns if col != SECRET_COLUMN_NAME]

    numeric_plots_saved = 0
    categorical_plots_saved = 0
    
    # Apply font scaling context to all plots generated in this block
    with sns.plotting_context("notebook", font_scale=font_scaling):
        for col_name in columns_to_plot:
            try:
                is_numeric = is_numeric_dtype(plot_df[col_name])
                n_unique = plot_df[col_name].nunique()

                # --- Case 1: Continuous Numeric (Overlapping Histogram/KDE) ---
                if is_numeric:
                    plt.figure(figsize=(10, 6))
                    
                    # Drop NaNs for plotting numeric axes
                    plot_data = plot_df[[col_name, SECRET_COLUMN_NAME]].dropna()
                    
                    sns.histplot(
                        data=plot_data, 
                        x=col_name, 
                        hue=SECRET_COLUMN_NAME, 
                        kde=True, 
                        common_norm=False, 
                        bins=30, 
                        alpha=0.5
                    )
                    
                    plt.title(f"Distribution of '{col_name}' (Continuous Comparison)")
                    plt.xlabel(col_name)
                    plt.ylabel("Count")
                    
                    save_path = numeric_dir / f"{sanitize_filename(col_name)}.svg"
                    numeric_plots_saved += 1

                # --- Case 2: Categorical (Grouped Count Plot) ---
                else:
                    if n_unique > max_categories:
                        _LOGGER.warning(f"Skipping plot for '{col_name}': {n_unique} unique values > {max_categories}.")
                        continue

                    # Adaptive figure size
                    fig_width = max(10, n_unique * 0.8)
                    plt.figure(figsize=(fig_width, 8))
                    
                    plot_data = plot_df[[col_name, SECRET_COLUMN_NAME]].copy()
                    
                    if plot_data[col_name].isnull().any():
                        plot_data[col_name] = plot_data[col_name].astype(object).fillna(fill_na_with)
                    
                    plot_data[col_name] = plot_data[col_name].astype(str)
                    
                    # Get category order by total frequency across all datasets
                    order = plot_data[col_name].value_counts().index
                    
                    sns.countplot(
                        data=plot_data, 
                        x=col_name, 
                        hue=SECRET_COLUMN_NAME, 
                        order=order
                    )
                    
                    plt.title(f"Distribution of '{col_name}' (Categorical Comparison)")
                    plt.xlabel(col_name)
                    plt.ylabel("Count")
                    
                    max_label_len = max([len(str(s)) for s in order] + [0])
                    if max_label_len > 10 or n_unique > 15:
                        plt.xticks(rotation=45, ha='right')
                    
                    save_path = categorical_dir / f"{sanitize_filename(col_name)}.svg"
                    categorical_plots_saved += 1

                # --- 4. Save Plot ---
                plt.grid(True, linestyle='--', alpha=0.6, axis='y')
                plt.tight_layout()
                plt.savefig(save_path, format='svg', bbox_inches="tight")
                plt.close()

            except Exception as e:
                _LOGGER.error(f"Failed to plot distribution for '{col_name}'. Error: {e}")
                plt.close()
    
    _LOGGER.info(f"Saved {numeric_plots_saved} continuous distribution comparison plots to '{numeric_dir.name}'.")
    _LOGGER.info(f"Saved {categorical_plots_saved} categorical distribution comparison plots to '{categorical_dir.name}'.")


def plot_numeric_overview_boxplot(
    df: pd.DataFrame,
    save_dir: Union[str, Path],
    plot_title: str = "Distribution Overview",
    strategy: Literal["value", "log", "scale"] = "value",
    handle_zero_variance: Literal["drop", "constant"] = "drop",
    show_means: bool = True,
    font_scaling: float = 1.0
):
    """
    Creates a single boxplot showing the distribution and range of all numeric columns.
    
    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | Path): Directory path to save the plot.
        plot_title (str): The title of the plot.
        strategy (Literal["value", "log", "scale"]): Visualization strategy to handle varying ranges.
            - "value": Plots raw values (default).
            - "log": Applies log transformation to handle skewed distributions.
            - "scale": Applies Robust scaling (using Median and IQR) to handle different scales while ignoring extreme outliers.
        handle_zero_variance (Literal["drop", "constant"]): How to handle zero-variance columns when strategy="scale".
            - "drop": Exclude zero-variance columns from the plot (default).
            - "constant": Set zero-variance columns to a constant value (0.0) after scaling, allowing them to be plotted.
        show_means (bool): If True, shows the mean value as a distinct marker on the boxplot.
        font_scaling (float): Multiplier for all text elements in the plot.
    """
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.empty:
        _LOGGER.warning("No numeric columns found. Overview boxplot not generated.")
        return

    # Apply scaling if requested
    if strategy == "scale":
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        zero_iqr_cols = numeric_df.columns[iqr == 0]
        
        if not zero_iqr_cols.empty:
            if handle_zero_variance == "drop":
                numeric_df = numeric_df.drop(columns=zero_iqr_cols)
                _LOGGER.warning(f"Dropped zero-IQR columns during robust scaling: {list(zero_iqr_cols)}")
                if numeric_df.empty:
                    _LOGGER.warning("No columns left after dropping zero-IQR features.")
                    return
                # Recalculate after drop
                q1 = numeric_df.quantile(0.25)
                q3 = numeric_df.quantile(0.75)
                iqr = q3 - q1
        
        # Using pandas native operations to achieve Robust scaling
        median = numeric_df.median()
        numeric_df = (numeric_df - median) / iqr
        
        # Intercept the NaNs generated by division by zero and set to 0.0
        if handle_zero_variance == "constant" and not zero_iqr_cols.empty:
            numeric_df[zero_iqr_cols] = 0.0
            _LOGGER.warning(f"Set zero-IQR columns to 0 during robust scaling: {list(zero_iqr_cols)}")
            
        x_label = "Value (Robust Scaled)"
    elif strategy == "log":
        x_label = "Value (Log Scale)"
    else:
        x_label = "Value"

    # Setup save path
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # Dynamic figure size based on the number of features
    num_features = numeric_df.shape[1]
    fig_height = max(6, num_features * 0.5)
    
    plt.figure(figsize=(12, fig_height))
    
    # Using orient='h' for better label readability with many features
    ax = sns.boxplot(data=numeric_df, 
                     orient='h', 
                     palette="Set2", 
                     showmeans=show_means, 
                     meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6})
    
    # Robust scaling cannot use a fixed range
    if strategy == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    elif strategy == "scale":
        # Add a vertical reference line at 0.0 (the aligned median for all robust-scaled features)
        ax.axvline(x=0.0, color='red', linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    
    # Calculate scaled font sizes
    title_fs = 18 * font_scaling
    label_fs = 14 * font_scaling
    tick_fs = 12 * font_scaling
    
    plt.title(plot_title, fontsize=title_fs)
    plt.xlabel(x_label, fontsize=label_fs)
    plt.ylabel("", fontsize=0)
    
    # scale tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    
    safe_title = sanitize_filename(plot_title).replace(".", "_")
    
    if strategy not in safe_title.lower():
        plot_filename = f"{safe_title}_{strategy}.svg"
    else:
        plot_filename = f"{safe_title}.svg"
    
    full_path = save_path / plot_filename
    
    try:
        plt.savefig(full_path, bbox_inches="tight", format='svg')
        _LOGGER.info(f"Saved numeric overview boxplot: '{plot_filename}' to '{save_path.name}'.")
    except Exception as e:
        _LOGGER.error(f"Failed to save numeric overview boxplot. Error: {e}")
    
    plt.close()


# macro function to plot overview boxplots using all strategies in one go
def plot_numeric_overview_boxplot_macro(df: pd.DataFrame, 
                                        save_dir: Union[str, Path], 
                                        plot_title: str = "Distribution Overview",
                                        handle_zero_variance: Literal["drop", "constant"] = "drop",
                                        show_means: bool = True,
                                        font_scaling: float = 1.0):
    """
    Plots numeric overview boxplots using all strategies ("value", "log", "scale") in one go, saving each plot with a strategy-specific suffix.
    
    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | Path): Directory path to save the plot.
        plot_title (str): The title of the plot.
        handle_zero_variance (Literal["drop", "constant"]): How to handle zero-variance columns when strategy="scale".
            - "drop": Exclude zero-variance columns from the plot (default).
            - "constant": Set zero-variance columns to a constant value (0.0) after scaling, allowing them to be plotted.
        show_means (bool): If True, shows the mean value as a distinct marker on the boxplot.
        font_scaling (float): Multiplier for all text elements in the plot.
    """
    
    strategies: tuple[Literal["value", "log", "scale"], ...] = ("value", "log", "scale")
    
    for strategy in strategies:
        plot_numeric_overview_boxplot(
            df=df,
            save_dir=save_dir,
            plot_title=f"{plot_title} ({strategy.title()})",
            strategy=strategy,
            handle_zero_variance=handle_zero_variance,
            show_means=show_means,
            font_scaling=font_scaling
        )


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
        safe_target_name = sanitize_filename(target_name)
        # Create a sanitized subdirectory for this target
        safe_target_dir_name = f"{safe_target_name}_vs_Continuous"
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
            plot_filename = f"{safe_feature_name}_vs_{safe_target_name}.svg"
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
        safe_target_name = sanitize_filename(target_name)
        # Create a sanitized subdirectory for this target
        safe_target_dir_name = f"{safe_target_name}_vs_Categorical"
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
            plot_filename = f"{safe_feature_name}_vs_{safe_target_name}.svg"
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
        save_dir (str | Path | None): If provided, the heatmap will be saved to this directory as a svg file instead of displaying it.
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
    if numeric_df.shape[1] < 2:
        _LOGGER.warning(f"Not enough numeric columns found ({numeric_df.shape[1]}). Heatmap requires at least 2.")
        return
    if method not in ["pearson", "kendall", "spearman"]:
        _LOGGER.error(f"'method' must be pearson, kendall, or spearman.")
        raise ValueError()
    
    corr = numeric_df.corr(method=method)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Remove the top row and rightmost column to drop redundant empty axis labels
    corr = corr.iloc[1:, :-1]
    mask = mask[1:, :-1]

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
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        # sanitize the plot title to save the file
        sanitized_plot_title = sanitize_filename(plot_title)
        # prepend method to filename
        sanitized_plot_title = f"{method}_{sanitized_plot_title}"
        
        plot_filename = sanitized_plot_title + ".svg"
        
        full_path = save_path / plot_filename
        
        plt.savefig(full_path, bbox_inches="tight", format='svg')
        _LOGGER.info(f"Saved correlation heatmap: '{plot_filename}'")
    else:
        plt.show()
    
    plt.close()

