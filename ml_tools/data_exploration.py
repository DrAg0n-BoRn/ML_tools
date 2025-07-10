import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
from IPython.display import clear_output
import time
from typing import Union, Literal, Dict, Tuple, List, Optional
from pathlib import Path
from .utilities import sanitize_filename, _script_info, make_fullpath
import re


# Keep track of all available tools, show using `info()`
__all__ = [
    "summarize_dataframe",
    "drop_constant_columns",
    "drop_rows_with_missing_data",
    "split_features_targets", 
    "show_null_columns",
    "drop_columns_with_missing_data",
    "split_continuous_binary", 
    "plot_correlation_heatmap", 
    "check_value_distributions", 
    "plot_value_distributions", 
    "clip_outliers_single", 
    "clip_outliers_multi",
    "match_and_filter_columns_by_regex",
    "standardize_percentages"
]


def summarize_dataframe(df: pd.DataFrame, round_digits: int = 2):
    """
    Returns a summary DataFrame with data types, non-null counts, number of unique values,
    missing value percentage, and basic statistics for each column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Decimal places to round numerical statistics.

    Returns:
        pd.DataFrame: Summary table.
    """
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Unique Values': df.nunique(),
        'Missing %': (df.isnull().mean() * 100).round(round_digits)
    })

    # For numeric columns, add summary statistics
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        summary_numeric = df[numeric_cols].describe().T[
            ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        ].round(round_digits)
        summary = summary.join(summary_numeric, how='left')

    print(f"Shape: {df.shape}")
    return summary


def drop_constant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Removes columns from a pandas DataFrame that contain only a single unique 
    value or are entirely null/NaN.

    This utility is useful for cleaning data by removing constant features that 
    have no predictive value.

    Args:
        df (pd.DataFrame): 
            The pandas DataFrame to clean.
        verbose (bool): 
            If True, prints the names of the columns that were dropped. 
            Defaults to True.

    Returns:
        pd.DataFrame: 
            A new DataFrame with the constant columns removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    original_columns = set(df.columns)
    cols_to_keep = []

    for col_name in df.columns:
        column = df[col_name]
        
        # We can apply this logic to all columns or only focus on numeric ones.
        # if not is_numeric_dtype(column):
        #     cols_to_keep.append(col_name)
        #     continue
        
        # Keep a column if it has more than one unique value (nunique ignores NaNs by default)
        if column.nunique(dropna=True) > 1:
            cols_to_keep.append(col_name)

    dropped_columns = original_columns - set(cols_to_keep)
    if verbose:
        print(f"🧹 Dropped {len(dropped_columns)} constant columns.")
        if dropped_columns:
            for dropped_column in dropped_columns:
                print(f"    {dropped_column}")

    return df[cols_to_keep]


def drop_rows_with_missing_data(df: pd.DataFrame, targets: Optional[list[str]], threshold: float = 0.7) -> pd.DataFrame:
    """
    Drops rows from the DataFrame using a two-stage strategy:
    
    1. If `targets`, remove any row where all target columns are missing.
    2. Among features, drop those with more than `threshold` fraction of missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        targets (list[str] | None): List of target column names. 
        threshold (float): Maximum allowed fraction of missing values in feature columns.

    Returns:
        pd.DataFrame: A cleaned DataFrame with problematic rows removed.
    """
    df_clean = df.copy()

    # Stage 1: Drop rows with all target columns missing
    if targets is not None:
        target_na = df_clean[targets].isnull().all(axis=1)
        if target_na.any():
            print(f"🧹 Dropping {target_na.sum()} rows with all target columns missing.")
            df_clean = df_clean[~target_na]
        else:
            print("✅ No rows with all targets missing.")
    else:
        targets = []

    # Stage 2: Drop rows based on feature column missing values
    feature_cols = [col for col in df_clean.columns if col not in targets]
    if feature_cols:
        feature_na_frac = df_clean[feature_cols].isnull().mean(axis=1)
        rows_to_drop = feature_na_frac[feature_na_frac > threshold].index
        if len(rows_to_drop) > 0:
            print(f"📉 Dropping {len(rows_to_drop)} rows with more than {threshold*100:.0f}% missing feature data.")
            df_clean = df_clean.drop(index=rows_to_drop)
        else:
            print(f"✅ No rows exceed the {threshold*100:.0f}% missing feature data threshold.")
    else:
        print("⚠️ No feature columns available to evaluate.")

    return df_clean


def split_features_targets(df: pd.DataFrame, targets: list[str]):
    """
    Splits a DataFrame's columns into features and targets.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the dataset.
        targets (list[str]): List of column names to be treated as target variables.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Features dataframe.
            - pd.DataFrame: Targets dataframe.

    Prints:
        - Shape of the original dataframe.
        - Shape of the features dataframe.
        - Shape of the targets dataframe.
    """
    df_targets = df[targets]
    df_features = df.drop(columns=targets)
    print(f"Original shape: {df.shape}\nFeatures shape: {df_features.shape}\nTargets shape: {df_targets.shape}")
    return df_features, df_targets


def show_null_columns(df: pd.DataFrame, round_digits: int = 2):
    """
    Displays a table of columns with missing values, showing both the count and
    percentage of missing entries per column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Number of decimal places for the percentage.

    Returns:
        pd.DataFrame: A DataFrame summarizing missing values in each column.
    """
    null_counts = df.isnull().sum()
    null_percent = df.isnull().mean() * 100

    # Filter only columns with at least one null
    mask = null_counts > 0
    null_summary = pd.DataFrame({
        'Missing Count': null_counts[mask],
        'Missing %': null_percent[mask].round(round_digits)
    })

    # Sort by descending percentage of missing values
    null_summary = null_summary.sort_values(by='Missing %', ascending=False)
    # print(null_summary)
    return null_summary


def drop_columns_with_missing_data(df: pd.DataFrame, threshold: float = 0.7, show_nulls_after: bool = True) -> pd.DataFrame:
    """
    Drops columns with more than `threshold` fraction of missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Fraction of missing values above which columns are dropped.
        show_nulls_after (bool): Prints `show_null_columns` after dropping columns. 

    Returns:
        pd.DataFrame: A new DataFrame without the dropped columns.
    """
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index

    if len(cols_to_drop) > 0:
        print(f"Dropping columns with more than {threshold*100:.0f}% missing data:")
        print(list(cols_to_drop))
        
        result_df = df.drop(columns=cols_to_drop)
        if show_nulls_after:
            print(show_null_columns(df=result_df))
        
        return result_df
    else:
        print(f"No columns have more than {threshold*100:.0f}% missing data.")
        return df


def split_continuous_binary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into two DataFrames: one with continuous columns, one with binary columns.
    Normalize binary values like 0.0/1.0 to 0/1 if detected.

    Parameters:
        df (pd.DataFrame): Input DataFrame with only numeric columns.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): (continuous_columns_df, binary_columns_df)

    Raises:
        TypeError: If any column is not numeric.
    """
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric (int or float).")

    binary_cols = []
    continuous_cols = []

    for col in df.columns:
        series = df[col]
        unique_values = set(series[~series.isna()].unique())

        if unique_values.issubset({0, 1}):
            binary_cols.append(col)
        elif unique_values.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: 0 if x == 0.0 else (1 if x == 1.0 else x))
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    binary_cols.sort()

    df_cont = df[continuous_cols]
    df_bin = df[binary_cols]

    print(f"Continuous columns shape: {df_cont.shape}")
    print(f"Binary columns shape: {df_bin.shape}")

    return df_cont, df_bin # type: ignore


def plot_correlation_heatmap(df: pd.DataFrame, 
                             save_dir: Union[str, Path, None] = None, 
                             plot_title: str="Correlation Heatmap",
                             method: Literal["pearson", "kendall", "spearman"]="pearson"):
    """
    Plots a heatmap of pairwise correlations between numeric features in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | Path | None): If provided, the heatmap will be saved to this directory as a svg file.
        plot_title: To make different plots, or overwrite existing ones.
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
        print("No numeric columns found. Heatmap not generated.")
        return
    
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
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(plot_title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    
    if save_dir:
        save_path = make_fullpath(save_dir, make=True)
        # sanitize the plot title to save the file
        plot_title = sanitize_filename(plot_title)
        plot_title = plot_title + ".svg"
        
        full_path = save_path / plot_title
        
        plt.savefig(full_path, bbox_inches="tight", format='svg')
        print(f"Saved correlation heatmap: '{plot_title}.svg'")
    
    plt.show()
    plt.close()


def check_value_distributions(df: pd.DataFrame, view_frequencies: bool=True, bin_threshold: int=10, skip_cols_with_key: Union[str, None]=None):
    """
    Analyzes value counts for each column in a DataFrame, optionally plots distributions, 
    and saves them as .png files in the specified directory.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        view_frequencies (bool): Print relative frequencies instead of value counts.
        bin_threshold (int): Threshold of unique values to start using bins.
        skip_cols_with_key (str | None): Skip column names containing the key. If None, don't skip any column.
    
    Notes:
        - Binning is adaptive: if quantile binning results in ≤ 2 unique bins, raw values are used instead.
    """
    # cherry-pick columns
    if skip_cols_with_key is not None:
        columns = [col for col in df.columns if skip_cols_with_key not in col]
    else:
        columns = df.columns.to_list()
    
    for col in columns:
        if _is_notebook():
            clear_output(wait=False)
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > bin_threshold:
            bins_number = 10
            binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
            while binned.nunique() <= 2:
                bins_number -= 1
                binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
                if bins_number <= 2:
                    break
            
            if binned.nunique() <= 2:
                view_std = df[col].value_counts(ascending=False)
            else:
                view_std = binned.value_counts(sort=False)
            
        else:
            view_std = df[col].value_counts(ascending=False)

        view_std.name = col

        # unlikely scenario where the series is empty
        if view_std.sum() == 0:
            view_freq = view_std
        else:
            view_freq = view_std / view_std.sum()
        # view_freq = df[col].value_counts(normalize=True, bins=10)  # relative percentages
        view_freq.name = col

        # Print value counts
        print(view_freq if view_frequencies else view_std)
        
        time.sleep(1)
        user_input_ = input("Press enter to continue")


def plot_value_distributions(df: pd.DataFrame, save_dir: Union[str, Path], bin_threshold: int=10, skip_cols_with_key: Union[str, None]=None):
    """
    Plots and saves the value distributions for all (or selected) columns in a DataFrame, 
    with adaptive binning for numerical columns when appropriate.

    For each column both raw counts and relative frequencies are computed and plotted.

    Plots are saved as PNG files under two subdirectories in `save_dir`:
    - "Distribution_Counts" for absolute counts.
    - "Distribution_Frequency" for relative frequencies.

    Args:
        df (pd.DataFrame): The input DataFrame whose columns are to be analyzed.
        save_dir (str | Path): Directory path where the plots will be saved. Will be created if it does not exist.
        bin_threshold (int): Minimum number of unique values required to trigger binning
            for numerical columns.
        skip_cols_with_key (str | None): If provided, any column whose name contains this
            substring will be excluded from analysis.

    Notes:
        - Binning is adaptive: if quantile binning results in ≤ 2 unique bins, raw values are used instead.
        - All non-alphanumeric characters in column names are sanitized for safe file naming.
        - Colormap is automatically adapted based on the number of categories or bins.
    """
    save_path = make_fullpath(save_dir, make=True)
    
    dict_to_plot_std = dict()
    dict_to_plot_freq = dict()
    
    # cherry-pick columns
    if skip_cols_with_key is not None:
        columns = [col for col in df.columns if skip_cols_with_key not in col]
    else:
        columns = df.columns.to_list()
    
    saved_plots = 0
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > bin_threshold:
            bins_number = 10
            binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
            while binned.nunique() <= 2:
                bins_number -= 1
                binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
                if bins_number <= 2:
                    break
            
            if binned.nunique() <= 2:
                view_std = df[col].value_counts(sort=False).sort_index()
            else:
                view_std = binned.value_counts(sort=False)
            
        else:
            view_std = df[col].value_counts(sort=False).sort_index()

        # unlikely scenario where the series is empty
        if view_std.sum() == 0:
            view_freq = view_std
        else:
            view_freq = 100 * view_std / view_std.sum() # Percentage
        # view_freq = df[col].value_counts(normalize=True, bins=10)  # relative percentages
        
        dict_to_plot_std[col] = dict(view_std)
        dict_to_plot_freq[col] = dict(view_freq)
        saved_plots += 1
    
    # plot helper
    def _plot_helper(dict_: dict, target_dir: Path, ylabel: Literal["Frequency", "Counts"], base_fontsize: int=12):
        for col, data in dict_.items():
            safe_col = sanitize_filename(col)
            
            if isinstance(list(data.keys())[0], pd.Interval):
                labels = [str(interval) for interval in data.keys()]
            else:
                labels = data.keys()
                
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab20.colors if len(data) <= 20 else plt.cm.viridis(np.linspace(0, 1, len(data))) # type: ignore
                
            plt.bar(labels, data.values(), color=colors[:len(data)], alpha=0.85)
            plt.xlabel("Values", fontsize=base_fontsize)
            plt.ylabel(ylabel, fontsize=base_fontsize)
            plt.title(f"Value Distribution for '{col}'", fontsize=base_fontsize+2)
            plt.xticks(rotation=45, ha='right', fontsize=base_fontsize-2)
            plt.yticks(fontsize=base_fontsize-2)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.gca().set_facecolor('#f9f9f9')
            plt.tight_layout()
            
            plot_path = target_dir / f"{safe_col}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
    
    # Save plots
    freq_dir = save_path / "Distribution_Frequency"
    std_dir = save_path / "Distribution_Counts"
    freq_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    _plot_helper(dict_=dict_to_plot_std, target_dir=std_dir, ylabel="Counts")
    _plot_helper(dict_=dict_to_plot_freq, target_dir=freq_dir, ylabel="Frequency")

    print(f"Saved {saved_plots} plot(s)")


def clip_outliers_single(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float
) -> Union[pd.DataFrame, None]:
    """
    Clips values in the specified numeric column to the range [min_val, max_val],
    and returns a new DataFrame where the original column is replaced by the clipped version.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to clip.
        min_val (float): Minimum allowable value; values below are clipped to this.
        max_val (float): Maximum allowable value; values above are clipped to this.

    Returns:
        pd.DataFrame: A new DataFrame with the specified column clipped in place.
        
        None: if a problem with the dataframe column occurred.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' must be numeric.")
        return None

    new_df = df.copy(deep=True)
    new_df[column] = new_df[column].clip(lower=min_val, upper=max_val)

    print(f"Column '{column}' clipped to range [{min_val}, {max_val}].")
    return new_df


def clip_outliers_multi(
    df: pd.DataFrame,
    clip_dict: Dict[str, Tuple[Union[int, float], Union[int, float]]],
    verbose: bool=False
) -> pd.DataFrame:
    """
    Clips values in multiple specified numeric columns to given [min, max] ranges,
    updating values (deep copy) and skipping invalid entries.

    Args:
        df (pd.DataFrame): The input DataFrame.
        clip_dict (dict): A dictionary where keys are column names and values are (min_val, max_val) tuples.
        verbose (bool): prints clipped range for each column.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns clipped.

    Notes:
        - Invalid specifications (missing column, non-numeric type, wrong tuple length)
          will be reported but skipped.
    """
    new_df = df.copy()
    skipped_columns = []
    clipped_columns = 0

    for col, bounds in clip_dict.items():
        try:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric.")

            if not (isinstance(bounds, tuple) and len(bounds) == 2):
                raise ValueError(f"Bounds for '{col}' must be a tuple of (min, max).")

            min_val, max_val = bounds
            new_df[col] = new_df[col].clip(lower=min_val, upper=max_val)
            if verbose:
                print(f"Clipped '{col}' to range [{min_val}, {max_val}].")
            clipped_columns += 1

        except Exception as e:
            skipped_columns.append((col, str(e)))
            continue
        
    print(f"Clipped {clipped_columns} columns.")

    if skipped_columns:
        print("\n⚠️ Skipped columns:")
        for col, msg in skipped_columns:
            print(f" - {col}: {msg}")

    return new_df


def match_and_filter_columns_by_regex(
    df: pd.DataFrame,
    pattern: str,
    case_sensitive: bool = False,
    escape_pattern: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a tuple of (filtered DataFrame, matched column names) based on a regex pattern.

    Parameters:
        df (pd.DataFrame): The DataFrame to search.
        pattern (str): The regex pattern to match column names (use a raw string).
        case_sensitive (bool): Whether matching is case-sensitive.
        escape_pattern (bool): If True, the pattern is escaped with `re.escape()` to treat it literally.

    Returns:
        (Tuple[pd.DataFrame, list[str]]): A DataFrame filtered to matched columns, and a list of matching column names.
    """
    if escape_pattern:
        pattern = re.escape(pattern)

    mask = df.columns.str.contains(pattern, case=case_sensitive, regex=True)
    matched_columns = df.columns[mask].to_list()
    filtered_df = df.loc[:, mask]
    
    print(f"{len(matched_columns)} column(s) match the regex pattern '{pattern}'.")

    return filtered_df, matched_columns


def standardize_percentages(
    df: pd.DataFrame,
    columns: list[str],
    treat_one_as_proportion: bool = True,
    round_digits: int = 2
) -> pd.DataFrame:
    """
    Standardizes numeric columns containing mixed-format percentages.

    This function cleans columns where percentages might be entered as whole
    numbers (55) and as proportions (0.55). It assumes values
    between 0 and 1 are proportions and multiplies them by 100.

    Args:
        df (pd.Dataframe): The input pandas DataFrame.
        columns (list[str]): A list of column names to standardize.
        treat_one_as_proportion (bool):
            - If True (default): The value `1` is treated as a proportion and converted to `100%`.
            - If False: The value `1` is treated as `1%`.
        round_digits (int): The number of decimal places to round the final result to.

    Returns:
        (pd.Dataframe):
        A new DataFrame with the specified columns cleaned and standardized.
    """
    df_copy = df.copy()

    if df_copy.empty:
        return df_copy

    # This helper function contains the core cleaning logic
    def _clean_value(x: float) -> float:
        """Applies the standardization rule to a single value."""
        if pd.isna(x):
            return x

        # If treat_one_as_proportion is True, the range for proportions is [0, 1]
        if treat_one_as_proportion and 0 <= x <= 1:
            return x * 100
        # If False, the range for proportions is [0, 1) (1 is excluded)
        elif not treat_one_as_proportion and 0 <= x < 1:
            return x * 100

        # Otherwise, the value is assumed to be a correctly formatted percentage
        return x

    for col in columns:
        # --- Robustness Checks ---
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue

        if not is_numeric_dtype(df_copy[col]):
            print(f"Warning: Column '{col}' is not numeric. Skipping.")
            continue

        # --- Applying the Logic ---
        # Apply the cleaning function to every value in the column
        df_copy[col] = df_copy[col].apply(_clean_value)

        # Round the result
        df_copy[col] = df_copy[col].round(round_digits)

    return df_copy


def _is_notebook():
    return get_ipython() is not None


def info():
    _script_info(__all__)
