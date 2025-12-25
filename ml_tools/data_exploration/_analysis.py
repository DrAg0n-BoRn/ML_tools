import pandas as pd
from typing import Optional, Union
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Data Exploration: Analysis")


__all__ = [
    "summarize_dataframe",
    "show_null_columns",
    "match_and_filter_columns_by_regex",
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
        'Completeness %': (df.notnull().mean() * 100).round(2),
        'Unique Values': df.nunique(),
        # 'Missing %': (df.isnull().mean() * 100).round(2)
    })

    # For numeric columns, add summary statistics
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        stats = df[numeric_cols].describe(percentiles=[.10, .25, .50, .70, .80, .90])
        
        summary_numeric = stats.T[
            ['mean', 'std', 'min', '10%', '25%', '50%', '70%', '80%', '90%', 'max']
        ].round(round_digits)
        summary = summary.join(summary_numeric, how='left')

    print(f"DataFrame Shape: {df.shape}")
    return summary


def show_null_columns(
    df: pd.DataFrame, 
    round_digits: int = 2,
    plot_to_dir: Optional[Union[str, Path]] = None,
    plot_filename: Optional[str] = None,
    use_all_columns: bool = False
) -> pd.DataFrame:
    """
    Returns a table of columns with missing values, showing both the count and
    percentage of missing entries per column.
    
    Optionally generates a visualization of the missing data profile.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Number of decimal places for the percentage.
        plot_to_dir (str | Path | None): If provided, saves a visualization of the 
            missing data to this directory.
        plot_filename (str): The filename for the saved plot (without extension). 
            Used only if `plot_to_dir` is set.
        use_all_columns (bool): If True, includes all columns in the summary and plot,
            even those with no missing values.

    Returns:
        pd.DataFrame: A DataFrame summarizing missing values in each column.
    """
    null_counts = df.isnull().sum()
    null_percent = df.isnull().mean() * 100

    if use_all_columns:
        null_summary = pd.DataFrame({
            'Missing Count': null_counts,
            'Missing %': null_percent.round(round_digits)
        })
    else:
        # Filter only columns with at least one null
        mask = null_counts > 0
        null_summary = pd.DataFrame({
            'Missing Count': null_counts[mask],
            'Missing %': null_percent[mask].round(round_digits)
        })

    # Sort by descending percentage of missing values
    null_summary = null_summary.sort_values(by='Missing %', ascending=False)
    
    # --- Visualization Logic ---
    if plot_to_dir:
        if null_summary.empty:
            _LOGGER.info("No missing data found. Skipping plot generation.")
        else:
            try:
                # Validate and create save directory
                save_path = make_fullpath(plot_to_dir, make=True, enforce="directory")
                
                # Prepare data
                features = null_summary.index.tolist()
                missing_pct = np.array(null_summary['Missing %'].values)
                present_pct = 100 - missing_pct
                n_features = len(features)
                
                # Dynamic width
                width = max(10, n_features * 0.4)
                plt.figure(figsize=(width, 8))

                # Stacked Bar Chart Logic
                
                # Grid behind bars
                plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

                # 1. Present Data: Solid Green
                plt.bar(
                    features, 
                    present_pct, 
                    color='tab:green', 
                    label='Present', 
                    width=0.6, 
                    zorder=3
                )

                # 2. Missing Data: Transparent Red Fill + Solid Red Hatch
                # define facecolor (fill) with alpha, but edgecolor (lines) without alpha.
                plt.bar(
                    features, 
                    missing_pct, 
                    bottom=present_pct, 
                    facecolor=(1.0, 1.0, 1.0, 0.2), # RGBA
                    edgecolor='tab:red',             # Solid red for the hatch lines
                    hatch='///',                     # hatch pattern
                    linewidth=0.4,                   # Ensure lines are thick enough to see
                    label='Missing', 
                    width=0.6, 
                    zorder=3
                )

                # Styling
                plt.ylim(0, 100)
                plt.ylabel("Data Completeness (%)", fontsize=13)
                plt.yticks(np.arange(0, 101, 10))
                plot_title = f"Missing Data - {plot_filename.replace('_', ' ')}" if plot_filename else "Missing Data"
                plt.title(plot_title)
                plt.xticks(rotation=45, ha='right', fontsize=9)
                
                # Reference line
                plt.axhline(y=100, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
                
                plt.legend(loc='lower right', framealpha=0.95)
                plt.tight_layout()

                # Save
                if plot_filename is None or plot_filename.strip() == "":
                    plot_filename = "Missing_Data_Profile"
                else:
                    plot_filename =  "Missing_Data_" + sanitize_filename(plot_filename)
    
                full_filename = plot_filename + ".svg"
                plt.savefig(save_path / full_filename, format='svg', bbox_inches="tight")
                plt.close()
                
                _LOGGER.info(f"Saved missing data plot as '{full_filename}'")
                
            except Exception as e:
                _LOGGER.error(f"Failed to generate missing data plot. Error: {e}")
                plt.close()

    return null_summary


def match_and_filter_columns_by_regex(
    df: pd.DataFrame,
    pattern: str,
    case_sensitive: bool = False,
    escape_pattern: bool = False
) -> tuple[pd.DataFrame, list[str]]:
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
    
    _LOGGER.info(f"{len(matched_columns)} columns match the regex pattern '{pattern}'.")
    
    # if filtered df is a series, convert to dataframe
    if isinstance(filtered_df, pd.Series):
        filtered_df = filtered_df.to_frame()

    return filtered_df, matched_columns

