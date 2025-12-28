import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import Optional, Union
from pathlib import Path

from ..utilities import save_dataframe_filename

from ..path_manager import make_fullpath
from .._core import get_logger

from ._analysis import show_null_columns


_LOGGER = get_logger("Data Exploration: Cleaning")


__all__ = [
    "drop_constant_columns",
    "drop_rows_with_missing_data",
    "drop_columns_with_missing_data",
    "drop_macro",
    "clean_column_names",
    "clip_outliers_single",
    "clip_outliers_multi",
    "drop_outlier_samples",
    "standardize_percentages",
]


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
        _LOGGER.error("Input must be a pandas DataFrame.")
        raise TypeError()
    
    # make copy to avoid modifying original
    df_clean = df.copy()

    original_columns = set(df.columns)
    cols_to_keep = []

    for col_name in df_clean.columns:
        column = df_clean[col_name]
        
        # Keep a column if it has more than one unique value (nunique ignores NaNs by default)
        if column.nunique(dropna=True) > 1:
            cols_to_keep.append(col_name)

    dropped_columns = original_columns - set(cols_to_keep)
    if verbose:
        if dropped_columns:
            _LOGGER.info(f"🧹 Dropped {len(dropped_columns)} constant columns: {list(dropped_columns)}")
        else:
            _LOGGER.info("No constant columns found.")
            
    # Return a new DataFrame with only the columns to keep
    df_clean = df_clean[cols_to_keep]
    
    if isinstance(df_clean, pd.Series):
        df_clean = df_clean.to_frame()

    return df_clean


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
    valid_targets = []
    if targets:
        # validate targets
        missing_targets = [t for t in targets if t not in df_clean.columns]
        if missing_targets:
            _LOGGER.error(f"Target columns not found in DataFrame: {missing_targets}")
            raise ValueError()
        else:
            valid_targets = targets
        
        # Only proceed if we actually have columns to check
        if valid_targets:
            target_na = df_clean[valid_targets].isnull().all(axis=1)
            if target_na.any():
                _LOGGER.info(f"🧹 Dropping {target_na.sum()} rows with all target columns missing.")
                df_clean = df_clean[~target_na]
            else:
                _LOGGER.info("No rows found where all targets are missing.")
        else:
            _LOGGER.error("Targets list provided but no matching columns found in DataFrame.")
            raise ValueError()

    # Stage 2: Drop rows based on feature column missing values
    feature_cols = [col for col in df_clean.columns if col not in valid_targets]
    if feature_cols:
        feature_na_frac = df_clean[feature_cols].isnull().mean(axis=1)
        rows_to_drop = feature_na_frac[feature_na_frac > threshold].index # type: ignore
        if len(rows_to_drop) > 0:
            _LOGGER.info(f"🧹 Dropping {len(rows_to_drop)} rows with more than {threshold*100:.0f}% missing feature data.")
            df_clean = df_clean.drop(index=rows_to_drop)
        else:
            _LOGGER.info(f"No rows exceed the {threshold*100:.0f}% missing feature data threshold.")
    else:
        _LOGGER.warning("No feature columns available to evaluate.")

    return df_clean


def drop_columns_with_missing_data(df: pd.DataFrame, threshold: float = 0.7, show_nulls_after: bool = True, skip_columns: Optional[list[str]]=None) -> pd.DataFrame:
    """
    Drops columns with more than `threshold` fraction of missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Fraction of missing values above which columns are dropped.
        show_nulls_after (bool): Prints `show_null_columns` after dropping columns. 
        skip_columns (list[str] | None): If given, these columns wont be included in the drop process. 

    Returns:
        pd.DataFrame: A new DataFrame without the dropped columns.
    """
    # If skip_columns is provided, create a list of columns to check.
    # Otherwise, check all columns.
    cols_to_check = df.columns
    if skip_columns:
        # Use set difference for efficient exclusion
        cols_to_check = df.columns.difference(skip_columns)

    # Calculate the missing fraction only on the columns to be checked
    missing_fraction = df[cols_to_check].isnull().mean()
    
    
    cols_to_drop = missing_fraction[missing_fraction > threshold].index # type: ignore

    if len(cols_to_drop) > 0:
        _LOGGER.info(f"🧹 Dropping columns with more than {threshold*100:.0f}% missing data: {list(cols_to_drop)}")
        
        result_df = df.drop(columns=cols_to_drop)
        if show_nulls_after:
            print(show_null_columns(df=result_df))
        
        return result_df
    else:
        _LOGGER.info(f"No columns have more than {threshold*100:.0f}% missing data.")
        return df


def drop_macro(df: pd.DataFrame, 
               log_directory: Union[str,Path], 
               targets: list[str], 
               skip_targets: bool=False, 
               threshold: float=0.7) -> pd.DataFrame:
    """
    Iteratively removes rows and columns with excessive missing data.

    This function performs a comprehensive cleaning cycle on a DataFrame. It
    repeatedly drops columns with constant values, followed by rows and columns that exceed
    a specified threshold of missing values. The process continues until the
    DataFrame's dimensions stabilize, ensuring that the interdependency between
    row and column deletions is handled. 
    
    Initial and final missing data reports are saved to the specified log directory.

    Args:
        df (pd.DataFrame): The input pandas DataFrame to be cleaned.
        log_directory (Union[str, Path]): Path to the directory where the missing data reports
            and plots will be saved inside a "Missing Report" subdirectory.
        targets (list[str]): A list of column names to be treated as target
            variables. This list guides the row-dropping logic.
        skip_targets (bool, optional): If True, the columns listed in `targets`
            will be exempt from being dropped, even if they exceed the missing
            data threshold.
        threshold (float, optional): The proportion of missing data required to drop
            a row or column. For example, 0.7 means a row/column will be
            dropped if 70% or more of its data is missing.

    Returns:
        pd.DataFrame: A new, cleaned DataFrame with offending rows and columns removed.
    """
    # make a deep copy to work with
    df_clean = df.copy()
    
    base_dir_path = make_fullpath(log_directory, make=True, enforce="directory")
    full_path = base_dir_path / "Missing Report"
    
    # Log initial state + Plot
    missing_data_start = show_null_columns(
        df=df_clean, 
        plot_to_dir=full_path, 
        plot_filename="Original",
        use_all_columns=True
    )
    save_dataframe_filename(df=missing_data_start.reset_index(drop=False),
                   save_dir=full_path,
                   filename="Missing_Data_Original",
                   verbose=2)
    
    # Clean cycles for rows and columns
    master = True
    while master:
        # track rows and columns
        initial_rows, initial_columns = df_clean.shape
        
        # drop constant columns
        df_clean = drop_constant_columns(df=df_clean)
        
        # clean rows
        df_clean = drop_rows_with_missing_data(df=df_clean, targets=targets, threshold=threshold)
        
        # clean columns
        if skip_targets:
            df_clean = drop_columns_with_missing_data(df=df_clean, threshold=threshold, show_nulls_after=False, skip_columns=targets)
        else:
            df_clean = drop_columns_with_missing_data(df=df_clean, threshold=threshold, show_nulls_after=False)
        
        # cleaned?
        remaining_rows, remaining_columns = df_clean.shape
        if remaining_rows >= initial_rows and remaining_columns >= initial_columns:
            master = False
    
    # log final state + plot
    missing_data_final = show_null_columns(
        df=df_clean,
        plot_to_dir=full_path,
        plot_filename="Processed",
        use_all_columns=True
    )
    save_dataframe_filename(df=missing_data_final.reset_index(drop=False),
                   save_dir=full_path,
                   filename="Missing_Data_Processed",
                   verbose=2)
    
    # return cleaned dataframe
    return df_clean


def clean_column_names(df: pd.DataFrame, replacement_char: str = '-', replacement_pattern: str = r'[\[\]{}<>,:"]', verbose: bool = True) -> pd.DataFrame:
    """
    Cleans DataFrame column names by replacing special characters.

    This function is useful for ensuring compatibility with libraries like LightGBM,
    which do not support special JSON characters such as `[]{}<>,:"` in feature names.

    Args:
        df (pd.DataFrame): The input DataFrame.
        replacement_char (str): The character to use for replacing characters.
        replacement_pattern (str): Regex pattern to use for the replacement logic.
        verbose (bool): If True, prints the renamed columns.

    Returns:
        pd.DataFrame: A new DataFrame with cleaned column names.
    """
    new_df = df.copy()
    
    original_columns = new_df.columns
    new_columns = original_columns.str.replace(replacement_pattern, replacement_char, regex=True)
    
    # Create a map of changes for logging
    rename_map = {old: new for old, new in zip(original_columns, new_columns) if old != new}
    
    if verbose:
        if rename_map:
            _LOGGER.info(f"Cleaned {len(rename_map)} column name(s) containing special characters:")
            for old, new in rename_map.items():
                print(f"    '{old}' -> '{new}'")
        else:
            _LOGGER.info("No column names required cleaning.")
            
    new_df.columns = new_columns
    return new_df



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
        _LOGGER.warning(f"Column '{column}' not found in DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        _LOGGER.warning(f"Column '{column}' must be numeric.")
        return None

    new_df = df.copy(deep=True)
    new_df[column] = new_df[column].clip(lower=min_val, upper=max_val)

    _LOGGER.info(f"Column '{column}' clipped to range [{min_val}, {max_val}].")
    return new_df


def clip_outliers_multi(
    df: pd.DataFrame,
    clip_dict: Union[dict[str, tuple[int, int]], dict[str, tuple[float, float]]],
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
                _LOGGER.error(f"Column '{col}' not found in DataFrame.")
                raise ValueError()

            if not pd.api.types.is_numeric_dtype(df[col]):
                _LOGGER.error(f"Column '{col}' is not numeric.")
                raise TypeError()

            if not (isinstance(bounds, tuple) and len(bounds) == 2):
                _LOGGER.error(f"Bounds for '{col}' must be a tuple of (min, max).")
                raise ValueError()

            min_val, max_val = bounds
            new_df[col] = new_df[col].clip(lower=min_val, upper=max_val)
            if verbose:
                print(f"Clipped '{col}' to range [{min_val}, {max_val}].")
            clipped_columns += 1

        except Exception as e:
            skipped_columns.append((col, str(e)))
            continue
        
    _LOGGER.info(f"Clipped {clipped_columns} columns.")

    if skipped_columns:
        _LOGGER.warning("Skipped columns:")
        for col, msg in skipped_columns:
            print(f" - {col}")

    return new_df


def drop_outlier_samples(
    df: pd.DataFrame,
    bounds_dict: dict[str, tuple[Union[int, float], Union[int, float]]],
    drop_on_nulls: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drops entire rows where values in specified numeric columns fall outside
    a given [min, max] range.

    This function processes a copy of the DataFrame, ensuring the original is
    not modified. It skips columns with invalid specifications.

    Args:
        df (pd.DataFrame): The input DataFrame.
        bounds_dict (dict): A dictionary where keys are column names and values
                            are (min_val, max_val) tuples defining the valid range.
        drop_on_nulls (bool): If True, rows with NaN/None in a checked column
                           will also be dropped. If False, NaN/None are ignored.
        verbose (bool): If True, prints the number of rows dropped for each column.

    Returns:
        pd.DataFrame: A new DataFrame with the outlier rows removed.

    Notes:
        - Invalid specifications (e.g., missing column, non-numeric type,
          incorrectly formatted bounds) will be reported and skipped.
    """
    new_df = df.copy()
    skipped_columns: list[tuple[str, str]] = []
    initial_rows = len(new_df)

    for col, bounds in bounds_dict.items():
        try:
            # --- Validation Checks ---
            if col not in df.columns:
                _LOGGER.error(f"Column '{col}' not found in DataFrame.")
                raise ValueError()

            if not pd.api.types.is_numeric_dtype(df[col]):
                _LOGGER.error(f"Column '{col}' is not of a numeric data type.")
                raise TypeError()

            if not (isinstance(bounds, tuple) and len(bounds) == 2):
                _LOGGER.error(f"Bounds for '{col}' must be a tuple of (min, max).")
                raise ValueError()

            # --- Filtering Logic ---
            min_val, max_val = bounds
            rows_before_drop = len(new_df)
            
            # Create the base mask for values within the specified range
            # .between() is inclusive and evaluates to False for NaN
            mask_in_bounds = new_df[col].between(min_val, max_val)

            if drop_on_nulls:
                # Keep only rows that are within bounds.
                # Since mask_in_bounds is False for NaN, nulls are dropped.
                final_mask = mask_in_bounds
            else:
                # Keep rows that are within bounds OR are null.
                mask_is_null = new_df[col].isnull()
                final_mask = mask_in_bounds | mask_is_null
            
            # Apply the final mask
            new_df = new_df[final_mask]
            
            rows_after_drop = len(new_df)

            if verbose:
                dropped_count = rows_before_drop - rows_after_drop
                if dropped_count > 0:
                    print(
                        f"  - Column '{col}': Dropped {dropped_count} rows with values outside range [{min_val}, {max_val}]."
                    )

        except (ValueError, TypeError) as e:
            skipped_columns.append((col, str(e)))
            continue

    total_dropped = initial_rows - len(new_df)
    _LOGGER.info(f"Finished processing. Total rows dropped: {total_dropped}.")

    if skipped_columns:
        _LOGGER.warning("Skipped the following columns due to errors:")
        for col, msg in skipped_columns:
            # Only print the column name for cleaner output as the error was already logged
            print(f" - {col}")
            
    # if new_df is a series, convert to dataframe
    if isinstance(new_df, pd.Series):
        new_df = new_df.to_frame()

    return new_df


def standardize_percentages(
    df: pd.DataFrame,
    columns: list[str],
    treat_one_as_proportion: bool = True,
    round_digits: int = 2,
    verbose: bool=True
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
    
    fixed_columns: list[str] = list()

    for col in columns:
        # --- Robustness Checks ---
        if col not in df_copy.columns:
            _LOGGER.warning(f"Column '{col}' not found. Skipping.")
            continue

        if not is_numeric_dtype(df_copy[col]):
            _LOGGER.warning(f"Column '{col}' is not numeric. Skipping.")
            continue

        # --- Applying the Logic ---
        # Apply the cleaning function to every value in the column
        df_copy[col] = df_copy[col].apply(_clean_value)

        # Round the result
        df_copy[col] = df_copy[col].round(round_digits)
        
        fixed_columns.append(col)
        
    if verbose:
        _LOGGER.info(f"Columns standardized:")
        for fixed_col in fixed_columns:
            print(f"  '{fixed_col}'")

    return df_copy

