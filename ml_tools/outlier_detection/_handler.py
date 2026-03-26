import pandas as pd
from typing import Union, Optional, Literal

from .._core import get_logger


_LOGGER = get_logger("Outlier Detection")


__all__ = [
    "clip_outliers_single",
    "clip_outliers_multi",
    "drop_outliers_rule",
    "drop_outliers_mask",
    "replace_outliers_mask"
]



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


def drop_outliers_rule(
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


def drop_outliers_mask(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """
    Removes rows identified as outliers.

    Args:
        df (pd.DataFrame): The original dataset.
        outlier_mask (pd.Series): A boolean mask where True indicates an outlier row. Must be aligned with df's index.

    Returns:
        pd.DataFrame: A new DataFrame with the outliers removed.
    """
    initial_shape = df.shape[0]
    df_clean = df[~outlier_mask].copy()
    dropped_count = initial_shape - df_clean.shape[0]
    
    _LOGGER.info(f"Dropped {dropped_count} outlier samples. Remaining rows: {df_clean.shape[0]}.")
    return df_clean


def replace_outliers_mask(
    df_features: pd.DataFrame, 
    outlier_mask: pd.Series,
    columns: Optional[list[str]] = None,
    strategy: Literal["nan", "mean", "median"] = "nan"
) -> pd.DataFrame:
    """
    Replaces values in outlier rows using a specified strategy (NaN, mean, or median).
    
    ⚠️ If numeric target columns are included, their values will also be replaced according to the strategy.

    Args:
        df_features (pd.DataFrame): The input DataFrame containing the features to be modified.
        outlier_mask (pd.Series): A boolean mask where True indicates an outlier row. Must be aligned with df_features's index.
        columns (list[str] | None): Specific columns to apply the replacement. 
                                    If None, applies to all numeric columns.
        strategy (Literal["nan", "mean", "median"]): The replacement strategy.
            - "nan": Replaces outliers with NaN.
            - "mean": Replaces outliers with the mean of the inliers (non-outliers).
            - "median": Replaces outliers with the median of the inliers.

    Returns:
        pd.DataFrame: A new DataFrame with outliers replaced according to the strategy.
    """
    df_replaced = df_features.copy()
    
    # Default to all numeric columns if none are specified
    if columns is None:
        columns = df_replaced.select_dtypes(include='number').columns.tolist()
        
    if strategy == "nan":
        df_replaced.loc[outlier_mask, columns] = float('nan')
        _LOGGER.info(f"Replaced outlier values with NaN in {len(columns)} columns for {outlier_mask.sum()} rows.")
        
    elif strategy in ["mean", "median"]:
        # CRITICAL: Calculate the replacement statistics based ONLY on the clean data (~outlier_mask)
        inliers_data = df_features.loc[~outlier_mask, columns]
        
        if strategy == "mean":
            replacement_values = inliers_data.mean()
        else:
            replacement_values = inliers_data.median()
            
        # Replace the outliers column by column with the calculated safe statistics
        for col in columns:
            df_replaced.loc[outlier_mask, col] = replacement_values[col]
            
        _LOGGER.info(f"Replaced outlier values with {strategy} in {len(columns)} columns for {outlier_mask.sum()} rows.")
        
    else:
        _LOGGER.error(f"Invalid replacement strategy provided: {strategy}")
        raise ValueError()
        
    return df_replaced
