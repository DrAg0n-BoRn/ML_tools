import polars as pl
from pathlib import Path
from typing import Union, Optional, Literal

from ..utilities import load_dataframe

from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("ETL Binarize")


__all__ = [
    "binarize_single_column",
    "binarize_columns",
    "binarize_merge_columns"
]


def _load_helper(df_or_path: Union[pl.DataFrame, str, Path], verbose: int) -> pl.DataFrame:
    if isinstance(df_or_path, (str, Path)):
        _df_path = make_fullpath(df_or_path, enforce="file")
        df, _ = load_dataframe(df_path=_df_path,
                                kind="polars",
                                all_strings=False,
                                empty_as_nan=True,
                                verbose=False)
        if verbose >= 2:
            _LOGGER.info(f"📂 Loaded DataFrame from: {_df_path} with shape: {df.shape}")
    
    elif isinstance(df_or_path, pl.DataFrame):
        df = df_or_path
        if verbose >= 2:
            _LOGGER.info(f"📟 Using provided DataFrame with shape: {df.shape}")
    
    else:
        _LOGGER.error("df_or_path must be a polars DataFrame or a valid file path.")
        raise TypeError()
    
    return df

def _validate_columns(df: pl.DataFrame, columns: list[str], verbose: int) -> list[str]:
    valid_columns: list[str] = list()

    for col in columns:
        if col not in df.columns:
            if verbose >= 1:
                _LOGGER.warning(f"Column '{col}' does not exist in the DataFrame. Skipping...")
        elif not df[col].dtype.is_numeric():
            if verbose >= 1:
                _LOGGER.warning(f"Column '{col}' is not numeric and cannot be binarized. Skipping...")
        else:
            valid_columns.append(col)
    
    if not valid_columns:
        _LOGGER.error("No valid numeric columns found for binarization.")
        raise ValueError()
     
    return valid_columns

def _parse_none_as(none_as: Literal['None', 'False', 'True']) -> Optional[int]:
    if none_as == 'None':
        return None
    elif none_as == 'False':
        return 0
    elif none_as == 'True':
        return 1
    else:
        _LOGGER.error(f"Invalid value for 'none_as': {none_as}. Must be one of: 'None', 'False', 'True'.")
        raise ValueError()

def _standardize_threshold(threshold: Optional[float]) -> float:
    if threshold is None:
        return 0.0
    elif isinstance(threshold, (int, float)):
        return round(float(threshold), 6)
    else:
        _LOGGER.error(f"Invalid threshold value: {threshold}. Must be a numeric value or None.")
        raise TypeError()


def binarize_single_column(df_or_path: Union[pl.DataFrame, str, Path],
                           column: str, 
                           none_as: Literal['None', 'False', 'True'] = 'False', 
                           threshold: Optional[float] = None,
                           verbose: int = 2) -> pl.DataFrame:
    """
    Binarizes a single numeric column in a DataFrame based on a given threshold.

    Args:
        df_or_path: A Polars DataFrame or a valid file path pointing to the dataset.
        column: The exact string name of the column to be binarized.
        none_as: Literal string determining how missing values are processed. 
                 'None' keeps them as Null, 'False' converts them to 0, and 'True' converts them to 1.
        threshold: The numeric value serving as the cutoff. Values strictly greater than 
                   the threshold become 1; values less than or equal become 0.
        verbose: Integer controlling the logging output level.

    Returns:
        A Polars DataFrame containing the newly binarized column.
    """
    # Load df
    df = _load_helper(df_or_path, verbose=verbose)
    
    # validate column
    if column not in df.columns:
        _LOGGER.error(f"Column '{column}' does not exist in the DataFrame.")
        raise ValueError()
    elif not df[column].dtype.is_numeric():
        _LOGGER.error(f"Column '{column}' is not numeric and cannot be binarized.")
        raise TypeError()
    
    # none
    none_value = _parse_none_as(none_as)
    
    # threshold
    threshold_value = _standardize_threshold(threshold)
    
    # binarize column
    df_binarized = df.with_columns(
        pl.when(df[column].is_null())
          .then(none_value)
          .otherwise(
              pl.when(df[column].round(decimals=6) > threshold_value)
              .then(1)
              .otherwise(0)) 
    )
    
    if verbose >= 2:
        _LOGGER.info(f"🔟 Binarized column '{column}' with threshold {threshold_value} and None as '{none_value}'.")
    
    return df_binarized


def binarize_columns(df_or_path: Union[pl.DataFrame, str, Path],
                     columns_thresholds: dict[str, Optional[float]],
                     none_as: Literal['None', 'False', 'True'] = 'False',
                     verbose: int = 2) -> pl.DataFrame:
    """
    Binarizes multiple numeric columns in a DataFrame based on individual thresholds.

    Args:
        df_or_path: A Polars DataFrame or a valid file path pointing to the dataset.
        columns_thresholds: A dictionary mapping column names (strings) to their specific 
                            numeric thresholds (floats or None).
        none_as: Literal string determining how missing values are processed. 
                 'None' keeps them as Null, 'False' converts them to 0, and 'True' converts them to 1.
        verbose: Integer controlling the logging output level.

    Returns:
        A Polars DataFrame containing the newly binarized columns.
    """
    # Load df
    df = _load_helper(df_or_path, verbose=verbose)
    
    # check: exist in df and is numeric
    valid_columns = _validate_columns(df, list(columns_thresholds.keys()), verbose)
    
    # match valid columns with standardized thresholds
    valid_columns_thresholds = {col: _standardize_threshold(columns_thresholds[col]) for col in valid_columns}
     
    # none value
    none_value = _parse_none_as(none_as)
     
    # binarize valid columns in a single pass
    exprs = [
        pl.when(pl.col(col).is_null())
        .then(none_value)
        .otherwise(
            pl.when(pl.col(col).round(decimals=6) > thresh)
            .then(1)
            .otherwise(0)
        ).alias(col)
        for col, thresh in valid_columns_thresholds.items()
    ]
    
    df_binarized = df.with_columns(exprs)
    
    if verbose >= 3:
        # one column per line
        msg = "\n\t".join([f"{col}: threshold={valid_columns_thresholds[col]}" for col in valid_columns])
        _LOGGER.info(f"🔟 Binarized columns with None as '{none_value}':\n\t{msg}")
    elif verbose >= 2:
        _LOGGER.info(f"🔟 Binarized {len(valid_columns)} columns with None as '{none_value}'.")
    
    return df_binarized


def binarize_merge_columns(df_or_path: Union[pl.DataFrame, str, Path],
                           columns_thresholds: dict[str, Optional[float]],
                           output_col_name: str,
                           none_as: Literal['None', 'False', 'True'] = 'False',
                           drop_originals: bool = True,
                           verbose: int = 2) -> pl.DataFrame:
    """
    Binarizes multiple numeric columns and merges them into a single column.
    
    The merging logic performs a horizontal maximum across the binarized columns, 
    effectively acting as a logical OR operation (if any column evaluates to 1, 
    the merged result is 1).

    Args:
        df_or_path: A Polars DataFrame or a valid file path pointing to the dataset.
        columns_thresholds: A dictionary mapping column names (strings) to their specific 
                            numeric thresholds (floats or None).
        output_col_name: The string name of the new column that will contain the merged results.
        none_as: Literal string determining how missing values are processed. 
                 'None' keeps them as Null, 'False' converts them to 0, and 'True' converts them to 1.
        drop_originals: Boolean indicating whether to drop the original targeted columns after merging.
        verbose: Integer controlling the logging output level.

    Returns:
        A Polars DataFrame containing the newly merged binary column.
    """
    df_binarized = binarize_columns(df_or_path=df_or_path, 
                                    columns_thresholds=columns_thresholds, 
                                    none_as=none_as, 
                                    verbose=verbose)
    
    # Identify the valid columns that were actually processed
    valid_columns = [col for col in columns_thresholds.keys() if col in df_binarized.columns]
    
    # Merge binarized columns using max_horizontal (acts as a logical OR for 0/1 flags)
    df_merged = df_binarized.with_columns(
        pl.max_horizontal(valid_columns).alias(output_col_name)
    )
    
    if drop_originals:
        df_merged = df_merged.drop(valid_columns)
        
    if verbose >= 2:
        _LOGGER.info(f"⛓️ Merged {len(valid_columns)} binarized columns into '{output_col_name}'.")
        
    return df_merged
