from collections.abc import Mapping
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from typing import Any, Optional, Union
import re


from .._core import get_logger


_LOGGER = get_logger("Data Exploration: Filter")


__all__ = [
    "filter_subset_categorical",
    "filter_subset_continuous",
    "filter_subset_regex",
]


def filter_subset_categorical(
    df: pd.DataFrame,
    filters: Union[dict[str, Any], dict[str, list[Any]]],
    drop_filter_cols: bool = False,
    reset_index: bool = True,
    verbose: int = 3
) -> pd.DataFrame:
    """
    Filters the DataFrame based on a dictionary of column-value conditions.

    Supports:
    - Single value matching (e.g., {"Color": "Blue"})
    - Multiple value matching (e.g., {"Color": ["Blue", "Red"]}) -> OR logic within column.
    - Multiple column filtering (e.g., {"Color": "Blue", "Size": "Large"}) -> AND logic between columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        filters (dict[str, Any] | dict[str, list[Any]]): Dictionary where keys are column names and values are the target values (scalar or list).
        drop_filter_cols (bool): If True, drops the columns used for filtering from the result.
        reset_index (bool): If True, resets the index of the resulting DataFrame.
        verbose (int): Verbosity level.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df_filtered = df.copy()

    # Validate columns exist
    missing_cols = [col for col in filters.keys() if col not in df.columns]
    if missing_cols:
        _LOGGER.error(f"Filter columns not found: {missing_cols}")
        raise ValueError()

    if verbose >= 2:
        _LOGGER.info(f"Original shape: {df.shape}")

    for col, value in filters.items():
        # Handle list of values (OR logic within column)
        if isinstance(value, list):
            df_filtered = df_filtered[df_filtered[col].isin(value)]
        # Handle single value
        else:
            # Warn if the value is a floating point due to potential precision issues
            if isinstance(value, float) and verbose >= 1:
                _LOGGER.warning(f"Filtering on column '{col}' with float value '{value}'.")
            df_filtered = df_filtered[df_filtered[col] == value]

    if drop_filter_cols:
        if verbose >= 3:
            _LOGGER.info(f"Dropping filter columns: {list(filters.keys())}")
        df_filtered.drop(columns=list(filters.keys()), inplace=True)

    if reset_index:
        if verbose >= 3:
            _LOGGER.info("Resetting index of the filtered DataFrame.")
        df_filtered.reset_index(drop=True, inplace=True)

    if verbose >= 2:
        _LOGGER.info(f"Filtered shape: {df_filtered.shape}")

    return df_filtered


def filter_subset_continuous(
    df: pd.DataFrame,
    range_filters: dict[str, tuple[Optional[float], Optional[float]]],
    drop_filter_cols: bool = False,
    reset_index: bool = True,
    verbose: int = 3
) -> pd.DataFrame:
    """
    Filters the DataFrame based on numerical ranges for continuous columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        range_filters (dict[str, tuple[Optional[float], Optional[float]]]): 
            - Dictionary where keys are column names and values are tuples of (min_value, max_value). 
            - Use None for a one-sided bound.
        drop_filter_cols (bool): If True, drops the columns used for filtering.
        reset_index (bool): If True, resets the index of the resulting DataFrame.
        verbose (int): Verbosity level.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df_filtered = df.copy()

    # Validate columns exist
    missing_cols = [col for col in range_filters.keys() if col not in df.columns]
    if missing_cols:
        _LOGGER.error(f"Columns not found: {missing_cols}")
        raise ValueError()

    if verbose >= 2:
        _LOGGER.info(f"Original shape: {df.shape}")

    for col, (min_val, max_val) in range_filters.items():
        if not is_numeric_dtype(df_filtered[col]) and verbose >= 1:
            _LOGGER.warning(f"Column '{col}' is not numeric. Filtering may behave unexpectedly.")

        if min_val is not None:
            df_filtered = df_filtered[df_filtered[col] >= min_val]
        
        if max_val is not None:
            df_filtered = df_filtered[df_filtered[col] <= max_val]

    if drop_filter_cols:
        if verbose >= 3:
            _LOGGER.info(f"Dropping filter columns: {list(range_filters.keys())}")
        df_filtered.drop(columns=list(range_filters.keys()), inplace=True)

    if reset_index:
        if verbose >= 3:
            _LOGGER.info("Resetting index of the filtered DataFrame.")
        df_filtered.reset_index(drop=True, inplace=True)

    if verbose >= 2:
        _LOGGER.info(f"Filtered shape: {df_filtered.shape}")

    return df_filtered


_Regex = Union[str, re.Pattern[str]]
_RegexFilter = Union[_Regex, list[_Regex], tuple[_Regex, ...]]


def _compile_regex_patterns(
    col: str, condition: _RegexFilter, case_insensitive: bool
) -> list[re.Pattern[str]]:
    patterns = condition if isinstance(condition, (list, tuple)) else [condition]
    compiled = []
    for pattern in patterns:
        source = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
        if not isinstance(source, str):
            _LOGGER.error(f"Expected a text regex for column {col!r}, got {pattern!r}.")
            raise TypeError()
        try:
            if case_insensitive:
                flags = pattern.flags if isinstance(pattern, re.Pattern) else 0
                compiled.append(re.compile(source, flags | re.IGNORECASE))
            else:
                compiled.append(re.compile(pattern))
        except (re.error, OverflowError) as exc:
            _LOGGER.error(f"Invalid regex {pattern!r} for column {col!r}: {exc}")
            raise ValueError() from exc
    return compiled


def filter_subset_regex(
    df: pd.DataFrame,
    filters: Mapping[str, _RegexFilter],
    drop_filter_cols: bool = False,
    reset_index: bool = True,
    case_insensitive: bool = False,
    verbose: int = 3,
) -> pd.DataFrame:
    """
    Filters rows based on regex patterns for specified columns in a DataFrame.
    
    Using '(?i)' in a regex pattern is equivalent to setting case_insensitive=True.
    
    Args:
        df (pd.DataFrame): Input DataFrame to filter.
        filters (Mapping[str, RegexFilter]): A mapping of column names to regex patterns or
            - Keys are column names.
            - Values are regex patterns (str or re.Pattern) or lists/tuples of such patterns.
        drop_filter_cols (bool): If True, drops the columns used for filtering from the result.
        reset_index (bool): If True, resets the index of the resulting DataFrame.
        case_insensitive (bool): If True, applies case-insensitive matching to all patterns.
        verbose (int): Verbosity level for logging.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the rows that match the specified regex patterns
        
    Example:
        `filter_subset_regex(df, {"order_id": r"^42\\d{3}$", "region": ["East", "West"]})` 
    """
    if not isinstance(filters, Mapping):
        _LOGGER.error("Filters must be a mapping of column names to regex patterns.")
        raise TypeError()
    missing_cols = [col for col in filters if col not in df.columns]
    if missing_cols:
        _LOGGER.error(f"Filter columns not found: {missing_cols}")
        raise ValueError()

    if verbose >= 2:
        _LOGGER.info(f"Original shape: {df.shape}")

    keep = np.ones(len(df), dtype=bool)
    for col, condition in filters.items():
        column = df[col]
        if not isinstance(column, pd.Series):
            _LOGGER.error(f"Column {col!r} is not a valid Series in the DataFrame.")
            raise ValueError()
        patterns = _compile_regex_patterns(col, condition, case_insensitive)

        # Positional masks also work when several rows share an index label.
        eligible = keep & column.notna().to_numpy()
        texts = map(str, column.loc[eligible])
        keep = np.zeros(len(df), dtype=bool)
        keep[eligible] = [any(pattern.search(text) for pattern in patterns) for text in texts]

    result = df.loc[keep].copy()
    if drop_filter_cols:
        if verbose >= 3:
            _LOGGER.info(f"Dropping filter columns: {list(filters)}")
        result.drop(columns=list(filters), inplace=True)
    if reset_index:
        if verbose >= 3:
            _LOGGER.info("Resetting index of the filtered DataFrame.")
        result.reset_index(drop=True, inplace=True)
    if verbose >= 2:
        _LOGGER.info(f"Filtered shape: {result.shape}")
    return result
