import pandas as pd
import numpy as np
import math
from typing import Union, Sequence, Optional

from .._core import get_logger


_LOGGER = get_logger("Math Utilities")


__all__ = [
    "normalize_mixed_list",
    "threshold_binary_values",
    "threshold_binary_values_batch",
    "discretize_categorical_values",
]


def normalize_mixed_list(data: list, threshold: int = 2) -> list[float]:
    """
    Normalize a mixed list of numeric values and strings casted to floats so that the sum of the values equals 1.0,
    applying heuristic adjustments to correct for potential data entry scale mismatches.

    Parameters:
        data (list): 
            A list of values that may include strings, floats, integers, or None.
            None values are treated as 0.0.
        
        threshold (int, optional): 
            The number of log10 orders of magnitude below the median scale 
            at which a value is considered suspect and is scaled upward accordingly. 
            Default is 2.

    Returns:
        List[float]: A list of normalized float values summing to 1.0. 
    
    Notes:
        - Zeros and None values remain zero.
        - Input strings are automatically cast to floats if possible.

    Example:
        >>> normalize_mixed_list([1, "0.01", 4, None])
        [0.2, 0.2, 0.6, 0.0]
    """
    # Step 1: Convert all values to float, treat None as 0.0
    float_list = [float(x) if x is not None else 0.0 for x in data]
    
    # Raise for negative values
    if any(x < 0 for x in float_list):
        _LOGGER.error("Negative values are not allowed in the input list.")
        raise ValueError()
    
    # Step 2: Compute log10 of non-zero values
    nonzero = [x for x in float_list if x > 0]
    if not nonzero:
        return [0.0 for _ in float_list]
    
    log_scales = [math.log10(x) for x in nonzero]
    log_median = np.median(log_scales)
    
    # Step 3: Adjust values that are much smaller than median
    adjusted = []
    for x in float_list:
        if x == 0.0:
            adjusted.append(0.0)
        else:
            log_x = math.log10(x)
            if log_median - log_x > threshold:
                scale_diff = round(log_median - log_x)
                adjusted.append(x * (10 ** scale_diff))
            else:
                adjusted.append(x)
    
    # Step 4: Normalize to sum to 1.0
    total = sum(adjusted)
    if total == 0:
        return [0.0 for _ in adjusted]
    
    return [x / total for x in adjusted]


def threshold_binary_values(
    input_array: Union[Sequence[float], np.ndarray, pd.Series],
    binary_values: Optional[int] = None
) -> Union[np.ndarray, pd.Series, list[float], tuple[float]]:
    """
    Thresholds binary features in a 1D input. The number of binary features are counted starting from the end.
    
    Binary elements are converted to 0 or 1 using a 0.5 threshold.

    Parameters:
        input_array: 1D sequence, NumPy array, or pandas Series.
        binary_values (Optional[int]) :
            - If `None`, all values are treated as binary.
            - If `int`, only this many last `binary_values` are thresholded.

    Returns:
        Any:
        Same type as input
    """
    original_type = type(input_array)

    if isinstance(input_array, (pd.Series, np.ndarray)):
        array = np.asarray(input_array)
    elif isinstance(input_array, (list, tuple)):
        array = np.array(input_array)
    else:
        _LOGGER.error("Unsupported input type")
        raise TypeError()

    array = array.flatten()
    total = array.shape[0]

    bin_count = total if binary_values is None else binary_values
    if not (0 <= bin_count <= total):
        _LOGGER.error("'binary_values' must be between 0 and the total number of elements")
        raise ValueError()

    if bin_count == 0:
        result = array
    else:
        cont_part = array[:-bin_count] if bin_count < total else np.array([])
        bin_part = (array[-bin_count:] > 0.5).astype(int)
        result = np.concatenate([cont_part, bin_part])

    if original_type is pd.Series:
        return pd.Series(result, index=input_array.index if hasattr(input_array, 'index') else None) # type: ignore
    elif original_type is list:
        return result.tolist()
    elif original_type is tuple:
        return tuple(result)
    else:
        return result
    
    
def threshold_binary_values_batch(
    input_array: np.ndarray,
    binary_values: int
) -> np.ndarray:
    """
    Threshold the last `binary_values` columns of a 2D NumPy array to binary {0,1} using 0.5 cutoff.

    Parameters
    ----------
    input_array : np.ndarray
        2D array with shape (batch_size, n_features).
    binary_values : int
        Number of binary features located at the END of each row.

    Returns
    -------
    np.ndarray
        Thresholded array, same shape as input.
    """
    if input_array.ndim != 2:
        _LOGGER.error(f"Expected 2D array, got {input_array.ndim}D array.")
        raise AssertionError()
    
    batch_size, total_features = input_array.shape
    
    if not (0 <= binary_values <= total_features):
        _LOGGER.error("'binary_values' out of valid range.")
        raise AssertionError()

    if binary_values == 0:
        return input_array.copy()

    cont_part = input_array[:, :-binary_values] if binary_values < total_features else np.empty((batch_size, 0))
    bin_part = input_array[:, -binary_values:] > 0.5
    bin_part = bin_part.astype(np.int32)

    return np.hstack([cont_part, bin_part])


def discretize_categorical_values(
    input_array: np.ndarray,
    categorical_info: dict[int, int],
    start_at_zero: bool = True
) -> np.ndarray:
    """
    Rounds specified columns of a 2D NumPy array to the nearest integer and
    clamps the result to a valid categorical range.
    
    If a 1D array is provided, it is treated as a single batch.

    Parameters
    ----------
    input_array : np.ndarray
        1D array (n_features,) or 2D array with shape (batch_size, n_features) containing continuous values.
    categorical_info : dict[int, int]
        A dictionary mapping column indices to their cardinality (number of categories).
        Example: {3: 4} means column 3 will be clamped to its 4 valid categories.
    start_at_zero : bool
        If True, categories range from 0 to k-1.
        If False, categories range from 1 to k.

    Returns
    -------
    np.ndarray
        A new array with the specified columns converted to integer-like values.
        Shape matches the input array's original shape.
    """
    # --- Input Validation ---
    if not isinstance(input_array, np.ndarray):
         _LOGGER.error(f"Expected np.ndarray, got {type(input_array)}.")
         raise ValueError()
    
    if input_array.ndim == 1:
        # Reshape 1D array (n_features,) to 2D (1, n_features)
        working_array = input_array.reshape(1, -1)
        original_was_1d = True
    elif input_array.ndim == 2:
        working_array = input_array
        original_was_1d = False
    else:
        _LOGGER.error(f"Expected 1D or 2D array, got {input_array.ndim}D array.")
        raise ValueError()

    if not isinstance(categorical_info, dict) or not categorical_info:
        _LOGGER.error(f"'categorical_info' is not a dictionary, or is empty.")
        raise ValueError()

    _, total_features = working_array.shape
    for col_idx, cardinality in categorical_info.items():
        if not isinstance(col_idx, int):
             _LOGGER.error(f"Column index key {col_idx} is not an integer.")
             raise TypeError()
        if not (0 <= col_idx < total_features):
            _LOGGER.error(f"Column index {col_idx} is out of bounds for an array with {total_features} features.")
            raise ValueError()
        if not isinstance(cardinality, int) or cardinality < 2:
            _LOGGER.error(f"Cardinality for column {col_idx} must be an integer >= 2, but got {cardinality}.")
            raise ValueError()

    # --- Core Logic ---
    output_array = working_array.copy()

    for col_idx, cardinality in categorical_info.items():
        # 1. Round the column values using "round half up"
        rounded_col = np.floor(output_array[:, col_idx] + 0.5)

        # 2. Determine clamping bounds
        min_bound = 0 if start_at_zero else 1
        max_bound = cardinality - 1 if start_at_zero else cardinality

        # 3. Clamp the values and update the output array
        output_array[:, col_idx] = np.clip(rounded_col, min_bound, max_bound)
    
    # NOTE: do NOT cast to int32, return the array as is (floats).
    # Categorical columns are now "integer-like floats" (1.0, 2.0, etc), preserving the precision of the continuous columns in the same array.
    # final_output = output_array.astype(np.int32)
    final_output = output_array
    
    # --- Output Shape Handling ---
    if original_was_1d:
        # Squeeze the batch dimension to return a 1D array
        return final_output.squeeze(axis=0)
    else:
        return final_output

