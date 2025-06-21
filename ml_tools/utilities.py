import math
import numpy as np
import pandas as pd
import polars as pl
import os
from pathlib import Path
import re
from typing import Literal, Union, Sequence, Optional, Any, Iterator, Tuple
import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError


# Keep track of available tools
__all__ = [
    "list_csv_paths",
    "load_dataframe",
    "yield_dataframes_from_dir",
    "merge_dataframes",
    "save_dataframe",
    "normalize_mixed_list",
    "sanitize_filename",
    "threshold_binary_values",
    "serialize_object",
    "deserialize_object",
    "distribute_datasets_by_target"
]


def list_csv_paths(directory: str) -> dict[str, str]:
    """
    Lists all `.csv` files in the specified directory and returns a mapping: filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str): Path to the directory containing `.csv` files.

    Returns:
        (dict[str, str]): Mapping {name, path}.
    """
    dir_path = Path(directory).expanduser().resolve()

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    csv_paths = list(dir_path.glob("*.csv"))
    if not csv_paths:
        raise IOError(f"No CSV files found in directory: {dir_path}")
    
    # make a dictionary of paths and names
    name_path_dict = {p.stem: str(p) for p in csv_paths}
    
    print("🗂️ CSV files found:")
    for name in name_path_dict.keys():
        print(f"\t{name}")

    return name_path_dict


def load_dataframe(df_path: str) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV file into a pandas DataFrame and extract the base name (without extension) from the file path.

    Args:
        df_path (str): The path to the CSV file.

    Returns:
        Tuple ([pd.DataFrame, str]):
        A tuple containing the loaded pandas DataFrame and the base name of the file.
    """
    path = Path(df_path).expanduser().resolve()
    df = pd.read_csv(path, encoding='utf-8')
    df_name = path.stem
    if df.empty:
        raise ValueError(f"DataFrame '{df_name}' is empty.")
    print(f"\n💿 Loaded dataset: '{df_name}' with shape: {df.shape}")
    return df, df_name


def yield_dataframes_from_dir(datasets_dir: str):
    """
    Iterates over all CSV files in a given directory, loading each into a pandas DataFrame.

    Parameters:
        datasets_dir (str):
        The path to the directory containing `.csv` dataset files.

    Yields:
        Tuple: ([pd.DataFrame, str])
            - The loaded pandas DataFrame.
            - The base name of the file (without extension).

    Notes:
    - Files are expected to have a `.csv` extension.
    - CSV files are read using UTF-8 encoding.
    - Output is streamed via a generator to support lazy loading of multiple datasets.
    """
    for df_name, df_path in list_csv_paths(datasets_dir).items():
        df, _ = load_dataframe(df_path)
        yield df, df_name


def merge_dataframes(
    *dfs: pd.DataFrame,
    reset_index: bool = False,
    direction: Literal["horizontal", "vertical"] = "horizontal",
    verbose: bool=True
) -> pd.DataFrame:
    """
    Merges multiple DataFrames either horizontally or vertically.

    Parameters:
        *dfs (pd.DataFrame): Variable number of DataFrames to merge.
        reset_index (bool): Whether to reset index in the final merged DataFrame.
        direction (["horizontal" | "vertical"]):
            - "horizontal": Merge on index, adding columns.
            - "vertical": Append rows; all DataFrames must have identical columns.

    Returns:
        pd.DataFrame: A single merged DataFrame.

    Raises:
        ValueError:
            - If fewer than 2 DataFrames are provided.
            - If indexes do not match for horizontal merge.
            - If column names or order differ for vertical merge.
    """
    if len(dfs) < 2:
        raise ValueError("At least 2 DataFrames must be provided.")
    
    if verbose:
        for i, df in enumerate(dfs, start=1):
            print(f"DataFrame {i} shape: {df.shape}")
    

    if direction == "horizontal":
        reference_index = dfs[0].index
        for i, df in enumerate(dfs, start=1):
            if not df.index.equals(reference_index):
                raise ValueError(f"Indexes do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=1)

    elif direction == "vertical":
        reference_columns = dfs[0].columns
        for i, df in enumerate(dfs, start=1):
            if not df.columns.equals(reference_columns):
                raise ValueError(f"Column names/order do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=0)

    else:
        raise ValueError(f"Invalid merge direction: {direction}")

    if reset_index:
        merged_df = merged_df.reset_index(drop=True)
    
    if verbose:
        print(f"Merged DataFrame shape: {merged_df.shape}")

    return merged_df


def save_dataframe(df: pd.DataFrame, save_dir: str, filename: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
        df: pandas.DataFrame to save
        save_dir: str, directory where the CSV file will be saved.
        filename: str, CSV filename, extension will be added if missing.
    """
    if df.empty:
        print(f"⚠️ Attempting to save an empty DataFrame: '{filename}'. Process Skipped.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = sanitize_filename(filename)
    
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    output_path = os.path.join(save_dir, filename)
        
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Saved dataset: '{filename}' with shape: {df.shape}")


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
        raise ValueError("Negative values are not allowed in the input list.")
    
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


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes the name by:
    - Stripping leading/trailing whitespace.
    - Replacing all internal whitespace characters with underscores.
    - Removing or replacing characters invalid in filenames.

    Args:
        name (str): Base filename.

    Returns:
        str: A sanitized string suitable to use as a filename.
    """
    # Strip leading/trailing whitespace
    sanitized = filename.strip()
    
    # Replace all whitespace sequences (space, tab, etc.) with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)

    # Conservative filter to keep filenames safe across platforms
    sanitized = re.sub(r'[^\w\-.]', '', sanitized)

    return sanitized


def threshold_binary_values(
    input_array: Union[Sequence[float], np.ndarray, pd.Series, pl.Series],
    binary_values: Optional[int] = None
) -> Union[np.ndarray, pd.Series, pl.Series, list[float], tuple[float]]:
    """
    Thresholds binary features in a 1D input. The number of binary features are counted starting from the end.

    Parameters:
        input_array: 1D sequence, NumPy array, pandas Series, or polars Series.
        binary_values (Optional[int]) :
            - If `None`, all values are treated as binary.
            - If `int`, only this many last `binary_values` are thresholded.

    Returns:
        Same type as input, with binary elements binarized to 0 or 1 using a 0.5 threshold.
    """
    original_type = type(input_array)

    if isinstance(input_array, pl.Series):
        array = input_array.to_numpy()
    elif isinstance(input_array, (pd.Series, np.ndarray)):
        array = np.asarray(input_array)
    elif isinstance(input_array, (list, tuple)):
        array = np.array(input_array)
    else:
        raise TypeError("Unsupported input type")

    array = array.flatten()
    total = array.shape[0]

    bin_count = total if binary_values is None else binary_values
    if not (0 <= bin_count <= total):
        raise ValueError("binary_values must be between 0 and the total number of elements")

    if bin_count == 0:
        result = array
    else:
        cont_part = array[:-bin_count] if bin_count < total else np.array([])
        bin_part = (array[-bin_count:] > 0.5).astype(int)
        result = np.concatenate([cont_part, bin_part])

    if original_type is pd.Series:
        return pd.Series(result, index=input_array.index if hasattr(input_array, 'index') else None) # type: ignore
    elif original_type is pl.Series:
        return pl.Series(input_array.name if hasattr(input_array, 'name') else "binary", result) # type: ignore
    elif original_type is list:
        return result.tolist()
    elif original_type is tuple:
        return tuple(result)
    else:
        return result


def serialize_object(obj: Any, save_dir: str, filename: str, verbose: bool=True, raise_on_error: bool=False) -> Optional[str]:
    """
    Serializes a Python object using joblib; suitable for Python built-ins, numpy, and pandas.

    Parameters:
        obj (Any) : The Python object to serialize.
        save_dir (str) : Directory path where the serialized object will be saved.
        filename (str) : Name for the output file, extension will be appended if needed.

    Returns:
        (str | None) : The full file path where the object was saved if successful; otherwise, None.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        sanitized_name = sanitize_filename(filename)
        if not sanitized_name.endswith('.joblib'):
            sanitized_name = sanitized_name + ".joblib"
        full_path = os.path.join(save_dir, sanitized_name)
        joblib.dump(obj, full_path)
    except (IOError, OSError, TypeError, TerminatedWorkerError) as e:
        message = f"❌ Failed to serialize object of type '{type(obj)}': {e}"
        if raise_on_error:
            raise Exception(message)
        else:
            print(message)
        return None
    else:
        if verbose:
            print(f"✅ Object of type '{type(obj)}' saved to '{full_path}'")
        return full_path


def deserialize_object(filepath: str, verbose: bool=True, raise_on_error: bool=True) -> Optional[Any]:
    """
    Loads a serialized object from a .joblib file.

    Parameters:
        filepath (str): Full path to the serialized .joblib file.

    Returns:
        (Any | None): The deserialized Python object, or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return None
    try:
        obj = joblib.load(filepath)
    except (IOError, OSError, EOFError, TypeError, ValueError) as e:
        message = f"❌ Failed to deserialize object from '{filepath}': {e}"
        if raise_on_error:
            raise Exception(message)
        else:
            print(message)
        return None
    else:
        if verbose:
            print(f"✅ Loaded object of type '{type(obj)}'")
        return obj


def distribute_datasets_by_target(
    df_or_path: Union[pd.DataFrame, str],
    target_columns: list[str],
    verbose: bool = False
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Yields cleaned DataFrames for each target column, where rows with missing
    target values are removed. The target column is placed at the end.

    Parameters
    ----------
    df_or_path : [pd.DataFrame | str]
        Dataframe or path to Dataframe with all feature and target columns ready to split and train a model.
    target_columns : List[str]
        List of target column names to generate per-target DataFrames.
    verbose: bool
        Whether to print info for each yielded dataset.

    Yields
    ------
    Tuple[str, pd.DataFrame]
        * First element is the target column name.
        * Second element is the corresponding cleaned DataFrame.
    """
    # Validate path
    if isinstance(df_or_path, str):
        df, _ = load_dataframe(df_or_path)
    else:
        df = df_or_path
    
    valid_targets = [col for col in df.columns if col in target_columns]
    feature_columns = [col for col in df.columns if col not in valid_targets]

    for target in valid_targets:
        subset = df[feature_columns + [target]].dropna(subset=[target])
        if verbose:
            print(f"Target: '{target}' - Dataframe shape: {subset.shape}")
        yield target, subset


def _script_info(all_data: list[str]):
    """
    List available names.
    """
    print("Available functions and objects:")
    for i, name in enumerate(all_data, start=1):
            print(f"{i} - {name}")


def info():
    _script_info(__all__)
