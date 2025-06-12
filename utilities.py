import math
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re


def list_csv_paths(directory: str) -> tuple[list[str], list[str]]:
    """
    Lists all CSV files in a given directory and returns their paths with corresponding base names.

    Parameters:
        directory (str): Path to the directory containing `.csv` files.

    Returns:
        Tuple ([List[str], List[str]]):
        - List of absolute paths to `.csv` files.
        - List of corresponding base names (without extensions).
    """
    dir_path = Path(directory).expanduser().resolve()

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    csv_paths = list(dir_path.glob("*.csv"))
    if not csv_paths:
        raise IOError(f"No CSV files found in directory: {dir_path}")

    paths = [str(p) for p in csv_paths]
    names = [p.stem for p in csv_paths]

    return paths, names


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
    print(f"Loaded dataset: '{df_name}' with shape: {df.shape}")
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
    for df_path, df_name in list_csv_paths(datasets_dir):
        df = pd.read_csv(df_path)
        print(f"Loaded dataset: '{df_name}' with shape: {df.shape}")
        yield df, df_name
        
        
def normalize_mixed_list(data: list, threshold: int = 2) -> list[float]:
    """
    Normalize a mixed list of numeric values and strings so that the sum of the values equals 1.0,
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
            Values significantly smaller than the median scale are scaled up 
            before normalization to correct likely input errors.
    
    Notes:
        - Zeros and None values remain zero.
        - If all input values are zero or None, the function returns a list of zeros.
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

