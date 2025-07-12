import math
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import re
from typing import Literal, Union, Sequence, Optional, Any, Iterator, Tuple
import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError


# Keep track of available tools
__all__ = [
    "make_fullpath",
    "list_csv_paths",
    "list_files_by_extension",
    "load_dataframe",
    "yield_dataframes_from_dir",
    "merge_dataframes",
    "save_dataframe",
    "normalize_mixed_list",
    "sanitize_filename",
    "threshold_binary_values",
    "threshold_binary_values_batch",
    "serialize_object",
    "deserialize_object",
    "distribute_datasets_by_target",
    "train_dataset_orchestrator",
]


def make_fullpath(
        input_path: Union[str, Path],
        make: bool = False,
        verbose: bool = False
    ) -> Path:
    """
    Resolves a string or Path into an absolute Path.

    - If the path exists, it is returned.
    - If the path does not exist and `make=True`, it will:
        - Create the file if the path has a suffix (i.e., is treated as a file)
        - Create the directory if it has no suffix
    - If `make=False` and the path does not exist, an error is raised.
    - Optionally prints whether the resolved path is a file or directory.

    Parameters:
        input_path (str | Path): Path to resolve.
        make (bool): If True, attempt to create file or directory.
        verbose (bool): Print classification after resolution.

    Returns:
        Path: Resolved absolute path.

    Raises:
        ValueError: If the path doesn't exist and can't be created.
    """
    path = Path(input_path).expanduser()

    is_file = path.suffix != ""

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        if not make:
            raise ValueError(f"❌ Path does not exist: '{path}'")

        try:
            if is_file:
                # Create parent directories first
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=False)
            else:
                path.mkdir(parents=True, exist_ok=True)
            resolved = path.resolve(strict=True)
        except Exception as e:
            raise ValueError(f"❌ Failed to create {'file' if is_file else 'directory'} '{path}': {e}")

    if verbose:
        if resolved.is_file():
            print("📄 Path is a File")
        elif resolved.is_dir():
            print("📁 Path is a Directory")
        else:
            print("❓ Path exists but is neither file nor directory")

    return resolved


def list_csv_paths(directory: Union[str,Path]) -> dict[str, Path]:
    """
    Lists all `.csv` files in the specified directory and returns a mapping: filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory containing `.csv` files.

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}.
    """
    dir_path = make_fullpath(directory)

    csv_paths = list(dir_path.glob("*.csv"))
    if not csv_paths:
        raise IOError(f"No CSV files found in directory: {dir_path.name}")
    
    # make a dictionary of paths and names
    name_path_dict = {p.stem: p for p in csv_paths}
    
    print("\n🗂️ CSV files found:")
    for name in name_path_dict.keys():
        print(f"\t{name}")

    return name_path_dict


def list_files_by_extension(directory: Union[str,Path], extension: str) -> dict[str, Path]:
    """
    Lists all files with the specified extension in the given directory and returns a mapping: 
    filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory to search in.
        extension (str): File extension to search for (e.g., 'json', 'txt').

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}.
    """
    dir_path = make_fullpath(directory)
    
    # Normalize the extension (remove leading dot if present)
    normalized_ext = extension.lstrip(".").lower()
    pattern = f"*.{normalized_ext}"
    
    matched_paths = list(dir_path.glob(pattern))
    if not matched_paths:
        raise IOError(f"No '.{normalized_ext}' files found in directory: {dir_path}")

    name_path_dict = {p.stem: p for p in matched_paths}
    
    print(f"\n📂 '{normalized_ext.upper()}' files found:")
    for name in name_path_dict:
        print(f"\t{name}")
    
    return name_path_dict


def load_dataframe(
    df_path: Union[str, Path], 
    kind: Literal["pandas", "polars"] = "pandas",
    all_strings: bool = False
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], str]:
    """
    Load a CSV file into a DataFrame and extract its base name.

    Can load data as either a pandas or a polars DataFrame. Allows for loading all
    columns as string types to prevent type inference errors.

    Args:
        df_path (Union[str, Path]): 
            The path to the CSV file.
        kind (Literal["pandas", "polars"], optional): 
            The type of DataFrame to load. Defaults to "pandas".
        all_strings (bool, optional): 
            If True, loads all columns as string data types. This is useful for
            ETL tasks and to avoid type-inference errors. Defaults to False.

    Returns:
        (Tuple[DataFrameType, str]):
            A tuple containing the loaded DataFrame (either pandas or polars)
            and the base name of the file (without extension).
            
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the DataFrame is empty or an invalid 'kind' is provided.
    """
    path = make_fullpath(df_path)
    
    df_name = path.stem

    if kind == "pandas":
        if all_strings:
            df = pd.read_csv(path, encoding='utf-8', dtype=str)
        else:
            df = pd.read_csv(path, encoding='utf-8')
            
    elif kind == "polars":
        if all_strings:
            df = pl.read_csv(path, infer_schema=False)
        else:
            # Default behavior: infer the schema.
            df = pl.read_csv(path, infer_schema_length=1000)
            
    else:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of 'pandas' or 'polars'.")

    # This check works for both pandas and polars DataFrames
    if df.shape[0] == 0:
        raise ValueError(f"DataFrame '{df_name}' loaded from '{path}' is empty.")

    print(f"\n💿 Loaded {kind} dataset: '{df_name}' with shape: {df.shape}")
    
    return df, df_name


def yield_dataframes_from_dir(datasets_dir: Union[str,Path]):
    """
    Iterates over all CSV files in a given directory, loading each into a pandas DataFrame.

    Parameters:
        datasets_dir (str | Path):
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
    datasets_path = make_fullpath(datasets_dir)
    for df_name, df_path in list_csv_paths(datasets_path).items():
        df: pd.DataFrame
        df, _ = load_dataframe(df_path, kind="pandas") # type: ignore
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


def save_dataframe(df: Union[pd.DataFrame, pl.DataFrame], save_dir: Union[str,Path], filename: str) -> None:
    """
    Saves a pandas or polars DataFrame to a CSV file.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): 
            The DataFrame to save.
        save_dir (Union[str, Path]): 
            The directory where the CSV file will be saved.
        filename (str): 
            The CSV filename. The '.csv' extension will be added if missing.
    """
    # This check works for both pandas and polars
    if df.shape[0] == 0:
        print(f"⚠️ Attempting to save an empty DataFrame: '{filename}'. Process Skipped.")
        return
    
    # Create the directory if it doesn't exist
    save_path = make_fullpath(save_dir, make=True)
    
    # Clean the filename
    filename = sanitize_filename(filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    output_path = save_path / filename
        
    # --- Type-specific saving logic ---
    if isinstance(df, pd.DataFrame):
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif isinstance(df, pl.DataFrame):
        df.write_csv(output_path) # Polars defaults to utf8 and no index
    else:
        # This error handles cases where an unsupported type is passed
        raise TypeError(f"Unsupported DataFrame type: {type(df)}. Must be pandas or polars.")
    
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
    assert input_array.ndim == 2, f"Expected 2D array, got {input_array.ndim}D"
    batch_size, total_features = input_array.shape
    assert 0 <= binary_values <= total_features, "binary_values out of valid range"

    if binary_values == 0:
        return input_array.copy()

    cont_part = input_array[:, :-binary_values] if binary_values < total_features else np.empty((batch_size, 0))
    bin_part = input_array[:, -binary_values:] > 0.5
    bin_part = bin_part.astype(np.int32)

    return np.hstack([cont_part, bin_part])


def serialize_object(obj: Any, save_dir: Union[str,Path], filename: str, verbose: bool=True, raise_on_error: bool=False) -> None:
    """
    Serializes a Python object using joblib; suitable for Python built-ins, numpy, and pandas.

    Parameters:
        obj (Any) : The Python object to serialize.
        save_dir (str | Path) : Directory path where the serialized object will be saved.
        filename (str) : Name for the output file, extension will be appended if needed.
    """
    try:
        save_path = make_fullpath(save_dir, make=True)
        sanitized_name = sanitize_filename(filename)
        if not sanitized_name.endswith('.joblib'):
            sanitized_name = sanitized_name + ".joblib"
        full_path = save_path / sanitized_name
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
        return None


def deserialize_object(filepath: Union[str,Path], verbose: bool=True, raise_on_error: bool=True) -> Optional[Any]:
    """
    Loads a serialized object from a .joblib file.

    Parameters:
        filepath (str | Path): Full path to the serialized .joblib file.

    Returns:
        (Any | None): The deserialized Python object, or None if loading fails.
    """
    true_filepath = make_fullpath(filepath)
    
    try:
        obj = joblib.load(true_filepath)
    except (IOError, OSError, EOFError, TypeError, ValueError) as e:
        message = f"❌ Failed to deserialize object from '{true_filepath}': {e}"
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
    df_or_path: Union[pd.DataFrame, str, Path],
    target_columns: list[str],
    verbose: bool = False
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Yields cleaned DataFrames for each target column, where rows with missing
    target values are removed. The target column is placed at the end.

    Parameters
    ----------
    df_or_path : [pd.DataFrame | str | Path]
        Dataframe or path to Dataframe with all feature and target columns ready to split and train a model.
    target_columns : List[str]
        List of target column names to generate per-target DataFrames.
    verbose: bool
        Whether to print info for each yielded dataset.

    Yields
    ------
    Tuple[str, pd.DataFrame]
        * Target name.
        * Pandas DataFrame.
    """
    # Validate path or dataframe
    if isinstance(df_or_path, str) or isinstance(df_or_path, Path):
        df_path = make_fullpath(df_or_path)
        df, _ = load_dataframe(df_path)
    else:
        df = df_or_path
    
    valid_targets = [col for col in df.columns if col in target_columns]
    feature_columns = [col for col in df.columns if col not in valid_targets]

    for target in valid_targets:
        subset = df[feature_columns + [target]].dropna(subset=[target]) # type: ignore
        if verbose:
            print(f"Target: '{target}' - Dataframe shape: {subset.shape}")
        yield target, subset


def train_dataset_orchestrator(list_of_dirs: list[Union[str,Path]], 
                               target_columns: list[str], 
                               save_dir: Union[str,Path],
                               safe_mode: bool=False):
    """
    Orchestrates the creation of single-target datasets from multiple directories each with a variable number of CSV datasets.

    This function iterates through a list of directories, finds all CSV files,
    and splits each dataframe based on the provided target columns. Each resulting
    single-target dataframe is then saved to a specified directory.

    Parameters
    ----------
    list_of_dirs : list[str | Path]
        A list of directory paths where the source CSV files are located.
    target_columns : list[str]
        A list of column names to be used as targets for splitting the datasets.
    save_dir : str | Path
        The directory where the newly created single-target datasets will be saved.
    safe_mode : bool
        If True, prefixes the saved filename with the source directory name to prevent overwriting files with the same name from different sources.
    """
    all_dir_paths: list[Path] = list()
    for dir in list_of_dirs:
        dir_path = make_fullpath(dir)
        if not dir_path.is_dir():
            raise IOError(f"'{dir}' is not a directory.")
        all_dir_paths.append(dir_path)
    
    # main loop
    total_saved = 0
    for df_dir in all_dir_paths:
        for df_name, df_path in list_csv_paths(df_dir).items():
            try:
                for target_name, df in distribute_datasets_by_target(df_or_path=df_path, target_columns=target_columns, verbose=False):
                    if safe_mode:
                        filename = df_dir.name + '_' + target_name + '_' + df_name
                    else:
                        filename = target_name + '_' + df_name
                    save_dataframe(df=df, save_dir=save_dir, filename=filename)
                    total_saved += 1
            except Exception as e:
                print(f"⚠️ Failed to process file '{df_path}'. Reason: {e}")
                continue 

    print(f"\n✅ {total_saved} single-target datasets were created.")


def _script_info(all_data: list[str]):
    """
    List available names.
    """
    print("Available functions and objects:")
    for i, name in enumerate(all_data, start=1):
            print(f"{i} - {name}")


def info():
    _script_info(__all__)
