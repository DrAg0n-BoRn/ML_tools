import math
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Literal, Union, Sequence, Optional, Any, Iterator, Tuple, overload, TypeVar, get_origin, Type
import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError
from .path_manager import sanitize_filename, make_fullpath, list_csv_paths, list_files_by_extension, list_subdirectories
from ._script_info import _script_info
from ._logger import _LOGGER
from .keys import DatasetKeys, PytorchModelArchitectureKeys, PytorchArtifactPathKeys, SHAPKeys


# Keep track of available tools
__all__ = [
    "load_dataframe",
    "yield_dataframes_from_dir",
    "merge_dataframes",
    "save_dataframe",
    "normalize_mixed_list",
    "threshold_binary_values",
    "threshold_binary_values_batch",
    "serialize_object",
    "deserialize_object",
    "distribute_dataset_by_target",
    "train_dataset_orchestrator",
    "train_dataset_yielder",
    "find_model_artifacts",
    "select_features_by_shap"
]


# Overload 1: When kind='pandas'
@overload
def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None, 
    kind: Literal["pandas"] = "pandas",
    all_strings: bool = False,
    verbose: bool = True
) -> Tuple[pd.DataFrame, str]:
    ... # for overload stubs

# Overload 2: When kind='polars'
@overload
def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["polars"] = "polars",
    all_strings: bool = False,
    verbose: bool = True
) -> Tuple[pl.DataFrame, str]:
    ... # for overload stubs

def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["pandas", "polars"] = "pandas",
    all_strings: bool = False,
    verbose: bool = True
) -> Union[Tuple[pd.DataFrame, str], Tuple[pl.DataFrame, str]]:
    """
    Load a CSV file into a DataFrame and extract its base name.

    Can load data as either a pandas or a polars DataFrame. Allows for loading all
    columns or a subset of columns as string types to prevent type inference errors.

    Args:
        df_path (str, Path): 
            The path to the CSV file.
        use_columns (list[str] | None):
            If provided, only these columns will be loaded from the CSV.
        kind ("pandas", "polars"): 
            The type of DataFrame to load. Defaults to "pandas".
        all_strings (bool): 
            If True, loads all columns as string data types. This is useful for
            ETL tasks and to avoid type-inference errors.

    Returns:
        (Tuple[DataFrameType, str]):
            A tuple containing the loaded DataFrame (either pandas or polars)
            and the base name of the file (without extension).
            
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the DataFrame is empty, an invalid 'kind' is provided, or a column in 'use_columns' is not found in the file.
    """
    path = make_fullpath(df_path)
    
    df_name = path.stem

    try:
        if kind == "pandas":
            pd_kwargs: dict[str,Any]
            pd_kwargs = {'encoding': 'utf-8'}
            if use_columns:
                pd_kwargs['usecols'] = use_columns
            if all_strings:
                pd_kwargs['dtype'] = str
                
            df = pd.read_csv(path, **pd_kwargs)

        elif kind == "polars":
            pl_kwargs: dict[str,Any]
            pl_kwargs = {}
            if use_columns:
                pl_kwargs['columns'] = use_columns
                
            if all_strings:
                pl_kwargs['infer_schema'] = False
            else:
                pl_kwargs['infer_schema_length'] = 1000
                
            df = pl.read_csv(path, **pl_kwargs)

        else:
            _LOGGER.error(f"Invalid kind '{kind}'. Must be one of 'pandas' or 'polars'.")
            raise ValueError()
            
    except (ValueError, pl.exceptions.ColumnNotFoundError) as e:
        _LOGGER.error(f"Failed to load '{df_name}'. A specified column may not exist in the file.")
        raise e

    # This check works for both pandas and polars DataFrames
    if df.shape[0] == 0:
        _LOGGER.error(f"DataFrame '{df_name}' loaded from '{path}' is empty.")
        raise ValueError()

    if verbose:
        _LOGGER.info(f"💾 Loaded {kind.upper()} dataset: '{df_name}' with shape: {df.shape}")
    
    return df, df_name # type: ignore

def yield_dataframes_from_dir(datasets_dir: Union[str,Path], verbose: bool=True):
    """
    Iterates over all CSV files in a given directory, loading each into a Pandas DataFrame.

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
    files_dict = list_csv_paths(datasets_path, verbose=verbose)
    for df_name, df_path in files_dict.items():
        df: pd.DataFrame
        df, _ = load_dataframe(df_path, kind="pandas", verbose=verbose) # type: ignore
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
        raise ValueError("❌ At least 2 DataFrames must be provided.")
    
    if verbose:
        for i, df in enumerate(dfs, start=1):
            print(f"➡️ DataFrame {i} shape: {df.shape}")
    

    if direction == "horizontal":
        reference_index = dfs[0].index
        for i, df in enumerate(dfs, start=1):
            if not df.index.equals(reference_index):
                raise ValueError(f"❌ Indexes do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=1)

    elif direction == "vertical":
        reference_columns = dfs[0].columns
        for i, df in enumerate(dfs, start=1):
            if not df.columns.equals(reference_columns):
                raise ValueError(f"❌ Column names/order do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=0)

    else:
        _LOGGER.error(f"Invalid merge direction: {direction}")
        raise ValueError()

    if reset_index:
        merged_df = merged_df.reset_index(drop=True)
    
    if verbose:
        _LOGGER.info(f"Merged DataFrame shape: {merged_df.shape}")

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
        _LOGGER.warning(f"Attempting to save an empty DataFrame: '{filename}'. Process Skipped.")
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
        _LOGGER.error(f"Unsupported DataFrame type: {type(df)}. Must be pandas or polars.")
        raise TypeError()
    
    _LOGGER.info(f"Saved dataset: '{filename}' with shape: {df.shape}")


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
    input_array: Union[Sequence[float], np.ndarray, pd.Series, pl.Series],
    binary_values: Optional[int] = None
) -> Union[np.ndarray, pd.Series, pl.Series, list[float], tuple[float]]:
    """
    Thresholds binary features in a 1D input. The number of binary features are counted starting from the end.
    
    Binary elements are converted to 0 or 1 using a 0.5 threshold.

    Parameters:
        input_array: 1D sequence, NumPy array, pandas Series, or polars Series.
        binary_values (Optional[int]) :
            - If `None`, all values are treated as binary.
            - If `int`, only this many last `binary_values` are thresholded.

    Returns:
        Any:
        Same type as input
    """
    original_type = type(input_array)

    if isinstance(input_array, pl.Series):
        array = input_array.to_numpy()
    elif isinstance(input_array, (pd.Series, np.ndarray)):
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
        _LOGGER.error(f"Failed to serialize object of type '{type(obj)}'.")
        if raise_on_error:
            raise e
        return None
    else:
        if verbose:
            _LOGGER.info(f"Object of type '{type(obj)}' saved to '{full_path}'")
        return None

# Define a TypeVar to link the expected type to the return type of deserialization
T = TypeVar('T')
    
def deserialize_object(
    filepath: Union[str, Path],
    expected_type: Optional[Type[T]] = None,
    verbose: bool = True,
    raise_on_error: bool = True
    ) -> Optional[T]:
    """
    Loads a serialized object from a .joblib file.

    Parameters:
        filepath (str | Path): Full path to the serialized .joblib file.
        expected_type (Type[T] | None): The expected type of the object.
            If provided, the function raises a TypeError if the loaded object
            is not an instance of this type. It correctly handles generics
            like `list[str]` by checking the base type (e.g., `list`).
            Defaults to None, which skips the type check.
        verbose (bool): If True, logs success messages.
        raise_on_error (bool): If True, raises exceptions on errors. If False, returns None instead.

    Returns:
        (Any | None): The deserialized Python object, which will match the
            `expected_type` if provided. Returns None if an error
            occurs and `raise_on_error` is False.
    """
    true_filepath = make_fullpath(filepath)
    
    try:
        obj = joblib.load(true_filepath)
    except (IOError, OSError, EOFError, TypeError, ValueError) as e:
        _LOGGER.error(f"Failed to deserialize object from '{true_filepath}'.")
        if raise_on_error:
            raise e
        return None
    else:
        # --- Type Validation Step ---
        if expected_type:
            # get_origin handles generics (e.g., list[str] -> list)
            # If it's not a generic, get_origin returns None, so we use the type itself.
            type_to_check = get_origin(expected_type) or expected_type
            
            # Can't do an isinstance check on 'Any', skip it.
            if type_to_check is not Any and not isinstance(obj, type_to_check):
                error_msg = (
                    f"Type mismatch: Expected an instance of '{expected_type}', "
                    f"but found '{type(obj)}' in '{true_filepath}'."
                )
                _LOGGER.error(error_msg)
                if raise_on_error:
                    raise TypeError()
                return None
        
        if verbose:
            _LOGGER.info(f"Loaded object of type '{type(obj)}' from '{true_filepath}'.")
        
        return obj


def distribute_dataset_by_target(
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
            _LOGGER.error(f"'{dir}' is not a directory.")
            raise IOError()
        all_dir_paths.append(dir_path)
    
    # main loop
    total_saved = 0
    for df_dir in all_dir_paths:
        for df_name, df_path in list_csv_paths(df_dir).items():
            try:
                for target_name, df in distribute_dataset_by_target(df_or_path=df_path, target_columns=target_columns, verbose=False):
                    if safe_mode:
                        filename = df_dir.name + '_' + target_name + '_' + df_name
                    else:
                        filename = target_name + '_' + df_name
                    save_dataframe(df=df, save_dir=save_dir, filename=filename)
                    total_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to process file '{df_path}'. Reason: {e}")
                continue 

    _LOGGER.info(f"{total_saved} single-target datasets were created.")


def train_dataset_yielder(
    df: pd.DataFrame,
    target_cols: list[str]
) -> Iterator[Tuple[pd.DataFrame, pd.Series, list[str], str]]:
    """ 
    Yields one tuple at a time:
        (features_dataframe, target_series, feature_names, target_name)

    Skips any target columns not found in the DataFrame.
    """
    # Determine which target columns actually exist in the DataFrame
    valid_targets = [col for col in target_cols if col in df.columns]

    # Features = all columns excluding valid target columns
    df_features = df.drop(columns=valid_targets)
    feature_names = df_features.columns.to_list()

    for target_col in valid_targets:
        df_target = df[target_col]
        yield (df_features, df_target, feature_names, target_col)


def find_model_artifacts(target_directory: Union[str,Path], load_scaler: bool, verbose: bool=False) -> list[dict[str,Any]]:
    """
    Scans subdirectories to find paths to model weights, target names, feature names, and model architecture. Optionally an scaler path if `load_scaler` is True.

    This function operates on a specific directory structure. It expects the
    `target_directory` to contain one or more subdirectories, where each
    subdirectory represents a single trained model result.

    The expected directory structure for each model is as follows:
    ```
        target_directory
        ├── model_1
        │   ├── *.pth
        │   ├── scaler_*.pth          (Required if `load_scaler` is True)
        │   ├── feature_names.txt
        │   ├── target_names.txt
        │   └── architecture.json
        └── model_2/
            └── ...
    ```

    Args:
        target_directory (str | Path): The path to the root directory that contains model subdirectories.
        load_scaler (bool): If True, the function requires and searches for a scaler file (`.pth`) in each model subdirectory.
        verbose (bool): If True, enables detailed logging during the file paths search process.

    Returns:
        (list[dict[str, Path]]): A list of dictionaries, where each dictionary
            corresponds to a model found in a subdirectory. The dictionary
            maps standardized keys to the absolute paths of the model's
            artifacts (weights, architecture, features, targets, and scaler).
            The scaler path will be `None` if `load_scaler` is False.
    """
    # validate directory
    root_path = make_fullpath(target_directory, enforce="directory")
    
    # store results
    all_artifacts: list[dict] = list()
    
    # find model directories
    result_dirs_dict = list_subdirectories(root_dir=root_path, verbose=verbose)
    for dir_name, dir_path in result_dirs_dict.items():
        # find files
        model_pth_dict = list_files_by_extension(directory=dir_path, extension="pth", verbose=verbose)
        
        # restriction
        if load_scaler:
            if len(model_pth_dict) != 2:
                _LOGGER.error(f"Directory {dir_path} should contain exactly 2 '.pth' files: scaler and weights.")
                raise IOError()
        else:
            if len(model_pth_dict) != 1:
                _LOGGER.error(f"Directory {dir_path} should contain exactly 1 '.pth' file: weights.")
                raise IOError()
        
        ##### Scaler and Weights #####
        scaler_path = None
        weights_path = None
        
        # load weights and scaler if present
        for pth_filename, pth_path in model_pth_dict.items():
            if load_scaler and pth_filename.lower().startswith(DatasetKeys.SCALER_PREFIX):
                scaler_path = pth_path
            else:
                weights_path = pth_path
        
        # validation
        if not weights_path:
            _LOGGER.error(f"Error parsing the model weights path from '{dir_name}'")
            raise IOError()
        
        if load_scaler and not scaler_path:
            _LOGGER.error(f"Error parsing the scaler path from '{dir_name}'")
            raise IOError()
        
        ##### Target and Feature names #####
        target_names_path = None
        feature_names_path = None
        
        # load feature and target names
        model_txt_dict = list_files_by_extension(directory=dir_path, extension="txt", verbose=verbose)
        
        for txt_filename, txt_path in model_txt_dict.items():
            if txt_filename == DatasetKeys.FEATURE_NAMES:
                feature_names_path = txt_path
            elif txt_filename == DatasetKeys.TARGET_NAMES:
                target_names_path = txt_path
        
        # validation
        if not target_names_path or not feature_names_path:
            _LOGGER.error(f"Error parsing features path or targets path from '{dir_name}'")
            raise IOError()
        
        ##### load model architecture path #####
        architecture_path = None
        
        model_json_dict = list_files_by_extension(directory=dir_path, extension="json", verbose=verbose)
        
        for json_filename, json_path in model_json_dict.items():
            if json_filename == PytorchModelArchitectureKeys.SAVENAME:
                architecture_path = json_path
        
        # validation
        if not architecture_path:
            _LOGGER.error(f"Error parsing the model architecture path from '{dir_name}'")
            raise IOError()
        
        ##### Paths dictionary #####
        parsing_dict = {
            PytorchArtifactPathKeys.WEIGHTS_PATH: weights_path,
            PytorchArtifactPathKeys.ARCHITECTURE_PATH: architecture_path,
            PytorchArtifactPathKeys.FEATURES_PATH: feature_names_path,
            PytorchArtifactPathKeys.TARGETS_PATH: target_names_path,
            PytorchArtifactPathKeys.SCALER_PATH: scaler_path
        }
        
        all_artifacts.append(parsing_dict)
    
    return all_artifacts


def select_features_by_shap(
    root_directory: Union[str, Path],
    shap_threshold: float = 1.0,
    verbose: bool = True) -> list[str]:
    """
    Scans subdirectories to find SHAP summary CSVs, then extracts feature
    names whose mean absolute SHAP value meets a specified threshold.

    This function is useful for automated feature selection based on feature
    importance scores aggregated from multiple models.

    Args:
        root_directory (Union[str, Path]):
            The path to the root directory that contains model subdirectories.
        shap_threshold (float):
            The minimum mean absolute SHAP value for a feature to be included
            in the final list.

    Returns:
        list[str]:
            A single, sorted list of unique feature names that meet the
            threshold criteria across all found files.
    """
    if verbose:
        _LOGGER.info(f"Starting feature selection with SHAP threshold >= {shap_threshold}")
    root_path = make_fullpath(root_directory, enforce="directory")

    # --- Step 2: Directory and File Discovery ---
    subdirectories = list_subdirectories(root_dir=root_path, verbose=False)
    
    shap_filename = SHAPKeys.SAVENAME + ".csv"

    valid_csv_paths = []
    for dir_name, dir_path in subdirectories.items():
        expected_path = dir_path / shap_filename
        if expected_path.is_file():
            valid_csv_paths.append(expected_path)
        else:
            _LOGGER.warning(f"No '{shap_filename}' found in subdirectory '{dir_name}'.")
    
    if not valid_csv_paths:
        _LOGGER.error(f"Process halted: No '{shap_filename}' files were found in any subdirectory.")
        return []

    if verbose:
        _LOGGER.info(f"Found {len(valid_csv_paths)} SHAP summary files to process.")

    # --- Step 3: Data Processing and Feature Extraction ---
    master_feature_set = set()
    for csv_path in valid_csv_paths:
        try:
            df, _ = load_dataframe(csv_path, kind="pandas", verbose=False)
            
            # Validate required columns
            required_cols = {SHAPKeys.FEATURE_COLUMN, SHAPKeys.SHAP_VALUE_COLUMN}
            if not required_cols.issubset(df.columns):
                _LOGGER.warning(f"Skipping '{csv_path}': missing required columns.")
                continue

            # Filter by threshold and extract features
            filtered_df = df[df[SHAPKeys.SHAP_VALUE_COLUMN] >= shap_threshold]
            features = filtered_df[SHAPKeys.FEATURE_COLUMN].tolist()
            master_feature_set.update(features)

        except (ValueError, pd.errors.EmptyDataError):
            _LOGGER.warning(f"Skipping '{csv_path}' because it is empty or malformed.")
            continue
        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred while processing '{csv_path}': {e}")
            continue

    # --- Step 4: Finalize and Return ---
    final_features = sorted(list(master_feature_set))
    if verbose:
        _LOGGER.info(f"Selected {len(final_features)} unique features across all files.")
    
    return final_features


def info():
    _script_info(__all__)
