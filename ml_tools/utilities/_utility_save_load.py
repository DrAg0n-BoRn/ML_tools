import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Literal, Union, Optional, Any, overload

from ..schema import FeatureSchema

from ..path_manager import make_fullpath, list_csv_paths, sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Save/Load Utilities")


__all__ = [
    "load_dataframe",
    "load_dataframe_greedy",
    "load_dataframe_with_schema",
    "yield_dataframes_from_dir",
    "save_dataframe_filename",
    "save_dataframe",
    "save_dataframe_with_schema"
]



# Overload 1: When kind='pandas'
@overload
def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None, 
    kind: Literal["pandas"] = "pandas",
    all_strings: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, str]:
    ... # for overload stubs

# Overload 2: When kind='polars'
@overload
def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["polars"] = "polars",
    all_strings: bool = False,
    verbose: bool = True
) -> tuple[pl.DataFrame, str]:
    ... # for overload stubs

def load_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["pandas", "polars"] = "pandas",
    all_strings: bool = False,
    verbose: bool = True
) -> Union[tuple[pd.DataFrame, str], tuple[pl.DataFrame, str]]:
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
            pl_kwargs['null_values'] = ["", " "]
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


def load_dataframe_greedy(directory: Union[str, Path],
                          use_columns: Optional[list[str]] = None,
                          all_strings: bool = False,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Greedily loads the first found CSV file from a directory into a Pandas DataFrame.

    This function scans the specified directory for any CSV files. It will
    attempt to load the *first* CSV file it finds using the `load_dataframe`
    function as a Pandas DataFrame.

    Args:
        directory (str, Path): 
            The path to the directory to search for a CSV file.
        use_columns (list[str] | None):
            A list of column names to load. If None, all columns are loaded.
        all_strings (bool): 
            If True, loads all columns as string data types.

    Returns:
        pd.DataFrame: 
            A pandas DataFrame loaded from the first CSV file found.

    Raises:
        FileNotFoundError: 
            If the specified directory does not exist or the CSV file path
            found is invalid.
        ValueError: 
            If the loaded DataFrame is empty or `use_columns` contains
            invalid column names.
    """
    # validate directory
    dir_path = make_fullpath(directory, enforce="directory")
    
    # list all csv files and grab one (should be the only one)
    csv_dict = list_csv_paths(directory=dir_path, verbose=False, raise_on_empty=True)
    
    # explicitly check that there is only one csv file
    if len(csv_dict) > 1:
        _LOGGER.warning(f"Multiple CSV files found in '{dir_path}'. Only one will be loaded.")
        
    for df_path in csv_dict.values():
        df , _df_name = load_dataframe(df_path=df_path,
                                    use_columns=use_columns,
                                    kind="pandas",
                                    all_strings=all_strings,
                                    verbose=verbose)
        break
    
    return df


def load_dataframe_with_schema(
    df_path: Union[str, Path], 
    schema: "FeatureSchema",
    all_strings: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Loads a CSV file into a Pandas DataFrame, strictly validating its
    feature columns against a FeatureSchema.

    This function wraps `load_dataframe`. After loading, it validates
    that the first N columns of the DataFrame (where N =
    len(schema.feature_names)) contain *exactly* the set of features
    specified in the schema.

    - If the columns are present but out of order, they are reordered.
    - If any required feature is missing from the first N columns, it fails.
    - If any extra column is found within the first N columns, it fails.

    Columns *after* the first N are considered target columns and are
    logged for verification.

    Args:
        df_path (str, Path): 
            The path to the CSV file.
        schema (FeatureSchema): 
            The schema object to validate against.
        all_strings (bool): 
            If True, loads all columns as string data types.

    Returns:
        (Tuple[pd.DataFrame, str]):
            A tuple containing the loaded, validated (and possibly
            reordered) pandas DataFrame and the base name of the file.
            
    Raises:
        ValueError: 
            - If the DataFrame is missing columns required by the schema
              within its first N columns.
            - If the DataFrame's first N columns contain unexpected
              columns that are not in the schema.
        FileNotFoundError: 
            If the file does not exist at the given path.
    """
    # Step 1: Load the dataframe using the original function
    try:
        df, df_name = load_dataframe(
            df_path=df_path, 
            use_columns=None,  # Load all columns for validation
            kind="pandas", 
            all_strings=all_strings,
            verbose=True
        )
    except Exception as e:
        _LOGGER.error(f"Failed during initial load for schema validation: {e}")
        raise e
    
    # Step 2: Call the helper to validate and reorder
    df_validated = _validate_and_reorder_schema(df=df, schema=schema)

    return df_validated, df_name


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
    files_dict = list_csv_paths(datasets_path, verbose=verbose, raise_on_empty=True)
    for df_name, df_path in files_dict.items():
        df: pd.DataFrame
        df, _ = load_dataframe(df_path, kind="pandas", verbose=verbose) # type: ignore
        yield df, df_name


def save_dataframe_filename(df: Union[pd.DataFrame, pl.DataFrame], save_dir: Union[str,Path], filename: str, verbose: int=3) -> None:
    """
    Saves a pandas or polars DataFrame to a CSV file.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): 
            The DataFrame to save.
        save_dir (Union[str, Path]): 
            The directory where the CSV file will be saved.
        filename (str): 
            The CSV filename. The '.csv' extension will be added if missing.
        verbose (int): 
            Verbosity level for logging.
                - 0: Error level
                - 1: Warning level
                - 2: Info level
                - 3: Detailed process info
    """
    # This check works for both pandas and polars
    if df.shape[0] == 0:
        # Warning instead of error to allow graceful skipping
        _LOGGER.warning(f"Attempting to save an empty DataFrame: '{filename}'. Process Skipped.")
        return
    
    # Create the directory if it doesn't exist
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # Clean the filename
    filename = sanitize_filename(filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    output_path = save_path / filename
        
    # --- Type-specific saving logic ---
    if isinstance(df, pd.DataFrame):
        # Transform "" to np.nan before saving
        df_to_save = df.replace(r'^\s*$', np.nan, regex=True)
        # Save
        df_to_save.to_csv(output_path, index=False, encoding='utf-8')
    elif isinstance(df, pl.DataFrame):
        # Transform empty strings to Null
        df_to_save = df.with_columns(
            pl.when(pl.col(pl.String).str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col(pl.String))
            .name.keep()
        )
        # Save
        df_to_save.write_csv(output_path)
    else:
        # This error handles cases where an unsupported type is passed
        _LOGGER.error(f"Unsupported DataFrame type: {type(df)}. Must be pandas or polars.")
        raise TypeError()
    
    if verbose >= 2:
        _LOGGER.info(f"Saved dataset: '{filename}' with shape: {df_to_save.shape}")


def save_dataframe(df: Union[pd.DataFrame, pl.DataFrame], full_path: Path, verbose: int=3) -> None:
    """
    Saves a DataFrame to a specified full path.

    This function is a wrapper for `save_dataframe_filename()`. It takes a
    single `pathlib.Path` object pointing to a `.csv` file.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): The pandas or polars DataFrame to save.
        full_path (Path): The complete file path, including the filename and `.csv` extension, where the DataFrame will be saved.
        verbose (int): Verbosity level for logging.
            - 0: Error level
            - 1: Warning level
            - 2: Info level
            - 3: Detailed process info
    """
    if not isinstance(full_path, Path) or not full_path.suffix.endswith(".csv"):
        _LOGGER.error('A path object pointing to a .csv file must be provided.')
        raise ValueError()

    save_dataframe_filename(df=df, 
                            save_dir=full_path.parent,
                            filename=full_path.name,
                            verbose=verbose)


def save_dataframe_with_schema(
    df: pd.DataFrame, 
    full_path: Path,
    schema: "FeatureSchema",
    verbose: int=3
) -> None:
    """
    Saves a pandas DataFrame to a CSV, strictly enforcing that the
    first N columns match the FeatureSchema.

    This function validates that the first N columns of the DataFrame
    (where N = len(schema.feature_names)) contain *exactly* the set
    of features specified in the schema.
    
    - If the columns are present but out of order, they are reordered.
    - If any required feature is missing from the first N columns, it fails.
    - If any extra column is found within the first N columns, it fails.

    Columns *after* the first N are considered target columns and are
    logged for verification.

    Args:
        df (pd.DataFrame): 
            The DataFrame to save.
        full_path (Path): 
            The complete file path where the DataFrame will be saved.
        schema (FeatureSchema): 
            The schema object to validate against.
        verbose (int): 
            Verbosity level for logging.
                - 0: Error level
                - 1: Warning level
                - 2: Info level
                - 3: Detailed process info
    """
    if not isinstance(full_path, Path) or not full_path.suffix.endswith(".csv"):
        _LOGGER.error('A path object pointing to a .csv file must be provided.')
        raise ValueError()
    
    # Call the helper to validate and reorder
    df_to_save = _validate_and_reorder_schema(df=df, schema=schema, verbose=verbose)
    
    # Call the original save function
    save_dataframe(df=df_to_save, full_path=full_path, verbose=verbose)


def _validate_and_reorder_schema(
    df: pd.DataFrame, 
    schema: "FeatureSchema",
    verbose:int=3
) -> pd.DataFrame:
    """
    Internal helper to validate and reorder a DataFrame against a schema.

    Checks for missing, extra, and out-of-order feature columns
    (the first N columns). Returns a reordered DataFrame if necessary.
    Logs all actions.

    Raises:
        ValueError: If validation fails.
    """
    # Get schema and DataFrame column info
    expected_features = list(schema.feature_names)
    expected_set = set(expected_features)
    n_features = len(expected_features)
    
    all_df_columns = df.columns.to_list()

    # --- Strict Validation ---

    # 0. Check if DataFrame is long enough
    if len(all_df_columns) < n_features:
        _LOGGER.error(f"DataFrame has only {len(all_df_columns)} columns, but schema requires {n_features} features.")
        raise ValueError()
    
    df_feature_cols = all_df_columns[:n_features]
    df_feature_set = set(df_feature_cols)
    df_target_cols = all_df_columns[n_features:]

    # 1. Check for missing features
    missing_from_df = expected_set - df_feature_set
    if missing_from_df:
        _LOGGER.error(f"DataFrame's first {n_features} columns are missing required schema features: {missing_from_df}")
        raise ValueError()

    # 2. Check for extra (unexpected) features
    extra_in_df = df_feature_set - expected_set
    if extra_in_df:
        _LOGGER.error(f"DataFrame's first {n_features} columns contain unexpected columns: {extra_in_df}")
        raise ValueError()

    # --- Reordering ---
    
    df_to_process = df

    # If we pass validation, the sets are equal. Now check order.
    if df_feature_cols == expected_features:
        if verbose >= 2:
            _LOGGER.info("DataFrame feature columns already match schema order.")
    else:
        if verbose >= 1:
            _LOGGER.warning("DataFrame feature columns do not match schema order. Reordering...")
        
        # Rebuild the DataFrame with the correct feature order + target columns
        new_order = expected_features + df_target_cols
        df_to_process = df[new_order]

    # Log the presumed target columns for user verification
    if not df_target_cols:
        if verbose >= 1:
            _LOGGER.warning(f"No target columns were found after index {n_features-1}.")
    else:
        if verbose >= 2:
            _LOGGER.info(f"Target Columns: {df_target_cols}")
    
    return df_to_process # type: ignore

