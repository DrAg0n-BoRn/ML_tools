import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Literal, Union, Optional, overload

from ..schema import FeatureSchema
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

from ._utility_save_load import _validate_and_reorder_schema


_LOGGER = get_logger("Parquet Save/Load")


__all__ = [
    "load_parquet_dataframe",
    "load_parquet_header",
    "load_parquet_with_schema",
    "save_parquet_dataframe_filename",
    "save_parquet_dataframe",
    "save_parquet_with_schema"
]


# Overload 1: When kind='pandas'
@overload
def load_parquet_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None, 
    kind: Literal["pandas"] = "pandas",
    empty_as_nan: bool = True,
    verbose: bool = True
) -> tuple[pd.DataFrame, str]:
    ...

# Overload 2: When kind='polars'
@overload
def load_parquet_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["polars"] = "polars",
    empty_as_nan: bool = True,
    verbose: bool = True
) -> tuple[pl.DataFrame, str]:
    ...

def load_parquet_dataframe(
    df_path: Union[str, Path], 
    use_columns: Optional[list[str]] = None,
    kind: Literal["pandas", "polars"] = "pandas",
    empty_as_nan: bool = True,
    verbose: bool = True
) -> Union[tuple[pd.DataFrame, str], tuple[pl.DataFrame, str]]:
    """
    Load a Parquet file into a DataFrame and extract its base name.

    Can load data as either a pandas or a polars DataFrame.

    Args:
        df_path (str, Path): 
            The path to the Parquet file.
        use_columns (list[str] | None):
            If provided, only these columns will be loaded from the Parquet file.
        kind ("pandas", "polars"): 
            The type of DataFrame to load. Defaults to "pandas".
        empty_as_nan (bool):
            If True, converts empty strings to NaN (for pandas) or Null (for polars).
        verbose (bool):
            If True, logs the loading process and the resulting shape.

    Returns:
        (Tuple[DataFrame, str]):
            A tuple containing the loaded DataFrame (either pandas or polars)
            and the base name of the file (without extension).
            
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the DataFrame is empty or an invalid 'kind' is provided.
        TypeError: If the loaded object is not of the expected DataFrame type.
    """
    path = make_fullpath(df_path, enforce="file")
    df_name = path.stem

    try:
        if kind == "pandas":
            df_pandas = pd.read_parquet(path, engine='pyarrow', columns=use_columns)
            if not isinstance(df_pandas, pd.DataFrame):
                _LOGGER.error(f"Loaded object is not a pandas DataFrame. Got type: {type(df_pandas)}")
                raise TypeError()
                
            if empty_as_nan:
                # Replace empty strings in string columns with NaN, object dtype columns are not considered for safe replacement in parquet.
                str_cols = df_pandas.select_dtypes(include=['string']).columns
                if not str_cols.empty:
                    df_pandas[str_cols] = df_pandas[str_cols].replace(r'^\s*$', np.nan, regex=True)
                    
            df = df_pandas

        elif kind == "polars":
            df_polars = pl.read_parquet(path, columns=use_columns)
            
            if empty_as_nan:
                df_polars = df_polars.with_columns(
                    pl.when(pl.col(pl.String).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(pl.String))
                    .name.keep()
                )
                
            df = df_polars
            
        else:
            _LOGGER.error(f"Invalid kind '{kind}'. Must be one of 'pandas' or 'polars'.")
            raise ValueError()
            
    except Exception as e:
        _LOGGER.error(f"Failed to load '{df_name}': {e}")
        raise

    if df.shape[0] == 0:
        _LOGGER.error(f"DataFrame '{df_name}' loaded from '{path}' is empty.")
        raise ValueError()

    if verbose:
        _LOGGER.info(f"💾 Loaded {kind.upper()} parquet dataset: '{df_name}' with shape: {df.shape}")
    
    return df, df_name # type: ignore


def load_parquet_header(df_path: Union[str, Path], verbose: int = 3) -> tuple[str, ...]:
    """
    Loads the header of a Parquet file and returns the column names as a tuple of strings.
    
    Args:
        df_path (str, Path): The path to the Parquet file.
        verbose (int): Logs the loaded header information.

    Returns:
        tuple[str, ...]: A tuple of column names.
    """
    path = make_fullpath(df_path, enforce="file")
    try:
        # Polars lazy scanning is extremely fast for reading just the schema from the metadata
        header_tuple = tuple(pl.scan_parquet(path).columns)
    except Exception as e:
        _LOGGER.error(f"Failed to load header from '{path}': {e}")
        raise
    
    if not header_tuple:
        if verbose >= 1:
            _LOGGER.warning(f"No columns found in the Parquet file '{path}'. The header is empty.")
        return tuple()
    
    if verbose >= 3:
        header_str = '\n\t'.join(header_tuple)
        _LOGGER.info(f"📃 Loaded header with {len(header_tuple)} columns from '{path}':\n\t{header_str}")
    elif verbose >= 2:
        _LOGGER.info(f"📃 Loaded header with {len(header_tuple)} columns.")
    
    return header_tuple


def load_parquet_with_schema(
    df_path: Union[str, Path], 
    schema: "FeatureSchema",
    empty_as_nan: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Loads a Parquet file into a Pandas DataFrame, strictly validating its
    feature columns against a FeatureSchema.

    This function wraps `load_parquet_dataframe`. After loading, it validates
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
            The path to the Parquet file.
        schema (FeatureSchema): 
            The schema object to validate against.
        empty_as_nan (bool):
            If True, converts empty strings to NaN.

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
    try:
        df, df_name = load_parquet_dataframe(
            df_path=df_path, 
            use_columns=None, 
            kind="pandas", 
            empty_as_nan=empty_as_nan,
            verbose=True
        )
    except Exception as e:
        _LOGGER.error(f"Failed during initial load for schema validation: {e}")
        raise e
    
    # Validate and reorder using the helper imported from _utility_save_load
    df_validated = _validate_and_reorder_schema(df=df, schema=schema)

    return df_validated, df_name


def save_parquet_dataframe_filename(
    df: Union[pd.DataFrame, pl.DataFrame], 
    save_dir: Union[str, Path], 
    filename: str, 
    empty_as_nan: bool = True,
    verbose: int = 3
) -> None:
    """
    Saves a pandas or polars DataFrame to a Parquet file.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): 
            The DataFrame to save.
        save_dir (Union[str, Path]): 
            The directory where the Parquet file will be saved.
        filename (str): 
            The Parquet filename. The '.parquet' extension will be added if missing.
        empty_as_nan (bool):
            If True, converts empty strings to NaN (for pandas) or Null (for polars) before saving.
        verbose (int): 
            Verbosity level for logging.
                - 0: Error level
                - 1: Warning level
                - 2: Info level
                - 3: Detailed process info
    """
    if df.shape[0] == 0:
        _LOGGER.warning(f"Attempting to save an empty DataFrame: '{filename}'. Process Skipped.")
        return
    
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    filename = sanitize_filename(filename)
    
    if not filename.lower().endswith('.parquet'):
        filename += '.parquet'
        
    output_path = save_path / filename
        
    if isinstance(df, pd.DataFrame):
        if empty_as_nan:
            # Replace empty strings in string columns with NaN before saving, object dtype columns are not considered for safe replacement in parquet.
            str_cols = df.select_dtypes(include=['string']).columns
            if not str_cols.empty:
                df_to_save = df.copy()
                df_to_save[str_cols] = df_to_save[str_cols].replace(r'^\s*$', np.nan, regex=True)
            else:
                df_to_save = df
        else:
            df_to_save = df
            
        df_to_save.to_parquet(output_path, engine='pyarrow', index=False)
        
    elif isinstance(df, pl.DataFrame):
        if empty_as_nan:
            df_to_save = df.with_columns(
                pl.when(pl.col(pl.String).str.strip_chars() == "")
                .then(None)
                .otherwise(pl.col(pl.String))
                .name.keep()
            )
        else:
            df_to_save = df
            
        df_to_save.write_parquet(output_path)
        
    else:
        _LOGGER.error(f"Unsupported DataFrame type: {type(df)}. Must be pandas or polars.")
        raise TypeError()
    
    if verbose >= 2:
        _LOGGER.info(f"💾 Saved parquet dataset: '{filename}' with shape: {df_to_save.shape}")


def save_parquet_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame], 
    full_path: Path, 
    empty_as_nan: bool = True,
    verbose: int = 3
) -> None:
    """
    Saves a DataFrame to a specified full path.

    This function is a wrapper for `save_parquet_dataframe_filename()`. It takes a
    single `pathlib.Path` object pointing to a `.parquet` file.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): The pandas or polars DataFrame to save.
        full_path (Path): The complete file path, including the filename and `.parquet` extension, where the DataFrame will be saved.
        empty_as_nan (bool): If True, converts empty strings to NaN (for pandas) or Null (for polars) before saving.
        verbose (int): Verbosity level for logging.
            - 0: Error level
            - 1: Warning level
            - 2: Info level
            - 3: Detailed process info
    """
    if not isinstance(full_path, Path) or not full_path.suffix.endswith(".parquet"):
        _LOGGER.error('A path object pointing to a .parquet file must be provided.')
        raise ValueError()

    save_parquet_dataframe_filename(
        df=df, 
        save_dir=full_path.parent,
        filename=full_path.name,
        empty_as_nan=empty_as_nan,
        verbose=verbose
    )


def save_parquet_with_schema(
    df: pd.DataFrame, 
    full_path: Path,
    schema: "FeatureSchema",
    empty_as_nan: bool = True,
    verbose: int = 3
) -> None:
    """
    Saves a pandas DataFrame to a Parquet file, strictly enforcing that the
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
        empty_as_nan (bool):
            If True, converts empty strings to NaN before saving.
        verbose (int): 
            Verbosity level for logging.
                - 0: Error level
                - 1: Warning level
                - 2: Info level
                - 3: Detailed process info
    """
    if not isinstance(full_path, Path) or not full_path.suffix.endswith(".parquet"):
        _LOGGER.error('A path object pointing to a .parquet file must be provided.')
        raise ValueError()
    
    df_to_save = _validate_and_reorder_schema(df=df, schema=schema, verbose=verbose)
    
    save_parquet_dataframe_filename(
        df=df_to_save, 
        save_dir=full_path.parent,
        filename=full_path.name,
        empty_as_nan=empty_as_nan,
        verbose=verbose
    )
