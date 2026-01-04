import json
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Literal

from ..path_manager import make_fullpath
from .._core import get_logger

from ._utility_save_load import load_dataframe


_LOGGER = get_logger("Translation Tools")


__all__ = [
    "translate_dataframe_columns",
    "create_translation_template",
    "audit_column_translation"
]


def translate_dataframe_columns(
    df: Union[pd.DataFrame, pl.DataFrame],
    mapper: Union[dict[str, str], str, Path],
    direction: Literal["A_to_B", "B_to_A"] = "A_to_B",
    verbose: int = 3
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Translates the column names of a DataFrame (Pandas or Polars) using a provided mapping source.

    The mapping can be a python dictionary, a JSON file, or a CSV file.
    
    Translation Logic:
    -----------------
    The DataFrame currently has columns in 'Language A'.
    
    - "A_to_B" (Standard): 
      The mapper is structured as {Language A : Language B}.
      Keys match the current DataFrame columns.
      
    - "B_to_A" (Inverted Source):
      The mapper is structured as {Language B : Language A}.
      Values match the current DataFrame columns.

    Parameters
    ----------
    df : (pd.DataFrame | pl.DataFrame)
        The input DataFrame to be translated.
    mapper : (dict[str, str] | str | Path)
        The source of the translation mapping:
        - Dict: {'original_name': 'new_name'}
        - JSON path: File containing a single JSON object (dict).
        - CSV path: File with two columns.
    direction : Literal["A_to_B", "B_to_A"]
        Specifies the structure of the provided mapper relative to the DataFrame.
    verbose : int
        Whether to log warnings and information about the process.

    Returns
    -------
    Dataframe:
        The polars or pandas DataFrame with renamed columns.
    """
    # df type validation
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        _LOGGER.error(f"Input df must be a pandas or polars DataFrame. Got: {type(df)}")
        raise TypeError()
    
    # 1. Load and Standardize the Mapping
    translation_map = _load_translation_mapping(mapper, direction)

    # 2. Validation: Check intersection between DF columns and Map keys
    df_cols = set(df.columns)
    map_keys = set(translation_map.keys())
    
    # Calculate overlap
    common_cols = df_cols.intersection(map_keys)
    
    if not common_cols:
        if verbose >= 1:
            _LOGGER.warning("No column names matched the provided translation mapping. Returning original DataFrame.")
        return df

    missing_in_map = df_cols - map_keys
    if missing_in_map and verbose >= 1:
        _LOGGER.warning(f"Columns not found in translation map: {list(missing_in_map)}")
    
    if verbose >= 3:
        _LOGGER.info(f"Translating {len(common_cols)} columns...")

    # 3. Apply Translation
    try:
        if isinstance(df, pd.DataFrame):
            resulting_df = df.rename(columns=translation_map)
        elif isinstance(df, pl.DataFrame):
            resulting_df = df.rename(translation_map, strict=False)
    except Exception as e:
        _LOGGER.error(f"Failed to rename columns: {e}")
        raise e
    else:
        if verbose >= 2:
            _LOGGER.info(f"Successfully translated {len(common_cols)} columns.")
        return resulting_df


def create_translation_template(
    df_or_path: Union[pd.DataFrame, pl.DataFrame, str, Path],
    save_path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Generates a JSON translation template from a DataFrame's column names.
    
    Creates a 'translation_template.json' file where keys are the dataframe column names and values 
    are empty strings, ready for manual translation.

    Parameters
    ----------
    df_or_path : [DataFrame | str | Path]
        The DataFrame or path to a CSV file to extract column names from.
    save_path : [str | Path]
        The destination directory for the .json template.
    """
    # 1. Get Columns
    if isinstance(df_or_path, (str, Path)):
        df, _ = load_dataframe(df_or_path, kind="pandas", verbose=False)
        columns = df.columns.tolist()
    elif isinstance(df_or_path, pd.DataFrame):
        columns = df_or_path.columns.tolist()
    elif isinstance(df_or_path, pl.DataFrame):
        columns = df_or_path.columns
    else:
        _LOGGER.error("Input must be a DataFrame or a path to a dataset.")
        raise TypeError()

    # 2. Create Dictionary {ColName : ""}
    template_dict = {col: "" for col in columns}
    
    # 3. Save to JSON
    out_path = make_fullpath(save_path, enforce="directory")
    full_out_path = out_path / "translation_template.json"
    
    try:
        with open(full_out_path, 'w', encoding='utf-8') as f:
            json.dump(template_dict, f, indent=4, ensure_ascii=False)
        
        if verbose:
            _LOGGER.info(f"Translation template created at '{out_path.name}' with {len(columns)} entries.")
    except Exception as e:
        _LOGGER.error(f"Failed to save template: {e}")
        raise e


def audit_column_translation(
    df_or_path: Union[pd.DataFrame, pl.DataFrame, str, Path],
    mapper: Union[dict[str, str], str, Path],
    direction: Literal["A_to_B", "B_to_A"] = "A_to_B"
) -> None:
    """
    Audits the coverage of a translation map against a DataFrame WITHOUT applying changes.
    
    Logs a detailed report of:
    - How many columns will be renamed.
    - Which DataFrame columns are NOT in the map (will remain unchanged).
    - Which Map keys are NOT in the DataFrame (unused mappings).

    Parameters
    ----------
    df_or_path : [DataFrame | str | Path]
        The target dataset to audit.
    mapper : [Dict | str | Path]
        The translation source.
    direction : ["A_to_B" | "B_to_A"]
        Direction logic (see translate_dataframe_columns).
    """
    # 1. Get DataFrame Columns
    if isinstance(df_or_path, (str, Path)):
        df, df_name = load_dataframe(df_or_path, kind="pandas", verbose=False)
        cols = set(df.columns)
        source_name = f"File: '{df_name}'"
    elif isinstance(df_or_path, pd.DataFrame):
        cols = set(df_or_path.columns)
        source_name = "DataFrame (Pandas)"
    elif isinstance(df_or_path, pl.DataFrame):
        cols = set(df_or_path.columns)
        source_name = "DataFrame (Polars)"
    else:
        _LOGGER.error("Input must be a DataFrame or a path to a dataset.")
        raise TypeError()

    # 2. Load Map
    try:
        trans_map = _load_translation_mapping(mapper, direction)
        map_keys = set(trans_map.keys())
    except Exception as e:
        _LOGGER.error(f"Could not load mapper. {e}")
        return

    # 3. Analyze Sets
    matched = cols.intersection(map_keys)
    missing_in_map = cols - map_keys
    unused_map_keys = map_keys - cols
    
    coverage_pct = (len(matched) / len(cols) * 100) if len(cols) > 0 else 0.0

    # 4. Report
    report_string = f"\n--- 🔍 Translation Audit Report: {source_name} ---\n \
    Direction: {direction}\n \
    Total Columns: {len(cols)}\n \
    Map Coverage: {coverage_pct:.1f}%\n"
    
    if matched:
        report_string += f"\n✅ Will Translate: {len(matched)} columns"
        
    if missing_in_map:
        report_string += f"\n⚠️  Not in Map: {len(missing_in_map)} columns: {list(missing_in_map)}"
        
    if unused_map_keys:
        report_string += f"\n➡️ Unused Map Keys: {len(unused_map_keys)}"

    _LOGGER.info(report_string)


def _load_translation_mapping(
    source: Union[dict[str, str], str, Path],
    direction: Literal["A_to_B", "B_to_A"]
) -> dict[str, str]:
    """
    Internal helper to load mapping from Dict, JSON, or CSV and handle direction inversion.
    """
    raw_map: dict[str, str] = {}

    # --- Load Source ---
    if isinstance(source, dict):
        raw_map = source.copy()

    elif isinstance(source, (str, Path)):
        path = make_fullpath(source, enforce="file")
        
        if path.suffix.lower() == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if not isinstance(content, dict):
                    _LOGGER.error(f"JSON file '{path.name}' does not contain a dictionary.")
                    raise ValueError()
                raw_map = content

        elif path.suffix.lower() == ".csv":
            # Load CSV using pandas for robustness
            try:
                df_map = pd.read_csv(path)
                
                # STRICT VALIDATION: Must be exactly 2 columns
                if df_map.shape[1] != 2:
                    _LOGGER.error(f"CSV file '{path.name}' must have exactly 2 columns for mapping. Found {df_map.shape[1]}.")
                    raise ValueError()
                
                key_col = df_map.columns[0]
                val_col = df_map.columns[1]
                
                # Convert to dictionary (drop NaNs to be safe)
                raw_map = df_map.dropna(subset=[key_col, val_col]).set_index(key_col)[val_col].to_dict()
                
            except Exception as e:
                _LOGGER.error(f"Error reading CSV mapping file: {e}")
                raise e
        else:
            _LOGGER.error(f"Unsupported file extension for mapping source: {path.suffix}")
            raise ValueError()
    else:
        _LOGGER.error("Mapper must be a Dictionary, or a Path/String to a JSON/CSV file.")
        raise TypeError()

    # --- Handle Direction ---
    # Case: The mapper is A->B, and DF is A. (Keys match DF). Return as is.
    if direction == "A_to_B":
        return raw_map
    
    # Case: The mapper is B->A, but DF is A. (Values match DF).
    # swap the mapper to A->B so the Keys match the DF.
    elif direction == "B_to_A":
        # Inversion requires unique values to be lossless
        reversed_map = {v: k for k, v in raw_map.items()}
        
        if len(reversed_map) < len(raw_map):
            _LOGGER.warning("Direction 'B_to_A' resulted in fewer keys than original. Duplicate target values existed in the source map; some collisions were overwritten.")
            
        return reversed_map
    
    else:
        _LOGGER.error("Direction must be 'A_to_B' or 'B_to_A'.")
        raise ValueError()
