import polars as pl
from pathlib import Path
from typing import Union

from ..data_exploration import show_null_columns
from ..utilities import save_dataframe_filename, load_dataframe

from ..path_manager import make_fullpath
from .._core import get_logger

from ._dragon_cleaner import DragonColumnCleaner, DragonDataFrameCleaner

_LOGGER = get_logger("ETL Basic Clean")


__all__ = [
    "basic_clean",
    "basic_clean_drop",
    "drop_macro_polars",
]


########## Basic cleaners #############
def _cleaner_core(df_in: pl.DataFrame, all_lowercase: bool) -> pl.DataFrame:
    # Cleaning rules
    cleaning_rules = {
        # 1. Comprehensive Punctuation & Symbol Normalization
        # Remove invisible control characters
        r'\p{C}+': '',
        
        # Full-width to half-width
        # Numbers
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        # Superscripts & Subscripts
        '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
        '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁰': '0',
        '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
        '₆': '6', '₇': '7', '₈': '8', '₉': '9', '₀': '0',
        '⁺': '', '⁻': '', '₊': '', '₋': '',
        # Uppercase Alphabet
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F',
        'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L',
        'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R',
        'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X',
        'Ｙ': 'Y', 'Ｚ': 'Z',
        # Lowercase Alphabet
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f',
        'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l',
        'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r',
        'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
        'ｙ': 'y', 'ｚ': 'z',
        # Punctuation
        '》': '>', '《': '<', '：': ':', '。': '.', '；': ';', '【': '[', '】': ']', '∼': '~',
        '（': '(', '）': ')', '？': '?', '！': '!', '～': '~', '＠': '@', '＃': '#', '＋': '+', '－': '-',
        '＄': '$', '％': '%', '＾': '^', '＆': '&', '＊': '*', '＼': '-', '｜': '|', '≈':'=', '·': '', '⋅': '',
        '¯': '-', '＿': '-',
        
        # Commas (avoid commas in entries)
        '，': ';',
        ',': ';',
        '、':';',
        
        # Others
        'σ': '',
        '□': '',
        '©': '',
        '®': '',
        '™': '',
        r'[°˚]': '',
        
        # Replace special characters in entries
        r'\\': '_',
        
        # Typographical standardization
        # Unify various dashes and hyphens to a standard hyphen
        r'[—–―]': '-',
        r'−': '-',
        # remove various quote types
        r'[“”"]': '',
        r"[‘’′']": '',
        
        # Collapse repeating punctuation
        r'\.{2,}': '.',      # Replace two or more dots with a single dot
        r'\?{2,}': '?',      # Replace two or more question marks with a single question mark
        r'!{2,}': '!',      # Replace two or more exclamation marks with a single one
        r';{2,}': ';',
        r'-{2,}': '-',
        r'/{2,}': '/',
        r'%{2,}': '%',
        r'&{2,}': '&',

        # 2. Internal Whitespace Consolidation
        # Collapse any sequence of whitespace chars (including non-breaking spaces) to a single space
        r'\s+': ' ',

        # 3. Leading/Trailing Whitespace Removal
        # Strip any whitespace from the beginning or end of the string
        r'^\s+|\s+$': '',
        
        # 4. Textual Null Standardization (New Step)
        # Convert common null-like text to actual nulls.
        r'^(N/A|无|NA|NULL|NONE|NIL|-|\.|;|/|%|&)$': None,

        # 5. Final Nullification of Empty Strings
        # After all cleaning, if a string is now empty, convert it to a null
        r'^\s*$': None,
        r'^$': None,
    }
    
    # Clean data
    try:
        # Create a cleaner for every column in the dataframe
        all_columns = df_in.columns
        column_cleaners = [
            DragonColumnCleaner(col, rules=cleaning_rules, case_insensitive=True) for col in all_columns
        ]
        
        # Instantiate and run the main dataframe cleaner
        df_cleaner = DragonDataFrameCleaner(cleaners=column_cleaners)
        df_cleaned = df_cleaner.clean(df_in) 
        
        # apply lowercase to all string columns
        if all_lowercase:
            df_final = df_cleaned.with_columns(
                pl.col(pl.String).str.to_lowercase()
            )
        else:
            df_final = df_cleaned

    except Exception as e:
        _LOGGER.error(f"An error occurred during the cleaning process.")
        raise e
    else:
        return df_final


def _local_path_manager(path_in: Union[str,Path], path_out: Union[str,Path]):
    # Handle paths
    input_path = make_fullpath(path_in, enforce="file")
    
    parent_dir = make_fullpath(Path(path_out).parent, make=True, enforce="directory")
    output_path = parent_dir / Path(path_out).name
    
    return input_path, output_path


def basic_clean(input_filepath: Union[str,Path], output_filepath: Union[str,Path], all_lowercase: bool=False):
    """
    Performs a comprehensive, standardized cleaning on all columns of a CSV file.

    The cleaning process includes:
    - Normalizing full-width and typographical punctuation to standard equivalents.
    - Consolidating all internal whitespace (spaces, tabs, newlines) into a single space.
    - Stripping any leading or trailing whitespace.
    - Converting common textual representations of null (e.g., "N/A", "NULL") to true null values.
    - Converting strings that become empty after cleaning into true null values.
    - Normalizing all text to lowercase (Optional).

    Args:
        input_filepath (str | Path):
            The path to the source CSV file to be cleaned.
        output_filepath (str | Path):
            The path to save the cleaned CSV file.
        all_lowercase (bool):
            Whether to normalize all text to lowercase.
        
    """
    # Handle paths
    input_path, output_path = _local_path_manager(path_in=input_filepath, path_out=output_filepath)
        
    # load polars df
    df, _ = load_dataframe(df_path=input_path, kind="polars", all_strings=True)
    
    # CLEAN
    df_final = _cleaner_core(df_in=df, all_lowercase=all_lowercase)
    
    # Save cleaned dataframe
    save_dataframe_filename(df=df_final, save_dir=output_path.parent, filename=output_path.name)
    
    _LOGGER.info(f"Data successfully cleaned.")
    

def basic_clean_drop(input_filepath: Union[str,Path], 
                     output_filepath: Union[str,Path], 
                     log_directory: Union[str,Path], 
                     targets: list[str], 
                     skip_targets: bool=False, 
                     threshold: float=0.8, 
                     all_lowercase: bool=False):
    """
    Performs standardized cleaning followed by iterative removal of rows and 
    columns with excessive missing data.

    This function combines the functionality of `basic_clean` and `drop_macro_polars`. It first 
    applies a comprehensive normalization process to all columns in the input CSV file.
    Then it applies iterative row and column dropping to remove redundant or incomplete data.

    Args:
        input_filepath (str | Path):
            The path to the source CSV file to be cleaned.
        output_filepath (str | Path):
            The path to save the fully cleaned CSV file after cleaning 
            and missing-data-based pruning.
        log_directory (str | Path):
            Path to the directory where missing data reports will be stored.
        targets (list[str]):
            A list of column names to be treated as target variables. 
            This list guides the row-dropping logic.
        skip_targets (bool):
            If True, the columns listed in `targets` will be exempt from being dropped, 
            even if they exceed the missing data threshold.
        threshold (float):
            The proportion of missing data required to drop a row or column. 
            For example, 0.8 means a row/column will be dropped if 80% or more 
            of its data is missing.
        all_lowercase (bool):
            Whether to normalize all text to lowercase.
    """
    # handle log path
    log_path = make_fullpath(log_directory, make=True, enforce="directory")
    
    # Handle df paths
    input_path, output_path = _local_path_manager(path_in=input_filepath, path_out=output_filepath)
    
    # load polars df
    df, _ = load_dataframe(df_path=input_path, kind="polars", all_strings=True)
    
    # CLEAN
    df_cleaned = _cleaner_core(df_in=df, all_lowercase=all_lowercase)
    
    # Drop macro (Polars implementation)
    df_final = drop_macro_polars(df=df_cleaned,
                                  log_directory=log_path,
                                  targets=targets,
                                  skip_targets=skip_targets,
                                  threshold=threshold)
    
    # Save cleaned dataframe
    save_dataframe_filename(df=df_final, save_dir=output_path.parent, filename=output_path.name)
    
    _LOGGER.info(f"Data successfully cleaned.")


########## EXTRACT and CLEAN ##########
def _generate_null_report(df: pl.DataFrame, save_dir: Path, filename: str):
    """
    Internal helper to generate and save a CSV report of missing data percentages using Polars.
    """
    total_rows = df.height
    if total_rows == 0:
        return

    null_stats = df.null_count()
    
    # Construct a report DataFrame
    report = pl.DataFrame({
        "column": df.columns,
        "null_count": null_stats.transpose().to_series(),
    }).with_columns(
        (pl.col("null_count") / total_rows * 100).round(2).alias("missing_percent")
    ).sort("missing_percent", descending=True)
    
    save_dataframe_filename(df=report, save_dir=save_dir, filename=filename, verbose=2)


def drop_macro_polars(df: pl.DataFrame, 
                       log_directory: Path, 
                       targets: list[str], 
                       skip_targets: bool, 
                       threshold: float) -> pl.DataFrame:
    """
    High-performance implementation of iterative row/column pruning using Polars.
    Includes temporary Pandas conversion for visualization.
    """
    df_clean = df.clone()
    
    # --- Helper to generate plot safely ---
    def _plot_safe(df_pl: pl.DataFrame, filename: str):
        try:
            # converting to pandas just for the plot
            # use_pyarrow_extension_array=True is  faster
            df_pd = df_pl.to_pandas(use_pyarrow_extension_array=True)
            show_null_columns(df_pd, plot_to_dir=log_directory, plot_filename=filename, use_all_columns=True)
        except Exception as e:
            _LOGGER.warning(f"Skipping plot generation due to error: {e}")
    
    # 1. Log Initial State
    _generate_null_report(df_clean, log_directory, "Missing_Data_Original")
    _plot_safe(df_clean, "Original")
    
    master = True
    while master:
        initial_rows, initial_cols = df_clean.shape
        
        # --- A. Drop Constant Columns ---
        # Keep columns where n_unique > 1. 
        # Note: n_unique in Polars ignores nulls by default (similar to pandas dropna=True).
        # We assume if a column is all nulls, it should also be dropped (n_unique=0).
        cols_to_keep = [
            col for col in df_clean.columns 
            if df_clean[col].n_unique() > 1
        ]
        df_clean = df_clean.select(cols_to_keep)
        
        # --- B. Drop Rows (Targets) ---
        # Drop rows where ALL target columns are null
        valid_targets = [t for t in targets if t in df_clean.columns]
        if valid_targets:
            df_clean = df_clean.filter(
                ~pl.all_horizontal(pl.col(valid_targets).is_null())
            )
            
        # --- C. Drop Rows (Features Threshold) ---
        # Drop rows where missing data fraction in FEATURE columns > threshold
        feature_cols = [c for c in df_clean.columns if c not in valid_targets]
        if feature_cols:
            # We want to KEEP rows where (null_count / total_features) <= threshold
            df_clean = df_clean.filter(
                (pl.sum_horizontal(pl.col(feature_cols).is_null()) / len(feature_cols)) <= threshold
            )
            
        # --- D. Drop Columns (Threshold) ---
        # Drop columns where missing data fraction > threshold
        current_height = df_clean.height
        if current_height > 0:
            null_counts = df_clean.null_count().row(0) # tuple of counts
            cols_to_drop = []
            
            for col_idx, col_name in enumerate(df_clean.columns):
                # Check if we should skip this column (if it's a target and skip_targets=True)
                if skip_targets and col_name in valid_targets:
                    continue
                
                missing_frac = null_counts[col_idx] / current_height
                if missing_frac > threshold:
                    cols_to_drop.append(col_name)
            
            if cols_to_drop:
                df_clean = df_clean.drop(cols_to_drop)

        # --- E. Check Convergence ---
        remaining_rows, remaining_cols = df_clean.shape
        if remaining_rows >= initial_rows and remaining_cols >= initial_cols:
            master = False

    # 2. Log Final State
    _generate_null_report(df_clean, log_directory, "Missing_Data_Processed")
    _plot_safe(df_clean, "Processed")
    
    return df_clean
