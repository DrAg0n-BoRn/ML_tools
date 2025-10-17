import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
from .path_manager import sanitize_filename, make_fullpath
from .data_exploration import drop_macro
from .utilities import save_dataframe, load_dataframe
from ._script_info import _script_info
from ._logger import _LOGGER


__all__ = [
    "save_unique_values",
    "basic_clean",
    "basic_clean_drop",
    "ColumnCleaner",
    "DataFrameCleaner"
]


################ Unique Values per column #################
def save_unique_values(csv_path: Union[str, Path], 
                       output_dir: Union[str, Path], 
                       verbose: bool=False,
                       keep_column_order: bool = True) -> None:
    """
    Loads a CSV file, then analyzes it and saves the unique non-null values
    from each column into a separate text file exactly as they appear.

    This is useful for understanding the raw categories or range of values
    within a dataset before and after cleaning.

    Args:
        csv_path (str | Path):
            The file path to the input CSV file.
        output_dir (str | Path):
            The path to the directory where the .txt files will be saved.
            The directory will be created if it does not exist.
        keep_column_order (bool):
            If True, prepends a numeric prefix (e.g., '1_', '2_') to each
            output filename to maintain the original column order.
    """
    # --- 1. Input Validation ---
    csv_path = make_fullpath(input_path=csv_path, enforce="file")
    output_dir = make_fullpath(input_path=output_dir, make=True)

    # --- 2. Load Data ---
    try:
        # Load all columns as strings to preserve original formatting
        df = pd.read_csv(csv_path, dtype=str, encoding='utf-8')
    except FileNotFoundError as e:
        _LOGGER.error(f"The file was not found at '{csv_path}'.")
        raise e
    except Exception as e2:
        _LOGGER.error(f"An error occurred while reading the CSV file.")
        raise e2
    else:
        _LOGGER.info(f"Data loaded from '{csv_path}'")
        
    # --- 3. Process Each Column ---
    counter = 0
    for i, column_name in enumerate(df.columns):
        # _LOGGER.info(f"Processing column: '{column_name}'...")

        # --- Get unique values AS IS ---
        try:
            # Drop nulls, get unique values, and sort them.
            # The values are preserved exactly as they are in the cells.
            unique_values = df[column_name].dropna().unique()
            sorted_uniques = sorted(unique_values)
        except Exception:
            _LOGGER.exception(f"Could not process column '{column_name}'.")
            continue

        if not sorted_uniques:
            _LOGGER.warning(f"Column '{column_name}' has no unique non-null values. Skipping.")
            continue

        # --- Sanitize column name to create a valid filename ---
        sanitized_name = sanitize_filename(column_name)
        if not sanitized_name.strip('_'):
            sanitized_name = f'column_{i}'
        
        # --- create filename prefix ---
        # If keep_column_order is True, create a prefix like "1_", "2_", etc.
        prefix = f"{i + 1}_" if keep_column_order else ''
        
        file_path = output_dir / f"{prefix}{sanitized_name}_unique_values.txt"

        # --- Write to file ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Unique values for column: '{column_name}'\n")
                f.write(f"# Total unique non-null values: {len(sorted_uniques)}\n")
                f.write("-" * 30 + "\n")
                for value in sorted_uniques:
                    f.write(f"{value}\n")
                    f.write("-" * 30 + "\n")
        except IOError:
            _LOGGER.exception(f"Error writing to file {file_path}.")
        else:
            if verbose:
                _LOGGER.info(f"Successfully saved {len(sorted_uniques)} unique values from '{column_name}'.")
            counter += 1

    _LOGGER.info(f"{counter} files of unique values created.")


########## Basic df cleaners #############
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
        '¯': '-',
        
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
            ColumnCleaner(col, rules=cleaning_rules, case_insensitive=True) for col in all_columns
        ]
        
        # Instantiate and run the main dataframe cleaner
        df_cleaner = DataFrameCleaner(cleaners=column_cleaners)
        df_cleaned = df_cleaner.clean(df_in, clone_df=False) # Use clone_df=False for efficiency
        
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


def _path_manager(path_in: Union[str,Path], path_out: Union[str,Path]):
    # Handle paths
    input_path = make_fullpath(path_in, enforce="file")
    
    parent_dir = make_fullpath(Path(path_out).parent, make=True, enforce="directory")
    output_path = parent_dir / Path(path_out).name
    
    return input_path, output_path


def basic_clean(input_filepath: Union[str,Path], output_filepath: Union[str,Path], all_lowercase: bool=True):
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
    input_path, output_path = _path_manager(path_in=input_filepath, path_out=output_filepath)
        
    # load polars df
    df, _ = load_dataframe(df_path=input_path, kind="polars", all_strings=True)
    
    # CLEAN
    df_final = _cleaner_core(df_in=df, all_lowercase=all_lowercase)
    
    # Save cleaned dataframe
    save_dataframe(df=df_final, save_dir=output_path.parent, filename=output_path.name)
    
    _LOGGER.info(f"Data successfully cleaned.")
    

def basic_clean_drop(input_filepath: Union[str,Path], output_filepath: Union[str,Path], log_directory: Union[str,Path], targets: list[str], 
                     skip_targets: bool=False, threshold: float=0.8, all_lowercase: bool=True):
    """
    Performs standardized cleaning followed by iterative removal of rows and 
    columns with excessive missing data.

    This function combines the functionality of `basic_clean` and `drop_macro`. It first 
    applies a comprehensive normalization process to all columns in the input CSV file, 
    ensuring consistent formatting and proper null value handling. The cleaned data is then 
    converted to a pandas DataFrame, where iterative row and column dropping is applied 
    to remove redundant or incomplete data.  

    The iterative dropping cycle continues until no further rows or columns meet the 
    removal criteria, ensuring that dependencies between row and column deletions are 
    fully resolved. Logs documenting the missing data profile before and after the 
    dropping process are saved to the specified log directory.  

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
    input_path, output_path = _path_manager(path_in=input_filepath, path_out=output_filepath)
    
    # load polars df
    df, _ = load_dataframe(df_path=input_path, kind="polars", all_strings=True)
    
    # CLEAN
    df_cleaned = _cleaner_core(df_in=df, all_lowercase=all_lowercase)
    
    # switch to pandas
    df_cleaned_pandas = df_cleaned.to_pandas()
    
    # Drop macro
    df_final = drop_macro(df=df_cleaned_pandas,
                          log_directory=log_path,
                          targets=targets,
                          skip_targets=skip_targets,
                          threshold=threshold)
    
    # Save cleaned dataframe
    save_dataframe(df=df_final, save_dir=output_path.parent, filename=output_path.name)
    
    _LOGGER.info(f"Data successfully cleaned.")


########## EXTRACT and CLEAN ##########
class ColumnCleaner:
    """
    A configuration object that defines cleaning rules for a single Polars DataFrame column.

    This class holds a dictionary of regex-to-replacement rules, the target column name,
    and the case-sensitivity setting. It is intended to be used with the DataFrameCleaner.
    
    Notes:
        - Define rules from most specific to more general to create a fallback system.
        - Beware of chain replacements (rules matching strings that have already been
          changed by a previous rule in the same cleaner).

    Args:
        column_name (str):
            The name of the column to be cleaned.
        rules (Dict[str, str]):
            A dictionary of regex patterns to replacement strings. Can use
            backreferences (e.g., r'$1 $2') for captured groups. Note that Polars
            uses a '$' prefix for backreferences.
        case_insensitive (bool):
            If True (default), regex matching ignores case.

    ## Usage Example

    ```python
    id_rules = {
        # Matches 'ID-12345' or 'ID 12345' and reformats to 'ID:12345'
        r'ID[- ](\\d+)': r'ID:$1'
    }

    id_cleaner = ColumnCleaner(column_name='user_id', rules=id_rules)
    # This object would then be passed to a DataFrameCleaner.
    ```
    """
    def __init__(self, column_name: str, rules: Dict[str, str], case_insensitive: bool = True):
        if not isinstance(column_name, str) or not column_name:
            _LOGGER.error("The 'column_name' must be a non-empty string.")
            raise TypeError()
        if not isinstance(rules, dict):
            _LOGGER.error("The 'rules' argument must be a dictionary.")
            raise TypeError()

        self.column_name = column_name
        self.rules = rules
        self.case_insensitive = case_insensitive


class DataFrameCleaner:
    """
    Orchestrates cleaning multiple columns in a Polars DataFrame.

    This class takes a list of ColumnCleaner objects and applies their defined
    rules to the corresponding columns of a DataFrame using high-performance
    Polars expressions.

    Args:
        cleaners (List[ColumnCleaner]):
            A list of ColumnCleaner configuration objects.

    Raises:
        TypeError: If 'cleaners' is not a list or contains non-ColumnCleaner objects.
        ValueError: If multiple ColumnCleaner objects target the same column.
    """
    def __init__(self, cleaners: List[ColumnCleaner]):
        if not isinstance(cleaners, list):
            _LOGGER.error("The 'cleaners' argument must be a list of ColumnCleaner objects.")
            raise TypeError()

        seen_columns = set()
        for cleaner in cleaners:
            if not isinstance(cleaner, ColumnCleaner):
                _LOGGER.error(f"All items in 'cleaners' list must be ColumnCleaner objects, but found an object of type {type(cleaner).__name__}.")
                raise TypeError()
            if cleaner.column_name in seen_columns:
                _LOGGER.error(f"Duplicate ColumnCleaner found for column '{cleaner.column_name}'. Each column should only have one cleaner.")
                raise ValueError()
            seen_columns.add(cleaner.column_name)

        self.cleaners = cleaners

    def clean(self, df: pl.DataFrame, clone_df: bool=True) -> pl.DataFrame:
        """
        Applies all defined cleaning rules to the Polars DataFrame.

        Args:
            df (pl.DataFrame): The Polars DataFrame to clean.
            clone_df (bool): Whether to work on a clone to prevent undesired changes.

        Returns:
            pl.DataFrame: A new, cleaned Polars DataFrame.

        Raises:
            ValueError: If any columns specified in the cleaners are not found
                        in the input DataFrame.
        """
        rule_columns = {c.column_name for c in self.cleaners}
        df_columns = set(df.columns)
        missing_columns = rule_columns - df_columns

        if missing_columns:
            _LOGGER.error("The following columns specified in cleaning rules were not found in the DataFrame:")
            for miss_col in sorted(list(missing_columns)):
                print(f"\t- {miss_col}")
            raise ValueError()

        if clone_df:
            df_cleaned = df.clone()
        else:
            df_cleaned = df
        
        # Build and apply a series of expressions for each column
        for cleaner in self.cleaners:
            col_name = cleaner.column_name
            
            # Start with the column, cast to String for replacement operations
            col_expr = pl.col(col_name).cast(pl.String)

            # Sequentially chain 'replace_all' expressions for each rule
            for pattern, replacement in cleaner.rules.items():
                final_pattern = f"(?i){pattern}" if cleaner.case_insensitive else pattern
                
                if replacement is None:
                    # If replacement is None, use a when/then expression to set matching values to null
                    col_expr = pl.when(col_expr.str.contains(final_pattern)) \
                                .then(None) \
                                .otherwise(col_expr)
                else:
                    col_expr = col_expr.str.replace_all(final_pattern, replacement)
            
            # Execute the expression chain for the column
            df_cleaned = df_cleaned.with_columns(col_expr.alias(col_name))
            
        _LOGGER.info(f"Cleaned {len(self.cleaners)} columns.")
            
        return df_cleaned
    
    def load_clean_save(self, input_filepath: Union[str,Path], output_filepath: Union[str,Path]):
        """
        This convenience method encapsulates the entire cleaning process into a
        single call. It loads a DataFrame from a specified file, applies all
        cleaning rules configured in the `DataFrameCleaner` instance, and saves
        the resulting cleaned DataFrame to a new file.

        The method ensures that all data is loaded as string types to prevent
        unintended type inference issues before cleaning operations are applied.

        Args:
            input_filepath (Union[str, Path]):
                The path to the input data file.
            output_filepath (Union[str, Path]):
                The full path, where the cleaned data file will be saved.
        """
        df, _ = load_dataframe(df_path=input_filepath, kind="polars", all_strings=True)
        
        df_clean = self.clean(df=df, clone_df=False)
        
        if isinstance(output_filepath, str):
            output_filepath = make_fullpath(input_path=output_filepath, enforce="file")
        
        save_dataframe(df=df_clean, save_dir=output_filepath.parent, filename=output_filepath.name)
        
        return None


def info():
    _script_info(__all__)
