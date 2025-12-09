import polars as pl
from pathlib import Path
from typing import Union, List, Dict, Optional

from ._path_manager import sanitize_filename, make_fullpath
from ._data_exploration import show_null_columns
from ._utilities import save_dataframe_filename, load_dataframe
from ._script_info import _script_info
from ._logger import get_logger


_LOGGER = get_logger("ETL Cleaning")


__all__ = [
    "DragonColumnCleaner",
    "DragonDataFrameCleaner",
    "save_unique_values",
    "basic_clean",
    "basic_clean_drop",
    "drop_macro_polars",
]


################ Unique Values per column #################
def save_unique_values(csv_path_or_df: Union[str, Path, pl.DataFrame], 
                       output_dir: Union[str, Path], 
                       use_columns: Optional[List[str]] = None,
                       verbose: bool=False,
                       keep_column_order: bool = True,
                       add_value_separator: bool = False) -> None:
    """
    Loads a CSV file or Polars DataFrame, then analyzes it and saves the unique non-null values
    from each column into a separate text file exactly as they appear.

    This is useful for understanding the raw categories or range of values
    within a dataset before and after cleaning.

    Args:
        csv_path_or_df (str | Path | pl.DataFrame):
            The file path to the input CSV file or a Polars DataFrame.
        output_dir (str | Path):
            The path to the directory where the .txt files will be saved.
            The directory will be created if it does not exist.
        keep_column_order (bool):
            If True, prepends a numeric prefix to each
            output filename to maintain the original column order.
        add_value_separator (bool):
            If True, adds a separator line between each unique value.
        use_columns (List[str] | None):
            If provided, only these columns will be processed. If None, all columns will be processed.
        verbose (bool):
            If True, prints the number of unique values saved for each column.
    """
    # 1 Handle input DataFrame or path
    if isinstance(csv_path_or_df, pl.DataFrame):
        df = csv_path_or_df
        if use_columns is not None:
            # Validate columns exist
            valid_cols = [c for c in use_columns if c in df.columns]
            if not valid_cols:
                _LOGGER.error("None of the specified columns in 'use_columns' exist in the provided DataFrame.")
                raise ValueError()
            df = df.select(valid_cols)
    else:
        csv_path = make_fullpath(input_path=csv_path_or_df, enforce="file")
        df = load_dataframe(df_path=csv_path, use_columns=use_columns, kind="polars", all_strings=True)[0]
        
    output_dir = make_fullpath(input_path=output_dir, make=True, enforce='directory')
    
    if df.height == 0:
        _LOGGER.warning("The input DataFrame is empty. No unique values to save.")
        return
    
    # --- 2. Process Each Column ---
    counter = 0
    
    # Iterate over columns using Polars methods
    for i, column_name in enumerate(df.columns):
        try:
            # Efficiently get unique non-null values and sort them
            # We use drop_nulls() on the specific series
            unique_series = df.select(pl.col(column_name)).drop_nulls().unique().sort(column_name)
            
            # Convert to a python list for writing
            sorted_uniques = unique_series.to_series().to_list()
        
        except Exception:
            _LOGGER.error(f"Could not process column '{column_name}'.")
            continue

        if not sorted_uniques:
            _LOGGER.warning(f"Column '{column_name}' has no unique non-null values. Skipping.")
            continue

        # --- 3. Filename Generation ---
        sanitized_name = sanitize_filename(column_name)
        if not sanitized_name.strip('_'):
            sanitized_name = f'column_{i}'
        
        prefix = f"{i + 1}_" if keep_column_order else ''
        file_path = output_dir / f"{prefix}{sanitized_name}_unique_values.txt"

        # --- 4. Write to File ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Unique values for column: '{column_name}'\n")
                f.write(f"# Total unique non-null values: {len(sorted_uniques)}\n")
                f.write("-" * 30 + "\n")
                
                for value in sorted_uniques:
                    f.write(f"{value}\n")
                    if add_value_separator:
                        f.write("-" * 30 + "\n")
                        
        except IOError:
            _LOGGER.exception(f"Error writing to file {file_path}.")
        else:
            if verbose:
                print(f"    Successfully saved {len(sorted_uniques)} unique values from '{column_name}'.")
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
    

def basic_clean_drop(input_filepath: Union[str,Path], output_filepath: Union[str,Path], log_directory: Union[str,Path], targets: list[str], 
                     skip_targets: bool=False, threshold: float=0.8, all_lowercase: bool=False):
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
class DragonColumnCleaner:
    """
    A configuration object that defines cleaning rules for a single Polars DataFrame column.

    This class holds a dictionary of regex-to-replacement rules, the target column name,
    and the case-sensitivity setting. It is intended to be used with the DragonDataFrameCleaner.
    
    Notes:
        - Define rules from most specific to more general to create a fallback system.
        - Beware of chain replacements (rules matching strings that have already been
          changed by a previous rule in the same cleaner).
    """
    def __init__(self, column_name: str, rules: Dict[str, str], case_insensitive: bool = True):
        """
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

        id_cleaner = DragonColumnCleaner(column_name='user_id', rules=id_rules)
        # This object would then be passed to a DragonDataFrameCleaner.
        ```
        """
        if not isinstance(column_name, str) or not column_name:
            _LOGGER.error("The 'column_name' must be a non-empty string.")
            raise TypeError()
        if not isinstance(rules, dict):
            _LOGGER.error("The 'rules' argument must be a dictionary.")
            raise TypeError()
        # validate rules
        for pattern, replacement in rules.items():
            if not isinstance(pattern, str):
                _LOGGER.error("All keys in 'rules' must be strings representing regex patterns.")
                raise TypeError()
            if replacement is not None and not isinstance(replacement, str):
                _LOGGER.error("All values in 'rules' must be strings or None (for nullification).")
                raise TypeError()

        self.column_name = column_name
        self.rules = rules
        self.case_insensitive = case_insensitive

    def preview(self, csv_path: Union[str, Path], report_dir: Union[str, Path], add_value_separator: bool=False):
        """
        Generates a preview report of unique values in the specified column after applying the current cleaning rules.
        
        Args:
            csv_path (str | Path):
                The path to the CSV file containing the data to clean.
            report_dir (str | Path):
                The directory where the preview report will be saved.
            add_value_separator (bool):
                If True, adds a separator line between each unique value in the report.
        """
        # Load DataFrame
        df, _ = load_dataframe(df_path=csv_path, use_columns=[self.column_name], kind="polars", all_strings=True)
        
        preview_cleaner = DragonDataFrameCleaner(cleaners=[self])
        df_preview = preview_cleaner.clean(df)
        
        # Apply cleaning rules to a copy of the column for preview
        save_unique_values(csv_path_or_df=df_preview, 
                           output_dir=report_dir, 
                           use_columns=[self.column_name], 
                           verbose=False,
                           keep_column_order=False,
                           add_value_separator=add_value_separator)


class DragonDataFrameCleaner:
    """
    Orchestrates cleaning multiple columns in a Polars DataFrame.
    """
    def __init__(self, cleaners: List[DragonColumnCleaner]):
        """
        Takes a list of DragonColumnCleaner objects and applies their defined
        rules to the corresponding columns of a DataFrame using high-performance
        Polars expressions wit memory optimization.

        Args:
            cleaners (List[DragonColumnCleaner]):
                A list of DragonColumnCleaner configuration objects.
        """
        if not isinstance(cleaners, list):
            _LOGGER.error("The 'cleaners' argument must be a list of DragonColumnCleaner objects.")
            raise TypeError()

        seen_columns = set()
        for cleaner in cleaners:
            if not isinstance(cleaner, DragonColumnCleaner):
                _LOGGER.error(f"All items in 'cleaners' list must be DragonColumnCleaner objects, but found an object of type {type(cleaner).__name__}.")
                raise TypeError()
            if cleaner.column_name in seen_columns:
                _LOGGER.error(f"Duplicate DragonColumnCleaner found for column '{cleaner.column_name}'. Each column should only have one cleaner.")
                raise ValueError()
            seen_columns.add(cleaner.column_name)

        self.cleaners = cleaners

    def clean(self, df: Union[pl.DataFrame, pl.LazyFrame], 
              rule_batch_size: int = 50) -> pl.DataFrame:
        """
        Applies cleaning rules. Supports Lazy execution to handle OOM issues.

        Args:
            df (pl.DataFrame | pl.LazyFrame): 
                The data to clean.
            rule_batch_size (int): 
                Splits the regex rules into chunks of this size. Helps prevent memory errors or memory spikes 
                during query optimization.

        Returns:
            pl.DataFrame: The cleaned, collected DataFrame.
        """
        # 1. Validate Columns (Only if eager, or simple schema check if lazy)
        # Note: For LazyFrames, we assume columns exist or let it fail at collection.
        if isinstance(df, pl.DataFrame):
            df_cols = set(df.columns)
            rule_cols = {c.column_name for c in self.cleaners}
            missing = rule_cols - df_cols
            if missing:
                _LOGGER.error(f"The following columns specified in cleaners are missing from the DataFrame: {missing}")
                raise ValueError()
            
            
            # lazy internally
            lf = df.lazy()
        else:
            # It should be a LazyFrame, check type
            if not isinstance(df, pl.LazyFrame):
                _LOGGER.error("The 'df' argument must be a Polars DataFrame or LazyFrame.")
                raise TypeError()
            # It is already a LazyFrame
            lf = df

        # 2. Build Expression Chain
        final_lf = lf
        
        for cleaner in self.cleaners:
            col_name = cleaner.column_name
            
            # Get all rules as a list of items
            all_rules = list(cleaner.rules.items())
            
            # Process in batches of 'rule_batch_size'
            for i in range(0, len(all_rules), rule_batch_size):
                rule_batch = all_rules[i : i + rule_batch_size]
                
                # Start expression for this batch
                col_expr = pl.col(col_name).cast(pl.String)
                
                for pattern, replacement in rule_batch:
                    final_pattern = f"(?i){pattern}" if cleaner.case_insensitive else pattern
                    
                    if replacement is None:
                        col_expr = pl.when(col_expr.str.contains(final_pattern)) \
                                    .then(None) \
                                    .otherwise(col_expr)
                    else:
                        col_expr = col_expr.str.replace_all(final_pattern, replacement)
                
                # Apply this batch of rules to the LazyFrame
                final_lf = final_lf.with_columns(col_expr.alias(col_name))

        # 3. Collect Results
        try:
            return final_lf.collect(engine="streaming")
        except Exception as e:
            _LOGGER.error("An error occurred during the cleaning process.")
            raise e
    
    def load_clean_save(self, input_filepath: Union[str,Path], output_filepath: Union[str,Path]):
        """
        This convenience method encapsulates the entire cleaning process into a
        single call. It loads a DataFrame from a specified file, applies all
        cleaning rules configured in the `DragonDataFrameCleaner` instance, and saves
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
        
        df_clean = self.clean(df=df)
        
        if isinstance(output_filepath, str):
            output_filepath = make_fullpath(input_path=output_filepath, enforce="file")
        
        save_dataframe_filename(df=df_clean, save_dir=output_filepath.parent, filename=output_filepath.name)
        
        return None


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
    
    save_dataframe_filename(df=report, save_dir=save_dir, filename=filename)


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


def info():
    _script_info(__all__)
