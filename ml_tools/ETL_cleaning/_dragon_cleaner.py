import polars as pl
from pathlib import Path
from typing import Union, Optional

from ..utilities import save_dataframe_filename, load_dataframe

from .._core import get_logger
from ..path_manager import make_fullpath

from ._clean_tools import save_unique_values, save_category_counts


_LOGGER = get_logger("DragonCleaner")


__all__ = [
    "DragonColumnCleaner",
    "DragonDataFrameCleaner",
]


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
    def __init__(self, 
                 column_name: str, 
                 exact_matches: Optional[Union[dict[str, Union[str, None]], dict[str, str]]] = None,
                 rules: Optional[Union[dict[str, Union[str, None]], dict[str, str]]] = None, 
                 case_insensitive: bool = False):
        """
        Args:
            column_name (str):
                The name of the column to be cleaned.
            exact_matches (Dict[str, str | None]):
                A dictionary of EXACT string matches to replacement strings.
                - Uses a hash map, which is significantly faster than regex.
                - Used for simple 1-to-1 mappings (e.g., {'Aluminum': 'Al'}).
                - Runs BEFORE the regex rules.
            rules (Dict[str, str | None]):
                A dictionary of regex patterns to replacement strings. 
                - Replacement can be None to indicate that matching values should be converted to null.
                - Can use backreferences (e.g., r'$1 $2') for captured groups. Note that Polars uses a '$' prefix for backreferences.
            case_insensitive (bool):
                If True, regex matching ignores case.

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
        
        # Validate Regex Rules
        if rules is not None:
            if not isinstance(rules, dict):
                _LOGGER.error("The 'rules' argument must be a dictionary.")
                raise TypeError()
            for pattern, replacement in rules.items():
                if not isinstance(pattern, str):
                    _LOGGER.error("All keys in 'rules' must be strings representing regex patterns.")
                    raise TypeError()
                if replacement is not None and not isinstance(replacement, str):
                    _LOGGER.error("All values in 'rules' must be strings or None (for nullification).")
                    raise TypeError()
        
        # Validate Exact Matches
        if exact_matches is not None:
            if not isinstance(exact_matches, dict):
                _LOGGER.error("The 'exact_matches' argument must be a dictionary.")
                raise TypeError()
            for key, val in exact_matches.items():
                if not isinstance(key, str):
                    _LOGGER.error("All keys in 'exact_matches' must be strings.")
                    raise TypeError()
                if val is not None and not isinstance(val, str):
                    _LOGGER.error("All values in 'exact_matches' must be strings or None.")
                    raise TypeError()
                
        # Raise if both are None or empty
        if not rules and not exact_matches:
            _LOGGER.error("At least one of 'rules' or 'exact_matches' must be provided.")
            raise ValueError()

        self.column_name = column_name
        self.rules = rules if rules else {}
        self.exact_matches = exact_matches if exact_matches else {}
        self.case_insensitive = case_insensitive

    def preview(self, 
                csv_path: Union[str, Path], 
                report_dir: Union[str, Path], 
                show_distribution: bool = True,
                add_value_separator: bool=False,
                rule_batch_size: int = 150):
        """
        Generates a preview report of unique values in the specified column after applying the current cleaning rules.
        
        Args:
            csv_path (str | Path):
                The path to the CSV file containing the data to clean.
            report_dir (str | Path):
                The directory where the preview report will be saved.
            show_distribution (bool):
                If True, generates a category count report for the column after cleaning.    
            add_value_separator (bool):
                If True, adds a separator line between each unique value in the report.
            rule_batch_size (int):
                Splits the regex rules into chunks of this size. Helps prevent memory errors.
        """
        # Load DataFrame
        df, _ = load_dataframe(df_path=csv_path, use_columns=[self.column_name], kind="polars", all_strings=True)
        
        preview_cleaner = DragonDataFrameCleaner(cleaners=[self])
        df_preview = preview_cleaner.clean(df, rule_batch_size=rule_batch_size)
        
        # Apply cleaning rules and save reports
        save_unique_values(csv_path_or_df=df_preview, 
                           output_dir=report_dir, 
                           use_columns=[self.column_name], 
                           verbose=False,
                           keep_column_order=False,
                           add_value_separator=add_value_separator)
        
        # Optionally save category counts
        if show_distribution:
            save_category_counts(csv_path_or_df=df_preview,
                                 output_dir=report_dir,
                                 use_columns=[self.column_name],
                                 verbose=False,
                                 keep_column_order=False)


class DragonDataFrameCleaner:
    """
    Orchestrates cleaning multiple columns in a Polars DataFrame.
    """
    def __init__(self, cleaners: list[DragonColumnCleaner]):
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
              rule_batch_size: int = 150) -> pl.DataFrame:
        """
        Applies cleaning rules. Supports Lazy execution to handle OOM issues.

        Args:
            df (pl.DataFrame | pl.LazyFrame): 
                The data to clean.
            rule_batch_size (int): 
                Splits the regex rules into chunks of this size. Helps prevent memory errors.

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
            
            # Start expression for this batch
            col_expr = pl.col(col_name).cast(pl.String)
            
            # --- PHASE 1: EXACT MATCHES ---
            # Apply dictionary-based replacement first (faster than regex)
            if cleaner.exact_matches:
                # 'replace' handles dictionary mapping safely. If value is mapped to None, it becomes null.
                col_expr = col_expr.replace(cleaner.exact_matches)

            # --- PHASE 2: REGEX PATTERNS ---
            all_rules = list(cleaner.rules.items())
            
            # Process in batches of 'rule_batch_size'
            for i in range(0, len(all_rules), rule_batch_size):
                rule_batch = all_rules[i : i + rule_batch_size]
                
                # continue chaining operations on the same col_expr
                for pattern, replacement in rule_batch:
                    final_pattern = f"(?i){pattern}" if cleaner.case_insensitive else pattern
                    
                    if replacement is None:
                        col_expr = pl.when(col_expr.str.contains(final_pattern)) \
                                    .then(None) \
                                    .otherwise(col_expr)
                    else:
                        col_expr = col_expr.str.replace_all(final_pattern, replacement)
                
                # Apply this batch of rules to the LazyFrame
                # apply partially here to keep the logical plan size under control
                final_lf = final_lf.with_columns(col_expr.alias(col_name))
                
                # Reset col_expr for the next batch, but pointing to the 'new' column
                # This ensures the next batch works on the result of the previous batch
                col_expr = pl.col(col_name)
                
            # If we had exact matches but NO regex rules, we still need to apply the expression once
            if cleaner.exact_matches and not all_rules:
                final_lf = final_lf.with_columns(col_expr.alias(col_name))

        # 3. Collect Results
        try:
            return final_lf.collect(engine="streaming")
        except Exception as e:
            _LOGGER.error("An error occurred during the cleaning process.")
            raise e
    
    def load_clean_save(self, 
                        input_filepath: Union[str,Path], 
                        output_filepath: Union[str,Path],
                        rule_batch_size: int = 150):
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
            rule_batch_size (int):
                Splits the regex rules into chunks of this size. Helps prevent memory errors.
        """
        df, _ = load_dataframe(df_path=input_filepath, kind="polars", all_strings=True)
        
        df_clean = self.clean(df=df, rule_batch_size=rule_batch_size)
        
        if isinstance(output_filepath, str):
            output_filepath = make_fullpath(input_path=output_filepath, enforce="file")
        
        save_dataframe_filename(df=df_clean, save_dir=output_filepath.parent, filename=output_filepath.name)
        
        return None
