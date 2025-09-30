import polars as pl
import pandas as pd
import re
from pathlib import Path
from typing import Literal, Union, Optional, Any, Callable, List, Dict, Tuple
from .path_manager import sanitize_filename, make_fullpath
from .utilities import save_dataframe, load_dataframe
from ._script_info import _script_info
from ._logger import _LOGGER


__all__ = [
    "save_unique_values",
    "ColumnCleaner",
    "DataFrameCleaner",
    "TransformationRecipe",
    "DataProcessor",
    "BinaryTransformer",
    "MultiBinaryDummifier",
    "AutoDummifier",
    "KeywordDummifier",
    "NumberExtractor",
    "MultiNumberExtractor",
    "RatioCalculator",
    "CategoryMapper",
    "RegexMapper",
    "ValueBinner",
    "DateFeatureExtractor"
]

################ Unique Values per column #################
def save_unique_values(csv_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Loads a CSV file, then analyzes it and saves the unique non-null values
    from each column into a separate text file exactly as they appear.

    This is useful for understanding the raw categories or range of values
    within a dataset before cleaning.

    Args:
        csv_path (Union[str, Path]):
            The file path to the input CSV file.
        output_dir (Union[str, Path]):
            The path to the directory where the .txt files will be saved.
            The directory will be created if it does not exist.
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
        file_path = output_dir / f"{sanitized_name}_unique_values.txt"

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
            _LOGGER.info(f"Successfully saved {len(sorted_uniques)} unique values from '{column_name}'.")
            
    _LOGGER.info("Process complete.")


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

        # Validate each regex pattern for correctness
        for pattern in rules.keys():
            try:
                re.compile(pattern)
            except re.error:
                _LOGGER.error(f"Invalid regex pattern '{pattern}'.")
                raise

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


############ TRANSFORM MAIN ####################

# Magic word for rename-only transformation
_RENAME = "rename"

class TransformationRecipe:
    """
    A builder class for creating a data transformation recipe.

    This class provides a structured way to define a series of transformation
    steps, with validation performed at the time of addition. It is designed
    to be passed to a `DataProcessor`.
    
    Use the method `add()` to add recipes.
    """
    def __init__(self):
        self._steps: List[Dict[str, Any]] = []

    def add(
        self,
        input_col_name: str,
        output_col_names: Union[str, List[str]],
        transform: Union[str, Callable],
    ) -> "TransformationRecipe":
        """
        Adds a new transformation step to the recipe.

        Args:
            input_col: The name of the column from the source DataFrame.
            output_col: The desired name(s) for the output column(s).
                        A string for a 1-to-1 mapping, or a list of strings
                        for a 1-to-many mapping.
            transform: The transformation to apply: 
                - Use "rename" for simple column renaming
                - If callable, must accept a `pl.Series` as the only parameter and return either a `pl.Series` or `pl.DataFrame`.

        Returns:
            The instance of the recipe itself to allow for method chaining.
        """
        # --- Validation ---
        if not isinstance(input_col_name, str) or not input_col_name:
            _LOGGER.error("'input_col' must be a non-empty string.")
            raise TypeError()
            
        if transform == _RENAME:
            if not isinstance(output_col_names, str):
                _LOGGER.error("For a RENAME operation, 'output_col' must be a string.")
                raise TypeError()
        elif not isinstance(transform, Callable):
            _LOGGER.error(f"'transform' must be a callable function or the string '{_RENAME}'.")
            raise TypeError()

        if isinstance(output_col_names, list) and transform == _RENAME:
            _LOGGER.error("A RENAME operation cannot have a list of output columns.")
            raise ValueError()
        
        # --- Add Step ---
        step = {
            "input_col": input_col_name,
            "output_col": output_col_names,
            "transform": transform,
        }
        self._steps.append(step)
        return self  # Allow chaining: recipe.add(...).add(...)

    def __iter__(self):
        """Allows the class to be iterated over, like a list."""
        return iter(self._steps)

    def __len__(self):
        """Allows the len() function to be used on an instance."""
        return len(self._steps)


class DataProcessor:
    """
    Transforms a Polars DataFrame based on a provided `TransformationRecipe` object.
    
    Use the method `transform()`.
    """
    def __init__(self, recipe: TransformationRecipe):
        """
        Initializes the DataProcessor with a transformation recipe.

        Args:
            recipe: An instance of the `TransformationRecipe` class that has
                    been populated with transformation steps.
        """
        if not isinstance(recipe, TransformationRecipe):
            _LOGGER.error("The recipe must be an instance of TransformationRecipe.")
            raise TypeError()
        if len(recipe) == 0:
            _LOGGER.error("The recipe cannot be empty.")
            raise ValueError()
        self._recipe = recipe

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the transformation recipe to the input DataFrame.
        """
        processed_columns = []
        # Recipe object is iterable
        for step in self._recipe:
            input_col_name = step["input_col"]
            output_col_spec = step["output_col"]
            transform_action = step["transform"]

            if input_col_name not in df.columns:
                _LOGGER.error(f"Input column '{input_col_name}' not found in DataFrame.")
                raise ValueError()

            input_series = df.get_column(input_col_name)

            if transform_action == _RENAME:
                processed_columns.append(input_series.alias(output_col_spec))
                continue

            if isinstance(transform_action, Callable):
                result = transform_action(input_series)

                if isinstance(result, pl.Series):
                    if not isinstance(output_col_spec, str):
                        _LOGGER.error(f"Function for '{input_col_name}' returned a Series but 'output_col' is not a string.")
                        raise TypeError()
                    processed_columns.append(result.alias(output_col_spec))
                
                elif isinstance(result, pl.DataFrame):
                    # 1. Handle list-based renaming
                    if isinstance(output_col_spec, list):
                        if len(result.columns) != len(output_col_spec):
                            _LOGGER.error(f"Mismatch in '{input_col_name}': function produced {len(result.columns)} columns, but recipe specifies {len(output_col_spec)} output names.")
                            raise ValueError()
                        
                        renamed_df = result.rename(dict(zip(result.columns, output_col_spec)))
                        processed_columns.extend(renamed_df.get_columns())

                    # 2. Handle a string prefix for AutoDummifier
                    elif isinstance(output_col_spec, str):
                        prefix = output_col_spec
                        # Replace the original name part with the desired prefix.
                        new_names = {
                            col: f"{prefix}{col[len(input_col_name):]}" for col in result.columns
                        }
                        renamed_df = result.rename(new_names)
                        processed_columns.extend(renamed_df.get_columns())
                    
                    else:
                        _LOGGER.error(f"Function for '{input_col_name}' returned a DataFrame, so 'output_col' must be a list of names or a string prefix.")
                        raise TypeError()
                
                else:
                    _LOGGER.error(f"Function for '{input_col_name}' returned an unexpected type: {type(result)}.")
                    raise TypeError()
            
            else: # This case is unlikely due to builder validation.
                _LOGGER.error(f"Invalid 'transform' action for '{input_col_name}': {transform_action}")
                raise TypeError()

        if not processed_columns:
            _LOGGER.error("The transformation resulted in an empty DataFrame.")
            return pl.DataFrame()
            
        return pl.DataFrame(processed_columns)
    
    def __str__(self) -> str:
        """
        Provides a detailed, human-readable string representation of the
        entire processing pipeline.
        """
        header = "DataProcessor Pipeline"
        divider = "-" * len(header)
        num_steps = len(self._recipe)
        
        lines = [
            header,
            divider,
            f"Number of steps: {num_steps}\n"
        ]

        if num_steps == 0:
            lines.append("No transformation steps defined.")
            return "\n".join(lines)

        for i, step in enumerate(self._recipe, 1):
            transform_action = step["transform"]
            
            # Get a clean name for the transformation action
            if transform_action == _RENAME: # "rename"
                transform_name = "Rename"
            else:
                # This works for both functions and class instances
                transform_name = type(transform_action).__name__

            lines.append(f"[{i}] Input: '{step['input_col']}'")
            lines.append(f"    - Transform: {transform_name}")
            lines.append(f"    - Output(s): {step['output_col']}")
            if i < num_steps:
                lines.append("") # Add a blank line between steps

        return "\n".join(lines)

    def inspect(self) -> None:
        """
        Prints the detailed string representation of the pipeline to the console.
        """
        print(self)

############ TRANSFORMERS ####################

class BinaryTransformer:
    """
    A transformer that maps string values to a binary 1 or 0 based on keyword matching.

    Must supply a list of keywords for either the 'true' case (1) or the 'false' case (0), but not both.

    Args:
        true_keywords (List[str] | None):
            If a string contains any of these keywords, the output is 1, otherwise 0.
        false_keywords (List[str] | None):
            If a string contains any of these keywords, the output is 0, otherwise 1.
    """
    def __init__(
        self,
        true_keywords: Optional[List[str]] = None,
        false_keywords: Optional[List[str]] = None,
        case_insensitive: bool = True,
    ):
        # --- Validation: Enforce one and only one option ---
        if true_keywords is not None and false_keywords is not None:
            _LOGGER.error("Provide either 'true_keywords' or 'false_keywords', but not both.")
            raise ValueError()
        if true_keywords is None and false_keywords is None:
            _LOGGER.error("You must provide either 'true_keywords' or 'false_keywords'.")
            raise ValueError()

        # --- Configuration ---
        self.keywords: List[str] = true_keywords if true_keywords is not None else false_keywords # type: ignore
        if not self.keywords:
            _LOGGER.error("Keyword list cannot be empty.")
            raise ValueError()

        self.mode: str = "true_mode" if true_keywords is not None else "false_mode"
        
        # --- Create the regex string pattern ---
        # Escape keywords to treat them as literals
        base_pattern = "|".join(re.escape(k) for k in self.keywords)

        # For polars, add case-insensitivity flag `(?i)` to the pattern string itself
        if case_insensitive:
            self.pattern = f"(?i){base_pattern}"
        else:
            self.pattern = base_pattern
        

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the binary mapping logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of string data.

        Returns:
            pl.Series: A new Series of type UInt8 containing 1s and 0s.
        """
        # Create a boolean Series: True if any keyword is found, else False
        contains_keyword = column.str.contains(self.pattern)

        # Apply logic and cast directly to integer type
        if self.mode == "true_mode":
            # True -> 1, False -> 0
            return contains_keyword.cast(pl.UInt8)
        else: # false_mode
            # We want the inverse: True -> 0, False -> 1
            return (~contains_keyword).cast(pl.UInt8)


class AutoDummifier:
    """
    A transformer that performs one-hot encoding on a categorical column,
    automatically detecting the unique categories from the data.
    """
    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Executes the one-hot encoding logic.

        Args:
            column (pl.Series): The input Polars Series of categories.

        Returns:
            pl.DataFrame: A DataFrame with one-hot encoded columns.
                          Column names are auto-generated by Polars as
                          '{original_col_name}_{category_value}'.
        """
        # Ensure the column is treated as a string before creating dummies
        return column.cast(pl.Utf8).to_dummies()


class MultiBinaryDummifier:
    """
    A one-to-many transformer that creates multiple binary columns from a single
    text column based on a list of keywords.

    For each keyword provided, this transformer generates a corresponding column
    with a value of 1 if the keyword is present in the input string, and 0 otherwise.
    It is designed to be used within the DataProcessor pipeline.

    Args:
        keywords (List[str]):
            A list of strings, where each string is a keyword to search for. A separate
            binary column will be created for each keyword.
        case_insensitive (bool):
            If True, keyword matching ignores case. Defaults to True.
    """
    def __init__(self, keywords: List[str], case_insensitive: bool = True):
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            _LOGGER.error("The 'keywords' argument must be a list of strings.")
            raise TypeError()
        if not keywords:
            _LOGGER.error("The 'keywords' list cannot be empty.")
            raise ValueError()

        self.keywords = keywords
        self.case_insensitive = case_insensitive

    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Executes the dummification logic.

        Args:
            column (pl.Series): The input Polars Series to transform.

        Returns:
            pl.DataFrame: A DataFrame where each column corresponds to a keyword.
        """
        # Ensure the input is treated as a string, preserving nulls
        str_column = column.cast(pl.Utf8)
        
        output_expressions = []
        for i, keyword in enumerate(self.keywords):
            # Escape keyword to treat it as a literal, not a regex pattern
            base_pattern = re.escape(keyword)

            # Add case-insensitivity flag `(?i)` if needed
            pattern = f"(?i){base_pattern}" if self.case_insensitive else base_pattern

            # Create the binary expression
            expr = (
                pl.when(str_column.is_null())
                .then(None) # Propagate nulls from original column
                .when(str_column.str.contains(pattern))
                .then(pl.lit(1, dtype=pl.UInt8))
                .otherwise(pl.lit(0, dtype=pl.UInt8))
                .alias(f"col_{i}") # Generic name for DataProcessor
            )
            output_expressions.append(expr)

        return pl.select(output_expressions)


class KeywordDummifier:
    """
    A configurable transformer that creates one-hot encoded columns based on
    keyword matching in a Polars Series.

    Operates on a "first match wins" principle.

    Args:
        group_names (List[str]): 
            A list of strings, where each string is the name of a category.
            This defines the matching priority and the base column names of the
            DataFrame returned by the transformation.
        group_keywords (List[List[str]]): 
            A list of lists of strings. Each inner list corresponds to a 
            `group_name` at the same index and contains the keywords to search for.
        case_insensitive (bool):
            If True, keyword matching ignores case.
    """
    def __init__(self, group_names: List[str], group_keywords: List[List[str]], case_insensitive: bool = True):
        if len(group_names) != len(group_keywords):
            _LOGGER.error("Initialization failed: 'group_names' and 'group_keywords' must have the same length.")
            raise ValueError()
        
        self.group_names = group_names
        self.group_keywords = group_keywords
        self.case_insensitive = case_insensitive

    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Executes the one-hot encoding logic.

        Args:
            column (pl.Series): The input Polars Series to transform.

        Returns:
            pl.DataFrame: A DataFrame with one-hot encoded columns.
        """
        column = column.cast(pl.Utf8)
        
        categorize_expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
        
        for name, keywords in zip(self.group_names, self.group_keywords):
            # Create the base regex pattern by escaping and joining keywords
            base_pattern = "|".join(re.escape(k) for k in keywords)
            
            # Add the case-insensitive flag `(?i)` to the pattern string
            if self.case_insensitive:
                pattern = f"(?i){base_pattern}"
            else:
                pattern = base_pattern
            
            categorize_expr = categorize_expr.when(
                column.str.contains(pattern)
            ).then(pl.lit(name))
        
        categorize_expr = categorize_expr.otherwise(None).alias("category")

        temp_df = pl.select(categorize_expr)
        df_with_dummies = temp_df.to_dummies(columns=["category"])
        
        final_columns = []
        for name in self.group_names:
            dummy_col_name = f"category_{name}"
            if dummy_col_name in df_with_dummies.columns:
                # The alias here uses the group name as the temporary column name
                final_columns.append(
                    df_with_dummies.get_column(dummy_col_name).alias(name)
                )
            else:
                # If a group had no matches, create a column of zeros
                final_columns.append(pl.lit(0, dtype=pl.UInt8).alias(name))

        return pl.select(final_columns)


class NumberExtractor:
    """
    A configurable transformer that extracts a single number from a Polars string series using a regular expression.

    An instance can be used as a 'transform' callable within the
    `DataProcessor` pipeline.

    Args:
        regex_pattern (str):
            The regular expression used to find the number. This pattern
            MUST contain exactly one capturing group `(...)`. Defaults to a standard pattern for integers and floats.
        dtype (str):
            The desired data type for the output column. Defaults to "float".
        round_digits (int | None):
            If the dtype is 'float', you can specify the number of decimal
            places to round the result to. This parameter is ignored if
            dtype is 'int'. Defaults to None (no rounding).
    """
    def __init__(
        self,
        regex_pattern: str = r"(\d+\.?\d*)",
        dtype: Literal["float", "int"] = "float",
        round_digits: Optional[int] = 2,
    ):
        # --- Validation ---
        if not isinstance(regex_pattern, str):
            _LOGGER.error("regex_pattern must be a string.")
            raise TypeError()
        
        # Validate that the regex has exactly one capturing group
        try:
            if re.compile(regex_pattern).groups != 1:
                _LOGGER.error("regex_pattern must contain exactly one capturing group '(...)'")
                raise ValueError()
        except re.error as e:
            _LOGGER.error(f"Invalid regex pattern provided: {e}")
            raise ValueError()

        if dtype not in ["float", "int"]:
            _LOGGER.error("dtype must be either 'float' or 'int'.")
            raise ValueError()
            
        if round_digits is not None:
            if not isinstance(round_digits, int):
                _LOGGER.error("round_digits must be an integer.")
                raise TypeError()
            if dtype == "int":
                _LOGGER.warning(f"'round_digits' is specified but dtype is 'int'. Rounding will be ignored.")

        self.regex_pattern = regex_pattern
        self.dtype = dtype
        self.round_digits = round_digits
        self.polars_dtype = pl.Float64 if dtype == "float" else pl.Int64

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Executes the number extraction logic.

        Args:
            column (pl.Series): The input Polars Series to transform.

        Returns:
            pl.Series: A new Series containing the extracted numbers.
        """
        # Extract the first (and only) capturing group
        extracted = column.str.extract(self.regex_pattern, 1)
        
        # Cast to the desired numeric type. Non-matching strings become null.
        casted = extracted.cast(self.polars_dtype, strict=False)
        
        # Apply rounding only if it's a float and round_digits is set
        if self.dtype == "float" and self.round_digits is not None:
            return casted.round(self.round_digits)
            
        return casted


class MultiNumberExtractor:
    """
    Extracts multiple numbers from a single polars string column into several new columns.

    This transformer is designed for one-to-many mappings, such as parsing coordinates (10, 25) into separate columns.

    Args:
        num_outputs (int):
            Number of numeric columns to create.
        regex_pattern (str):
            The regex pattern to find all numbers. Must contain one
            capturing group around the number part.
            Defaults to a standard pattern for integers and floats.
        dtype (str):
            The desired data type for the output columns. Defaults to "float".
        fill_value (int | float | None):
            A value to fill in if a number is not found at a given position (if positive match).
            - For example, if `num_outputs=2` and only one number is found in a string, the second output column will be filled with this value. If None, it will be filled with null.
    """
    def __init__(
        self,
        num_outputs: int,
        regex_pattern: str = r"(\d+\.?\d*)",
        dtype: Literal["float", "int"] = "float",
        fill_value: Optional[Union[int, float]] = None
    ):
        # --- Validation ---
        if not isinstance(num_outputs, int) or num_outputs <= 0:
            _LOGGER.error("num_outputs must be a positive integer.")
            raise ValueError()
        
        if not isinstance(regex_pattern, str):
            _LOGGER.error("regex_pattern must be a string.")
            raise TypeError()
        
        # Validate that the regex has exactly one capturing group
        try:
            if re.compile(regex_pattern).groups != 1:
                _LOGGER.error("regex_pattern must contain exactly one capturing group '(...)'")
                raise ValueError()
        except re.error as e:
            _LOGGER.error(f"Invalid regex pattern provided: {e}")
            raise ValueError()
        
        # Validate dtype
        if dtype not in ["float", "int"]:
            _LOGGER.error("dtype must be either 'float' or 'int'.")
            raise ValueError()
        
        self.num_outputs = num_outputs
        self.regex_pattern = regex_pattern
        self.fill_value = fill_value
        self.polars_dtype = pl.Float64 if dtype == "float" else pl.Int64

    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Executes the multi-number extraction logic. Preserves nulls from the input column.
        """
        output_expressions = []
        for i in range(self.num_outputs):
            # Define the core extraction logic for the i-th number
            extraction_expr = (
                column.str.extract_all(self.regex_pattern)
                .list.get(i)
                .cast(self.polars_dtype, strict=False)
            )

            # Apply the fill value if provided
            if self.fill_value is not None:
                extraction_expr = extraction_expr.fill_null(self.fill_value)

            # Only apply the logic when the input is not null.
            # Otherwise, the result should also be null.
            final_expr = (
                pl.when(column.is_not_null())
                .then(extraction_expr)
                .otherwise(None)
                .alias(f"col_{i}") # Name the final output expression
            )
            
            output_expressions.append(final_expr)
        
        return pl.select(output_expressions)


class RatioCalculator:
    """
    A transformer that parses a string ratio (e.g., "40:5" or "30/2") and
    computes the result of the division. It gracefully handles strings that
    do not match the pattern by returning null.
    """
    def __init__(
        self,
        # Default pattern includes the full-width colon '：'
        regex_pattern: str = r"(\d+\.?\d*)\s*[:：/]\s*(\d+\.?\d*)"
    ):
        # --- Robust Validation ---
        try:
            compiled_pattern = re.compile(regex_pattern)
            if compiled_pattern.groups != 2:
                _LOGGER.error("RatioCalculator regex_pattern must contain exactly two capturing groups '(...)'.")
                raise ValueError()
            if compiled_pattern.groupindex:
                _LOGGER.error("RatioCalculator must be initialized with unnamed capturing groups (e.g., '(\\d+)'), not named groups (e.g., '(?P<name>\\d+)').")
                raise ValueError()
        except re.error as e:
            _LOGGER.error(f"Invalid regex pattern provided: {e}")
            raise ValueError()

        self.regex_pattern = regex_pattern

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the ratio calculation logic to the input column.
        This version uses .str.extract() for maximum stability.
        """
        # Extract numerator (group 1) and denominator (group 2) separately.
        numerator_expr = column.str.extract(self.regex_pattern, 1).cast(pl.Float64, strict=False)
        denominator_expr = column.str.extract(self.regex_pattern, 2).cast(pl.Float64, strict=False)

        # Calculate the ratio, handling division by zero.
        final_expr = pl.when(denominator_expr != 0).then(
            numerator_expr / denominator_expr
        ).otherwise(
            None # Handles both null denominators and division by zero
        )

        return pl.select(final_expr.round(4)).to_series()


class CategoryMapper:
    """
    A transformer that maps string categories to specified numerical values using a dictionary.

    Ideal for ordinal encoding.

    Args:
        mapping (Dict[str, [int | float]]):
            A dictionary that defines the mapping from a string category (key)
            to a numerical value (value).
        unseen_value (int | float | None):
            The numerical value to use for categories that are present in the
            data but not in the mapping dictionary. If not provided or set
            to None, unseen categories will be mapped to a null value.
    """
    def __init__(
        self,
        mapping: Dict[str, Union[int, float]],
        unseen_value: Optional[Union[int, float]] = None,
    ):
        if not isinstance(mapping, dict):
            _LOGGER.error("The 'mapping' argument must be a dictionary.")
            raise TypeError()
        
        self.mapping = mapping
        self.default_value = unseen_value

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the dictionary mapping to the input column.

        Args:
            column (pl.Series): The input Polars Series of categories.

        Returns:
            pl.Series: A new Series with categories mapped to numbers.
        """
        # Ensure the column is treated as a string for matching keys
        str_column = column.cast(pl.Utf8)

        # Create a list of 'when/then' expressions, one for each mapping
        mapping_expressions = [
            pl.when(str_column == from_val).then(pl.lit(to_val))
            for from_val, to_val in self.mapping.items()
        ]

        # Use coalesce to find the first non-null value.
        # The default_value acts as the final fallback.
        final_expr = pl.coalesce(
            *mapping_expressions, # Unpack the list of expressions
            pl.lit(self.default_value)
        )
        
        return pl.select(final_expr).to_series()


class RegexMapper:
    """
    A transformer that maps string categories to numerical values based on a
    dictionary of regular expression patterns.

    The class iterates through the mapping dictionary in order, and the first
    pattern that matches a given string determines the output value. This
    "first match wins" logic makes the order of the mapping important.

    Args:
        mapping (Dict[str, [int | float]]):
            An ordered dictionary where keys are regex patterns and values are
            the numbers to map to if the pattern is found.
        unseen_value (int | float | None):
            The numerical value to use for strings that do not match any
            of the regex patterns. If None, unseen values are mapped to null.
        case_insensitive (bool):
            If True , the regex matching for all patterns will ignore case.
    """
    def __init__(
        self,
        mapping: Dict[str, Union[int, float]],
        unseen_value: Optional[Union[int, float]] = None,
        case_insensitive: bool = True,
    ):
        # --- Validation ---
        if not isinstance(mapping, dict):
            _LOGGER.error("The 'mapping' argument must be a dictionary.")
            raise TypeError()

        self.unseen_value = unseen_value
        
        # --- Process and validate patterns ---
        # Process patterns here to be more efficient, avoiding reprocessing on every __call__.
        self.processed_mapping: List[Tuple[str, Union[int, float]]] = []
        for pattern, value in mapping.items():
            final_pattern = f"(?i){pattern}" if case_insensitive else pattern

            # Validate the final pattern that will actually be used by Polars
            try:
                re.compile(final_pattern)
            except re.error as e:
                _LOGGER.error(f"Invalid regex pattern '{final_pattern}': {e}")
                raise ValueError()
            if not isinstance(value, (int, float)):
                _LOGGER.error(f"Mapping values must be int or float, but got {type(value)} for pattern '{pattern}'.")
                raise TypeError()
            
            self.processed_mapping.append((final_pattern, value))

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the regex mapping logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of string data.

        Returns:
            pl.Series: A new Series with strings mapped to numbers based on
                       the first matching regex pattern.
        """
        # pl.String is the modern alias for pl.Utf8
        str_column = column.cast(pl.String)

        # Start with the fallback value for non-matches.
        mapping_expr = pl.lit(self.unseen_value)

        # Iterate through the processed mapping in reverse to construct the nested expression
        for pattern, value in reversed(self.processed_mapping):
            mapping_expr = (
                pl.when(str_column.str.contains(pattern))
                .then(pl.lit(value))
                .otherwise(mapping_expr)
            )
        
        return pl.select(mapping_expr).to_series()


class ValueBinner:
    """
    A transformer that discretizes a continuous numerical column into a finite number of bins.

    Each bin is assigned an integer label (0, 1, 2, ...).

    Args:
        breaks (List[int | float]):
            A list of numbers defining the boundaries of the bins. The list
            must be sorted in ascending order and contain at least two values.
            For example, `breaks=[0, 18, 40, 65]` creates three bins.
        left_closed (bool):
            Determines which side of the interval is inclusive.
            - If `False` (default): Intervals are (lower, upper].
            - If `True`: Intervals are [lower, upper).
    """
    def __init__(
        self,
        breaks: List[Union[int, float]],
        left_closed: bool = False,
    ):
        # --- Validation ---
        if not isinstance(breaks, list) or len(breaks) < 2:
            _LOGGER.error("The 'breaks' argument must be a list of at least two numbers.")
            raise ValueError()
        
        # Check if the list is sorted
        if not all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1)):
            _LOGGER.error("The 'breaks' list must be sorted in ascending order.")
            raise ValueError()

        self.breaks = breaks
        self.left_closed = left_closed
        # Generate numerical labels [0, 1, 2, ...] for the bins
        self.labels = [str(i) for i in range(len(breaks) - 1)]

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the binning logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of numerical data.

        Returns:
            pl.Series: A new Series of integer labels for the bins. Values
                       outside the specified breaks will become null.
        """
        # `cut` creates a new column of type Categorical
        binned_column = column.cut(
            breaks=self.breaks,
            labels=self.labels,
            left_closed=self.left_closed
        )
        
        # to_physical() converts the Categorical type to its underlying
        # integer representation (u32), which is perfect for ML.
        return binned_column.to_physical()


class DateFeatureExtractor:
    """
    A one-to-many transformer that extracts multiple numerical features from a date or datetime column.

    It can handle columns that are already in a Polars Date/Datetime format,
    or it can parse string columns if a format is provided.

    Args:
        features (List[str]):
            A list of the date/time features to extract. Supported features are:
            'year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond',
            'microsecond', 'nanosecond', 'ordinal_day' (day of year),
            'weekday' (Mon=1, Sun=7), 'week' (week of year), and 'timestamp'.
        format (str | None):
            The format code used to parse string dates (e.g., "%Y-%m-%d %H:%M:%S").
            Use if the input column is not a Date or Datetime type.
    """
    
    ALLOWED_FEATURES = {
        'year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond',
        'microsecond', 'nanosecond', 'ordinal_day', 'weekday', 'week', 'timestamp'
    }

    def __init__(
        self,
        features: List[str],
        format: Optional[str] = None,
    ):
        # --- Validation ---
        if not isinstance(features, list) or not features:
            _LOGGER.error("'features' must be a non-empty list of strings.")
            raise ValueError()
        
        for feature in features:
            if feature not in self.ALLOWED_FEATURES:
                _LOGGER.error(f"Feature '{feature}' is not supported. Allowed features are: {self.ALLOWED_FEATURES}")
                raise ValueError()

        self.features = features
        self.format = format

    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Applies the feature extraction logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of dates.

        Returns:
            pl.DataFrame: A DataFrame with columns for each extracted feature.
        """
        date_col = column
        # First, parse strings into a datetime object if a format is given
        if self.format is not None:
            date_col = date_col.str.to_datetime(format=self.format, strict=False)

        output_expressions = []
        for i, feature in enumerate(self.features):
            # Build the expression based on the feature name
            if feature == 'timestamp':
                expr = date_col.dt.timestamp(time_unit="ms")
            else:
                # getattr is a clean way to call methods like .dt.year(), .dt.month(), etc.
                expr = getattr(date_col.dt, feature)()
            
            # Alias with a generic name for the processor to handle
            output_expressions.append(expr.alias(f"col_{i}"))
            
        return pl.select(output_expressions)


def info():
    _script_info(__all__)
