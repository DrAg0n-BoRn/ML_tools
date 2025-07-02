import polars as pl
import re
from typing import Literal, Union, Optional, Any, Callable, List, Dict
from .utilities import _script_info
import pandas as pd


__all__ = [
    "ColumnCleaner",
    "DataFrameCleaner"
    "TransformationRecipe",
    "DataProcessor",
    "KeywordDummifier",
    "NumberExtractor",
    "MultiNumberExtractor",
    "RatioCalculator"
    "CategoryMapper",
    "RegexMapper",
    "ValueBinner",
    "DateFeatureExtractor"
]

########## EXTRACT and CLEAN ##########

class ColumnCleaner:
    """
    Cleans and standardizes a single pandas Series based on a dictionary of regex-to-value replacement rules.
    
    Args:
        rules (Dict[str, str]):
            A dictionary where each key is a regular expression pattern and
            each value is the standardized string to replace matches with.
    """
    def __init__(self, rules: Dict[str, str]):
        if not isinstance(rules, dict):
            raise TypeError("The 'rules' argument must be a dictionary.")

        # Validate that all keys are valid regular expressions
        for pattern in rules.keys():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        self.rules = rules

    def clean(self, series: pd.Series) -> pd.Series:
        """
        Applies the standardization rules to the provided Series (requires string data).
        
        Non-matching values are kept as they are.

        Args:
            series (pd.Series): The pandas Series to clean.

        Returns:
            pd.Series: A new Series with the values cleaned and standardized.
        """
        return series.astype(str).replace(self.rules, regex=True)


class DataFrameCleaner:
    """
    Orchestrates the cleaning of multiple columns in a pandas DataFrame using a nested dictionary of rules and `ColumnCleaner` objects.

    Args:
        rules (Dict[str, Dict[str, str]]):
            A nested dictionary where each top-level key is a column name,
            and its value is a dictionary of regex rules for that column, as expected by `ColumnCleaner`.
    """
    def __init__(self, rules: Dict[str, Dict[str, str]]):
        if not isinstance(rules, dict):
            raise TypeError("The 'rules' argument must be a nested dictionary.")
        
        for col_name, col_rules in rules.items():
            if not isinstance(col_rules, dict):
                raise TypeError(
                    f"The value for column '{col_name}' must be a dictionary "
                    f"of rules, but got type {type(col_rules).__name__}."
                )
        
        self.rules = rules

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all defined cleaning rules to the DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame to clean.

        Returns:
            pd.DataFrame: A new, cleaned DataFrame.
        """
        rule_columns = set(self.rules.keys())
        df_columns = set(df.columns)
        
        missing_columns = rule_columns - df_columns
        
        if missing_columns:
            # Report all missing columns in a single, clear error message
            raise ValueError(
                f"The following columns specified in the cleaning rules "
                f"were not found in the DataFrame: {sorted(list(missing_columns))}"
            )
        
        # Start the process
        df_cleaned = df.copy()
        
        for column_name, column_rules in self.rules.items():
            # Create and apply the specific cleaner for the column
            cleaner = ColumnCleaner(rules=column_rules)
            df_cleaned[column_name] = cleaner.clean(df_cleaned[column_name])
            
        return df_cleaned


############ TRANSFORM ####################

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
            raise TypeError("'input_col' must be a non-empty string.")
            
        if transform == _RENAME:
            if not isinstance(output_col_names, str):
                raise TypeError("For a RENAME operation, 'output_col' must be a string.")
        elif not isinstance(transform, Callable):
            raise TypeError(f"'transform' must be a callable function or the string '{_RENAME}'.")

        if isinstance(output_col_names, list) and transform == _RENAME:
            raise ValueError("A RENAME operation cannot have a list of output columns.")
        
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
            raise TypeError("The recipe must be an instance of TransformationRecipe.")
        if len(recipe) == 0:
            raise ValueError("The recipe cannot be empty.")
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
                raise ValueError(f"Input column '{input_col_name}' not found in DataFrame.")

            input_series = df.get_column(input_col_name)

            if transform_action == _RENAME:
                processed_columns.append(input_series.alias(output_col_spec))
                continue

            if isinstance(transform_action, Callable):
                result = transform_action(input_series)

                if isinstance(result, pl.Series):
                    if not isinstance(output_col_spec, str):
                        raise TypeError(f"Function for '{input_col_name}' returned a Series but 'output_col' is not a string.")
                    processed_columns.append(result.alias(output_col_spec))
                
                elif isinstance(result, pl.DataFrame):
                    if not isinstance(output_col_spec, list):
                        raise TypeError(f"Function for '{input_col_name}' returned a DataFrame but 'output_col' is not a list.")
                    if len(result.columns) != len(output_col_spec):
                        raise ValueError(
                            f"Mismatch in '{input_col_name}': function produced {len(result.columns)} columns, "
                            f"but recipe specifies {len(output_col_spec)} output names."
                        )
                    
                    renamed_df = result.rename(dict(zip(result.columns, output_col_spec)))
                    processed_columns.extend(renamed_df.get_columns())
                
                else:
                    raise TypeError(f"Function for '{input_col_name}' returned an unexpected type: {type(result)}.")
            
            else: # This case is now unlikely due to builder validation.
                raise TypeError(f"Invalid 'transform' action for '{input_col_name}': {transform_action}")

        if not processed_columns:
            print("Warning: The transformation resulted in an empty DataFrame.")
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


class KeywordDummifier:
    """
    A configurable transformer that creates one-hot encoded columns based on
    keyword matching in a Polars Series.

    Instantiate this class with keyword configurations. The instance can be used as a 'transform' callable compatible with the `TransformationRecipe`.

    Args:
        group_names (List[str]): 
            A list of strings, where each string is the name of a category.
            This defines the matching priority and the base column names of the
            DataFrame returned by the transformation.
        group_keywords (List[List[str]]): 
            A list of lists of strings. Each inner list corresponds to a 
            `group_name` at the same index and contains the keywords to search for.
    """
    def __init__(self, group_names: List[str], group_keywords: List[List[str]]):
        if len(group_names) != len(group_keywords):
            raise ValueError("Initialization failed: 'group_names' and 'group_keywords' must have the same length.")
        
        self.group_names = group_names
        self.group_keywords = group_keywords

    def __call__(self, column: pl.Series) -> pl.DataFrame:
        """
        Executes the one-hot encoding logic.

        Args:
            column (pl.Series): The input Polars Series to transform.

        Returns:
            pl.DataFrame: A DataFrame with one-hot encoded columns.
        """
        column = column.cast(pl.Utf8)
        
        categorize_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for name, keywords in zip(self.group_names, self.group_keywords):
            pattern = "|".join(re.escape(k) for k in keywords)
            categorize_expr = categorize_expr.when(
                column.str.contains(pattern)
            ).then(pl.lit(name))
        
        categorize_expr = categorize_expr.otherwise(None).alias("category")

        temp_df = pl.DataFrame(categorize_expr)
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
                final_columns.append(pl.lit(0, dtype=pl.UInt8).alias(name))

        return pl.DataFrame(final_columns)


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
        round_digits: Optional[int] = None,
    ):
        # --- Validation ---
        if not isinstance(regex_pattern, str):
            raise TypeError("regex_pattern must be a string.")
        
        # Validate that the regex has exactly one capturing group
        try:
            if re.compile(regex_pattern).groups != 1:
                raise ValueError("regex_pattern must contain exactly one capturing group '(...)'")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        if dtype not in ["float", "int"]:
            raise ValueError("dtype must be either 'float' or 'int'.")
            
        if round_digits is not None:
            if not isinstance(round_digits, int):
                raise TypeError("round_digits must be an integer.")
            if dtype == "int":
                print(f"Warning: 'round_digits' is specified but dtype is 'int'. Rounding will be ignored.")

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
            raise ValueError("num_outputs must be a positive integer.")
        
        if not isinstance(regex_pattern, str):
            raise TypeError("regex_pattern must be a string.")
        
        # Validate that the regex has exactly one capturing group
        try:
            if re.compile(regex_pattern).groups != 1:
                raise ValueError("regex_pattern must contain exactly one capturing group '(...)'")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided: {e}") from e
        
        # Validate dtype
        if dtype not in ["float", "int"]:
            raise ValueError("dtype must be either 'float' or 'int'.")
        
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
    A transformer that parses a string ratio (e.g., "40:5" or "30/2") and computes the result of the division.

    Args:
        regex_pattern (str, optional):
            The regex pattern to find the numerator and denominator. It MUST
            contain exactly two capturing groups: the first for the
            numerator and the second for the denominator. Defaults to a
            pattern that handles common delimiters like ':' and '/'.
    """
    def __init__(
        self,
        regex_pattern: str = r"(\d+\.?\d*)\s*[:/]\s*(\d+\.?\d*)"
    ):
        # --- Validation ---
        try:
            if re.compile(regex_pattern).groups != 2:
                raise ValueError(
                    "regex_pattern must contain exactly two "
                    "capturing groups '(...)'."
                )
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

        self.regex_pattern = regex_pattern

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the ratio calculation logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of ratio strings.

        Returns:
            pl.Series: A new Series of floats containing the division result.
                       Returns null for invalid formats or division by zero.
        """
        # .extract_groups returns a struct with a field for each capture group
        # e.g., {"group_1": "40", "group_2": "5"}
        groups = column.str.extract_groups(self.regex_pattern)

        # Extract numerator and denominator, casting to float
        # strict=False ensures that non-matches become null
        numerator = groups.struct.field("group_1").cast(pl.Float64, strict=False)
        denominator = groups.struct.field("group_2").cast(pl.Float64, strict=False)

        # Safely perform division, returning null if denominator is 0
        return pl.when(denominator != 0).then(
            numerator / denominator
        ).otherwise(None)


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
            raise TypeError("The 'mapping' argument must be a dictionary.")
        
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
        mapping (Dict[str, Union[int, float]]):
            An ordered dictionary where keys are regex patterns and values are
            the numbers to map to if the pattern is found.
        unseen_value (Optional[Union[int, float]], optional):
            The numerical value to use for strings that do not match any
            of the regex patterns. If None (default), unseen values are
            mapped to null.
    """
    def __init__(
        self,
        mapping: Dict[str, Union[int, float]],
        unseen_value: Optional[Union[int, float]] = None,
    ):
        # --- Validation ---
        if not isinstance(mapping, dict):
            raise TypeError("The 'mapping' argument must be a dictionary.")
        
        for pattern, value in mapping.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
            if not isinstance(value, (int, float)):
                raise TypeError(f"Mapping values must be int or float, but got {type(value)} for pattern '{pattern}'.")

        self.mapping = mapping
        self.unseen_value = unseen_value

    def __call__(self, column: pl.Series) -> pl.Series:
        """
        Applies the regex mapping logic to the input column.

        Args:
            column (pl.Series): The input Polars Series of string data.

        Returns:
            pl.Series: A new Series with strings mapped to numbers based on
                       the first matching regex pattern.
        """
        # Ensure the column is treated as a string for matching
        str_column = column.cast(pl.Utf8)

        # Build the when/then/otherwise chain from the inside out.
        # Start with the final fallback value for non-matches.
        mapping_expr = pl.lit(self.unseen_value)

        # Iterate through the mapping in reverse to construct the nested expression
        for pattern, value in reversed(list(self.mapping.items())):
            mapping_expr = (
                pl.when(str_column.str.contains(pattern))
                .then(pl.lit(value))
                .otherwise(mapping_expr)
            )
        
        # Execute the complete expression chain and return the resulting Series
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
            raise ValueError("The 'breaks' argument must be a list of at least two numbers.")
        
        # Check if the list is sorted
        if not all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1)):
            raise ValueError("The 'breaks' list must be sorted in ascending order.")

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
            raise ValueError("'features' must be a non-empty list of strings.")
        
        for feature in features:
            if feature not in self.ALLOWED_FEATURES:
                raise ValueError(
                    f"Feature '{feature}' is not supported. "
                    f"Allowed features are: {self.ALLOWED_FEATURES}"
                )

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
