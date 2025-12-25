import polars as pl
from pathlib import Path
from typing import Union, Optional

from ..utilities import load_dataframe

from ..path_manager import sanitize_filename, make_fullpath
from .._core import get_logger


_LOGGER = get_logger("ETL Clean Tools")


__all__ = [
    "save_unique_values",
]


################ Unique Values per column #################
def save_unique_values(csv_path_or_df: Union[str, Path, pl.DataFrame], 
                       output_dir: Union[str, Path], 
                       use_columns: Optional[list[str]] = None,
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
            col_expr = pl.col(column_name)
            
            # Check if the column is string-based (String or Utf8)
            dtype = df.schema[column_name]
            if dtype in (pl.String, pl.Utf8):
                 # Filter out actual empty strings AND whitespace-only strings
                dataset = df.select(col_expr).filter(
                    col_expr.str.strip_chars().str.len_chars() > 0
                )
            else:
                dataset = df.select(col_expr)

            # Efficiently get unique non-null values and sort them
            unique_series = dataset.drop_nulls().unique().sort(column_name)
            
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
