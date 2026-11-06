"""Convenience functions for loading one frame, a workbook, or an iterator."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator, Literal, overload
import pandas as pd
import polars as pl

from ._reader import ExcelReader
from ._types import ExcelReadOptions, Frame, Kind, SheetSelection


@overload
def load_dataframe_excel(
    df_path: str | Path,
    *,
    use_columns: list[str] | tuple[str, ...] | None = None,
    kind: Literal["pandas"] = "pandas",
    all_strings: bool = False,
    empty_as_nan: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[pd.DataFrame, str]: ...


@overload
def load_dataframe_excel(
    df_path: str | Path,
    *,
    use_columns: list[str] | tuple[str, ...] | None = None,
    kind: Literal["polars"],
    all_strings: bool = False,
    empty_as_nan: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[pl.DataFrame, str]: ...


def load_dataframe_excel(
    df_path: str | Path,
    *,
    use_columns: list[str] | tuple[str, ...] | None = None,
    kind: Kind = "pandas",
    all_strings: bool = False,
    empty_as_nan: bool = True,
    verbose: bool = True,
    sheet_name: SheetSelection = 0,
    schema_mode: Literal["strict", "union"] = "strict",
    source_sheet_column: str | None = None,
    empty_sheets: Literal["raise", "skip"] = "raise",
    type_conflicts: Literal["raise", "string"] = "raise",
    **read_options: Any,
) -> tuple[Frame, str]:
    """Load Excel contents as (DataFrame, file_stem).
    
    Args:
        df_path: Path to the Excel file.
        use_columns: List of column names to read; None reads all columns.
        kind: "pandas" or "polars" for the returned frame type.
        all_strings: If True, convert non-null values to strings.
        empty_as_nan: If True, treat empty cells as NaN/None.
        verbose: If True, print progress messages.
        sheet_name: Sheet selection (zero-based index, name, list, or None for all).
        schema_mode: "strict" or "union" for column alignment across sheets.
        source_sheet_column: Optional column name to store source sheet names.
        empty_sheets: How to handle empty sheets ("raise", "skip").
        type_conflicts: How to handle type conflicts ("raise", "string").
        **read_options: Additional options passed to ExcelReadOptions.
        
    Returns:
        tuple[DataFrame, str]: A tuple containing the loaded DataFrame and the file stem.
    """
    if read_options.get("schema_overrides", {}) is None:
        read_options["schema_overrides"] = {}
    options = ExcelReadOptions(
        kind=kind,
        use_columns=use_columns,
        all_strings=all_strings,
        empty_as_nan=empty_as_nan,
        verbose=verbose,
        empty_sheets=empty_sheets,
        **read_options,
    )
    with ExcelReader(df_path, options=options) as reader:
        return reader.read(
            sheet_name,
            schema_mode=schema_mode,
            source_sheet_column=source_sheet_column,
            type_conflicts=type_conflicts,
        ), reader.path.stem


def load_workbook_dataframes(
    df_path: str | Path,
    *,
    sheet_name: SheetSelection = None,
    options: ExcelReadOptions | None = None,
    source_sheet_column: str | None = None,
    **read_options: Any,
) -> dict[str, Frame]:
    """
    Load unrelated sheets separately, preserving selection order and names.
    
    Args:
        df_path: Path to the Excel file.
        sheet_name: Sheet selection (zero-based index, name, list, or None for all).
        options: Optional ExcelReadOptions instance.
        source_sheet_column: Optional column name to store source sheet names.
        **read_options: Additional options passed to ExcelReader.
    
    Returns:
        dict[str, DataFrame]: A dictionary of sheet names to DataFrames.
    """
    with ExcelReader(df_path, options=options, **read_options) as reader:
        return reader.read_sheets(sheet_name, source_sheet_column=source_sheet_column)


def yield_dataframes_from_excel(
    df_path: str | Path,
    *,
    sheet_name: SheetSelection = None,
    options: ExcelReadOptions | None = None,
    source_sheet_column: str | None = None,
    **read_options: Any,
) -> Iterator[tuple[Frame, str]]:
    """Yield (DataFrame, sheet_name); close this generator when abandoning it early.

    For guaranteed early-exit cleanup use ``contextlib.closing(generator)`` or
    ``with ExcelReader(...) as reader: ... reader.iter_sheets()``. Each sheet is
    materialized in memory; calamine may also materialize sheet cells internally.
    
    Args:
        df_path: Path to the Excel file.
        sheet_name: Sheet selection (zero-based index, name, list, or None for all).
        options: Optional ExcelReadOptions instance.
        source_sheet_column: Optional column name to store source sheet names.
        **read_options: Additional options passed to ExcelReader.
        
    Yields:
        tuple[DataFrame, str]: A tuple containing the loaded DataFrame and the sheet name
    """
    with ExcelReader(df_path, options=options, **read_options) as reader:
        yield from reader.iter_sheets(sheet_name, source_sheet_column=source_sheet_column)
