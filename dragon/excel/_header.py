"""Worksheet header inspection without constructing a dataframe."""

from pathlib import Path
from typing import Any

from ._reader import ExcelReader


def load_excel_header(
    df_path: str | Path, 
    sheet_name: str | int = 0, 
    verbose: bool = True, 
    **read_options: Any
) -> tuple[str, ...]:
    """
    Return normalized header names; header_row and engine are configurable.
    
    Args:
        df_path (str | Path): Path to the Excel file.
        sheet_name (str | int, optional): Name or index of the sheet to read.
        verbose (bool, optional): Whether to print progress messages.
        **read_options: Additional keyword arguments to pass to the ExcelReader.
        
    Returns:
        tuple[str, ...]: A tuple of normalized header names.
    """
    with ExcelReader(df_path, verbose=verbose, **read_options) as reader:
        return reader.header(sheet_name)
