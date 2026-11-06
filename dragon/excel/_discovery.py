"""Deterministic spreadsheet discovery and worksheet listing."""

from pathlib import Path
from typing import Literal

from ..path_manager import make_fullpath

from ._common import OOXML_EXTENSIONS, READ_EXTENSIONS, LOGGER


def find_excel_files(
    directory: str | Path,
    *,
    extensions: tuple[str, ...] | None = None,
    exclude_temp: bool = True,
    recursive: bool = False,
    raise_on_empty: bool = True,
    engine: Literal["auto", "openpyxl", "calamine"] = "auto",
) -> list[Path]:
    """
    Find supported files in a stable order; omit Excel '~$' lock files.
    
    Args:
        directory: Path to search for files.
        extensions: File extensions to include (default: all supported).
        exclude_temp: Exclude temporary lock files (default: True).
        recursive: Search subdirectories (default: False).
        raise_on_empty: Raise if no files found (default: True).
        engine: Excel engine to use ('auto', 'openpyxl', 'calamine').

    Returns:
        list[Path]: A list of Path objects representing the found files.
    """
    directory = Path(make_fullpath(directory, enforce="directory"))
    if engine not in ("auto", "openpyxl", "calamine"):
        LOGGER.error(f"Unknown Excel engine: {engine}")
        raise ValueError()
    supported = OOXML_EXTENSIONS if engine == "openpyxl" else READ_EXTENSIONS
    allowed = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or supported)
    }
    if not allowed.issubset(supported):
        LOGGER.error(f"Extensions {allowed - set(supported)} are unsupported by {engine}.")
        raise ValueError()
    paths = directory.rglob("*") if recursive else directory.iterdir()
    result = sorted(
        (
            p
            for p in paths
            if p.is_file()
            and p.suffix.lower() in allowed
            and (not exclude_temp or not p.name.startswith("~"))
        ),
        key=lambda p: (str(p).casefold(), str(p)),
    )
    if not result and raise_on_empty:
        LOGGER.error(f"No supported spreadsheets found in {directory}.")
        raise FileNotFoundError()
    return result


def list_excel_sheets(
    filepath: str | Path, 
    *, 
    include_hidden: bool = True, 
    engine: Literal["auto", "openpyxl", "calamine"] = "auto"
) -> list[str]:
    """
    Return worksheet names in workbook order, excluding chart sheets.
    
    Args:
        filepath: Path to the Excel file.
        include_hidden: Include hidden sheets.
        engine: Excel engine to use ('auto', 'openpyxl', 'calamine').
        
    Returns:
        list[str]: A list of worksheet names.
    """
    from ._reader import ExcelReader

    with ExcelReader(filepath, engine=engine, include_hidden=include_hidden) as reader:
        return reader.sheet_names
