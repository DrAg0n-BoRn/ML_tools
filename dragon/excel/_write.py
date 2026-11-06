"""Transactional single/multi-sheet dataframe exports and sheet replacement."""

from __future__ import annotations
import math
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Literal
from itertools import chain
import polars as pl

from ..path_manager import make_fullpath

from ._common import atomic_output, require, to_polars, LOGGER
from ._types import Frame


def _validate_names(names: list[str]) -> None:
    if not names:
        LOGGER.error("At least one worksheet is required.")
        raise ValueError()
    if any(not isinstance(n, str) for n in names):
        LOGGER.error("Worksheet names must be strings.")
        raise TypeError()
    if len({n.casefold() for n in names}) != len(names):
        LOGGER.error("Worksheet names must be unique (case-insensitive).")
        raise ValueError()
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 31
            or re.search(r"[\[\]:*?/\\\x00-\x1f]", name)
            or name.startswith("'")
            or name.endswith("'")
        ):
            LOGGER.error(f"Invalid Excel worksheet name: {name}")
            raise ValueError()


def _cell_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Excel stores numeric cells as floating point. Keep Decimal values exact.
        value = str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53:
        return str(value)  # Excel numbers cannot represent these IDs exactly.
    if isinstance(value, (datetime, time)) and value.tzinfo is not None:
        LOGGER.error("Excel cannot store timezone-aware date/time values; remove/convert the timezone first.")
        raise ValueError()
    if isinstance(value, (dict, list, tuple, bytes)):
        LOGGER.error("Nested/binary cells must be serialized explicitly before Excel export.")
        raise TypeError()
    if isinstance(value, str) and len(value) > 32767:
        LOGGER.error("Excel cell text exceeds the 32767-character limit.")
        raise ValueError()
    return value


def save_dataframe_excel(df: Frame, 
                         full_path: str | Path, 
                         *, 
                         sheet_name: str = "Data",
                         **write_options: Any
                        ) -> Path:
    """
    Save one Excel sheet. 
    
    Accepts all `save_workbook_dataframes` write options.
    
    Args:
        df: DataFrame to save.
        full_path: Path to save the Excel file.
        sheet_name: Name of the single worksheet to create.
        write_options: Additional options passed to `save_workbook_dataframes`:
            - overwrite: If True, allow overwriting the output file if it already exists.
            - mode: 'w' to write a new workbook, 'a' to append/replace sheets in an existing workbook.
            - if_sheet_exists: 'error' to raise an error if a sheet already exists, 'replace' to overwrite the existing sheet.
            - freeze_panes: Optional (row, column) tuple to freeze panes at the specified position. If None, no panes are frozen.
            - autofilter: If True, apply an autofilter to the header row.
            - autofit: If True, automatically adjust column widths based on content.
            - column_widths: Optional mapping of column names to explicit widths. Overrides autofit.
            - start_row: Zero-based row index to start writing data.
            - start_column: Zero-based column index to start writing data.
            - include_header: If True, include the DataFrame's column names as the first row.
            - formulas: If True, treat strings starting with '=' as formulas; otherwise, treat them as literal text.
    
    Returns:
        Path: Absolute path to the saved Excel file.
    """
    
    return save_workbook_dataframes({sheet_name: df}, 
                                    full_path, 
                                    **write_options)


def save_workbook_dataframes(
    dataframes: Mapping[str, Frame],
    full_path: str | Path,
    *,
    overwrite: bool = False,
    mode: Literal["w", "a"] = "w",
    if_sheet_exists: Literal["error", "replace"] = "error",
    freeze_panes: tuple[int, int] | None = (1, 0),
    autofilter: bool = True,
    autofit: bool = False,
    column_widths: Mapping[str, float] | None = None,
    start_row: int = 0,
    start_column: int = 0,
    include_header: bool = True,
    formulas: bool = False,
) -> Path:
    """
    Save named pandas/Polars frames to .xlsx and return its absolute path.
    
    Args:
        dataframes: Mapping of sheet names to DataFrames to save.
        full_path: Path to save the Excel file.
        overwrite: If True, allow overwriting the output file if it already exists.
        mode: 'w' to write a new workbook, 'a' to append/replace sheets in an existing workbook.
        if_sheet_exists: 'error' to raise an error if a sheet already exists, 'replace' to overwrite the existing sheet.
        freeze_panes: Optional (row, column) tuple to freeze panes at the specified position. If None, no panes are frozen.
        autofilter: If True, apply an autofilter to the header row.
        autofit: If True, automatically adjust column widths based on content.
        column_widths: Optional mapping of column names to explicit widths. Overrides autofit.
        start_row: Zero-based row index to start writing data.
        start_column: Zero-based column index to start writing data.
        include_header: If True, include the DataFrame's column names as the first row.
        formulas: If True, treat strings starting with '=' as formulas; otherwise, treat them as literal text.
        
    Returns:
        Path: Absolute path to the saved Excel file.
    """
    path = make_fullpath(full_path)
    if path.suffix.lower() != ".xlsx":
        LOGGER.error("Dataframe export writes .xlsx only.")
        raise ValueError()
    if mode not in ("w", "a") or if_sheet_exists not in ("error", "replace"):
        LOGGER.error("mode must be 'w'/'a'; if_sheet_exists must be 'error'/'replace'.")
        raise ValueError()
    for value in (start_row, start_column):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            LOGGER.error("Output row/column offsets must be nonnegative integers.")
            raise ValueError()
    if freeze_panes is not None and (
        len(freeze_panes) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in freeze_panes)
    ):
        LOGGER.error("freeze_panes must be two nonnegative integers or None.")
        raise ValueError()
    _validate_names(list(dataframes))
    frames = {name: to_polars(frame) for name, frame in dataframes.items()}
    for name, frame in frames.items():
        if include_header:
            for column in frame.columns:
                _cell_value(column)
        if not frame.width:
            LOGGER.error(f"Cannot export a zero-column DataFrame: {name}")
            raise ValueError()
        if (
            start_row + frame.height + int(include_header) > 1_048_576
            or start_column + frame.width > 16_384
        ):
            LOGGER.error(f"{name!r} exceeds Excel's worksheet dimensions.")
            raise ValueError()
    if mode == "a" and (not path.is_file() or not overwrite):
        LOGGER.error("mode='a' requires an existing workbook and overwrite=True.")
        raise ValueError()
    widths = dict(column_widths or {})
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in widths.values()):
        LOGGER.error("Column widths must be positive numbers.")
        raise ValueError()
    if mode == "a":
        _append(
            frames,
            path,
            if_sheet_exists=if_sheet_exists,
            freeze_panes=freeze_panes,
            autofilter=autofilter,
            autofit=autofit,
            widths=widths,
            start_row=start_row,
            start_column=start_column,
            include_header=include_header,
            formulas=formulas,
        )
        return path
    xlsxwriter = require("xlsxwriter")
    with atomic_output(path, overwrite=overwrite) as temp:
        with xlsxwriter.Workbook(
            str(temp), {"strings_to_formulas": formulas, "strings_to_urls": False}
        ) as workbook:
            header_format = workbook.add_format({"bold": True, "bg_color": "#E8EEF7"})
            date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
            datetime_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
            time_format = workbook.add_format({"num_format": "hh:mm:ss"})
            duration_format = workbook.add_format({"num_format": "[h]:mm:ss"})
            for name, frame in frames.items():
                ws = workbook.add_worksheet(name)
                if include_header:
                    for c, column in enumerate(frame.columns, start_column):
                        result = ws.write_string(start_row, c, column, header_format)
                        if result != 0:
                            raise ValueError(f"Excel writer rejected header {name}:{c + 1} (code {result}).")
                sizes = [len(c) for c in frame.columns]
                for r, row in enumerate(frame.iter_rows(), start_row + int(include_header)):
                    for j, raw in enumerate(row):
                        value = _cell_value(raw)
                        sizes[j] = max(sizes[j], len(str(value)) if value is not None else 0)
                        fmt = (
                            datetime_format
                            if isinstance(value, datetime)
                            else date_format
                            if isinstance(value, date)
                            else time_format
                            if isinstance(value, time)
                            else duration_format
                            if isinstance(value, timedelta)
                            else None
                        )
                        result = ws.write(r, start_column + j, value, fmt)
                        if result != 0:
                            LOGGER.error(f"Excel writer rejected cell {name}:{r + 1},{start_column + j + 1} (code {result}).")
                            raise ValueError()
                if freeze_panes is not None:
                    ws.freeze_panes(*freeze_panes)
                if autofilter and include_header:
                    ws.autofilter(
                        start_row,
                        start_column,
                        start_row + frame.height,
                        start_column + frame.width - 1,
                    )
                for j, column in enumerate(frame.columns):
                    width = widths.get(column, min(sizes[j] + 2, 60) if autofit else None)
                    if width is not None:
                        ws.set_column(start_column + j, start_column + j, width)
    return path


def _append(frames: dict[str, pl.DataFrame], path: Path, **options: Any) -> None:
    ox = require("openpyxl")
    workbook = ox.load_workbook(path)
    try:
        for name, frame in frames.items():
            matching = next(
                (n for n in workbook.sheetnames if n.casefold() == name.casefold()), None
            )
            position = len(workbook.sheetnames)
            if matching is not None:
                if options["if_sheet_exists"] == "error":
                    LOGGER.error(f"Worksheet {name!r} already exists.")
                    raise FileExistsError()
                position = workbook.sheetnames.index(matching)
                del workbook[matching]
            ws = workbook.create_sheet(name, position)
            r0, c0 = options["start_row"] + 1, options["start_column"] + 1
            rows = [tuple(frame.columns)] if options["include_header"] else []

            sizes = [len(c) for c in frame.columns]
            for r, row in enumerate(chain(rows, frame.iter_rows()), r0):
                for j, raw in enumerate(row):
                    value = _cell_value(raw)
                    cell = ws.cell(r, c0 + j, value)
                    if isinstance(value, str):
                        is_header = options["include_header"] and r == r0
                        cell.data_type = (
                            "f" if options["formulas"] and not is_header and value.startswith("=")
                            else "s"
                        )
                    elif isinstance(value, timedelta):
                        cell.number_format = "[h]:mm:ss"
                    sizes[j] = max(sizes[j], len(str(value)) if value is not None else 0)
            if options["freeze_panes"] is not None:
                r, c = options["freeze_panes"]
                ws.freeze_panes = ws.cell(r + 1, c + 1).coordinate
            if options["autofilter"] and options["include_header"]:
                ws.auto_filter.ref = f"{ws.cell(r0, c0).coordinate}:{ws.cell(r0 + frame.height, c0 + frame.width - 1).coordinate}"
            for j, column in enumerate(frame.columns):
                width = options["widths"].get(
                    column, min(sizes[j] + 2, 60) if options["autofit"] else None
                )
                if width is not None:
                    ws.column_dimensions[ox.utils.get_column_letter(c0 + j)].width = width
        with atomic_output(path, overwrite=True) as temp:
            workbook.save(temp)
    finally:
        workbook.close()
