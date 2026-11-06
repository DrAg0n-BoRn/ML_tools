"""Worksheet unmerging and splitting."""

from __future__ import annotations
from copy import copy
from pathlib import Path
from typing import Any, Literal

from ..path_manager import make_fullpath

from ._common import (
    OOXML_EXTENSIONS,
    atomic_output,
    input_path,
    require,
    safe_stem,
    select_sheets,
    unique_names,
    LOGGER,
)
from ._types import SheetSelection


__all__ = [
    "unmerge_excel_cells",
    "split_excel_sheets",
]


def _open_editable(filepath: str | Path):
    path = input_path(filepath)
    if path.suffix.lower() not in OOXML_EXTENSIONS:
        LOGGER.error(f"Cell editing requires an OOXML workbook: {OOXML_EXTENSIONS}")
        raise ValueError()
    return path, require("openpyxl").load_workbook(
        path, keep_vba=path.suffix.lower() in (".xlsm", ".xltm")
    )


def _unmerge_sheet(ws: Any, *, fill: bool = True, copy_styles: bool = True) -> None:
    for region in list(ws.merged_cells.ranges):
        top = ws.cell(region.min_row, region.min_col)
        value, data_type, style = top.value, top.data_type, copy(top._style)
        comment, hyperlink = copy(top.comment), copy(top.hyperlink)
        ws.unmerge_cells(str(region))
        for row in ws.iter_rows(
            min_row=region.min_row,
            max_row=region.max_row,
            min_col=region.min_col,
            max_col=region.max_col,
        ):
            for cell in row:
                if fill:
                    cell.value = value
                    cell.data_type = data_type
                if copy_styles:
                    cell._style = copy(style)
                    cell.comment = copy(comment)
                    if hyperlink is not None and cell is not top:
                        link = copy(hyperlink)
                        link.ref = cell.coordinate
                        # The property setter may fill an empty cell with the URL.
                        old_value, old_type = cell.value, cell.data_type
                        cell.hyperlink = link
                        cell.value = old_value
                        cell.data_type = old_type


def unmerge_excel_cells(
    filepath: str | Path,
    output_path: str | Path | None = None,
    *,
    sheet_name: SheetSelection = None,
    fill: bool = True,
    copy_styles: bool = True,
    overwrite: bool = False,
) -> Path:
    """Unmerge selected sheets, optionally filling each region's top-left value.

    Preserve other sheets. Formulas are copied literally, not translated.
    The default output is '<stem>_unmerged<suffix>'; the input is never silently
    overwritten.
    
    Args:
        filepath: Path to the input workbook.
        output_path: Optional path for the output workbook. If None, a new file
            is created in the same directory with '_unmerged' appended to the stem.
        sheet_name: Optional sheet name or list of names to unmerge. If None,
            all sheets are processed.
        fill: If True, fill each unmerged cell with the value from the top-left cell of the merged region. If False, unmerged cells will be empty.
        copy_styles: If True, copy cell styles, comments, and hyperlinks from the top-left cell of each merged region to the unmerged cells. If False, styles will not be copied.
        overwrite: If True, allow overwriting the output file if it already exists. If False, raise an error if the output file exists.
        
    Returns:
        Path: The path to the created output workbook.
    """
    path, workbook = _open_editable(filepath)
    target = (
        Path(make_fullpath(output_path))
        if output_path
        else path.with_name(f"{path.stem}_unmerged{path.suffix}")
    )
    try:
        if target.suffix.lower() != path.suffix.lower():
            LOGGER.error("Unmerge output must keep the source extension (including macro/template format).")
            raise ValueError()
        names = select_sheets([s.title for s in workbook.worksheets], sheet_name)
        for name in names:
            _unmerge_sheet(workbook[name], fill=fill, copy_styles=copy_styles)
        with atomic_output(target, overwrite=overwrite) as temp:
            workbook.save(temp)
    finally:
        workbook.close()
    return target


def _copy_style(source: Any, destination: Any) -> None:
    """Register style components in the destination workbook, not source IDs."""
    for attribute in ("font", "fill", "border", "alignment", "protection"):
        setattr(destination, attribute, copy(getattr(source, attribute)))
    destination.number_format = source.number_format
    destination.quotePrefix = source.quotePrefix
    destination.pivotButton = source.pivotButton


def _copy_sheet(source: Any, destination: Any, *, copy_styles: bool) -> None:
    for row in source.iter_rows():
        for cell in row:
            target = destination.cell(cell.row, cell.column, cell.value)
            target.data_type = cell.data_type
            if copy_styles:
                _copy_style(cell, target)
                target.comment = copy(cell.comment)
                if cell.hyperlink:
                    target.hyperlink = copy(cell.hyperlink)
    for region in source.merged_cells.ranges:
        destination.merge_cells(str(region))
    if copy_styles:
        for key, dim in source.row_dimensions.items():
            copied = copy(dim)
            copied.parent = destination
            copied._style = None
            _copy_style(dim, copied)
            destination.row_dimensions[key] = copied
        for key, dim in source.column_dimensions.items():
            copied = copy(dim)
            copied.parent = destination
            copied._style = None
            _copy_style(dim, copied)
            destination.column_dimensions[key] = copied
        destination.freeze_panes = source.freeze_panes
        destination.auto_filter = copy(source.auto_filter)


def _split_output_paths(path: Path, names: list[str], directory: Path) -> list[Path]:
    stems = unique_names([safe_stem(f"{path.stem}_{name}") for name in names])
    used: set[str] = set()
    outputs: list[Path] = []
    for stem in stems:
        base, index = stem, 1
        while stem.casefold() in used:
            stem = f"{base}_{index}"
            index += 1
        used.add(stem.casefold())
        outputs.append(directory / f"{stem}.xlsx")
    return outputs


def split_excel_sheets(
    filepath: str | Path,
    output_dir: str | Path | None = None,
    *,
    sheet_name: SheetSelection = None,
    unmerge: bool = False,
    fill: bool = True,
    copy_styles: bool = True,
    formula_mode: Literal["values", "formulas"] = "values",
    overwrite: bool = False,
) -> list[Path]:
    """Split worksheets into .xlsx files, optionally unmerging in the same pass.

    Charts, images, macros, tables, validations and workbook defined names are
    NOT copied. Use this for table extraction, not lossless workbook cloning.
    
    Each output is atomic.
    
    Args:
        filepath: Path to the input workbook.
        output_dir: Optional directory for the output workbooks. If None, files are created in the same directory as the input workbook.
        sheet_name: Optional sheet name or list of names to split. If None, all sheets are processed.
        unmerge: If True, unmerge cells in the output sheets. If False, merged cells are preserved.
        fill: If True and unmerge is True, fill each unmerged cell with the value from the top-left cell of the merged region. If False, unmerged cells will be empty.
        copy_styles: If True, copy cell styles, comments, and hyperlinks from the source sheets to the output sheets. If False, styles will not be copied.
        formula_mode: 'values' to export cached formula results, 'formulas' to copy expressions literally.
        overwrite: If True, allow overwriting the output files if they already exist. If False, raise an error if any output file exists.
    
    Returns:
        list[Path]: A list of Paths to the created output workbooks.
    """
    if formula_mode not in ("values", "formulas"):
        LOGGER.error("formula_mode must be 'values' or 'formulas'.")
        raise ValueError()
    path = input_path(filepath)
    if path.suffix.lower() not in OOXML_EXTENSIONS:
        raise ValueError("Worksheet copying requires an OOXML workbook.")
    ox = require("openpyxl")
    workbook = ox.load_workbook(path, data_only=formula_mode == "values")
    directory = Path(make_fullpath(output_dir)) if output_dir else path.parent
    try:
        names = select_sheets([s.title for s in workbook.worksheets], sheet_name)
        outputs = _split_output_paths(path, names, directory)
        for target in outputs:
            if target == path or (target.exists() and not overwrite):
                raise FileExistsError(
                    f"Split output would overwrite an input/existing file: {target}"
                )
        for name, target in zip(names, outputs, strict=True):
            new_book = ox.Workbook()
            try:
                # Theme-based colors and indexed colors belong to the workbook.
                new_book.loaded_theme = workbook.loaded_theme
                new_book._colors = copy(workbook._colors)
                ws = new_book.active
                ws.title = name
                _copy_sheet(workbook[name], ws, copy_styles=copy_styles)
                if unmerge:
                    _unmerge_sheet(ws, fill=fill, copy_styles=copy_styles)
                with atomic_output(target, overwrite=overwrite) as temp:
                    new_book.save(temp)
            finally:
                new_book.close()
    finally:
        workbook.close()
    return outputs
