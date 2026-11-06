"""One workbook handle, shared parsing rules, either dataframe backend."""

from __future__ import annotations
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Literal
import polars as pl
from collections.abc import Generator

from ._common import (
    LOGGER,
    OOXML_EXTENSIONS,
    add_source,
    input_path,
    output_kind,
    require,
    select_sheets,
    stack_frames,
    unique_names,
    to_polars,
)
from ._types import ExcelReadOptions, Frame, SheetSelection


class ExcelReader:
    """
    Read a workbook repeatedly without reopening it.

    Use as a context manager or call `close()`. 
    
    Defaults to use openpyxl for OOXML and python-calamine for .xls/.xlsb/.ods. The I/O engine is independent
    of the output kind, so pandas and Polars share sheet and cleaning semantics.
    
    Hidden filtering applies before numeric sheet selection.
    """

    def __init__(self, 
                 filepath: str | Path, 
                 *, 
                 options: ExcelReadOptions | None = None, 
                 **read_options: Any
    ) -> None:
        """
        Initialize a reader for a workbook. The file is opened immediately, and the
        workbook is closed when the reader is closed or the context manager exits.
        
        Args:
            filepath (str | Path): Path to the Excel file to read.
            options (ExcelReadOptions | None): Optional read options to customize reading behavior.
            **read_options: Additional keyword arguments to override specific read options.
        """
        self.path = input_path(filepath)
        self.options = (options or ExcelReadOptions()).with_changes(**read_options)
        self.engine = self.options.engine
        if self.engine == "auto":
            self.engine = "openpyxl" if self.path.suffix.lower() in OOXML_EXTENSIONS else "calamine"
        self._book: Any = None
        self._names: list[str] = []
        self._closed = False
        
        if self.engine == "openpyxl":
            if self.path.suffix.lower() not in OOXML_EXTENSIONS:
                LOGGER.error(f"openpyxl cannot read {self.path.suffix}; use engine='calamine'.")
                raise ValueError()
            self._book = require("openpyxl").load_workbook(
                self.path, read_only=True, data_only=self.options.data_only, keep_links=False
            )
            self._names = [
                s.title
                for s in self._book.worksheets
                if self.options.include_hidden or s.sheet_state == "visible"
            ]
        else:
            if not self.options.data_only:
                LOGGER.error("Calamine cannot return formula expressions; use openpyxl for OOXML.")
                raise ValueError()
            calamine = require("python_calamine")
            self._book = calamine.CalamineWorkbook.from_path(str(self.path))
            self._names = [
                s.name
                for s in self._book.sheets_metadata
                if s.typ == calamine.SheetTypeEnum.WorkSheet
                and (self.options.include_hidden or s.visible == calamine.SheetVisibleEnum.Visible)
            ]

    def __enter__(self) -> ExcelReader:
        self._check_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying workbook; safe to call repeatedly."""
        if not self._closed:
            self._book.close()
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            LOGGER.error("Attempted to operate on a closed ExcelReader.")
            raise RuntimeError()

    @property
    def sheet_names(self) -> list[str]:
        """Return the names of visible sheets in the workbook."""
        self._check_open()
        return list(self._names)

    def _rows(self, sheet: str) -> Generator[tuple[Any, ...], None, None]:
        self._check_open()
        if self.engine == "openpyxl":
            # Ignore stale producer dimensions and let the XML supply the extent.
            ws = self._book[sheet]
            ws.reset_dimensions()
            yield from ws.iter_rows(values_only=True)
        else:
            ws = self._book.get_sheet_by_name(sheet)
            # Keep leading blank rows/columns so coordinates match openpyxl.
            for row in ws.to_python(skip_empty_area=False):
                yield tuple(
                    None
                    if value == ""
                    else int(value)
                    if isinstance(value, float) and value.is_integer()
                    else value
                    for value in row
                )

    @contextmanager
    def _header_and_rows(self,
                    sheet: str,
                    options: ExcelReadOptions) -> Generator[tuple[list[str], Iterator[tuple[Any, ...]]], None, None]:
        rows = self._rows(sheet)
        try:
            if options.header_row is None:
                first = next(rows, ())
                header = [f"column_{i}" for i in range(len(first))]
                from itertools import chain

                data = chain([first], rows) if first else rows
            else:
                raw = next(islice(rows, options.header_row, options.header_row + 1), ())
                header = [
                    str(v) if v is not None and str(v).strip() else f"column_{i}"
                    for i, v in enumerate(raw)
                ]
                data = rows
            data = islice(data, options.skip_rows, None)
            if options.n_rows is not None:
                data = islice(data, options.n_rows)
            yield header, data
        finally:
            rows.close()

    def header(self, 
               sheet_name: str | int = 0) -> tuple[str, ...]:
        """Read only the header row
        
        Args:
            sheet_name (str | int): The name or index of the sheet to read the header from. Defaults to the first sheet (index 0).
            
        Returns:
            tuple[str, ...]: The header row as a tuple of strings.
        """
        if not isinstance(sheet_name, (str, int)):
            LOGGER.error("Header inspection requires one worksheet name or index.")
            raise TypeError()
        sheet = select_sheets(self.sheet_names, sheet_name)[0]
        with self._header_and_rows(sheet, self.options.with_changes(n_rows=0)) as (header, _):
            return tuple(self._normalize_header(header, sheet))

    def _normalize_header(self, header: list[str], sheet: str) -> list[str]:
        if len(header) != len(set(header)):
            if self.options.duplicate_headers == "raise":
                LOGGER.error(f"Duplicate column headers in {sheet!r}: {header}")
                raise ValueError()
            return unique_names(header)
        return header

    def _read_frame(self, 
                    sheet: str, 
                    options: ExcelReadOptions) -> pl.DataFrame:
        with self._header_and_rows(sheet, options) as (header, data):
            rows = [list(row) for row in data]
        width = max([len(header)] + [len(row) for row in rows])
        header.extend(f"column_{i}" for i in range(len(header), width))
        header = self._normalize_header(header, sheet)
        rows = [row + [None] * (width - len(row)) for row in rows]
        if options.empty_as_nan:
            rows = [
                [None if isinstance(v, str) and not v.strip() else v for v in row] for row in rows
            ]
        if options.drop_empty_rows:
            rows = [row for row in rows if any(v is not None for v in row)]
        selected = list(range(width))
        if options.use_columns is not None:
            missing = [name for name in options.use_columns if name not in header]
            if missing:
                if options.missing_columns == "raise":
                    LOGGER.error(f"Missing requested columns in {sheet!r}: {missing}")
                    raise ValueError()
                LOGGER.warning(f"Ignoring missing columns in {sheet!r}: {missing}")
            selected = [header.index(name) for name in options.use_columns if name in header]
            if not selected:
                LOGGER.error(f"None of the requested columns exist in {sheet!r}.")
                raise ValueError()
        series = []
        for index in selected:
            name = header[index]
            values = [row[index] for row in rows]
            if options.drop_empty_columns and not any(v is not None for v in values):
                continue
            override = options.schema_overrides.get(name)
            if options.all_strings:
                series.append(
                    pl.Series(
                        name, [None if v is None else str(v) for v in values], dtype=pl.String
                    )
                )
                continue
            groups = {_value_group(v) for v in values if v is not None}
            if len(groups) > 1 and override is None:
                LOGGER.error(f"Mixed types in {sheet!r}, column {name!r}: {groups}")
                raise TypeError()
            try:
                if groups == {"number"}:
                    if any(isinstance(v, float) for v in values if v is not None):
                        if any(
                            isinstance(v, int) and abs(v) > 2**53 for v in values if v is not None
                        ):
                            LOGGER.error(f"Integer/float mixture would lose precision in {sheet!r}, column {name!r}.")
                            raise ValueError()
                        values = [float(v) if v is not None else None for v in values]
                if override is not None:
                    # Build the source without null-coercion, then cast strictly.
                    source = (
                        pl.Series(
                            name, [None if v is None else str(v) for v in values], dtype=pl.String
                        )
                        if len(groups) > 1
                        else pl.Series(name, values)
                    )
                    series.append(source.cast(override, strict=True))
                else:
                    series.append(pl.Series(name, values, strict=True))
            except Exception as exc:
                LOGGER.error(f"Cannot parse {sheet!r}, column {name!r}: {exc}")
                raise TypeError() from exc
            
        unknown = set(options.schema_overrides) - {header[i] for i in selected}
        if unknown:
            LOGGER.error(f"schema_overrides refers to absent/unselected columns: {sorted(unknown)}")
            raise ValueError()
        
        frame = pl.DataFrame(series)
        if options.verbose:
            LOGGER.info(f"Loaded {self.path.name}:{sheet} with shape {frame.shape}.")
        return frame

    def iter_sheets(self, 
                    sheet_name: SheetSelection = None, 
                    *, 
                    source_sheet_column: str | None = None
    ) -> Iterator[tuple[Frame, str]]:
        """
        Yield one whole sheet at a time.
        
        Args:
            sheet_name (SheetSelection): The name(s) or index(es) of the sheet(s) to read. If None, all sheets are read.
            source_sheet_column (str | None): If provided, adds a column with the sheet name to each frame.
            
        Returns:
            Iterator[tuple[Frame, str]]: An iterator of tuples containing the DataFrame and the corresponding sheet name.
        """
        yielded = False
        for name in select_sheets(self.sheet_names, sheet_name):
            frame = self._read_frame(name, self.options)
            if not frame.height:
                if self.options.empty_sheets == "skip":
                    LOGGER.warning(f"Skipping empty worksheet {self.path.name}:{name}.")
                    continue
                LOGGER.error(f"Empty worksheet {self.path.name}:{name}.")
                raise ValueError()
            
            frame = add_source(frame, source_sheet_column, name)
            yielded = True
            yield output_kind(frame, self.options.kind), name
            
        if not yielded:
            LOGGER.error("No nonempty worksheets remain.")
            raise ValueError()

    def read_sheets(self, 
                    sheet_name: SheetSelection = None, 
                    *, 
                    source_sheet_column: str | None = None
    ) -> dict[str, Frame]:
        """
        Read separate frames keyed by worksheet name.
        
        Args:
            sheet_name (SheetSelection): The name(s) or index(es) of the sheet(s) to read. If None, all sheets are read.
            source_sheet_column (str | None): If provided, adds a column with the sheet name to each frame.
            
        Returns:
            dict[str, Frame]: A dictionary mapping sheet names to their corresponding DataFrames.
        """
        result = {
            name: frame
            for frame, name in self.iter_sheets(sheet_name, source_sheet_column=source_sheet_column)
        }
        if not result:
            LOGGER.error("No nonempty worksheets remain.")
            raise ValueError()
        return result

    def read(
        self,
        sheet_name: SheetSelection = 0,
        *,
        schema_mode: Literal["strict", "union"] = "strict",
        source_sheet_column: str | None = None,
        type_conflicts: Literal["raise", "string"] = "raise",
    ) -> Frame:
        """
        Read one sheet or stack selected sheets into a single frame.
        
        Args:
            sheet_name (SheetSelection): The name(s) or index(es) of the sheet(s) to read. If None, the first sheet is read.
            schema_mode ("strict" | "union"): The mode to use when resolving schema conflicts.
            source_sheet_column (str | None): If provided, adds a column with the sheet name to the resulting frame.
            type_conflicts ("raise" | "string"): The mode to use when resolving type conflicts.

        Returns:
            Frame: A single DataFrame containing the data from the selected sheet(s).
        """
        frames = self.read_sheets(sheet_name, source_sheet_column=source_sheet_column)
        combined = stack_frames(
            [to_polars(f) for f in frames.values()],
            list(frames),
            schema_mode=schema_mode,
            type_conflicts=type_conflicts,
        )
        if source_sheet_column is not None:
            combined = combined.select(
                [c for c in combined.columns if c != source_sheet_column] + [source_sheet_column]
            )
        return output_kind(combined, self.options.kind)


def _value_group(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, timedelta):
        return "duration"
    if isinstance(value, Decimal):
        return "decimal"
    return type(value).__name__
