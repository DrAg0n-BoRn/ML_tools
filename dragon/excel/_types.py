"""Public types and immutable defaults for Excel ingestion."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias
import pandas as pd
import polars as pl

from .._core import get_logger

_LOGGER = get_logger("Excel Config")

Frame: TypeAlias = pd.DataFrame | pl.DataFrame
SheetSelection: TypeAlias = str | int | list[str | int] | list[str] | list[int] | tuple[str | int, ...] | None
Kind: TypeAlias = Literal["pandas", "polars"]


@dataclass(frozen=True, slots=True)
class ExcelReadOptions:
    """
    Reusable reader configuration. Row offsets are zero-based worksheet rows.
    
    Components:
        kind: "pandas" or "polars" for the returned frame type.
        use_columns: Explicit column names to read, or None for all columns.
        all_strings: If True, read all columns as strings.
        empty_as_nan: If True, treat empty cells as NaN.
        verbose: If True, log warnings and info messages.
        engine: "auto", "openpyxl", or "calamine" for the Excel reader backend.
        header_row: Zero-based row index for column names, or None for auto-names.
        skip_rows: Zero-based row count to skip after the header row.
        n_rows: Maximum number of physical rows to read, or None for all rows.
        drop_empty_rows: If True, drop rows that are entirely empty.
        drop_empty_columns: If True, drop columns that are entirely empty.
        duplicate_headers: "raise" or "suffix" for duplicate column names.
        missing_columns: "raise" or "ignore" for missing columns in use_columns.
        empty_sheets: "raise" or "skip" for sheets with no data rows.
        include_hidden: If True, include hidden sheets in the read.
        data_only: If True, read cached formula results without recalculation.
        schema_overrides: Mapping of column names to Polars dtypes for strict casting.
    """

    kind: Kind = "pandas"
    use_columns: tuple[str, ...] | list[str] | None = None
    all_strings: bool = False
    empty_as_nan: bool = True
    verbose: bool = True
    engine: Literal["auto", "openpyxl", "calamine"] = "auto"
    header_row: int | None = 0
    skip_rows: int = 0
    n_rows: int | None = None
    drop_empty_rows: bool = True
    drop_empty_columns: bool = False
    duplicate_headers: Literal["raise", "suffix"] = "raise"
    missing_columns: Literal["raise", "ignore"] = "raise"
    empty_sheets: Literal["raise", "skip"] = "raise"
    include_hidden: bool = True
    data_only: bool = True
    schema_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, choices in {
            "kind": ("pandas", "polars"),
            "engine": ("auto", "openpyxl", "calamine"),
            "duplicate_headers": ("raise", "suffix"),
            "missing_columns": ("raise", "ignore"),
            "empty_sheets": ("raise", "skip"),
        }.items():
            if getattr(self, field_name) not in choices:
                _LOGGER.error(f"{field_name} must be one of {choices}.")
                raise ValueError()
        for field_name in ("header_row", "skip_rows", "n_rows"):
            value = getattr(self, field_name)
            if value is None and field_name != "skip_rows":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _LOGGER.error(f"{field_name} must be a nonnegative integer" + (" or None." if field_name != "skip_rows" else "."))
                raise ValueError()
        if self.use_columns is not None:
            cols = tuple(self.use_columns)
            if not cols or any(not isinstance(c, str) for c in cols):
                _LOGGER.error("use_columns must be a nonempty sequence of names or None.")
                raise ValueError()
            if len(cols) != len(set(cols)):
                _LOGGER.error("use_columns contains duplicate names.")
                raise ValueError()
            object.__setattr__(self, "use_columns", cols)
        if self.all_strings and self.schema_overrides:
            _LOGGER.error("all_strings and schema_overrides are mutually exclusive.")
            raise ValueError()
        object.__setattr__(self, "schema_overrides", MappingProxyType(dict(self.schema_overrides)))

    def with_changes(self, **changes: Any) -> ExcelReadOptions:
        """Return validated options with selected fields replaced."""
        return replace(self, **changes)
