"""Shared validation, dataframe operations, and transactional output helpers."""

from __future__ import annotations

import importlib
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, Literal
import pandas as pd
import polars as pl

from .._core import get_logger
from ..path_manager import make_fullpath

from ._types import Frame, Kind, SheetSelection

LOGGER = get_logger("Excel")
OOXML_EXTENSIONS = (".xlsx", ".xlsm", ".xltx", ".xltm")
READ_EXTENSIONS = OOXML_EXTENSIONS + (".xls", ".xlsb", ".ods")


def require(module: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            LOGGER.debug("Module %r not found, but another module was missing: %s", module, exc.name)
            raise
        LOGGER.error(f"Excel operation requires {module}. Install dragon-ml-toolbox[excel].")
        raise ImportError(
            f"Excel operation requires {module!r}. Install dragon-ml-toolbox[excel]."
        ) from exc


def input_path(path: str | Path) -> Path:
    result = Path(make_fullpath(path, enforce="file"))
    if result.suffix.lower() not in READ_EXTENSIONS:
        LOGGER.error(f"Unsupported spreadsheet extension {result.suffix!r}: {result}")
        raise ValueError()
    return result


def select_sheets(names: list[str], selection: SheetSelection) -> list[str]:
    if selection is None:
        chosen = list(names)
    else:
        selectors = [selection] if isinstance(selection, (str, int)) else list(selection)
        chosen = []
        for item in selectors:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                LOGGER.error("Sheet selectors must be names or nonnegative integer positions.")
                raise TypeError()
            if isinstance(item, int):
                if item < 0 or item >= len(names):
                    LOGGER.error(f"Worksheet position {item} is outside 0..{len(names) - 1}.")
                    raise IndexError()
                item = names[item]
            if item not in names:
                LOGGER.error(f"Worksheet {item!r} not found. Available: {names}")
                raise KeyError()
            if item in chosen:
                LOGGER.error(f"Worksheet {item!r} selected more than once.")
                raise ValueError()
            chosen.append(item)
    if not chosen:
        LOGGER.error("No worksheets selected.")
        raise ValueError()
    return chosen


def unique_names(names: list[str]) -> list[str]:
    """Suffix duplicates without stealing an existing name later in the input."""
    reserved, used, result = set(names), set(), []
    for name in names:
        candidate, index = name, 1
        if candidate in used:
            while True:
                candidate = f"{name}_copy{index}"
                index += 1
                if candidate not in reserved and candidate not in used:
                    break
        used.add(candidate)
        result.append(candidate)
    return result


def to_polars(frame: Frame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    if isinstance(frame, pd.DataFrame):
        if not frame.columns.is_unique:
            LOGGER.error("DataFrame columns must be unique.")
            raise ValueError()
        if any(not isinstance(c, str) for c in frame.columns):
            LOGGER.error("DataFrame column names must be strings.")
            raise TypeError()
        try:
            return pl.from_pandas(frame, include_index=False)
        except Exception as exc:
            LOGGER.error("Cannot convert pandas columns; normalize mixed/object values first.")
            raise TypeError() from exc
    LOGGER.error(f"Expected a pandas or Polars DataFrame, got {type(frame).__name__}.")
    raise TypeError()


def output_kind(frame: pl.DataFrame, kind: Kind) -> Frame:
    if kind == "polars":
        return frame
    if kind == "pandas":
        # Arrow-backed pandas columns preserve nullable integers and strings.
        return frame.to_pandas(use_pyarrow_extension_array=True)
    LOGGER.error(f"Unsupported output kind '{kind}'.")
    raise ValueError()


def add_source(frame: pl.DataFrame, name: str | None, value: str) -> pl.DataFrame:
    if name is None:
        return frame
    if not name or name in frame.columns:
        LOGGER.error(f"Source column {name!r} is empty or collides with a data column.")
        raise ValueError()
    return frame.with_columns(pl.lit(value).alias(name))


def stack_frames(
    frames: list[pl.DataFrame],
    labels: list[str],
    *,
    schema_mode: Literal["strict", "union"] = "strict",
    type_conflicts: Literal["raise", "string"] = "raise",
) -> pl.DataFrame:
    if schema_mode not in ("strict", "union"):
        LOGGER.error("schema_mode must be 'strict' or 'union'.")
        raise ValueError()
    if type_conflicts not in ("raise", "string"):
        LOGGER.error("type_conflicts must be 'raise' or 'string'.")
        raise ValueError()
    if not frames or not any(f.height for f in frames):
        LOGGER.error("No nonempty worksheets remain to combine.")
        raise ValueError()
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    if schema_mode == "strict":
        for frame, label in zip(frames, labels, strict=True):
            if frame.columns != frames[0].columns:
                LOGGER.error(f"Column mismatch in {label}: expected {frames[0].columns}, got {frame.columns}.")
                raise ValueError()
    frames = [f.clone() for f in frames]
    for column in columns:
        dtypes = {
            f.schema[column] for f in frames if column in f.columns and f.schema[column] != pl.Null
        }
        numeric = bool(dtypes) and all(d.is_numeric() for d in dtypes)
        if len(dtypes) > 1 and not numeric:
            if type_conflicts == "raise":
                LOGGER.error(f"Incompatible types for {column!r}: {dtypes}. Use all_strings or type_conflicts='string'.")
                raise TypeError()
            frames = [
                f.with_columns(pl.col(column).cast(pl.String)) if column in f.columns else f
                for f in frames
            ]
        elif numeric and (
            any(d.is_float() for d in dtypes)
            or (
                any(d.is_unsigned_integer() for d in dtypes)
                and any(d.is_signed_integer() for d in dtypes)
            )
        ):
            for frame in frames:
                if column in frame.columns and frame.schema[column].is_integer():
                    if any(abs(v) > 2**53 for v in frame[column].drop_nulls().to_list()):
                        LOGGER.error(f"Combining {column!r} would risk integer precision loss; cast explicitly.")
                        raise TypeError()
    result = pl.concat(
        frames, how="diagonal_relaxed" if schema_mode == "union" else "vertical_relaxed"
    )
    return result.select(columns)


def safe_stem(value: str) -> str:
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return result or "sheet"


@contextmanager
def atomic_output(path: str | Path, *, overwrite: bool = False) -> Generator[Path]:
    """Write beside the destination; publish only after the writer succeeds."""
    target = Path(make_fullpath(path))
    if target.exists() and not overwrite:
        LOGGER.error(f"Output already exists: {target}")
        raise FileExistsError()
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as handle:
        temp = Path(handle.name)
    try:
        yield temp
        if overwrite:
            os.replace(temp, target)
        else:
            # Atomic no-clobber publication, including concurrent writers.
            os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)
