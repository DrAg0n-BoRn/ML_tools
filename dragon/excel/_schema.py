"""FeatureSchema validation for Excel dataframes."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import polars as pl

from ..schema import FeatureSchema

from ._common import output_kind, to_polars, LOGGER
from ._load import load_dataframe_excel
from ._types import Frame, Kind


def load_excel_with_schema(
    df_path: str | Path,
    schema: FeatureSchema,
    all_strings: bool = False,
    empty_as_nan: bool = True,
    *,
    kind: Kind = "pandas",
    **read_options: Any,
) -> tuple[Frame, str]:
    """Enforce existing FeatureSchema first-N-feature rules, for either backend.

    The first N columns must contain exactly schema.feature_names. Reorder those
    features to schema order, retain later target columns and source metadata.
    
    Args:
        df_path: Path to Excel file.
        schema: FeatureSchema to validate against.
        all_strings: If True, read all columns as strings.
        empty_as_nan: If True, treat empty cells as NaN.
        kind: "pandas" or "polars" for the returned frame type.
        **read_options: Additional options passed to ExcelReadOptions.
        
    Returns:
        tuple(DataFrame, file_stem) where DataFrame is the loaded and validated frame.
    """
    if read_options.get("use_columns") is not None:
        LOGGER.error("Schema validation requires all columns; omit use_columns extra option.")
        raise ValueError()
    
    read_options.pop("use_columns", None)
    features = list(schema.feature_names)
    if (
        not features
        or len(features) != len(set(features))
        or any(not isinstance(x, str) for x in features)
    ):
        LOGGER.error("FeatureSchema must have nonempty, unique string feature_names.")
        raise ValueError()
    
    frame, name = load_dataframe_excel(
        df_path, kind="polars", all_strings=all_strings, empty_as_nan=empty_as_nan, **read_options
    )
    frame = to_polars(frame)
    actual = frame.columns[: len(features)]
    if len(actual) != len(features) or set(actual) != set(features):
        LOGGER.error(f"First {len(features)} columns must contain {features}; got {actual}.")
        raise ValueError()
    ordered: pl.DataFrame = frame.select(features + frame.columns[len(features) :])
    return output_kind(ordered, kind), name
