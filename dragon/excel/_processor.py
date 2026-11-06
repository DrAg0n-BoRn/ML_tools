"""Reusable file/directory workflows built from the focused Excel helpers."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal
import polars as pl
import polars.selectors as cs

from ..path_manager import make_fullpath

from ._cells import _split_output_paths, split_excel_sheets, unmerge_excel_cells
from ._common import (
    OOXML_EXTENSIONS,
    add_source,
    atomic_output,
    input_path,
    output_kind,
    select_sheets,
    select_columns,
    stack_frames,
    to_polars,
    unique_names,
)
from ._discovery import find_excel_files
from ._reader import ExcelReader
from ._reports import ExcelValidationIssue, ExcelValidationReport
from ._types import ExcelReadOptions, Frame, SheetSelection
from ._write import save_dataframe_excel
from ._common import LOGGER


class ExcelProcessor:
    """Process a workbook, directory, or ordered iterable of workbook paths.

    it receives
    the configured backend and must return a pandas or Polars dataframe.
    File discovery is snapshotted at construction, preventing generated outputs
    from being picked up later in the same instance. Directory paths are sorted;
    an explicit input list preserves user order. No transformation mutates inputs.

    stack/concat_columns/join return frames and optionally save via output_path.
    split/unmerge return paths; validate returns an ExcelValidationReport.
    """

    def __init__(
        self,
        sources: str | Path | Iterable[str | Path],
        *,
        options: ExcelReadOptions | None = None,
        recursive: bool = False,
        transform: Callable[[Frame, Path, str], Frame] | None = None,
        **read_options: Any,
    ) -> None:
        """Initialize with sources, options, and an optional transform function.
        
        Args:
            sources: A single workbook path, a directory path, or an iterable of paths.
            options: Optional ExcelReadOptions instance.
            recursive: If True, search directories recursively for workbooks.
            transform: Optional function to transform each frame before combining.
            **read_options: Additional options passed to ExcelReader.
        """
        
        self.options = (options or ExcelReadOptions()).with_changes(**read_options)
        self.paths = tuple(self._resolve_sources(sources, recursive))
        self.transform = transform

    def _resolve_sources(self, sources, recursive: bool) -> list[Path]:
        if isinstance(sources, (str, Path)):
            path = make_fullpath(sources)
            if path.is_dir():
                return find_excel_files(path, recursive=recursive, engine=self.options.engine)
            paths = [input_path(path)]
        else:
            paths = [input_path(path) for path in sources]
        if not paths:
            LOGGER.error("At least one source workbook is required.")
            raise ValueError()
        if len({p.resolve() for p in paths}) != len(paths):
            LOGGER.error("Duplicate source workbooks are not allowed.")
            raise ValueError()
        return paths

    def _iter_frames(self, sheet_name: SheetSelection) -> Iterator[tuple[pl.DataFrame, Path, str]]:
        for path in self.paths:
            with ExcelReader(path, options=self.options) as reader:
                for frame, name in reader.iter_sheets(sheet_name):
                    if self.transform is not None:
                        frame = self.transform(frame, path, name)
                    yield to_polars(frame), path, name

    def _finish(self, frame: pl.DataFrame, output_path, overwrite: bool) -> Frame:
        if output_path is not None:
            target = make_fullpath(output_path)
            if target.resolve() in {p.resolve() for p in self.paths}:
                LOGGER.error("Transformation output must not replace a source workbook.")
                raise ValueError()
            self.save(frame, target, overwrite=overwrite)
        return output_kind(frame, self.options.kind)

    @staticmethod
    def save(
        frame: Frame, 
        output_path: str | Path, 
        *, 
        overwrite: bool = False, 
        **write_options: Any
    ) -> Path:
        """
        Save a result as .xlsx, .csv, or .parquet, inferring format from suffix.
        
        Args:
            frame: The DataFrame to save.
            output_path: The path where the DataFrame should be saved.
            overwrite: If True, allows overwriting existing files.
            **write_options: Additional options passed to the underlying write function.
                - For .xlsx, options are passed to local 'save_dataframe_excel' function.
                - For .csv, options are passed to Polars' write_csv.
                - For .parquet, options are passed to Polars' write_parquet.
                
        Returns:
            Path: The path where the DataFrame was saved.
        """
        path = make_fullpath(output_path)
        if path.suffix.lower() == ".xlsx":
            return save_dataframe_excel(frame, path, overwrite=overwrite, **write_options)
        
        if path.suffix.lower() not in (".csv", ".parquet"):
            LOGGER.error(f"Output extension must be .xlsx, .csv, or .parquet. Received: '{path.suffix}'")
            raise ValueError()
        
        data = to_polars(frame)
        
        with atomic_output(path, overwrite=overwrite) as temp:
            if path.suffix.lower() == ".csv":
                data.write_csv(temp, **write_options)
            else:
                data.write_parquet(temp, **write_options)
        return path

    def stack(
        self,
        *,
        sheet_name: SheetSelection = 0,
        schema_mode: Literal["strict", "union"] = "strict",
        type_conflicts: Literal["raise", "string"] = "raise",
        source_file_column: str | None = None,
        source_sheet_column: str | None = None,
        rename_columns: dict[str, str] | None = None,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Frame:
        """Stack rows from selected files/sheets. Source files use absolute paths.
        
        Args:
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            schema_mode: 'strict' requires identical columns; 'union' allows missing columns.
            type_conflicts: 'raise' errors on type conflicts; 'string' casts to string.
            source_file_column: Optional column name to add for source file paths.
            source_sheet_column: Optional column name to add for source sheet names.
            rename_columns: Optional mapping to rename columns in the final output.
            output_path: Optional path to save the stacked DataFrame.
            overwrite: If True, allows overwriting existing files at output_path.
            
        Returns:
            Dataframe: The stacked DataFrame, optionally saved to output_path.
        """
        frames, labels = [], []
        for frame, path, name in self._iter_frames(sheet_name):
            frame = add_source(frame, source_file_column, str(path))
            frame = add_source(frame, source_sheet_column, name)
            frames.append(frame)
            labels.append(f"{path.name}:{name}")
        result = stack_frames(
            frames, labels, schema_mode=schema_mode, type_conflicts=type_conflicts
        )
        metadata = [c for c in (source_file_column, source_sheet_column) if c is not None]
        result = select_columns(result, [c for c in result.columns if c not in metadata] + metadata)
        if rename_columns:
            result = result.rename(rename_columns, strict=True)
        return self._finish(result, output_path, overwrite)

    def concat_columns(
        self,
        *,
        sheet_name: SheetSelection = 0,
        row_mismatch: Literal["raise", "pad"] = "raise",
        duplicate_columns: Literal["suffix", "first", "raise"] = "suffix",
        drop_columns: Iterable[str] = (),
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Frame:
        """Concatenate columns by row POSITION, ignoring indexes.
        
        This method is useful for combining sheets with the same row order but different columns. 
        
        Args:
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            row_mismatch: 'raise' errors on differing row counts; 'pad' fills with nulls.
            duplicate_columns: 'suffix' adds suffixes to duplicates; 'first' keeps the first; 'raise' errors on duplicates.
            drop_columns: Iterable of column names to drop before concatenation.
            output_path: Optional path to save the concatenated DataFrame.
            overwrite: If True, allows overwriting existing files at output_path.
        Returns:
            Dataframe: The concatenated DataFrame, optionally saved to output_path.
        """
        if row_mismatch not in ("raise", "pad") or duplicate_columns not in (
            "suffix",
            "first",
            "raise",
        ):
            LOGGER.error("Invalid row_mismatch or duplicate_columns policy.")
            raise ValueError()
        
        drop = set(drop_columns)
        frames = [
            frame.drop(cs.by_name([c for c in drop if c in frame.columns], require_all=True))
            for frame, _, _ in self._iter_frames(sheet_name)
        ]
        if not frames or any(not f.width for f in frames):
            LOGGER.error("No frames, or a frame has no remaining columns.")
            raise ValueError()
        
        if row_mismatch == "raise" and len({f.height for f in frames}) > 1:
            LOGGER.error("Row counts differ across frames; use row_mismatch='pad'.")
            raise ValueError()
        
        names = [c for f in frames for c in f.columns]
        if duplicate_columns == "raise" and len(names) != len(set(names)):
            LOGGER.error("Duplicate columns across input sheets; use duplicate_columns='suffix' or 'first'.")
            raise ValueError()
        
        renamed = unique_names(names)
        series, seen, offset = [], set(), 0
        height = max(f.height for f in frames)
        for frame in frames:
            for column in frame.columns:
                replacement = renamed[offset]
                offset += 1
                if duplicate_columns == "first" and column in seen:
                    continue
                values = frame[column]
                if frame.height < height:
                    values = values.extend_constant(None, height - frame.height)
                series.append(
                    values.alias(replacement if duplicate_columns == "suffix" else column)
                )
                seen.add(column)
        return self._finish(pl.DataFrame(series), output_path, overwrite)

    def join(
        self,
        on: str | list[str],
        *,
        sheet_name: SheetSelection = 0,
        how: Literal["inner", "left", "full"] = "inner",
        validate: Literal["1:1", "1:m", "m:1", "m:m"] = "1:1",
        nulls_equal: bool = False,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Frame:
        """Join two or more frames on shared keys; 
        
        Args:
            on: Column name(s) to join on.
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            how: Type of join to perform ('inner', 'left', 'full').
                - 'inner': Only include rows with matching keys in both frames.
                - 'left': Include all rows from the left frame, with matching rows from the right frame.
                - 'full': Include all rows from both frames, with matching rows combined.
            validate: Join cardinality validation ('1:1', '1:m', 'm:1', 'm:m').
                - '1:1': Each key in the left frame matches at most one key in the right frame, and vice versa.
                - '1:m': Each key in the left frame matches at most one key in the right frame, but keys in the right frame can match multiple keys in the left frame.
                - 'm:1': Each key in the right frame matches at most one key in the left frame, but keys in the left frame can match multiple keys in the right frame.
                - 'm:m': Keys in both frames can match multiple keys in the other frame.
            nulls_equal: If True, treat null keys as equal during the join; otherwise, null keys do not match.
            output_path: Optional path to save the joined DataFrame.
            overwrite: If True, allows overwriting existing files at output_path.
            
        Returns:
            Dataframe: The joined DataFrame, optionally saved to output_path.
        """
        if how not in ("inner", "left", "full"):
            LOGGER.error("how must be 'inner', 'left', or 'full'.")
            raise ValueError()
        
        if validate not in ("1:1", "1:m", "m:1", "m:m"):
            LOGGER.error("Invalid join cardinality validation; must be '1:1', '1:m', 'm:1', or 'm:m'.")
            raise ValueError()
        
        keys = [on] if isinstance(on, str) else list(on)
        if not keys or len(keys) != len(set(keys)):
            LOGGER.error("Join keys must be nonempty and unique.")
            raise ValueError()
        
        frames = list(self._iter_frames(sheet_name))
        if len(frames) < 2:
            LOGGER.error("Join requires at least two sheets/frames.")
            raise ValueError()
        
        for frame, path, name in frames:
            missing = set(keys) - set(frame.columns)
            if missing:
                LOGGER.error(f"Missing join keys in {path.name}:{name}: {sorted(missing)}")
                raise ValueError()
        
        result = frames[0][0]
        for right, _, _ in frames[1:]:
            nonkeys = [c for c in right.columns if c not in keys]
            combined_names = result.columns + nonkeys
            renamed = unique_names(combined_names)[len(result.columns) :]
            right = right.rename(dict(zip(nonkeys, renamed, strict=True)))
            
            result = result.join(
                right,
                on=[cs.by_name(key, require_all=True) for key in keys],
                how=how,
                validate=validate,
                nulls_equal=nulls_equal,
                coalesce=True,
                maintain_order="left_right",
            )
        return self._finish(result, output_path, overwrite)

    def validate(
        self,
        expected_columns: Iterable[str],
        *,
        sheet_name: SheetSelection = None,
        strict: bool = False,
        raise_on_error: bool = False,
    ) -> ExcelValidationReport:
        """
        Validate each selected sheet's header; strict also checks extras/order.
        
        Args:
            expected_columns: Iterable of expected column names.
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            strict: If True, also checks for extra columns and order.
            raise_on_error: If True, raises an exception if any validation issues are found.
            
        Returns:
            ExcelValidationReport: A report detailing the validation results for each sheet.
        """
        expected = list(expected_columns)
        if (
            not expected
            or len(expected) != len(set(expected))
            or any(not isinstance(c, str) for c in expected)
        ):
            LOGGER.error("expected_columns must be nonempty, unique strings.")
            raise ValueError()
        
        report = ExcelValidationReport(files_checked=len(self.paths))
        
        for path in self.paths:
            try:
                with ExcelReader(path, options=self.options) as reader:
                    for name in select_sheets(reader.sheet_names, sheet_name):
                        report.sheets_checked += 1
                        try:
                            header = list(reader.header(name))
                            missing = tuple(c for c in expected if c not in header)
                            extra = tuple(c for c in header if c not in expected)
                            order = not missing and not extra and header != expected
                            if missing or (strict and (extra or order)):
                                message = f"Missing: {list(missing)}; extra: {list(extra)}; incorrect order: {order}"
                                report.issues.append(
                                    ExcelValidationIssue(path, name, message, missing, extra, order)
                                )
                            else:
                                report.valid_sheets += 1
                        except Exception as exc:
                            report.issues.append(
                                ExcelValidationIssue(path, name, f"Header could not be read: {exc}")
                            )
            except Exception as exc:
                report.issues.append(
                    ExcelValidationIssue(path, None, f"Workbook could not be read: {exc}")
                )
        if raise_on_error:
            report.raise_for_errors()
        return report

    def _output_directory(self, output_dir) -> Path | None:
        directory = make_fullpath(output_dir) if output_dir is not None else None
        if directory is not None and len({p.stem.casefold() for p in self.paths}) != len(
            self.paths
        ):
            LOGGER.error(f"Source stems collide in a flat output directory; process those workbooks separately.")
            raise ValueError()
        return directory

    def _cell_selections(self, sheet_name: SheetSelection) -> list[tuple[Path, list[str]]]:
        if any(path.suffix.lower() not in OOXML_EXTENSIONS for path in self.paths):
            LOGGER.error("Cell editing/splitting requires OOXML workbooks; use dataframe export for legacy files.")
            raise ValueError()
        selections = []
        for path in self.paths:
            with ExcelReader(path, include_hidden=self.options.include_hidden) as reader:
                selections.append((path, select_sheets(reader.sheet_names, sheet_name)))
        return selections

    def _check_outputs(self, targets: list[Path], overwrite: bool) -> None:
        normalized = [str(p.resolve()).casefold() for p in targets]
        if len(normalized) != len(set(normalized)):
            LOGGER.error("Generated output filenames collide; process the sources separately.")
            raise ValueError()
        inputs = {str(p.resolve()).casefold() for p in self.paths}
        for target, normalized_path in zip(targets, normalized, strict=True):
            if normalized_path in inputs or (target.exists() and not overwrite):
                LOGGER.error(f"Output would overwrite a source/existing file: {target}")
                raise FileExistsError()

    def split(
        self,
        output_dir: str | Path | None = None,
        *,
        sheet_name: SheetSelection = None,
        unmerge: bool = False,
        **options: Any,
    ) -> list[Path]:
        """
        Split selected worksheets; use unmerge=True for the old combined workflow.
        
        Args:
            output_dir: Optional directory to save the split files; defaults to source directory.
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            unmerge: If True, unmerge cells in the output; otherwise, keep merged cells if any.
            **options: Additional options passed to the underlying split/unmerge functions.
            
        Returns:
            List[Path]: Paths to the generated split files.
        """
        directory = self._output_directory(output_dir)
        selections = self._cell_selections(sheet_name)
        targets = [
            target
            for path, names in selections
            for target in _split_output_paths(path, names, directory or path.parent)
        ]
        self._check_outputs(targets, options.get("overwrite", False))
        return [
            output
            for path, names in selections
            for output in split_excel_sheets(
                    path, 
                    directory, 
                    sheet_name=names, 
                    unmerge=unmerge, 
                    **options
                )
        ]

    def unmerge(
        self,
        output_dir: str | Path | None = None,
        *,
        sheet_name: SheetSelection = None,
        **options: Any,
    ) -> list[Path]:
        """
        Keep whole workbooks while unmerging selected sheets.
        
        Args:
            output_dir: Optional directory to save the unmerged files; defaults to source directory.
            sheet_name: Sheet selection criteria (zero-based index, name, or list).
            **options: Additional options passed to the underlying unmerge function.
        
        Returns:
            list[Path]: Paths to the generated unmerged files.
        """
        directory = self._output_directory(output_dir)
        selections = self._cell_selections(sheet_name)
        
        targets = [
            (directory or path.parent) / f"{path.stem}_unmerged{path.suffix}"
            for path, _ in selections
        ]
        self._check_outputs(targets, options.get("overwrite", False))
        return [
            unmerge_excel_cells(
                path,
                target,
                sheet_name=names,
                **options,
            )
            for (path, names), target in zip(selections, targets, strict=True)
        ]
