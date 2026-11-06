"""Structured, serializable worksheet validation results."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._common import LOGGER


@dataclass(frozen=True, slots=True)
class ExcelValidationIssue:
    path: Path
    sheet: str | None
    message: str
    missing_columns: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    incorrect_order: bool = False


@dataclass(slots=True)
class ExcelValidationReport:
    """
    One issue per failed sheet/read, with explicit file and sheet counts.
    
    Properties:
        is_valid: True if no issues were found.
        invalid_files: Tuple of unique file paths with issues.
    
    Methods:
        raise_for_errors: Raises ValueError with a summary of issues if any exist.
        to_dict: Returns a serializable dictionary representation of the report.
    """

    files_checked: int = 0
    sheets_checked: int = 0
    valid_sheets: int = 0
    issues: list[ExcelValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def invalid_files(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(issue.path for issue in self.issues))

    def raise_for_errors(self) -> None:
        if self.issues:
            details = "\n".join(
                f"{i.path.name}:{i.sheet or '<workbook>'}: {i.message}" for i in self.issues
            )
            LOGGER.error(f"Excel validation failed:\n{details}")
            raise ValueError()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for issue in result["issues"]:
            issue["path"] = str(issue["path"])
        result["is_valid"] = self.is_valid
        return result
