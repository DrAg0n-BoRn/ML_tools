from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Any
import traceback
import json
import csv
from .path_manager import sanitize_filename, make_fullpath
from ._script_info import _script_info
from ._logger import _LOGGER


__all__ = [
    "custom_logger"
]


def custom_logger(
    data: Union[
        List[Any],
        Dict[Any, Any],
        str,
        BaseException
    ],
    save_directory: Union[str, Path],
    log_name: str,
) -> None:
    """
    Logs various data types to corresponding output formats:

    - list[Any]                    → .txt
        Each element is written on a new line.

    - dict[str, list[Any]]        → .csv
        Dictionary is treated as tabular data; keys become columns, values become rows.

    - dict[str, scalar]           → .json
        Dictionary is treated as structured data and serialized as JSON.

    - str                         → .log
        Plain text string is written to a .log file.

    - BaseException               → .log
        Full traceback is logged for debugging purposes.

    Args:
        data: The data to be logged. Must be one of the supported types.
        save_directory: Directory where the log will be saved. Created if it does not exist.
        log_name: Base name for the log file. Timestamp will be appended automatically.

    Raises:
        ValueError: If the data type is unsupported.
    """
    try:
        save_path = make_fullpath(save_directory, make=True)
        
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        log_name = sanitize_filename(log_name)
        
        base_path = save_path / f"{log_name}_{timestamp}"

        if isinstance(data, list):
            _log_list_to_txt(data, base_path.with_suffix(".txt"))

        elif isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                _log_dict_to_csv(data, base_path.with_suffix(".csv"))
            else:
                _log_dict_to_json(data, base_path.with_suffix(".json"))

        elif isinstance(data, str):
            _log_string_to_log(data, base_path.with_suffix(".log"))

        elif isinstance(data, BaseException):
            _log_exception_to_log(data, base_path.with_suffix(".log"))

        else:
            raise ValueError("Unsupported data type. Must be list, dict, str, or BaseException.")

        _LOGGER.info(f"🗄️ Log saved to: '{base_path}'")

    except Exception as e:
        _LOGGER.error(f"❌ Log not saved: {e}")


def _log_list_to_txt(data: List[Any], path: Path) -> None:
    log_lines = []
    for item in data:
        try:
            log_lines.append(str(item).strip())
        except Exception:
            log_lines.append(f"(unrepresentable item of type {type(item)})")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


def _log_dict_to_csv(data: Dict[Any, List[Any]], path: Path) -> None:
    sanitized_dict = {}
    max_length = max(len(v) for v in data.values()) if data else 0

    for key, value in data.items():
        if not isinstance(value, list):
            raise ValueError(f"Dictionary value for key '{key}' must be a list.")
        sanitized_key = str(key).strip().replace('\n', '_').replace('\r', '_')
        padded_value = value + [None] * (max_length - len(value))
        sanitized_dict[sanitized_key] = padded_value

    # The `newline=''` argument is important to prevent extra blank rows
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # 1. Write the header row from the sanitized dictionary keys
        header = list(sanitized_dict.keys())
        writer.writerow(header)

        # 2. Transpose columns to rows and write them
        # zip(*sanitized_dict.values()) elegantly converts the column data
        # (lists in the dict) into row-by-row tuples.
        rows_to_write = zip(*sanitized_dict.values())
        writer.writerows(rows_to_write)


def _log_string_to_log(data: str, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data.strip() + '\n')


def _log_exception_to_log(exc: BaseException, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Exception occurred:\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


def _log_dict_to_json(data: Dict[Any, Any], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def info():
    _script_info(__all__)