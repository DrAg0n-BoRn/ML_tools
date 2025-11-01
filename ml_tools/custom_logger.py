from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Any, Literal
import traceback
import json
import csv
from itertools import zip_longest
from collections import Counter

from .path_manager import sanitize_filename, make_fullpath
from ._script_info import _script_info
from ._logger import _LOGGER


__all__ = [
    "custom_logger",
    "save_list_strings",
    "load_list_strings",
    "compare_lists"
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
    add_timestamp: bool=True,
    dict_as: Literal['auto', 'json', 'csv'] = 'auto',
) -> None:
    """
    Logs various data types to corresponding output formats:

    - list[Any]                    → .txt
        Each element is written on a new line.

    - dict[str, list[Any]]        → .csv    (if dict_as='auto' or 'csv')
        Dictionary is treated as tabular data; keys become columns, values become rows.

    - dict[str, scalar]           → .json   (if dict_as='auto' or 'json')
        Dictionary is treated as structured data and serialized as JSON.

    - str                         → .log
        Plain text string is written to a .log file.

    - BaseException               → .log
        Full traceback is logged for debugging purposes.

    Args:
        data (Any): The data to be logged. Must be one of the supported types.
        save_directory (str | Path): Directory where the log will be saved. Created if it does not exist.
        log_name (str): Base name for the log file.
        add_timestamp (bool): Whether to add a timestamp to the filename.
        dict_as ('auto'|'json'|'csv'): 
            - 'auto': Guesses format (JSON or CSV) based on dictionary content.
            - 'json': Forces .json format for any dictionary.
            - 'csv': Forces .csv format. Will fail if dict values are not all lists.

    Raises:
        ValueError: If the data type is unsupported.
    """
    try:
        if not isinstance(data, BaseException) and not data:
            _LOGGER.warning("Empty data received. No log file will be saved.")
            return
        
        save_path = make_fullpath(save_directory, make=True)
        
        sanitized_log_name = sanitize_filename(log_name)
        
        if add_timestamp:
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
            base_path = save_path / f"{sanitized_log_name}_{timestamp}"
        else:
            base_path = save_path / sanitized_log_name
        
        # Router
        if isinstance(data, list):
            _log_list_to_txt(data, base_path.with_suffix(".txt"))

        elif isinstance(data, dict):
            if dict_as == 'json':
                _log_dict_to_json(data, base_path.with_suffix(".json"))
            
            elif dict_as == 'csv':
                # This will raise a ValueError if data is not all lists
                _log_dict_to_csv(data, base_path.with_suffix(".csv"))
            
            else: # 'auto' mode
                if all(isinstance(v, list) for v in data.values()):
                    _log_dict_to_csv(data, base_path.with_suffix(".csv"))
                else:
                    _log_dict_to_json(data, base_path.with_suffix(".json"))

        elif isinstance(data, str):
            _log_string_to_log(data, base_path.with_suffix(".log"))

        elif isinstance(data, BaseException):
            _log_exception_to_log(data, base_path.with_suffix(".log"))

        else:
            _LOGGER.error("Unsupported data type. Must be list, dict, str, or BaseException.")
            raise ValueError()

        _LOGGER.info(f"Log saved as: '{base_path.name}'")

    except Exception:
        _LOGGER.exception(f"Log not saved.")


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
            _LOGGER.error(f"Dictionary value for key '{key}' must be a list.")
            raise ValueError()
        
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


def save_list_strings(list_strings: list[str], directory: Union[str,Path], filename: str, verbose: bool=True):
    """Saves a list of strings as a text file."""
    target_dir = make_fullpath(directory, make=True, enforce="directory")
    sanitized_name = sanitize_filename(filename)
    
    if not sanitized_name.endswith(".txt"):
        sanitized_name = sanitized_name + ".txt"
    
    full_path = target_dir / sanitized_name
    with open(full_path, 'w') as f:
        for string_data in list_strings:
            f.write(f"{string_data}\n")
    
    if verbose:
        _LOGGER.info(f"Text file saved as '{full_path.name}'.")


def load_list_strings(text_file: Union[str,Path], verbose: bool=True) -> list[str]:
    """Loads a text file as a list of strings."""
    target_path = make_fullpath(text_file, enforce="file")
    loaded_strings = []

    with open(target_path, 'r') as f:
        loaded_strings = [line.strip() for line in f]
    
    if len(loaded_strings) == 0:
        _LOGGER.error("The text file is empty.")
        raise ValueError()
    
    if verbose:
        _LOGGER.info(f"Loaded '{target_path.name}' as list of strings.")
        
    return loaded_strings


class _RobustEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects.

    This handles:
    1.  `type` objects (e.g., <class 'int'>) which result from
        `check_type_only=True`.
    2.  Any other custom class or object by falling back to its
        string representation.
    """
    def default(self, o):
        if isinstance(o, type):
            return str(o)
        try:
            return super().default(o)
        except TypeError:
            return str(o)

def compare_lists(
    list_A: list,
    list_B: list,
    save_dir: Union[str, Path],
    strict: bool = False,
    check_type_only: bool = False
) -> dict:
    """
    Compares two lists and saves a JSON report of the differences.

    Args:
        list_A (list): The first list to compare.
        list_B (list): The second list to compare.
        save_dir (str | Path): The directory where the resulting report will be saved.
        strict (bool):
            - If False: Performs a "bag" comparison. Order does not matter, but duplicates do.
            - If True: Performs a strict, positional comparison.
            
        check_type_only (bool):
            - If False: Compares items using `==` (`__eq__` operator).
            - If True: Compares only the `type()` of the items.

    Returns:
        dict: A dictionary detailing the differences. (saved to `save_dir`).
    """
    MISSING_A_KEY = "missing_in_A"
    MISSING_B_KEY = "missing_in_B"
    MISMATCH_KEY = "mismatch"
    
    results: dict[str, list] = {MISSING_A_KEY: [], MISSING_B_KEY: []}
    
    # make directory
    save_path = make_fullpath(input_path=save_dir, make=True, enforce="directory")

    if strict:
        # --- STRICT (Positional) Mode ---
        results[MISMATCH_KEY] = []
        sentinel = object()

        if check_type_only:
            compare_func = lambda a, b: type(a) == type(b)
        else:
            compare_func = lambda a, b: a == b

        for index, (item_a, item_b) in enumerate(
            zip_longest(list_A, list_B, fillvalue=sentinel)
        ):
            if item_a is sentinel:
                results[MISSING_A_KEY].append({"index": index, "item": item_b})
            elif item_b is sentinel:
                results[MISSING_B_KEY].append({"index": index, "item": item_a})
            elif not compare_func(item_a, item_b):
                results[MISMATCH_KEY].append(
                    {
                        "index": index,
                        "list_A_item": item_a,
                        "list_B_item": item_b,
                    }
                )

    else:
        # --- NON-STRICT (Bag) Mode ---
        if check_type_only:
            # Types are hashable, we can use Counter (O(N))
            types_A_counts = Counter(type(item) for item in list_A)
            types_B_counts = Counter(type(item) for item in list_B)

            diff_A_B = types_A_counts - types_B_counts
            for item_type, count in diff_A_B.items():
                results[MISSING_B_KEY].extend([item_type] * count)

            diff_B_A = types_B_counts - types_A_counts
            for item_type, count in diff_B_A.items():
                results[MISSING_A_KEY].extend([item_type] * count)

        else:
            # Items may be unhashable. Use O(N*M) .remove() method
            temp_B = list(list_B)
            missing_in_B = []

            for item_a in list_A:
                try:
                    temp_B.remove(item_a)
                except ValueError:
                    missing_in_B.append(item_a)
            
            results[MISSING_A_KEY] = temp_B
            results[MISSING_B_KEY] = missing_in_B

    # --- Save the Report ---
    try:
        full_path = save_path / "list_comparison.json"

        # Write the report dictionary to the JSON file
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, cls=_RobustEncoder)
            
    except Exception as e:
        _LOGGER.error(f"Failed to save comparison report to {save_path}: \n{e}")

    return results


def info():
    _script_info(__all__)