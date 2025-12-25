from typing import Any, Union, Literal, overload
from pathlib import Path
import json

from ..path_manager import sanitize_filename, make_fullpath
from .._core import get_logger

from ._IO_utils import _RobustEncoder


_LOGGER = get_logger("IO Save/Load")


__all__ = [
    "save_json",
    "load_json",
    "save_list_strings",
    "load_list_strings"
]


def save_json(
    data: Union[dict[Any, Any], list[Any]],
    directory: Union[str, Path],
    filename: str,
    verbose: bool = True
) -> None:
    """
    Saves a dictionary or list as a JSON file.

    Args:
        data (dict | list): The data to save.
        directory (str | Path): The directory to save the file in.
        filename (str): The name of the file (extension .json will be added if missing).
        verbose (bool): Whether to log success messages.
    """
    target_dir = make_fullpath(directory, make=True, enforce="directory")
    sanitized_name = sanitize_filename(filename)

    if not sanitized_name.endswith(".json"):
        sanitized_name += ".json"

    full_path = target_dir / sanitized_name

    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            # Using _RobustEncoder ensures compatibility with non-standard types (like 'type' objects)
            json.dump(data, f, indent=4, ensure_ascii=False, cls=_RobustEncoder)

        if verbose:
            _LOGGER.info(f"JSON file saved as '{full_path.name}'.")

    except Exception as e:
        _LOGGER.error(f"Failed to save JSON to '{full_path}': {e}")
        raise


# 1. Define Overloads (for the type checker)
@overload
def load_json(
    file_path: Union[str, Path], 
    expected_type: Literal["dict"] = "dict",
    verbose: bool = True
) -> dict[Any, Any]: ...

@overload
def load_json(
    file_path: Union[str, Path], 
    expected_type: Literal["list"],
    verbose: bool = True
) -> list[Any]: ...


def load_json(
    file_path: Union[str, Path], 
    expected_type: Literal["dict", "list"] = "dict",
    verbose: bool = True
) -> Union[dict[Any, Any], list[Any]]:
    """
    Loads a JSON file.

    Args:
        file_path (str | Path): The path to the JSON file.
        expected_type ('dict' | 'list'): strict check for the root type of the JSON.
        verbose (bool): Whether to log success/failure messages.

    Returns:
        dict | list: The loaded JSON data.
    """
    target_path = make_fullpath(file_path, enforce="file")
    
    # Map string literals to actual python types
    type_map = {"dict": dict, "list": list}
    target_type = type_map.get(expected_type, dict)

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, target_type):
            _LOGGER.error(f"JSON root is type {type(data)}, expected {expected_type}.")
            raise ValueError()

        if verbose:
            _LOGGER.info(f"Loaded JSON data from '{target_path.name}'.")
        
        return data

    except json.JSONDecodeError as e:
        _LOGGER.error(f"Failed to decode JSON from '{target_path}': {e.msg}")
        raise ValueError()
        
    except Exception as e:
        _LOGGER.error(f"Error loading JSON from '{target_path}': {e}")
        raise


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

