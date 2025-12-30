import json
import csv
import traceback
from pathlib import Path
from typing import Union, Any, Literal
from datetime import datetime

from ..path_manager import sanitize_filename, make_fullpath
from .._core import get_logger


_LOGGER = get_logger("IO logger")


__all__ = [
    "custom_logger",
    "train_logger"
]


def custom_logger(
    data: Union[
        list[Any],
        dict[Any, Any],
        str,
        BaseException
    ],
    save_directory: Union[str, Path],
    log_name: str,
    add_timestamp: bool=True,
    dict_as: Literal['auto', 'json', 'csv'] = 'auto',
    verbose: int = 3
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
            # Display a warning instead of error to allow program continuation
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
        
        if verbose >= 2:
            _LOGGER.info(f"Log saved as: '{base_path.name}'")

    except Exception:
        _LOGGER.exception(f"Log not saved.")


def _log_list_to_txt(data: list[Any], path: Path) -> None:
    log_lines = []
    for item in data:
        try:
            log_lines.append(str(item).strip())
        except Exception:
            log_lines.append(f"(unrepresentable item of type {type(item)})")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


def _log_dict_to_csv(data: dict[Any, list[Any]], path: Path) -> None:
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


def _log_dict_to_json(data: dict[Any, Any], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)        


def train_logger(train_config: Union[dict, Any],
                 model_parameters: Union[dict, Any, None],
                 train_history: Union[dict, None],
                 save_directory: Union[str, Path],
                 verbose: int = 3) -> None:
    """
    Logs training data to JSON, adding a timestamp to the filename.
    
    Args:
        train_config (dict | Any): Training configuration parameters. If object, must have a `.to_log()` method returning a dict.
        model_parameters (dict | Any | None): Model parameters. If object, must have a `.to_log()` method returning a dict.
        train_history (dict | None): Training history log.
        save_directory (str | Path): Directory to save the log file.
    """
    # train_config should be a dict or a custom object with the ".to_log()" method
    if not isinstance(train_config, dict):
        if hasattr(train_config, "to_log") and callable(getattr(train_config, "to_log")):
            train_config_dict: dict = train_config.to_log()
            if not isinstance(train_config_dict, dict):
                _LOGGER.error("'train_config.to_log()' did not return a dictionary.")
                raise ValueError()
        else:
            _LOGGER.error("'train_config' must be a dict or an object with a 'to_log()' method.")
            raise ValueError()
    else:
        # check for empty dict
        if not train_config:
            _LOGGER.error("'train_config' dictionary is empty.")
            raise ValueError()
        
        train_config_dict = train_config
        
    # model_parameters should be a dict or a custom object with the ".to_log()" method or None
    model_parameters_dict = {}

    if model_parameters is not None:
        if not isinstance(model_parameters, dict):
            if hasattr(model_parameters, "to_log") and callable(getattr(model_parameters, "to_log")):
                params_result: dict = model_parameters.to_log()
                if not isinstance(params_result, dict):
                    _LOGGER.error("'model_parameters.to_log()' did not return a dictionary.")
                    raise ValueError()
                model_parameters_dict = params_result
            else:
                _LOGGER.error("'model_parameters' must be a dict, None, or an object with a 'to_log()' method.")
                raise ValueError()
        else:
            # check for empty dict
            if not model_parameters:
                _LOGGER.error("'model_parameters' dictionary is empty.")
                raise ValueError()
            
            model_parameters_dict = model_parameters
    
    # make base dictionary
    data: dict = train_config_dict | model_parameters_dict
    
    # add training history if provided and is not empty
    if train_history is not None:
        if not train_history:
            _LOGGER.error("'train_history' dictionary was provided but is empty.")
            raise ValueError()
        data.update(train_history)
    
    custom_logger(
        data=data,
        save_directory=save_directory,
        log_name="Training_Log",
        add_timestamp=True,
        dict_as='json',
        verbose=1
    )
    
    if verbose >= 2:
        _LOGGER.info(f"Training log saved to '{save_directory}/Training_Log.json'")

