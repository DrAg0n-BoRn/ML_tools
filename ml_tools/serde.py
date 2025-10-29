import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError
from typing import Any, Union, TypeVar, get_origin, Type, Optional
from pathlib import Path

from .path_manager import make_fullpath, sanitize_filename
from ._script_info import _script_info
from ._logger import _LOGGER
from ._schema import FeatureSchema


__all__ = [
    "serialize_object_filename",
    "serialize_object",
    "deserialize_object",
    "serialize_schema",
    "deserialize_schema"
]


# Base types that have a generic `type()` log.
_SIMPLE_TYPES = (list, dict, tuple, set, str, int, float, bool)


def serialize_object_filename(obj: Any, save_dir: Union[str,Path], filename: str, verbose: bool=True, raise_on_error: bool=False) -> None:
    """
    Serializes a Python object using joblib; suitable for Python built-ins, numpy, and pandas.

    Parameters:
        obj (Any) : The Python object to serialize.
        save_dir (str | Path) : Directory path where the serialized object will be saved.
        filename (str) : Name for the output file, extension will be appended if needed.
    """
    if obj is None:
        _LOGGER.warning(f"Attempted to serialize a None object. Skipping save for '{filename}'.")
        return
    
    try:
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        sanitized_name = sanitize_filename(filename)
        full_path = save_path / sanitized_name
    except (IOError, OSError, TypeError) as e:
        _LOGGER.error(f"Failed to construct save path from dir='{save_dir}' and filename='{filename}'. Error: {e}")
        if raise_on_error:
            raise e
        return None
    
    # call serialize_object with the fully constructed path.
    serialize_object(obj=obj,
                     file_path=full_path,
                     verbose=verbose,
                     raise_on_error=raise_on_error)


def serialize_object(obj: Any, file_path: Path, verbose: bool = True, raise_on_error: bool = False) -> None:
    """
    Serializes a Python object using joblib to a specific file path.

    Suitable for Python built-ins, numpy, and pandas.

    Parameters:
        obj (Any) : The Python object to serialize.
        file_path (Path) : The full file path to save the object to.
                           '.joblib' extension will be appended if missing.
        raise_on_error (bool) : If True, raises exceptions on failure.
    """
    if obj is None:
        _LOGGER.warning(f"Attempted to serialize a None object. Skipping save for '{file_path}'.")
        return
    
    try:
        # Ensure the extension is correct
        file_path = file_path.with_suffix('.joblib')

        # Ensure the parent directory exists
        _save_dir = make_fullpath(file_path.parent, make=True, enforce="directory")

        # Dump the object
        joblib.dump(obj, file_path)

    except (IOError, OSError, TypeError, TerminatedWorkerError) as e:
        _LOGGER.error(f"Failed to serialize object of type '{type(obj)}' to '{file_path}'. Error: {e}")
        if raise_on_error:
            raise e
        return None
    else:
        if verbose:
            if type(obj) in _SIMPLE_TYPES:
                _LOGGER.info(f"Object of type '{type(obj)}' saved to '{file_path}'")
            else:
                _LOGGER.info(f"Object '{obj}' saved to '{file_path}'")

        return None


# Define a TypeVar to link the expected type to the return type of deserialization
T = TypeVar('T')
    
def deserialize_object(
    filepath: Union[str, Path],
    expected_type: Optional[Type[T]] = None,
    verbose: bool = True,
    ) -> T:
    """
    Loads a serialized object from a .joblib file.

    Parameters:
        filepath (str | Path): Full path to the serialized .joblib file.
        expected_type (Type[T] | None): The expected type of the object.
            If provided, the function raises a TypeError if the loaded object
            is not an instance of this type. It correctly handles generics
            like `list[str]` by checking the base type (e.g., `list`).
            Defaults to None, which skips the type check.
        verbose (bool): If True, logs success messages.

    Returns:
        (Any): The deserialized Python object, which will match the `expected_type` if provided.
    """
    true_filepath = make_fullpath(filepath, enforce="file")
    
    try:
        obj = joblib.load(true_filepath)
    except (IOError, OSError, EOFError, TypeError, ValueError) as e:
        _LOGGER.error(f"Failed to deserialize object from '{true_filepath}'.")
        raise e
    else:
        # --- Type Validation Step ---
        if expected_type:
            # get_origin handles generics (e.g., list[str] -> list)
            # If it's not a generic, get_origin returns None, so we use the type itself.
            type_to_check = get_origin(expected_type) or expected_type
            
            # Can't do an isinstance check on 'Any', skip it.
            if type_to_check is not Any and not isinstance(obj, type_to_check):
                error_msg = (
                    f"Type mismatch: Expected an instance of '{expected_type}', but found '{type(obj)}' in '{true_filepath}'."
                )
                _LOGGER.error(error_msg)
                raise TypeError()
        
        if verbose:
            # log special objects
            if type(obj) in _SIMPLE_TYPES:
                _LOGGER.info(f"Loaded object of type '{type(obj)}' from '{true_filepath}'.")
            else:
                _LOGGER.info(f"Loaded object '{obj}' from '{true_filepath}'.")
        
        return obj # type: ignore


def serialize_schema(schema: FeatureSchema, file_path: Path):
    """
    Serializes a FeatureSchema object to a .joblib file.

    This is a high-level wrapper around `serialize_object` that
    specifically handles `FeatureSchema` instances and ensures
    errors are raised on failure.

    Args:
        schema (FeatureSchema): The schema object to serialize.
        file_path (Path): The full file path to save the schema to.
    """
    serialize_object(obj=schema,
                     file_path=file_path,
                     verbose=True,
                     raise_on_error=True)
    
    
def deserialize_schema(file_path: Path):
    """
    Deserializes a FeatureSchema object from a .joblib file.

    This is a high-level wrapper around `deserialize_object` that
    validates the loaded object is an instance of `FeatureSchema`.

    Args:
        file_path (Path): The full file path of the serialized schema.

    Returns:
        FeatureSchema: The deserialized schema object.
    
    Raises:
        TypeError: If the deserialized object is not an instance of `FeatureSchema`.
    """
    schema = deserialize_object(filepath=file_path,
                                expected_type=FeatureSchema,
                                verbose=True)
    return schema


def info():
    _script_info(__all__)
