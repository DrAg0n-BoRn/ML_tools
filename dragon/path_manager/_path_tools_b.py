from typing import Optional, Union
from pathlib import Path
import shutil

from .._core import get_logger

from ._path_tools import make_fullpath, sanitize_filename


_LOGGER = get_logger("Path Ops")


__all__ = [
    "clean_directory",
    "safe_move",
]


def clean_directory(directory: Union[str, Path], verbose: bool = True) -> None:
    """
    ⚠️  DANGER: DESTRUCTIVE OPERATION ⚠️

    Deletes all files and subdirectories inside the specified directory. It is designed to empty a folder, not delete the folder itself.

    Safety: It skips hidden files and directories (those starting with a period '.'). This works for macOS/Linux hidden files and dot-config folders on Windows.

    Args:
        directory (str | Path): The directory path to clean.
        verbose (bool): If True, prints the name of each top-level item deleted.
    """
    target_dir = make_fullpath(directory, enforce="directory")

    if verbose:
        _LOGGER.warning(f"Starting cleanup of directory: {target_dir}")

    for item in target_dir.iterdir():
        # Safety Check: Skip hidden files/dirs
        if item.name.startswith("."):
            continue

        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                if verbose:
                    print(f"    🗑️  Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                if verbose:
                    print(f"    🗑️  Deleted directory: {item.name}")
        except Exception as e:
            _LOGGER.warning(f"Failed to delete item '{item.name}': {e}")
            continue


def safe_move(
    source: Union[str, Path], 
    destination_directory: Union[str, Path], 
    rename: Optional[str] = None, 
    overwrite: bool = False
) -> Path:
    """
    Moves a file or directory to a destination directory with safety checks.

    Features:
    - Supports optional renaming (sanitized automatically).
    - PRESERVES file extensions during renaming (cannot be modified).
    - Prevents accidental overwrites unless explicit.

    Args:
        source (str | Path): The file or directory to move.
        destination_directory (str | Path): The destination DIRECTORY where the item will be moved. It will be created if it does not exist.
        rename (Optional[str]): If provided, the moved item will be renamed to this. Note: For files, the extension is strictly preserved.
        overwrite (bool): If True, overwrites the destination path if it exists.
    
    Returns:
        Path: The new absolute path of the moved item.
    """
    # 1. Validation and Setup
    src_path = make_fullpath(source, make=False)

    # Ensure destination directory exists
    dest_dir_path = make_fullpath(destination_directory, make=True, enforce="directory")

    # 2. Determine Target Name
    if rename:
        # Prevent double extensions if the user proactively included it
        if src_path.is_file() and src_path.suffix and rename.lower().endswith(src_path.suffix.lower()):
            rename = rename[:-len(src_path.suffix)]
            
        sanitized_name = sanitize_filename(rename)
        if src_path.is_file():
            # Strict Extension Preservation
            final_name = f"{sanitized_name}{src_path.suffix}"
        else:
            final_name = sanitized_name
    else:
        final_name = src_path.name

    final_path = dest_dir_path / final_name

    # 3. Safety Checks (Collision Detection)
    if final_path.exists():
        if not overwrite:
            _LOGGER.error(f"Destination already exists: '{final_path}'. Use overwrite=True to force.")
            raise FileExistsError()
        
        # Smart Overwrite Handling
        if final_path.is_dir():
            if src_path.is_file():
                _LOGGER.error(f"Cannot overwrite directory '{final_path}' with file '{src_path}'")
                raise IsADirectoryError()
            # If overwriting a directory, we must remove the old one first to avoid nesting/errors
            shutil.rmtree(final_path)
        else:
            # Destination is a file
            if src_path.is_dir():
                _LOGGER.error(f"Cannot overwrite file '{final_path}' with directory '{src_path}'")
                raise FileExistsError()
            final_path.unlink()

    # 4. Perform Move
    try:
        shutil.move(str(src_path), str(final_path))
        return final_path
    except Exception as e:
        _LOGGER.exception(f"Failed to move '{src_path}' to '{final_path}'")
        raise e

