from typing import Optional, Union, Literal
from pathlib import Path
import re
import shutil

from .._core import get_logger


_LOGGER = get_logger("Path Ops")


__all__ = [
    "make_fullpath",
    "sanitize_filename",
    "list_csv_paths",
    "list_files_by_extension",
    "list_subdirectories",
    "clean_directory",
    "safe_move",
]


def make_fullpath(
        input_path: Union[str, Path],
        make: bool = False,
        verbose: bool = False,
        enforce: Optional[Literal["directory", "file"]] = None
    ) -> Path:
    """
    Resolves a string or Path into an absolute Path, optionally creating it.

    - If the path exists, it is returned.
    - If the path does not exist and `make=True`, it will:
        - Create the file if the path has a suffix
        - Create the directory if it has no suffix
    - If `make=False` and the path does not exist, an error is raised.
    - If `enforce`, raises an error if the resolved path is not what was enforced.
    - Optionally prints whether the resolved path is a file or directory.

    Parameters:
        input_path (str | Path): 
            Path to resolve.
        make (bool): 
            If True, attempt to create file or directory.
        verbose (bool): 
            Print classification after resolution.
        enforce ("directory" | "file" | None):
            Raises an error if the resolved path is not what was enforced.

    Returns:
        Path: Resolved absolute path.

    Raises:
        ValueError: If the path doesn't exist and can't be created.
        TypeError: If the final path does not match the `enforce` parameter.
        
    ## 🗒️ Note:
    
    Directories with dots will be treated as files.
    
    Files without extension will be treated as directories.
    """
    path = Path(input_path).expanduser()

    is_file = path.suffix != ""

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        if not make:
            _LOGGER.error(f"Path does not exist: '{path}'.")
            raise FileNotFoundError()

        try:
            if is_file:
                # Create parent directories first
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=False)
            else:
                path.mkdir(parents=True, exist_ok=True)
            resolved = path.resolve(strict=True)
        except Exception:
            _LOGGER.exception(f"Failed to create {'file' if is_file else 'directory'} '{path}'.")
            raise IOError()
    
    if enforce == "file" and not resolved.is_file():
        _LOGGER.error(f"Path was enforced as a file, but it is not: '{resolved}'")
        raise TypeError()
    
    if enforce == "directory" and not resolved.is_dir():
        _LOGGER.error(f"Path was enforced as a directory, but it is not: '{resolved}'")
        raise TypeError()

    if verbose:
        if resolved.is_file():
            print("📄 Path is a File")
        elif resolved.is_dir():
            print("📁 Path is a Directory")
        else:
            print("❓ Path exists but is neither file nor directory")

    return resolved


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes the name by:
    - Stripping leading/trailing whitespace.
    - Replacing all internal whitespace characters with underscores.
    - Removing or replacing characters invalid in filenames.

    Args:
        filename (str): Base filename.

    Returns:
        str: A sanitized string suitable to use as a filename.
    """
    # Strip leading/trailing whitespace
    sanitized = filename.strip()
    
    # Replace all whitespace sequences (space, tab, etc.) with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)

    # Conservative filter to keep filenames safe across platforms
    sanitized = re.sub(r'[^\w\-.]', '', sanitized)
    
    # Check for empty string after sanitization
    if not sanitized:
        _LOGGER.error("The sanitized filename is empty. The original input may have contained only invalid characters.")
        raise ValueError()

    return sanitized


def list_csv_paths(directory: Union[str, Path], verbose: bool = True, raise_on_empty: bool = True) -> dict[str, Path]:
    """
    Lists all `.csv` files in the specified directory and returns a mapping: filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory containing `.csv` files.
        verbose (bool): If True, prints found files.
        raise_on_empty (bool): If True, raises IOError if no files are found.

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}.
    """
    # wraps the more general function
    return list_files_by_extension(directory=directory, extension="csv", verbose=verbose, raise_on_empty=raise_on_empty)


def list_files_by_extension(
    directory: Union[str, Path], 
    extension: str, 
    verbose: bool = True,
    raise_on_empty: bool = True
) -> dict[str, Path]:
    """
    Lists all files with the specified extension in the given directory and returns a mapping: 
    filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory to search in.
        extension (str): File extension to search for (e.g., 'json', 'txt').
        verbose (bool): If True, logs the files found.
        raise_on_empty (bool): If True, raises IOError if no matching files are found.

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}. Returns empty dict if none found and raise_on_empty is False.
    """
    dir_path = make_fullpath(directory, enforce="directory")
    
    # Normalize the extension (remove leading dot if present)
    normalized_ext = extension.lstrip(".").lower()
    pattern = f"*.{normalized_ext}"
    
    matched_paths = list(dir_path.glob(pattern))
    
    if not matched_paths:
        msg = f"No '.{normalized_ext}' files found in directory: {dir_path}."
        if raise_on_empty:
            _LOGGER.error(msg)
            raise IOError()
        else:
            if verbose:
                _LOGGER.warning(msg)
            return {}

    name_path_dict = {p.stem: p for p in matched_paths}
    
    if verbose:
        _LOGGER.info(f"📂 '{normalized_ext.upper()}' files found:")
        for name in name_path_dict:
            print(f"\t{name}")
    
    return name_path_dict


def list_subdirectories(
    root_dir: Union[str, Path], 
    verbose: bool = True, 
    raise_on_empty: bool = True
) -> dict[str, Path]:
    """
    Scans a directory and returns a dictionary of its immediate subdirectories.

    Args:
        root_dir (str | Path): The path to the directory to scan.
        verbose (bool): If True, prints the number of directories found. 
        raise_on_empty (bool): If True, raises IOError if no subdirectories are found.

    Returns:
        dict[str, Path]: A dictionary mapping subdirectory names (str) to their full Path objects.
    """
    root_path = make_fullpath(root_dir, enforce="directory")
    
    directories = [p.resolve() for p in root_path.iterdir() if p.is_dir()]
    
    if len(directories) < 1:
        msg = f"No subdirectories found inside '{root_path}'"
        if raise_on_empty:
            _LOGGER.error(msg)
            raise IOError()
        else:
            if verbose:
                _LOGGER.warning(msg)
            return {}
    
    if verbose:
        count = len(directories)
        # Use pluralization for better readability
        plural = 'ies' if count != 1 else 'y'
        print(f"Found {count} subdirector{plural} in '{root_path.name}'.")
    
    # Create a dictionary where the key is the directory's name (a string)
    # and the value is the full Path object.
    dir_map = {p.name: p for p in directories}
    
    return dir_map


def clean_directory(directory: Union[str, Path], verbose: bool = False) -> None:
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
    final_destination: Union[str, Path], 
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
        final_destination (str | Path): The destination DIRECTORY where the item will be moved. It will be created if it does not exist.
        rename (Optional[str]): If provided, the moved item will be renamed to this. Note: For files, the extension is strictly preserved.
        overwrite (bool): If True, overwrites the destination path if it exists.
    
    Returns:
        Path: The new absolute path of the moved item.
    """
    # 1. Validation and Setup
    src_path = make_fullpath(source, make=False)

    # Ensure destination directory exists
    dest_dir_path = make_fullpath(final_destination, make=True, enforce="directory")

    # 2. Determine Target Name
    if rename:
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

