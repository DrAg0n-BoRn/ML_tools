from typing import Optional, Union, Literal
from pathlib import Path
import re

from .._core import get_logger


_LOGGER = get_logger("Path Ops")


__all__ = [
    "make_fullpath",
    "sanitize_filename",
    "list_csv_paths",
    "list_files_by_extension",
    "list_files_by_extension_global",
    "list_subdirectories",
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
        msg = f"No '.{normalized_ext}' files found in directory: '{dir_path}'."
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


def list_files_by_extension_global(
    directory: Union[str, Path], 
    extension: str, 
    depth: int = -1,
    verbose: int = 2,
    raise_on_empty: bool = True
) -> list[tuple[str, Path]]:
    """
    Lists all files with the specified extension in the given directory and its subdirectories.
    
    Returns a list of tuples containing the filename (without extension) and its absolute path.

    Parameters:
        directory (str | Path): Path to the directory to search in.
        extension (str): File extension to search for (e.g., 'json', 'txt').
        depth (int): Search depth limit. >= 1 for specific depths, or -1 for all subdirectories.
            - depth=1 only the specified directory
            - depth=2 immediate subdirectories, etc.
        verbose (int): Logs the process. 0=silent, 1=warnings, 2=info, 3=detailed information.
        raise_on_empty (bool): If True, raises IOError if no matching files are found.

    Returns:
        (list[tuple[str, Path]]): List of tuples containing (filename, filepath). 
    """
    dir_path = make_fullpath(directory, enforce="directory")
    
    # Validate depth parameter
    if not isinstance(depth, int):
        _LOGGER.error("Depth must be an integer.")
        raise TypeError()
    
    if depth < -1 or depth == 0:
        _LOGGER.error("Depth must be >= 1, or -1 for all subdirectories.")
        raise ValueError()

    normalized_ext = extension.lstrip(".").lower()
    matched_paths = []
    
    if depth == -1:
        # Infinite depth
        matched_paths = list(dir_path.rglob(f"*.{normalized_ext}"))
    else:
        # Controlled depth using BFS
        queue = [(dir_path, 1)]
        while queue:
            current_dir, current_depth = queue.pop(0)
            try:
                for item in current_dir.iterdir():
                    if item.is_file() and item.suffix.lower() == f".{normalized_ext}":
                        matched_paths.append(item)
                    elif item.is_dir() and current_depth < depth:
                        queue.append((item, current_depth + 1))
            except PermissionError:
                continue
    
    if not matched_paths:
        msg = f"No '.{normalized_ext}' files found in directory tree: '{dir_path}' (depth={depth})."
        if raise_on_empty:
            _LOGGER.error(msg)
            raise IOError()
        else:
            if verbose >= 1:
                _LOGGER.warning(msg)
            return []

    # List of tuples mapping (filename (no extension), absolute filepath)
    name_path_list = [(p.stem, p) for p in matched_paths]
    
    report_depth = '♾️' if depth == -1 else str(depth)
    
    if verbose >= 3:
        _LOGGER.info(f"📂 '{normalized_ext.upper()}' files found (depth={report_depth}):")
        for name, _ in name_path_list:
            print(f"\t{name}")
    elif verbose >= 2:
        _LOGGER.info(f"Found {len(name_path_list)} '.{normalized_ext}' files in '{dir_path}' (depth={report_depth}).")
    
    return name_path_list


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
        _LOGGER.info(f"Found {count} subdirector{plural} in '{root_path.name}'.")
    
    # Create a dictionary where the key is the directory's name (a string)
    # and the value is the full Path object.
    dir_map = {p.name: p for p in directories}
    
    return dir_map
