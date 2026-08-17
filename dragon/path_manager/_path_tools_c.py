from typing import Union, Optional
from pathlib import Path
import hashlib
from collections import defaultdict

from .._core import get_logger

from ._path_tools import make_fullpath


_LOGGER = get_logger("Path Ops")


__all__ = [
    "get_file_hash",
    "get_size",
    "find_duplicate_files",
]


def get_file_hash(filepath: Union[str, Path], algorithm: str = "sha256", print_result: bool = True) -> Optional[str]:
    """
    Calculates the hash of a file using the specified algorithm.

    Parameters:
        filepath (str | Path): The path to the file.
        algorithm (str): The hashing algorithm to use (e.g., 'md5', 'sha1', 'sha256').
        print_result (bool): If True, logs the hash to the console instead of returning it.

    Returns:
        str | None : The hex digest of the file's hash if not logged to console.
    """
    path = make_fullpath(filepath, enforce="file")
    
    # validate the algorithm
    if algorithm not in hashlib.algorithms_available:
        _LOGGER.error(f"Unsupported hashing algorithm: {algorithm}")
        raise ValueError()

    hash_func = getattr(hashlib, algorithm)()
    
    # Read in chunks to efficiently handle large files
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_func.update(chunk)
    
    # hexdigest returns the hash as a string of hexadecimal digits
    result = hash_func.hexdigest()
    if print_result:
        _LOGGER.info(result)
        return None
    else:
        return result


def get_size(path: Union[str, Path], human_readable: bool = True, print_result: bool = True) -> Optional[str]:
    """
    Calculates the size of a file or directory.

    Parameters:
        path (str | Path): The path to evaluate.
        human_readable (bool): If True, returns a formatted string (e.g., '1.50 MB'). 
                               If False, returns the raw byte count as a string.
        print_result (bool): If True, logs the size to the console instead of returning it.

    Returns:
        str | None: The calculated size if not logged to console.
    """
    target = make_fullpath(path)
    
    if target.is_file():
        size_bytes = target.stat().st_size
    elif target.is_dir():
        size_bytes = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
    else:
        _LOGGER.error(f"Path is neither a file nor a directory: '{target}'.")
        raise ValueError()

    if not human_readable:
        final_result = str(size_bytes)
        if print_result:
            _LOGGER.info(final_result)
            return None
        else:
            return final_result

    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']:
        if size_bytes < 1024.0:
            final_result = f"{size_bytes:.2f} {unit}".replace(".00", "")
            if print_result:
                _LOGGER.info(final_result)
                return None
            else:
                return final_result
        size_bytes /= 1024.0
    else:
        _LOGGER.warning("Size exceeds Yottabytes.")
        return None


def find_duplicate_files(directory: Union[str, Path], verbose: bool=True) -> dict[str, tuple[Path, ...]]:
    """
    Scans a directory recursively to find duplicate files based on their content.
    
    Uses file size as a preliminary filter before computing hashes to optimize performance.

    Parameters:
        directory (str | Path): The root directory to scan.
        verbose (bool): If True, logs whether duplicates were found and how many groups exist.

    Returns:
        dict[str, tuple[Path, ...]]: A dictionary mapping the first discovered filename of the duplicates 
                                     to a tuple containing the absolute paths of all identical files.
    """
    dir_path = make_fullpath(directory, enforce="directory")

    # Preliminary filter: Group files by their exact byte size
    size_map = defaultdict(list)
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
                size_map[size].append(file_path)
            except OSError:
                continue

    duplicates = {}
    
    # Secondary filter: Compute and compare hashes only for files that share the same size
    for size, paths in size_map.items():
        if len(paths) > 1:
            hash_map = defaultdict(list)
            for path in paths:
                try:
                    file_hash = get_file_hash(path)
                    hash_map[file_hash].append(path)
                except OSError:
                    continue
            
            for file_hash, identical_paths in hash_map.items():
                if len(identical_paths) > 1:
                    # Using the filename of the first duplicate as the shared key identifier
                    shared_name = identical_paths[0].name
                    
                    # Prevent key collisions if multiple distinct duplicate groups share the same initial filename
                    if shared_name in duplicates:
                        shared_name = f"{shared_name}_{file_hash[:8]}"
                        
                    duplicates[shared_name] = tuple(identical_paths)
    
    if verbose:
        if duplicates:
            _LOGGER.warning(f"Found {len(duplicates)} groups of duplicate files.")
        else:
            _LOGGER.info("No duplicate files found.")
    
    return duplicates
