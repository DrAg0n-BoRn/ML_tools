from pprint import pprint
from typing import Optional, List, Dict, Union, Literal
from pathlib import Path
import re
from ._script_info import _script_info
from ._logger import _LOGGER
import sys


__all__ = [
    "PathManager",
    "make_fullpath",
    "sanitize_filename",
    "list_csv_paths",
    "list_files_by_extension",
]


class PathManager:
    """
    Manages and stores a project's file paths, acting as a centralized
    "path database". It supports both development mode and applications
    bundled with Pyinstaller.
    
    Supports python dictionary syntax.
    """
    def __init__(
        self,
        anchor_file: str,
        base_directories: Optional[List[str]] = None
    ):
        resolved_anchor_path = Path(anchor_file).resolve()
        self._package_name = resolved_anchor_path.parent.name
        self._is_bundled, bundle_root = self._get_bundle_root()
        self._paths: Dict[str, Path] = {}

        if self._is_bundled:
            # In a PyInstaller bundle, the package is inside the temp _MEIPASS dir
            package_root = Path(bundle_root) / self._package_name # type: ignore
        else:
            # In dev mode, the package root is the directory containing the anchor file.
            package_root = resolved_anchor_path.parent

        # Register the root of the package itself
        self._paths["ROOT"] = package_root

        # Register all the base directories
        if base_directories:
            for dir_name in base_directories:
                # This logic works for both dev mode and bundled mode
                self._paths[dir_name] = package_root / dir_name
    
    def _get_bundle_root(self) -> tuple[bool, Optional[str]]:
        """
        Checks if the app is running in a PyInstaller or Nuitka bundle and returns the root path.
        
        Returns:
            A tuple (is_bundled, bundle_root_path).
        """
        # --- PyInstaller Check ---
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # The bundle root for PyInstaller is the temporary _MEIPASS directory
            return True, sys._MEIPASS # type: ignore
        
        # --- Nuitka Check ---
        elif '__nuitka_binary_dir' in sys.__dict__:
            # For Nuitka, the root is the directory of the binary.
            # Unlike PyInstaller's _MEIPASS, this is the final install location.
            return True, sys.__dict__['__nuitka_binary_dir']
            
        # --- Not Bundled ---
        else:
            return False, None

    def get(self, key: str) -> Path:
        """
        Retrieves a stored path by its key.

        Args:
            key (str): The key of the path to retrieve.

        Returns:
            Path: The resolved, absolute Path object.

        Raises:
            KeyError: If the key is not found in the manager.
        """
        try:
            return self._paths[key]
        except KeyError:
            _LOGGER.error(f"❌ Path key '{key}' not found.")
            raise

    def update(self, new_paths: Dict[str, Union[str, Path]], overwrite: bool = False) -> None:
        """
        Adds new paths or overwrites existing ones in the manager.

        Args:
            new_paths (Dict[str, Union[str, Path]]): A dictionary where keys are
                                    the identifiers and values are the
                                    Path objects or strings to store.
            overwrite (bool): If False (default), raises a KeyError if any
                            key in new_paths already exists. If True,
                            allows overwriting existing keys.
        """
        if not overwrite:
            for key in new_paths:
                if key in self._paths:
                    raise KeyError(
                        f"❌ Path key '{key}' already exists in the manager. To replace it, call update() with overwrite=True."
                    )

        # Resolve any string paths to Path objects before storing
        resolved_new_paths = {k: Path(v) for k, v in new_paths.items()}
        self._paths.update(resolved_new_paths)
        
    def make_dirs(self, keys: Optional[List[str]] = None, verbose: bool = False) -> None:
        """
        Creates directory structures for registered paths in writable locations.

        This method identifies paths that are directories (no file suffix) and creates them on the filesystem.

        In a bundled application, this method will NOT attempt to create directories inside the read-only app package, preventing crashes. It
        will only operate on paths outside of the package (e.g., user data dirs).

        Args:
            keys (Optional[List[str]]): If provided, only the directories
                                        corresponding to these keys will be
                                        created. If None (default), all
                                        registered directory paths are used.
            verbose (bool): If True, prints a message for each action.
        """
        path_items = []
        if keys:
            for key in keys:
                if key in self._paths:
                    path_items.append((key, self._paths[key]))
                elif verbose:
                    _LOGGER.warning(f"⚠️ Key '{key}' not found in PathManager, skipping.")
        else:
            path_items = self._paths.items()

        # Get the package root to check against.
        package_root = self._paths.get("ROOT")

        for key, path in path_items:
            if path.suffix:  # It's a file, not a directory
                continue

            # --- THE CRITICAL CHECK ---
            # Determine if the path is inside the main application package.
            is_internal_path = package_root and path.is_relative_to(package_root)

            if self._is_bundled and is_internal_path:
                if verbose:
                    _LOGGER.warning(f"⚠️ Skipping internal directory '{key}' in bundled app (read-only).")
                continue
            # -------------------------

            if verbose:
                _LOGGER.info(f"📁 Ensuring directory exists for key '{key}': {path}")

            path.mkdir(parents=True, exist_ok=True)
            
    def status(self) -> None:
        """
        Checks the status of all registered paths on the filesystem and prints a formatted report.
        """
        report = {}
        for key, path in self.items():
            if path.is_dir():
                report[key] = "📁 Directory"
            elif path.is_file():
                report[key] = "📄 File"
            else:
                report[key] = "❌ Not Found"

        print("\n--- Path Status Report ---")
        pprint(report)

    def __repr__(self) -> str:
        """Provides a string representation of the stored paths."""
        path_list = "\n".join(f"  '{k}': '{v}'" for k, v in self._paths.items())
        return f"PathManager(\n{path_list}\n)"
    
    # --- Dictionary-Style Methods ---
    def __getitem__(self, key: str) -> Path:
        """Allows dictionary-style getting, e.g., PM['my_key']"""
        return self.get(key)

    def __setitem__(self, key: str, value: Union[str, Path]):
        """Allows dictionary-style setting, does not allow overwriting, e.g., PM['my_key'] = path"""
        self.update({key: value}, overwrite=False)

    def __contains__(self, key: str) -> bool:
        """Allows checking for a key's existence, e.g., if 'my_key' in PM"""
        return key in self._paths

    def __len__(self) -> int:
        """Allows getting the number of paths, e.g., len(PM)"""
        return len(self._paths)

    def keys(self):
        """Returns all registered path keys."""
        return self._paths.keys()

    def values(self):
        """Returns all registered Path objects."""
        return self._paths.values()

    def items(self):
        """Returns all registered (key, Path) pairs."""
        return self._paths.items()


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
            raise ValueError(f"❌ Path does not exist: '{path}'")

        try:
            if is_file:
                # Create parent directories first
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=False)
            else:
                path.mkdir(parents=True, exist_ok=True)
            resolved = path.resolve(strict=True)
        except Exception as e:
            raise ValueError(f"❌ Failed to create {'file' if is_file else 'directory'} '{path}': {e}")
    
    if enforce == "file" and not resolved.is_file():
        raise TypeError(f"❌ Path was enforced as a file, but it is not: '{resolved}'")
    
    if enforce == "directory" and not resolved.is_dir():
        raise TypeError(f"❌ Path was enforced as a directory, but it is not: '{resolved}'")

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
        raise ValueError("The sanitized filename is empty. The original input may have contained only invalid characters.")

    return sanitized


def list_csv_paths(directory: Union[str,Path], verbose: bool=True) -> dict[str, Path]:
    """
    Lists all `.csv` files in the specified directory and returns a mapping: filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory containing `.csv` files.

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}.
    """
    dir_path = make_fullpath(directory)

    csv_paths = list(dir_path.glob("*.csv"))
    if not csv_paths:
        raise IOError(f"❌ No CSV files found in directory: {dir_path.name}")
    
    # make a dictionary of paths and names
    name_path_dict = {p.stem: p for p in csv_paths}
    
    if verbose:
        _LOGGER.info("🗂️ CSV files found:")
        for name in name_path_dict.keys():
            print(f"\t{name}")

    return name_path_dict


def list_files_by_extension(directory: Union[str,Path], extension: str, verbose: bool=True) -> dict[str, Path]:
    """
    Lists all files with the specified extension in the given directory and returns a mapping: 
    filenames (without extensions) to their absolute paths.

    Parameters:
        directory (str | Path): Path to the directory to search in.
        extension (str): File extension to search for (e.g., 'json', 'txt').

    Returns:
        (dict[str, Path]): Dictionary mapping {filename: filepath}.
    """
    dir_path = make_fullpath(directory)
    
    # Normalize the extension (remove leading dot if present)
    normalized_ext = extension.lstrip(".").lower()
    pattern = f"*.{normalized_ext}"
    
    matched_paths = list(dir_path.glob(pattern))
    if not matched_paths:
        raise IOError(f"❌ No '.{normalized_ext}' files found in directory: {dir_path}")

    name_path_dict = {p.stem: p for p in matched_paths}
    
    if verbose:
        _LOGGER.info(f"\n📂 '{normalized_ext.upper()}' files found:")
        for name in name_path_dict:
            print(f"\t{name}")
    
    return name_path_dict


def info():
    _script_info(__all__)
