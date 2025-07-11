from pprint import pprint
from typing import Optional, List, Dict, Callable, Union
from pathlib import Path
from .utilities import _script_info
from .logger import _LOGGER


__all__ = [
    "PathManager"
]


class PathManager:
    """
    Manages and stores a project's file paths, acting as a centralized
    "path database". It supports both development mode and applications
    bundled with Briefcase.
    
    Supports python dictionary syntax.
    """
    def __init__(
        self,
        anchor_file: str,
        base_directories: Optional[List[str]] = None
    ):
        """
        The initializer determines the project's root directory and can pre-register
        a list of base directories relative to that root.

        Args:
            anchor_file (str): The absolute path to a file whose parent directory will be considered the package root and name. Typically, `__file__`.
            base_directories (Optional[List[str]]): A list of directory names located at the same level as the anchor file to be registered immediately.
        """
        resolved_anchor_path = Path(anchor_file).resolve()
        self._package_name = resolved_anchor_path.parent.name
        self._is_bundled, self._resource_path_func = self._check_bundle_status()
        self._paths: Dict[str, Path] = {}

        if self._is_bundled:
            # In a bundle, resource_path gives the absolute path to the 'app_packages' dir
            # when given the package name.
            package_root = self._resource_path_func(self._package_name) # type: ignore
        else:
            # In dev mode, the package root is the directory containing the anchor file.
            package_root = resolved_anchor_path.parent

        # Register the root of the package itself
        self._paths["ROOT"] = package_root

        # Register all the base directories
        if base_directories:
            for dir_name in base_directories:
                # In dev mode, this is simple. In a bundle, we must resolve
                # each path from the package root.
                if self._is_bundled:
                     self._paths[dir_name] = self._resource_path_func(self._package_name, dir_name) # type: ignore
                else:
                     self._paths[dir_name] = package_root / dir_name
                     
    # A helper function to find the briefcase-injected resource function
    def _check_bundle_status(self) -> tuple[bool, Optional[Callable]]:
        """Checks if the app is running in a Briefcase bundle."""
        try:
            # This function is injected by Briefcase into the global scope
            from briefcase.platforms.base import resource_path # type: ignore
            return True, resource_path
        except (ImportError, NameError):
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
                        f"Path key '{key}' already exists in the manager. To replace it, call update() with overwrite=True."
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


def info():
    _script_info(__all__)
