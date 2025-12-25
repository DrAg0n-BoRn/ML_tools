from typing import Optional, Union
from pathlib import Path
import sys

from .._core import get_logger

from ._path_tools import sanitize_filename


_LOGGER = get_logger("DragonPathManager")


__all__ = [
    "DragonPathManager"
]


class DragonPathManager:
    """
    Manages and stores a project's file paths, acting as a centralized
    "path database". It supports both development mode and applications
    bundled with Pyinstaller or Nuitka.
    
    All keys provided to the manager are automatically sanitized to ensure
    they are valid Python identifiers. This allows for clean, attribute-style
    access. The sanitization process involves replacing whitespace with
    underscores and removing special characters.
    """
    def __init__(
        self,
        anchor_file: str,
        base_directories: Optional[list[str]] = None,
        strict_to_root: bool = True
    ):
        """
        Sets up the core paths for a project by anchoring to a specific file.

        The manager automatically registers a 'ROOT' path, which points to the
        root of the package, and can pre-register common subdirectories found
        directly within that root.

        Args:
            anchor_file (str): The path to a file within your package, typically
                            the `__file__` of the script where DragonPathManager
                            is instantiated. This is used to locate the
                            package root directory.
            base_directories (List[str] | None): An optional list of strings,
                                                    where each string is the name
                                                    of a subdirectory to register
                                                    relative to the package root.
            strict_to_root (bool): If True, checks that all registered paths are defined within the package ROOT.
        """
        resolved_anchor_path = Path(anchor_file).resolve()
        self._package_name = resolved_anchor_path.parent.name
        self._is_bundled, bundle_root = self._get_bundle_root()
        self._paths: dict[str, Path] = {}
        self._strict_to_root = strict_to_root

        if self._is_bundled:
            # In a PyInstaller/Nuitka bundle, the package is inside the temp _MEIPASS dir
            package_root = Path(bundle_root) / self._package_name # type: ignore
        else:
            # In dev mode, the package root is the directory containing the anchor file.
            package_root = resolved_anchor_path.parent

        # Register the root of the package itself
        self.ROOT = package_root

        # Register all the base directories
        if base_directories:
            for dir_name in base_directories:
                sanitized_dir_name = self._sanitize_key(dir_name)
                self._check_underscore_key(sanitized_dir_name)
                setattr(self, sanitized_dir_name, package_root / sanitized_dir_name)
        
        # Signal that initialization is complete.
        self._initialized = True
    
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
        
    def _check_underscore_key(self, key: str) -> None:
        if key.startswith("_"):
            _LOGGER.error(f"Path key '{key}' cannot start with underscores.")
            raise ValueError()

    def update(self, new_paths: dict[str, Union[str, Path]]) -> None:
        """
        Adds new paths in the manager.

        Args:
            new_paths (dict[str, Union[str, Path]]): A dictionary where keys are
                                    the identifiers and values are the
                                    Path objects to store.
        """
        # Pre-check
        for key in new_paths:
            sanitized_key = self._sanitize_key(key)
            self._check_underscore_key(sanitized_key)
            if hasattr(self, sanitized_key):
                _LOGGER.error(f"Cannot add path for key '{sanitized_key}' ({key}): an attribute with this name already exists.")
                raise KeyError()
        
        # If no conflicts, add new paths
        for key, value in new_paths.items():
            self.__setattr__(key, value)
        
    def _sanitize_key(self, key: str):
        return sanitize_filename(key)
        
    def make_dirs(self, keys: Optional[list[str]] = None, verbose: bool = False) -> None:
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
                    _LOGGER.warning(f"Key '{key}' not found in DragonPathManager, skipping.")
        else:
            path_items = self._paths.items()

        # Get the package root to check against.
        package_root = self._paths.get("ROOT")

        for key, path in path_items:
            if path.suffix:  # It's a file, not a directory
                continue

            # --- CRITICAL CHECK ---
            # Determine if the path is inside the main application package.
            is_internal_path = package_root and path.is_relative_to(package_root)

            if self._is_bundled and is_internal_path:
                if verbose:
                    _LOGGER.warning(f"Skipping internal directory '{key}' in bundled app (read-only).")
                continue
            # -------------------------

            if verbose:
                _LOGGER.info(f"📁 Ensuring directory exists for key '{key}': {path}")

            path.mkdir(parents=True, exist_ok=True)
            
    def status(self) -> None:
        """
        Checks the status of all registered paths on the filesystem and prints a formatted report.
        """
        # 1. Gather Data and determine max widths
        rows = []
        max_key_len = len("Key")  # Start with header width
        
        # Sort by key for readability
        for key, path in sorted(self.items()):
            if path.is_dir():
                stat_msg = "📁 Directory"
            elif path.is_file():
                stat_msg = "📄 File"
            elif not path.exists():
                stat_msg = "❌ Not Found"
            else:
                stat_msg = "❓ Unknown"
            
            rows.append((key, stat_msg, str(path)))
            max_key_len = max(max_key_len, len(key))

        # 2. Print Header
        mode_icon = "📦" if self._is_bundled else "🛠️"
        mode_text = "Bundled Mode" if self._is_bundled else "Development Mode"
        
        print(f"\n{'-'*80}")
        print(f" 🐉 DragonPathManager Status Report")
        print(f"    Context: {mode_icon} {mode_text}")
        print(f"    Root:    {self.ROOT}")
        print(f"{'-'*80}")

        # 3. Print Table Header
        # {variable:<width} aligns text to the left within the padding
        print(f" {'Key':<{max_key_len}} | {'Status':<12} | Path")
        print(f" {'-'*max_key_len} | {'-'*12} | {'-'*40}")

        # 4. Print Rows
        for key, stat, p_str in rows:
            print(f" {key:<{max_key_len}} | {stat:<12} | {p_str}")
        
        print(f"{'-'*80}\n")

    def __repr__(self) -> str:
        """Provides a string representation of the stored paths."""
        path_list = "\n".join(f"  '{k}': '{v}'" for k, v in self._paths.items())
        return f"DragonPathManager(\n{path_list}\n)"
    
    # --- Dictionary-Style Methods ---
    def __getitem__(self, key: str) -> Path:
        """Allows dictionary-style getting, e.g., PM['my_key']"""
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Union[str, Path]):
        """Allows dictionary-style setting, e.g., PM['my_key'] = path"""
        sanitized_key = self._sanitize_key(key)
        self._check_underscore_key(sanitized_key)
        self.__setattr__(sanitized_key, value)

    def __contains__(self, key: str) -> bool:
        """Allows checking for a key's existence, e.g., if 'my_key' in PM"""
        sanitized_key = self._sanitize_key(key)
        true_false = sanitized_key in self._paths
        # print(f"key {sanitized_key} in current path dictionary keys: {true_false}")
        return true_false

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
    
    def __getattr__(self, name: str) -> Path:
        """
        Allows attribute-style access to paths, e.g., PM.data.
        """
        # Block access to private attributes
        if name.startswith('_'):
            _LOGGER.error(f"Access to private attribute '{name}' is not allowed, remove leading underscore.")
            raise AttributeError()
        
        sanitized_name = self._sanitize_key(name)
        
        try:
            # Look for the key in our internal dictionary
            return self._paths[sanitized_name]
        except KeyError:
            # If not found, raise an AttributeError
            _LOGGER.error(f"'{type(self).__name__}' object has no attribute or path key '{sanitized_name}'")
            raise AttributeError()
    
    def __setattr__(self, name: str, value: Union[str, Path, bool, dict, str, int, tuple]):
        """Allows attribute-style setting of paths, e.g., PM.data = 'path/to/data'."""
        # Check for internal attributes, which are set directly on the object.
        if name.startswith('_'):
            # This check prevents setting new private attributes after __init__ is done.
            is_initialized = self.__dict__.get('_initialized', False)
            if is_initialized:
                _LOGGER.error(f"Cannot set private attribute '{name}' after initialization.")
                raise AttributeError()
            super().__setattr__(name, value)
            return

        # Sanitize the key for the public path.
        sanitized_name = self._sanitize_key(name)
        self._check_underscore_key(sanitized_name)

        # Prevent overwriting existing methods (e.g., PM.status = 'foo').
        # This check looks at the class, not the instance therefore won't trigger __getattr__.
        if hasattr(self.__class__, sanitized_name):
            _LOGGER.error(f"Cannot overwrite existing attribute or method '{sanitized_name}' ({name}).")
            raise AttributeError()
        
        if not isinstance(value, (str, Path)):
            _LOGGER.error(f"Cannot assign type '{type(value).__name__}' to a path. Must be str or Path.")
            raise TypeError()
        
        # Resolve the new path
        new_path = Path(value).expanduser().absolute()

        # --- STRICT CHECK ---
        # Only check if strict mode is on
        if self.__dict__.get("_strict_to_root", False) and sanitized_name != "ROOT":
            root_path = self._paths.get("ROOT")
            # Ensure ROOT exists and the new path is inside it
            if root_path and not new_path.is_relative_to(root_path):
                _LOGGER.error(f"Strict Mode Violation: '{name}' ({new_path}) is outside ROOT ({root_path})")
                raise ValueError()

        # Store absolute Path.
        self._paths[sanitized_name] = new_path

