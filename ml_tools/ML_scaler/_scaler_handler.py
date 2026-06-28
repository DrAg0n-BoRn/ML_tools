import torch
from pathlib import Path
from typing import Union, Optional

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import ScalerKeys, DatasetKeys

from ._ML_scaler import DragonScaler

_LOGGER = get_logger("Scaler Handler")

__all__ = ["DragonScalerHandler"]


class DragonScalerHandler:
    """
    Handles the loading and parsing of saved DragonScaler artifacts.
    
    This class manages the complexities of loading scaler states from disk. It can 
    resolve a specific `.pth` file or automatically find the appropriate scaler 
    file within a given directory. It safely handles both consolidated scaler 
    dictionaries (containing feature and/or target scalers) and legacy standalone 
    scalers, mapping tensors to the CPU to prevent cross-device loading errors.

    Attributes:
        source_path (Optional[Path]): The exact resolved path of the loaded scaler file.
        feature_scaler (DragonScaler): The loaded Feature DragonScaler, if available.
        target_scaler (DragonScaler): The loaded Target DragonScaler, if available.
    """
    def __init__(self, scaler_path: Union[str, Path]):
        """
        Initializes the handler and attempts to load scalers from the given path.

        Args:
            scaler_path (Union[str, Path]): The direct path to a `.pth` scaler file, 
                or a directory containing exactly one matching scaler artifact.

        Raises:
            FileNotFoundError: If the path does not exist, is neither a file nor 
                directory, or if a directory contains zero or multiple matching files.
        """
        self._feature_scaler: Optional[DragonScaler] = None
        self._target_scaler: Optional[DragonScaler] = None
        self.source_path: Optional[Path] = None
        
        self._load_scalers(scaler_path)

    def _load_scalers(self, scaler_path: Union[str, Path]) -> None:
        target_path = make_fullpath(scaler_path, make=False)

        if target_path.is_dir():
            expected_pattern = f"{DatasetKeys.SCALER_PREFIX}*.pth"
            matching_files = list(target_path.glob(expected_pattern))
            
            if not matching_files:
                _LOGGER.error(f"No files matching pattern '{expected_pattern}' found in directory '{target_path}'.")
                raise FileNotFoundError()
            elif len(matching_files) > 1:
                _LOGGER.error(f"Multiple files matching pattern '{expected_pattern}' found in directory '{target_path}'. Please specify the exact file.")
                raise FileNotFoundError()
            else:
                target_path = matching_files[0]
        elif not target_path.is_file():
            _LOGGER.error(f"Provided path '{target_path}' is neither a file nor a directory.")
            raise FileNotFoundError()
        
        self.source_path = target_path

        loaded_data = torch.load(target_path, map_location="cpu", weights_only=False)

        if ScalerKeys.FEATURE_SCALER in loaded_data or ScalerKeys.TARGET_SCALER in loaded_data:
            f_state = loaded_data.get(ScalerKeys.FEATURE_SCALER)
            t_state = loaded_data.get(ScalerKeys.TARGET_SCALER)
            
            if f_state is not None:
                self._feature_scaler = DragonScaler.load(f_state, verbose=False)
            if t_state is not None:
                self._target_scaler = DragonScaler.load(t_state, verbose=False)
        else:
            _LOGGER.warning(f"File '{target_path.name}' contains a standalone scaler without explicit keys. Assigning it to feature_scaler by default.")
            self._feature_scaler = DragonScaler.load(loaded_data, verbose=False)
            
        # Report the loaded scalers
        if self._feature_scaler is not None and self._target_scaler is not None:
            _LOGGER.info(f"Feature Scaler and Target Scaler loaded from '{target_path.name}'.")
        elif self._feature_scaler is not None:
            _LOGGER.info(f"Feature Scaler loaded from '{target_path.name}'. No Target Scaler found.")
        elif self._target_scaler is not None:
            _LOGGER.info(f"Target Scaler loaded from '{target_path.name}'. No Feature Scaler found.")
        else:
            _LOGGER.error(f"No valid scalers found in '{target_path.name}'.")
            raise ValueError()

    @property
    def feature_scaler(self) -> DragonScaler:
        """
        Returns the loaded Feature DragonScaler. If not loaded, raises a ValueError.
        """
        if self._feature_scaler is None:
            _LOGGER.error(f"Feature DragonScaler not loaded. Source: '{self.source_path.name if self.source_path else 'Unknown Path'}'.")
            raise ValueError()
        return self._feature_scaler

    @property
    def target_scaler(self) -> DragonScaler:
        """
        Returns the loaded Target DragonScaler. If not loaded, raises a ValueError.
        """
        if self._target_scaler is None:
            _LOGGER.error(f"Target DragonScaler not loaded. Source: '{self.source_path.name if self.source_path else 'Unknown Path'}'.")
            raise ValueError()
        return self._target_scaler
    
    def status(self) -> None:
        """
        Displays a string representation of the current status of the handler.
        """
        _LOGGER.info(repr(self))
    
    def __repr__(self) -> str:
        f_status = "✅ Loaded" if self._feature_scaler else "❌ None"
        t_status = "✅ Loaded" if self._target_scaler else "❌ None"
        file_name = self.source_path.name if self.source_path else "❓ Unknown"
        return f"DragonScalerHandler(feature_scaler={f_status}, target_scaler={t_status}, source='{file_name}')"
