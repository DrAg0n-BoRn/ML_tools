from torch import nn
from typing import Union, Any
from pathlib import Path
import json
from abc import ABC, abstractmethod

from ..schema import FeatureSchema

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import PytorchModelArchitectureKeys, SchemaKeys

from .._core._schema_load_ops import prepare_schema_from_json


_LOGGER = get_logger("DragonModel: Save/Load")


__all__ = [
    "_ArchitectureHandlerMixin",
    "_ArchitectureBuilder",
]

##################################
# Mixin class for saving and loading basic model architectures
class _ArchitectureHandlerMixin:
    """
    A mixin class to provide save and load functionality for model architectures.
    """
    # abstract method that must be implemented by children
    @abstractmethod
    def get_architecture_config(self) -> dict[str, Any]:
        "To be implemented by children"
        pass
    
    def save_architecture(self: nn.Module, directory: Union[str, Path], verbose: bool = True): # type: ignore
        """Saves the model's architecture to an "architecture.json" file."""
        if not hasattr(self, 'get_architecture_config'):
            _LOGGER.error(f"Model '{self.__class__.__name__}' must have a 'get_architecture_config()' method to use this functionality.")
            raise AttributeError()

        path_dir = make_fullpath(directory, make=True, enforce="directory")
        
        json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
        
        full_path = path_dir / json_filename

        config = {
            PytorchModelArchitectureKeys.MODEL: self.__class__.__name__,
            PytorchModelArchitectureKeys.CONFIG: self.get_architecture_config() # type: ignore
        }

        with open(full_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        if verbose:
            _LOGGER.info(f"Architecture for '{self.__class__.__name__}' saved as '{full_path.name}'")

    @classmethod
    def load_architecture(cls: type, file_or_dir: Union[str, Path], verbose: bool = True) -> nn.Module:
        """
        Loads a model architecture from a JSON file. 
        If a directory is provided, the function will attempt to load the JSON file "architecture.json" inside.
        """
        user_path = make_fullpath(file_or_dir)
        
        if user_path.is_dir():
            json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
            target_path = make_fullpath(user_path / json_filename, enforce="file")
        elif user_path.is_file():
            target_path = user_path
        else:
            _LOGGER.error(f"Invalid path: '{file_or_dir}'")
            raise IOError()

        with open(target_path, 'r') as f:
            saved_data = json.load(f)

        saved_class_name = saved_data[PytorchModelArchitectureKeys.MODEL]
        config = saved_data[PytorchModelArchitectureKeys.CONFIG]

        if saved_class_name != cls.__name__:
            _LOGGER.error(f"Model class mismatch. File specifies '{saved_class_name}', but '{cls.__name__}' was expected.")
            raise ValueError()

        # Hook to allow children classes to modify config before init (reconstruction)
        config = cls._prepare_config_for_load(config)

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model

    @classmethod
    def _prepare_config_for_load(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Hook method to process configuration data before model instantiation.
        Base implementation simply returns the config as-is.
        """
        return config


##################################
# Base class for loading and saving advanced models
##################################
class _ArchitectureBuilder(_ArchitectureHandlerMixin, nn.Module, ABC):
    """
    Base class for Dragon models that unifies architecture handling.
    
    Implements:
    - JSON serialization and JSON deserialization with automatic FeatureSchema reconstruction.
    - Standardized string representation (__repr__) showing hyperparameters.
    """
    def __init__(self):
        super().__init__()
        # Placeholder for hyperparameters, to be populated by child classes
        self.model_hparams: dict[str, Any] = {}

    @classmethod
    def _prepare_config_for_load(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Overrides the mixin hook to reconstruct the FeatureSchema object
        from the raw dictionary data found in the JSON.
        """
        if SchemaKeys.SCHEMA_DICT not in config:
            _LOGGER.error(f"The model architecture is missing the '{SchemaKeys.SCHEMA_DICT}' key.")
            raise ValueError()
            
        schema_data = config.pop(SchemaKeys.SCHEMA_DICT)
        
        # Use shared helper to prepare arguments (handles tuple/int conversion)
        schema_kwargs = prepare_schema_from_json(schema_data)
        
        schema = FeatureSchema(**schema_kwargs)
        
        config['schema'] = schema
        return config

    def __repr__(self):
        # 1. Format hyperparameters
        hparams_str = ",\n  ".join([f"{k}={v}" for k, v in self.model_hparams.items()])
        
        # 2. Format child modules
        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append(f"  ({name}): {mod_str}")

        # 3. Combine
        main_str = f"{self.__class__.__name__}(\n  {hparams_str}\n"
        if child_lines:
            main_str += "\n".join(child_lines) + "\n"
        main_str += ")"
        return main_str

