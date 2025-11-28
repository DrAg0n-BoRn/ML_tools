from torch import nn
from typing import Any, Dict, Union
import json
from pathlib import Path
from abc import ABC, abstractmethod

from ._path_manager import make_fullpath
from ._keys import PytorchModelArchitectureKeys
from ._schema import FeatureSchema
from ._logger import get_logger


_LOGGER = get_logger("DragonModel")


##################################
# Base class for loading and saving advanced models
##################################
class _ArchitectureBuilder(nn.Module, ABC):
    """
    Base class for Dragon models that unifies architecture handling.
    
    Implements:
    - JSON serialization and JSON deserialization with automatic FeatureSchema reconstruction.
    - Standardized string representation (__repr__) showing hyperparameters.
    """
    def __init__(self):
        super().__init__()
        # Placeholder for hyperparameters, to be populated by child classes
        self.model_hparams: Dict[str, Any] = {}
        
    # abstract method that must be implemented by children
    @abstractmethod
    def get_architecture_config(self) -> Dict[str, Any]:
        "To be implemented by children"
        pass

    def save(self: nn.Module, directory: Union[str, Path], verbose: bool = True): # type: ignore
        """Saves the model's architecture to a JSON file."""
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
    def load(cls, file_or_dir: Union[str, Path], verbose: bool = True) -> nn.Module:
        """
        Loads a model architecture from a JSON file.
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

        # --- Schema Reconstruction Logic (Unified) ---
        if 'schema_dict' not in config:
            raise ValueError("Missing 'schema_dict' in config.")
            
        schema_data = config.pop('schema_dict')
        
        raw_index_map = schema_data['categorical_index_map']
        if raw_index_map is not None:
            # JSON keys are strings, convert back to int
            rehydrated_index_map = {int(k): v for k, v in raw_index_map.items()}
        else:
            rehydrated_index_map = None

        schema = FeatureSchema(
            feature_names=tuple(schema_data['feature_names']),
            continuous_feature_names=tuple(schema_data['continuous_feature_names']),
            categorical_feature_names=tuple(schema_data['categorical_feature_names']),
            categorical_index_map=rehydrated_index_map,
            categorical_mappings=schema_data['categorical_mappings']
        )
        
        config['schema'] = schema
        # --- End Reconstruction ---

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model

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
