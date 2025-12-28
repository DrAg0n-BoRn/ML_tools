from typing import NamedTuple, Optional, Union, Any
from pathlib import Path
import json

from ..IO_tools import save_list_strings

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import SchemaKeys, DatasetKeys, PytorchModelArchitectureKeys

from .._core._schema_load_ops import prepare_schema_from_json


_LOGGER = get_logger("FeatureSchema")


__all__ = [
    "FeatureSchema",
]


class FeatureSchema(NamedTuple):
    """Holds the final, definitive schema for the model pipeline."""
    
    # The final, ordered list of all feature names
    feature_names: tuple[str, ...]
    
    # List of all continuous feature names
    continuous_feature_names: tuple[str, ...]
    
    # List of all categorical feature names
    categorical_feature_names: tuple[str, ...]
    
    # Map of {column_index: cardinality} for categorical features
    categorical_index_map: Optional[dict[int, int]]
    
    # Map string-to-int category values (e.g., {'color': {'red': 0, 'blue': 1}})
    categorical_mappings: Optional[dict[str, dict[str, int]]]
    
    def to_json(self, directory: Union[str, Path], verbose: bool = True) -> None:
        """
        Saves the schema as 'FeatureSchema.json' to the provided directory. 
        
        Handles conversion of Tuple->List and IntKeys->StrKeys automatically.
        """
        # validate path
        dir_path = make_fullpath(directory, make=True, enforce="directory")
        file_path = dir_path / SchemaKeys.SCHEMA_FILENAME
        
        try:
            # Convert named tuple to dict
            data = self._asdict()
            
            # Write to disk
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            if verbose:
                _LOGGER.info(f"{self} saved to '{dir_path.name}/{SchemaKeys.SCHEMA_FILENAME}'")
                
        except (IOError, TypeError) as e:
            _LOGGER.error(f"Failed to save FeatureSchema to JSON: {e}")
            raise e
        
    @classmethod
    def from_json(cls, directory: Union[str, Path], verbose: bool = True) -> 'FeatureSchema':
        """
        Loads a 'FeatureSchema.json' from the provided directory.
        
        Restores Tuples from Lists and Integer Keys from Strings.
        """
        # validate directory
        dir_path = make_fullpath(directory, enforce="directory")
        file_path = dir_path / SchemaKeys.SCHEMA_FILENAME
        
        if not file_path.exists():
            _LOGGER.error(f"FeatureSchema file not found at '{directory}'")
            raise FileNotFoundError()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data: dict[str, Any] = json.load(f)
            
            # Logic delegated to shared helper
            schema_kwargs = prepare_schema_from_json(data)
            schema = cls(**schema_kwargs)

            if verbose:
                _LOGGER.info(f"{schema} loaded from '{dir_path.name}/{SchemaKeys.SCHEMA_FILENAME}'")

            return schema

        except (IOError, ValueError, KeyError) as e:
            _LOGGER.error(f"Failed to load FeatureSchema from '{dir_path}': {e}")
            raise e
    
    @classmethod
    def from_model_architecture(cls, file_or_dir: Union[str, Path], verbose: bool = True) -> 'FeatureSchema':
        """
        Extracts and loads the FeatureSchema embedded within a model's 'architecture.json' file.

        Args:
            file_or_dir: Path to the JSON file or the directory containing 'architecture.json'.
            verbose: If True, prints a confirmation message upon loading.
        
        Raises:
            KeyError: If the architecture file does not contain a FeatureSchema configuration.
        """
        user_path = make_fullpath(file_or_dir)
        
        # 1. Resolve Path
        if user_path.is_dir():
            json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
            target_path = make_fullpath(user_path / json_filename, enforce="file")
        elif user_path.is_file():
            target_path = user_path
        else:
            _LOGGER.error(f"Invalid path: '{file_or_dir}'")
            raise IOError()

        # 2. Load Architecture JSON
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                arch_data: dict[str, Any] = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            _LOGGER.error(f"Failed to load architecture file from '{target_path}': {e}")
            raise e

        # 3. Validate and Extract Schema Dict
        config = arch_data.get(PytorchModelArchitectureKeys.CONFIG, {})
        
        if SchemaKeys.SCHEMA_DICT not in config:
            error_msg = f"The model architecture at '{target_path.name}' does not contain a '{SchemaKeys.SCHEMA_DICT}' key. This model might not use a FeatureSchema."
            _LOGGER.error(error_msg)
            raise KeyError()

        data = config[SchemaKeys.SCHEMA_DICT]

        # 4. Reconstruct Schema (Restore Types)
        try:
            schema_kwargs = prepare_schema_from_json(data)
            schema = cls(**schema_kwargs)

            if verbose:
                _LOGGER.info(f"{schema} extracted from architecture '{target_path.parent.name}/{target_path.name}'")

            return schema

        except (ValueError, KeyError) as e:
            _LOGGER.error(f"Failed to parse FeatureSchema from architecture file: {e}")
            raise e

    def _save_helper(self, artifact: tuple[str, ...], directory: Union[str,Path], filename: str, verbose: bool):
        to_save = list(artifact)
        
        # empty check
        if not to_save:
            _LOGGER.warning(f"Skipping save for '{filename}': The feature list is empty.")
            return
        
        save_list_strings(list_strings=to_save,
                          directory=directory,
                          filename=filename,
                          verbose=verbose)

    def save_all_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves all feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.feature_names,
                          directory=directory,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=verbose)
        
    def save_continuous_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves continuous feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.continuous_feature_names,
                          directory=directory,
                          filename=DatasetKeys.CONTINUOUS_NAMES,
                          verbose=verbose)
    
    def save_categorical_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves categorical feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.categorical_feature_names,
                          directory=directory,
                          filename=DatasetKeys.CATEGORICAL_NAMES,
                          verbose=verbose)
        
    def save_artifacts(self, directory: Union[str,Path]):
        """
        Saves feature names, categorical feature names, continuous feature names to separate text files.
        """
        self.save_all_features(directory=directory, verbose=True)
        self.save_continuous_features(directory=directory, verbose=True)
        self.save_categorical_features(directory=directory, verbose=True)
        
    def __repr__(self) -> str:
        """Returns a concise representation of the schema's contents."""
        total = len(self.feature_names)
        cont = len(self.continuous_feature_names)
        cat = len(self.categorical_feature_names)
        index_map = self.categorical_index_map is not None
        cat_map = self.categorical_mappings is not None
        return (
            f"FeatureSchema(total={total}, continuous={cont}, categorical={cat}, index_map={index_map}, categorical_map={cat_map})"
        )

