from typing import NamedTuple, Tuple, Optional, Dict, Union, Any
from pathlib import Path
import json

from ._custom_logger import save_list_strings
from ._keys import DatasetKeys
from ._logger import get_logger
from ._path_manager import make_fullpath
from ._script_info import _script_info


_LOGGER = get_logger("FeatureSchema")


__all__ = [
    "FeatureSchema"
]


_SCHEMA_FILENAME = "FeatureSchema.json"


class FeatureSchema(NamedTuple):
    """Holds the final, definitive schema for the model pipeline."""
    
    # The final, ordered list of all feature names
    feature_names: Tuple[str, ...]
    
    # List of all continuous feature names
    continuous_feature_names: Tuple[str, ...]
    
    # List of all categorical feature names
    categorical_feature_names: Tuple[str, ...]
    
    # Map of {column_index: cardinality} for categorical features
    categorical_index_map: Optional[Dict[int, int]]
    
    # Map string-to-int category values (e.g., {'color': {'red': 0, 'blue': 1}})
    categorical_mappings: Optional[Dict[str, Dict[str, int]]]
    
    def to_json(self, directory: Union[str, Path], verbose: bool = True) -> None:
        """
        Saves the schema as 'FeatureSchema.json' to the provided directory. 
        
        Handles conversion of Tuple->List and IntKeys->StrKeys automatically.
        """
        # validate path
        dir_path = make_fullpath(directory, enforce="directory")
        file_path = dir_path / _SCHEMA_FILENAME
        
        try:
            # Convert named tuple to dict
            data = self._asdict()
            
            # Write to disk
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            if verbose:
                _LOGGER.info(f"FeatureSchema saved to '{dir_path.name}/{_SCHEMA_FILENAME}'")
                
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
        file_path = dir_path / _SCHEMA_FILENAME
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)
            
            # 1. Restore Tuples (JSON loads them as lists)
            feature_names = tuple(data.get("feature_names", []))
            cont_names = tuple(data.get("continuous_feature_names", []))
            cat_names = tuple(data.get("categorical_feature_names", []))

            # 2. Restore Integer Keys for categorical_index_map
            raw_map = data.get("categorical_index_map")
            cat_index_map: Optional[Dict[int, int]] = None
            if raw_map is not None:
                cat_index_map = {int(k): v for k, v in raw_map.items()}

            # 3. Mappings (keys are strings, no conversion needed)
            cat_mappings = data.get("categorical_mappings")

            schema = cls(
                feature_names=feature_names,
                continuous_feature_names=cont_names,
                categorical_feature_names=cat_names,
                categorical_index_map=cat_index_map,
                categorical_mappings=cat_mappings
            )

            if verbose:
                _LOGGER.info(f"FeatureSchema loaded from '{dir_path.name}'")

            return schema

        except (IOError, ValueError, KeyError) as e:
            _LOGGER.error(f"Failed to load FeatureSchema from '{dir_path}': {e}")
            raise e

    def _save_helper(self, artifact: Tuple[str, ...], directory: Union[str,Path], filename: str, verbose: bool):
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


def info():
    _script_info(__all__)
