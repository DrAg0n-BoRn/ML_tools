from typing import NamedTuple, Tuple, Optional, Dict, Union
from pathlib import Path

from .custom_logger import save_list_strings
from ._keys import DatasetKeys
from ._logger import _LOGGER


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
            f"<FeatureSchema(total={total}, continuous={cont}, categorical={cat}, index_map={index_map}, categorical_map={cat_map})>"
        )
