from abc import ABC, abstractmethod
from typing import Union, Optional, Any
from pathlib import Path
from torch.utils.data import Dataset

from ..ML_vision_transformers._core_transforms import _save_recipe
from ..ML_vision_transformers._inspect_folder import inspect_folder

from ..IO_tools import save_json
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys


_LOGGER = get_logger("Vision Dataset")


class _BaseVisionDataset(ABC):
    def __init__(self):
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        
        self.class_map: dict[str, int] = {}
        self.classes: list[str] = []
        
        self._is_split: bool = False
        self._are_transforms_configured: bool = False
        self._has_mean_std: bool = False
        self._val_recipe_components: Optional[dict[str, Any]] = None

    @property
    def train_dataset(self) -> Dataset:
        """
        Returns the training dataset. Raises a RuntimeError if the training dataset has not been created.
        """
        if self._train_dataset is None: 
            _LOGGER.error("Train Dataset not created.")
            raise RuntimeError()
        return self._train_dataset
    
    @property
    def validation_dataset(self) -> Dataset:
        """
        Returns the validation dataset. Raises a RuntimeError if the validation dataset has not been created.
        """
        if self._val_dataset is None: 
            _LOGGER.error("Validation Dataset not yet created.")
            raise RuntimeError()
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        """
        Returns the test dataset. Raises a RuntimeError if the test dataset has not been created.
        """
        if self._test_dataset is None: 
            _LOGGER.error("Test Dataset not yet created.")
            raise RuntimeError()
        return self._test_dataset

    def images_per_dataset(self) -> str:
        """
        Returns a string representation of the number of images in each dataset (train, validation, test if applicable).
        
        If the datasets have not been split, it returns a message indicating that no datasets were found.
        """
        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0 # type: ignore
            val_len = len(self._val_dataset) if self._val_dataset else 0 # type: ignore
            test_len = len(self._test_dataset) if self._test_dataset else 0 # type: ignore
            return f"Train | Validation | Test: {train_len} | {val_len} | {test_len} images"
        
        if hasattr(self, '_full_dataset') and getattr(self, '_full_dataset') is not None:
            return f"Full Dataset: {len(getattr(self, '_full_dataset'))} images"

        return "No datasets found"

    def save_class_map(self, save_dir: Union[str, Path]) -> None:
        """
        Saves the class to index mapping as a JSON file in the specified directory. Raises a ValueError if the class_map is empty.
        
        Args:
            save_dir (Union[str, Path]): The directory where the class map JSON file will be saved. If the directory does not exist, it will be created.
        """
        if not self.class_map:
            _LOGGER.error("Class to index mapping is empty.")
            raise ValueError()
        
        save_json(data=self.class_map, 
                  directory=save_dir, 
                  filename="Class_to_Index", 
                  verbose=False)
        
        _LOGGER.info(f"Class to index mapping saved to '{save_dir}'.")

    def save_transform_recipe(self, filepath: Union[str, Path]) -> None:
        """
        Saves the transformation recipe for the validation set to a JSON file at the specified filepath. 
        
        Raises a RuntimeError if transforms are not configured.
        
        Raises a ValueError if the validation recipe components are not available.
        
        Args:
            filepath (Union[str, Path]): The filepath where the transformation recipe JSON file will be saved.
        """
        if not self._are_transforms_configured:
            _LOGGER.error("Transforms are not configured. Call .configure_transforms() first.")
            raise RuntimeError()
        
        if not self._val_recipe_components:
            _LOGGER.error("Error getting the transformers recipe for validation set.")
            raise ValueError()
        
        recipe: dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: self._get_task_name(),
            VisionTransformRecipeKeys.PIPELINE: self._build_recipe_pipeline()
        }
        if isinstance(filepath, str):
            if not filepath.endswith(".json"):
                filepath += ".json"
            file_path = Path(filepath)
        else:
            file_path = filepath.with_suffix(".json")
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        _save_recipe(recipe, file_path)
        
    def inspect_folder(self, path: Union[str, Path]):
        """
        Logs a report of the types, sizes, and channels of image files
        found in the directory and its subdirectories.

        This is a utility method to help diagnose potential dataset
        issues (e.g., mixed image modes, corrupted files) before loading.

        Args:
            path (str, Path): The directory path to inspect.
        """
        inspect_folder(path)

    @abstractmethod
    def _get_task_name(self) -> str:
        pass

    @abstractmethod
    def _build_recipe_pipeline(self) -> list[dict[str, Any]]:
        pass
