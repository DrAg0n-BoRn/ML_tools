from abc import ABC, abstractmethod
from typing import Union, Optional, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os

from ..ML_vision_transformers._core_transforms import _save_recipe, _load_recipe
from ..ML_vision_utilities._inspect_folder import inspect_folder

from ..IO_tools import save_json
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys, VisionDatasetManifestKeys


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
        
        self.transform_recipe: Optional[dict[str, Any]] = None
        self._config_kwargs: dict[str, Any] = {}
        
        # --- Manifest Tracking ---
        self._creation_mode: Optional[str] = None
        self._source_paths: dict[str, Path] = {}
        self._split_config: dict[str, Any] = {}
        self._callable_requirements: dict[str, bool] = {}

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
    
    @property
    def image_channels(self) -> int:
        """
        Dynamically extracts the number of image channels by inspecting the first sample 
        of the dataset. Requires transforms to be configured.
        """
        if not self._are_transforms_configured:
            _LOGGER.error("Cannot determine image channels before transforms are configured.")
            raise RuntimeError()
        
        # Prioritize train_dataset, fallback to val or test if train is somehow empty
        dataset = self._train_dataset or self._val_dataset or self._test_dataset
        
        if not dataset or len(dataset) == 0: # type: ignore
            _LOGGER.error("No data available to inspect. Datasets are empty.")
            raise RuntimeError()
            
        try:
            # All pipelines return (image_tensor, target)
            sample = dataset[0]
            image_tensor = sample[0]
            
            if not isinstance(image_tensor, torch.Tensor):
                _LOGGER.error(f"Expected a torch.Tensor for the image, but got {type(image_tensor)}. Ensure ToTensor is in the pipeline.")
                raise TypeError()
            
            # PyTorch image tensors are strictly formatted as [C, H, W]
            return int(image_tensor.shape[0])
            
        except Exception as e:
            _LOGGER.error(f"Failed to dynamically extract image channels: {e}")
            raise

    def save_class_map(self, save_dir: Union[str, Path], filename: str = "Class_to_Index") -> None:
        """
        Saves the class to index mapping as a JSON file in the specified directory. Raises a ValueError if the class_map is empty.
        
        Args:
            save_dir (Union[str, Path]): The directory where the class map JSON file will be saved. If the directory does not exist, it will be created.
            filename (str): The name of the JSON file to save.
        """
        if not self.class_map:
            _LOGGER.error("Class to index mapping is empty.")
            raise ValueError()
        
        sanitized_filename = sanitize_filename(filename)
        
        save_json(data=self.class_map, 
                  directory=save_dir, 
                  filename=sanitized_filename, 
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
            _LOGGER.error("Transforms are not configured.")
            raise RuntimeError()
        
        if not self.transform_recipe:
            _LOGGER.error("Error getting the transform recipe. It has not been populated.")
            raise ValueError()
        
        if isinstance(filepath, str):
            if not filepath.endswith(".json"):
                filepath += ".json"
            file_path = Path(filepath)
        else:
            file_path = filepath.with_suffix(".json")
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        _save_recipe(self.transform_recipe, file_path)
    
    def load_and_configure_transforms(self, filepath: Union[str, Path], **override_kwargs) -> None:
        """
        Loads the transformation recipe from a JSON file, extracts the configuration parameters,
        and dynamically calls configure_transforms().
        
        Args:
            filepath (Union[str, Path]): The filepath from where the transformation recipe JSON file will be loaded.
            **override_kwargs: Any configuration arguments to override or re-inject.
                - ⚠️ Any un-serializable callables like `pre_transforms` and `extra_train_transforms` in DragonDatasetImageClassification must be re-injected here, as they cannot be saved in the JSON recipe.
        """
        if not self._is_split:
            _LOGGER.error("Cannot apply transforms because datasets have not been split. Call .split_data() first.")
            raise RuntimeError()

        recipe = _load_recipe(filepath)
        
        # Safely extract configuration or default to empty dict
        config_kwargs: dict[str, Any] = recipe.get(VisionTransformRecipeKeys.CONFIGURATION, dict())
        
        # Inject dynamic overrides (like Callables) or overwrite loaded values
        config_kwargs.update(override_kwargs)
        
        if not config_kwargs:
            # methods include default values, so if the recipe is missing the configuration, we can still call configure_transforms with defaults
            _LOGGER.warning(f"No configuration parameters found in the recipe. Ensure that the recipe is valid and contains the '{VisionTransformRecipeKeys.CONFIGURATION}' key.")
        
        # Automatically configure the pipeline
        try:
            self.configure_transforms(**config_kwargs)
        except Exception as e:
            _LOGGER.error(f"Error occurred while configuring transforms: {e}")
            raise
    
    def inspect_folder(self, directory: Union[str, Path], save_dir_log: Optional[Union[str, Path]] = None) -> None:
        """
        Logs a report of the types, sizes, and channels of image files
        found in the directory and its subdirectories.

        This is a utility method to help diagnose potential dataset
        issues (e.g., mixed image modes, corrupted files) before loading.

        Args:
            directory (str, Path): The directory path to inspect.
            save_dir_log (str, Path, optional): The directory where the log file will be saved. If not provided, the log will be saved at the same level as the inspected folder.
        """
        inspect_folder(directory=directory, save_dir_log=save_dir_log)
    
    def _populate_transform_recipe(self) -> None:
        """
        Populates the transform_recipe dictionary using the subclass implementations.
        """
        self.transform_recipe = {
            VisionTransformRecipeKeys.TASK: self._get_task_name(),
            VisionTransformRecipeKeys.PIPELINE: self._build_recipe_pipeline(),
            VisionTransformRecipeKeys.CONFIGURATION: self._config_kwargs
        }
    
    def save_dataset_manifest(self, save_dir: Union[str, Path], filename: str = "dataset_manifest.json") -> None:
        """
        Saves a JSON manifest containing all necessary metadata to perfectly recreate the dataset.
        
        Must be called after splitting, configuring transforms, and setting the class map.
        
        Args:
            save_dir (Union[str, Path]): Directory to save the manifest. 
            filename (str): Name of the JSON manifest file.
        """
        if not (self._is_split and self._are_transforms_configured and self.class_map):
            _LOGGER.error("Cannot save manifest. Dataset must be fully finalized (split, transforms configured, and class map set).")
            raise RuntimeError()
        
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        # Convert source paths to be relative to the manifest's save directory
        relative_paths = {}
        for key, path in self._source_paths.items():
            try:
                rel_path = os.path.relpath(path.resolve(), save_path)
                # Force forward slashes for cross-platform (GitHub/Windows/Linux/macOS) compatibility
                relative_paths[key] = Path(rel_path).as_posix()
            except ValueError:
                # Fallback to absolute if on different drives (e.g., Windows edge case)
                # Also force posix format for consistency
                _LOGGER.warning(f"Could not make path '{path}' relative to '{save_path}'. Using absolute path instead.")
                relative_paths[key] = Path(path.resolve()).as_posix()
                
        manifest_data = {
            VisionDatasetManifestKeys.DATASET_CLASS: self.__class__.__name__,
            VisionDatasetManifestKeys.CREATION_MODE: self._creation_mode,
            VisionDatasetManifestKeys.PATHS: relative_paths,
            VisionDatasetManifestKeys.SPLIT_CONFIG: self._split_config,
            VisionDatasetManifestKeys.CLASS_MAP: self.class_map,
            VisionDatasetManifestKeys.TRANSFORM_RECIPE: self.transform_recipe,
            VisionDatasetManifestKeys.CALLABLE_REQUIREMENTS: self._callable_requirements
        }
        
        sanitized_filename = sanitize_filename(filename)
        
        manifest_filepath = save_path / sanitized_filename
        
        save_json(data=manifest_data, 
                  directory=save_path, 
                  filename=sanitized_filename, 
                  verbose=False)
        
        _LOGGER.info(f"Dataset manifest successfully saved to '{manifest_filepath}'.")    
    

    @abstractmethod
    def _get_task_name(self) -> str:
        pass

    @abstractmethod
    def _build_recipe_pipeline(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def configure_transforms(self, **kwargs) -> None:
        pass
