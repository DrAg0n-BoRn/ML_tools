import inspect
from typing import Union, Optional, Callable, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ..ML_vision_transformers._core_transforms import TRANSFORM_REGISTRY, _save_recipe
from ..ML_vision_transformers._inspect_folder import inspect_folder
from ..IO_tools import save_json

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys


_LOGGER = get_logger("Vision Dataset")


__all__ = [
    "DragonDatasetVision"
]


class DragonDatasetVision:
    """
    Creates processed PyTorch datasets for computer vision tasks from an
    image folder directory.
    
    Supports two modes:
    1. `from_folder()`: Loads from one directory and splits into train/val/test.
    2. `from_folders()`: Loads from pre-split train/val/test directories.
    
    Uses online augmentations per epoch (image augmentation without creating new files).
    """
    def __init__(self):
        """
        Typically not called directly. Use the class methods `from_folder()` or `from_folders()` to create an instance.
        """
        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None
        self._full_dataset: Optional[ImageFolder] = None
        self.labels: Optional[list[int]] = None
        self.class_map: dict[str,int] = dict()
        
        self._is_split = False
        self._are_transforms_configured = False
        self._val_recipe_components = None
        self._has_mean_std: bool = False

    @classmethod
    def from_folder(cls, root_dir: Union[str,Path]) -> 'DragonDatasetVision':
        """
        Creates a maker instance from a single root directory of images.
        
        This method assumes a single directory (e.g., 'data/') that
        contains class subfolders (e.g., 'data/cat/', 'data/dog/').
        
        The dataset will be loaded in its entirety, and you MUST call
        `.split_data()` afterward to create train/validation/test sets.

        Args:
            root_dir (str | Path): The path to the root directory containing class subfolders.

        Returns:
            Instance: A new instance with the full dataset loaded.
        """
        root_path = make_fullpath(root_dir, enforce="directory")
        # Load with NO transform. We get PIL Images.
        full_dataset = ImageFolder(root=root_path, transform=None)
        _LOGGER.info(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes.")
        
        maker = cls()
        maker._full_dataset = full_dataset
        maker.labels = [s[1] for s in full_dataset.samples]
        maker.class_map = full_dataset.class_to_idx
        return maker
    
    @classmethod
    def from_folders(cls, 
                     train_dir: Union[str,Path], 
                     val_dir: Union[str,Path], 
                     test_dir: Optional[Union[str,Path]] = None) -> 'DragonDatasetVision':
        """
        Creates a maker instance from separate, pre-split directories.
        
        This method is used when you already have 'train', 'val', and
        optionally 'test' folders, each containing class subfolders.
        It bypasses the need for `.split_data()`.

        Args:
            train_dir (str | Path): Path to the training data directory.
            val_dir (str | Path): Path to the validation data directory.
            test_dir (str | Path | None): Path to the test data directory.

        Returns:
            Instance: A new, pre-split instance.

        Raises:
            ValueError: If the classes found in train, val, or test directories are inconsistent.
        """
        train_path = make_fullpath(train_dir, enforce="directory")
        val_path = make_fullpath(val_dir, enforce="directory")
        
        _LOGGER.info("Loading data from separate directories.")
        # Load with NO transform. We get PIL Images.
        train_ds = ImageFolder(root=train_path, transform=None)
        val_ds = ImageFolder(root=val_path, transform=None)
        
        # Check for class consistency
        if train_ds.class_to_idx != val_ds.class_to_idx:
            _LOGGER.error("Train and validation directories have different or inconsistent classes.")
            raise ValueError()

        maker = cls()
        maker._train_dataset = train_ds
        maker._val_dataset = val_ds
        maker.class_map = train_ds.class_to_idx
        
        if test_dir:
            test_path = make_fullpath(test_dir, enforce="directory")
            test_ds = ImageFolder(root=test_path, transform=None)
            if train_ds.class_to_idx != test_ds.class_to_idx:
                _LOGGER.error("Train and test directories have different or inconsistent classes.")
                raise ValueError()
            maker._test_dataset = test_ds
            _LOGGER.info(f"Loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test images.")
        else:
            _LOGGER.info(f"Loaded: {len(train_ds)} train, {len(val_ds)} val images.")

        maker._is_split = True # Mark as "split" since data is pre-split
        return maker
        
    @staticmethod
    def inspect_folder(path: Union[str, Path]):
        """
        Logs a report of the types, sizes, and channels of image files
        found in the directory and its subdirectories.
        
        This is a utility method to help diagnose potential dataset
        issues (e.g., mixed image modes, corrupted files) before loading.

        Args:
            path (str, Path): The directory path to inspect.
        """
        inspect_folder(path)

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   stratify: bool = True, random_state: Optional[int] = None) -> 'DragonDatasetVision':
        """
        Splits the dataset into train, validation, and optional test sets.
        
        This method MUST be called if you used `from_folder()`. It has no effect if you used `from_folders()`.

        Args:
            val_size (float): Proportion of the dataset to reserve for
                              validation (e.g., 0.2 for 20%).
            test_size (float): Proportion of the dataset to reserve for
                               testing.
            stratify (bool): If True, splits are performed in a stratified
                             fashion, preserving the class distribution.
            random_state (int | None): Seed for the random number generator for reproducible splits.

        Returns:
            Self: The same instance, now with datasets split.
            
        Raises:
            ValueError: If `val_size` and `test_size` sum to 1.0 or more.
        """
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return self

        if val_size + test_size >= 1.0:
            _LOGGER.error("The sum of val_size and test_size must be less than 1.")
            raise ValueError()
        
        if not self._full_dataset:
            _LOGGER.error("There is no dataset to split.")
            raise ValueError()
        
        indices = list(range(len(self._full_dataset)))
        labels_for_split = self.labels if stratify else None

        train_indices, val_test_indices = train_test_split(
            indices, test_size=(val_size + test_size), random_state=random_state, stratify=labels_for_split
        )
        
        if not self.labels:
            _LOGGER.error("Error when getting full dataset labels.")
            raise ValueError()

        if test_size > 0:
            val_test_labels = [self.labels[i] for i in val_test_indices]
            stratify_val_test = val_test_labels if stratify else None
            val_indices, test_indices = train_test_split(
                val_test_indices, test_size=(test_size / (val_size + test_size)), 
                random_state=random_state, stratify=stratify_val_test
            )
            self._test_dataset = Subset(self._full_dataset, test_indices)
            _LOGGER.info(f"Test set created with {len(self._test_dataset)} images.")
        else:
            val_indices = val_test_indices
        
        self._train_dataset = Subset(self._full_dataset, train_indices)
        self._val_dataset = Subset(self._full_dataset, val_indices)
        self._is_split = True
        
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, 
                             resize_size: int = 256, 
                             crop_size: Optional[int] = 224, 
                             mean: Optional[list[float]] = [0.485, 0.456, 0.406], 
                             std: Optional[list[float]] = [0.229, 0.224, 0.225],
                             pre_transforms: Optional[list[Callable]] = None,
                             extra_train_transforms: Optional[list[Callable]] = None) -> 'DragonDatasetVision':
        """
        Configures and applies the image transformations and augmentations.
        
        This method must be called AFTER data is loaded and split.
        
        It sets up two pipelines:
        1.  **Training Pipeline:** Includes random transforms for online augmentation:
            - `Resize(resize_size)`
            - `RandomResizedCrop(crop_size)`
            - `RandomHorizontalFlip(0.5)`
            - `RandomRotation(90)` 
            - (Any `extra_train_transforms`)
            
        2.  **Validation/Test Pipeline:** A deterministic pipeline using `Resize` and `CenterCrop` for consistent evaluation.
            
        Both pipelines finish with `ToTensor` and `Normalize`.

        Args:
            resize_size (int): The size to resize the smallest edge of the image.
            crop_size (int): The target size (square) for the final cropped image. If None, then it will be the same value as `resize_size`, to avoid losing information from the image borders.
            mean (List[float] | None): The mean values for normalization (e.g., ImageNet mean).
            std (List[float] | None): The standard deviation values for normalization (e.g., ImageNet std).
            extra_train_transforms (List[Callable] | None): A list of additional torchvision transforms to add to the end of the training transformations.
            pre_transforms (List[Callable] | None): An list of transforms to be applied at the very beginning of the transformations for all sets.

        Returns:
            Self: The same instance, with transforms applied.
            
        Raises:
            RuntimeError: If called before data is split.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data (or using `from_folders`). Call .split_data() first if using `from_folder`.")
            raise RuntimeError()
        
        if (mean is None and std is not None) or (mean is not None and std is None):
            _LOGGER.error(f"'mean' and 'std' must be both None or both defined, but only one was provided.")
            raise ValueError()

        # --- Define Transform Pipelines ---
        if crop_size is None:
            crop_size = resize_size
        
        # --- Store components for validation recipe ---
        self._val_recipe_components = {
            VisionTransformRecipeKeys.PRE_TRANSFORMS: pre_transforms or [],
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            VisionTransformRecipeKeys.CROP_SIZE: crop_size,
        }
        
        if mean is not None and std is not None:
            self._val_recipe_components.update({
                VisionTransformRecipeKeys.MEAN: mean,
                VisionTransformRecipeKeys.STD: std
            })
            self._has_mean_std = True
        
        base_pipeline = []
        if pre_transforms:
            base_pipeline.extend(pre_transforms)

        # Base augmentations for training
        base_train_transforms = [
            transforms.Resize(resize_size), # Scale down
            transforms.RandomResizedCrop(size=crop_size), # Random crops over the image
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90)
        ]
        if extra_train_transforms:
            base_train_transforms.extend(extra_train_transforms)
        
        # Final conversion and normalization
        final_transforms: list[Callable] = [
            transforms.ToTensor()
        ]
        
        if self._has_mean_std:
            final_transforms.append(transforms.Normalize(mean=mean, std=std))

        # Build the val/test pipeline
        val_transform_list = [
            *base_pipeline, # Apply pre_transforms first
            transforms.Resize(resize_size), 
            transforms.CenterCrop(crop_size), 
            *final_transforms
        ]
        
        # Build the train pipeline
        train_transform_list = [
            *base_pipeline, # Apply pre_transforms first
            *base_train_transforms, 
            *final_transforms
        ]
        
        val_transform = transforms.Compose(val_transform_list)
        train_transform = transforms.Compose(train_transform_list)

        # --- Apply Transforms using the Wrapper ---
        # This correctly assigns the transform regardless of whether the dataset is a Subset (from_folder) or an ImageFolder (from_folders).
        
        self._train_dataset = _DatasetTransformer(self._train_dataset, train_transform, self.class_map) # type: ignore
        self._val_dataset = _DatasetTransformer(self._val_dataset, val_transform, self.class_map) # type: ignore
        if self._test_dataset:
            self._test_dataset = _DatasetTransformer(self._test_dataset, val_transform, self.class_map) # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Image transforms configured and applied.")
        return self

    def get_datasets(self) -> tuple[Dataset, ...]:
        """
        Returns the final train, validation, and optional test datasets.
        
        This is the final step, used to retrieve the datasets for use in
        a `MLTrainer` or `DataLoader`.

        Returns:
            (Tuple[Dataset, ...]): A tuple containing the (train, val)
                                 or (train, val, test) datasets.
                                 
        Raises:
            RuntimeError: If called before data is split.
            UserWarning: If called before transforms are configured.
        """
        if not self._is_split:
            _LOGGER.error("Data has not been split. Call .split_data() first.")
            raise RuntimeError()
        if not self._are_transforms_configured:
            _LOGGER.warning("Transforms have not been configured.")

        if self._test_dataset:
            return self._train_dataset, self._val_dataset, self._test_dataset # type: ignore
        return self._train_dataset, self._val_dataset # type: ignore

    def save_transform_recipe(self, filepath: Union[str, Path]) -> None:
        """
        Saves the validation transform pipeline as a JSON recipe file.
        
        This recipe can be loaded by the PyTorchVisionInferenceHandler
        to ensure identical preprocessing.

        Args:
            filepath (str | Path): The path to save the .json recipe file.
        """
        if not self._are_transforms_configured:
            _LOGGER.error("Transforms are not configured. Call .configure_transforms() first.")
            raise RuntimeError()

        recipe: dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "classification",
            VisionTransformRecipeKeys.PIPELINE: []
        }
        
        components = self._val_recipe_components
        
        if not components:
            _LOGGER.error(f"Error getting the transformers recipe for validation set.")
            raise ValueError()
        
        # validate path
        file_path = make_fullpath(filepath, make=True, enforce="file")

        # Handle pre_transforms
        for t in components[VisionTransformRecipeKeys.PRE_TRANSFORMS]:
            t_name = t.__class__.__name__
            t_class = t.__class__
            kwargs = {}
            
            # 1. Check custom registry first
            if t_name in TRANSFORM_REGISTRY:
                _LOGGER.debug(f"Found '{t_name}' in TRANSFORM_REGISTRY.")
                kwargs = getattr(t, VisionTransformRecipeKeys.KWARGS, {})

            # 2. Else, try to introspect for standard torchvision transforms
            else:
                _LOGGER.debug(f"'{t_name}' not in registry. Attempting introspection...")
                try:
                    # Get the __init__ signature of the transform's class
                    sig = inspect.signature(t_class.__init__)
                    
                    # Iterate over its __init__ parameters (e.g., 'num_output_channels')
                    for param in sig.parameters.values():
                        if param.name == 'self':
                            continue
                        
                        # Check if the *instance* 't' has that parameter as an attribute
                        attr_name_public = param.name
                        attr_name_private = '_' + param.name
                        
                        attr_to_get = ""
                        
                        if hasattr(t, attr_name_public):
                            attr_to_get = attr_name_public
                        elif hasattr(t, attr_name_private):
                            attr_to_get = attr_name_private
                        else:
                            # Parameter in __init__ has no matching attribute
                            continue 
                        
                        # Store the value under the __init__ parameter's name
                        kwargs[param.name] = getattr(t, attr_to_get)
                            
                    _LOGGER.debug(f"Introspection for '{t_name}' found kwargs: {kwargs}")

                except (ValueError, TypeError):
                    # Fails on some built-ins or C-implemented __init__
                    _LOGGER.warning(f"Could not introspect parameters for '{t_name}'. If this transform has parameters, they will not be saved.")
                    kwargs = {}

            # 3. Add to pipeline
            recipe[VisionTransformRecipeKeys.PIPELINE].append({
                VisionTransformRecipeKeys.NAME: t_name,
                VisionTransformRecipeKeys.KWARGS: kwargs
            })
                
        # 2. Add standard transforms
        recipe[VisionTransformRecipeKeys.PIPELINE].extend([
            {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}},
            {VisionTransformRecipeKeys.NAME: "CenterCrop", "kwargs": {"size": components[VisionTransformRecipeKeys.CROP_SIZE]}},
            {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}}
        ])
        
        if self._has_mean_std:
            recipe[VisionTransformRecipeKeys.PIPELINE].append(
                {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                "mean": components[VisionTransformRecipeKeys.MEAN],
                "std": components[VisionTransformRecipeKeys.STD]
                }}
            )
        
        # 3. Save the file
        _save_recipe(recipe, file_path)
        
    def save_class_map(self, save_dir: Union[str,Path]) -> dict[str,int]:
        """
        Saves the class to index mapping {str: int} to a directory.
        """
        if not self.class_map:
            _LOGGER.error(f"Class to index mapping is empty.")
            raise ValueError()
        
        save_json(data=self.class_map,
                  directory=save_dir,
                  filename="Class_to_Index",
                  verbose=False)
        
        _LOGGER.info(f"Class to index mapping saved to {save_dir}.")
        
        return self.class_map
    
    def images_per_dataset(self) -> str:
        """
        Get the number of images per dataset as a string.
        """
        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0
            val_len = len(self._val_dataset) if self._val_dataset else 0
            test_len = len(self._test_dataset) if self._test_dataset else 0
            return f"Train | Validation | Test: {train_len} | {val_len} | {test_len} images"
        elif self._full_dataset:
            return f"Full Dataset: {len(self._full_dataset)} images"
        else:
            _LOGGER.warning("No datasets found.")
            return "No datasets found"
    
    @property
    def train_dataset(self) -> Dataset:
        if self._train_dataset is None: 
            _LOGGER.error("Train Dataset not created.")
            raise RuntimeError()
        return self._train_dataset
    
    @property
    def validation_dataset(self) -> Dataset:
        if self._val_dataset is None: 
            _LOGGER.error("Validation Dataset not yet created.")
            raise RuntimeError()
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        if self._test_dataset is None: 
            _LOGGER.error("Test Dataset not yet created.")
            raise RuntimeError()
        return self._test_dataset
    
    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}>:\n"
        s += f"  Split: {self._is_split}\n"
        s += f"  Transforms Configured: {self._are_transforms_configured}\n"
        
        if self.class_map:
            s += f"  Classes: {len(self.class_map)}\n"

        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0
            val_len = len(self._val_dataset) if self._val_dataset else 0
            test_len = len(self._test_dataset) if self._test_dataset else 0
            s += f"  Datasets (Train|Val|Test): {train_len} | {val_len} | {test_len}\n"
        elif self._full_dataset:
            s += f"  Full Dataset Size: {len(self._full_dataset)} images\n"
            
        return s
    

class _DatasetTransformer(Dataset):
    """
    Internal wrapper class to apply a specific transform pipeline to any
    dataset (e.g., a full ImageFolder or a Subset).
    """
    def __init__(self, dataset: Dataset, transform: Optional[transforms.Compose] = None, class_map: dict[str,int]=dict()):
        self.dataset = dataset
        self.transform = transform
        self.class_map = class_map
        
        # --- Propagate attributes for inspection ---
        # For ImageFolder
        if hasattr(dataset, 'class_to_idx'):
            self.class_to_idx = getattr(dataset, 'class_to_idx')
        if hasattr(dataset, 'classes'):
            self.classes = getattr(dataset, 'classes')
        # For Subset
        if hasattr(dataset, 'indices'):
            self.indices = getattr(dataset, 'indices')
        if hasattr(dataset, 'dataset'):
            # This allows access to the *original* full dataset
            self.original_dataset = getattr(dataset, 'dataset')

    def __getitem__(self, index):
        # Get the original data (e.g., PIL Image, label)
        x, y = self.dataset[index] 
        
        # Apply the specific transform for this dataset
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.dataset) # type: ignore
