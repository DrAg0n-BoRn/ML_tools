import inspect
from typing import Union, Optional, Callable, Any
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ..ML_vision_transformers._core_transforms import TRANSFORM_REGISTRY

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys, MLTaskKeys

from ._base_vision_dataset import _BaseVisionDataset


_LOGGER = get_logger("Image Classification Dataset")


__all__ = [
    "DragonDatasetImageClassification"
]


class _DatasetTransformer(Dataset):
    """
    Internal wrapper class to apply a specific transform pipeline to any
    dataset (e.g., a full ImageFolder or a Subset).
    """
    def __init__(self, dataset: Dataset, transform: Optional[transforms.Compose] = None, class_map: dict[str,int]=dict()):
        self.dataset = dataset
        self.transform = transform
        self.class_map = class_map
        self.classes = list(class_map.keys())
        
        # --- Propagate attributes for inspection ---
        # For ImageFolder
        if hasattr(dataset, 'class_to_idx'):
            self.class_to_idx = getattr(dataset, 'class_to_idx')
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


class DragonDatasetImageClassification(_BaseVisionDataset):
    """
    Creates processed PyTorch datasets for computer vision classification tasks from an
    image folder directory.
    
    Supports two modes:
    1. `from_folder()`: Loads from one directory and splits into train/val/test.
    2. `from_folders()`: Loads from pre-split train/val/test directories.
    
    Uses online augmentations per epoch.
    
    Workflow:
    ```
    1. maker = DragonDatasetImageClassification.from_folder("data/", ...) # or from_folders(train_dir, val_dir, test_dir)
    3. maker.configure_transforms(resize_size=256, crop_size=224, mean=[...], std=[...], extra_train_transforms=[...])
    4. maker.save_transform_recipe('val_transform_recipe.json')
    5. maker.save_class_map("data/") # Saves class_to_index mapping as JSON
    ```
    """
    def __init__(self):
        super().__init__()
        self._full_dataset: Optional[ImageFolder] = None
        self.labels: Optional[list[int]] = None

    @classmethod
    def from_folder(cls, 
                    root_dir: Union[str,Path],
                    val_size: float = 0.2, 
                    test_size: float = 0.0, 
                    stratify: bool = True, 
                    random_state: Optional[int] = None) -> 'DragonDatasetImageClassification':
        """
        Creates a maker instance from a single root directory of images.
        
        This method assumes a single directory (e.g., 'data/') that
        contains class subfolders (e.g., 'data/cat/', 'data/dog/').
        
        The dataset will be loaded and split to create train, validation, and optional test sets.

        Args:
            root_dir (str | Path): The path to the root directory containing class subfolders.
            val_size (float): Proportion of the dataset to reserve for validation (e.g., 0.2 for 20%).
            test_size (float): Proportion of the dataset to reserve for testing.
            stratify (bool): If True, splits are performed in a stratified fashion, preserving the class distribution.
            random_state (int | None): Seed for the random number generator for reproducible splits.

        Returns:
            Instance: A new instance with the dataset loaded and split.
        """
        root_path = make_fullpath(root_dir, enforce="directory")
        # Load with NO transform. We get PIL Images.
        full_dataset = ImageFolder(root=root_path, transform=None)
        _LOGGER.info(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes.")
        
        maker = cls()
        maker._full_dataset = full_dataset
        maker.labels = [s[1] for s in full_dataset.samples]
        maker.class_map = full_dataset.class_to_idx
        maker.classes = list(maker.class_map.keys())
        
        # Automatically split the dataset
        maker._split_data(val_size=val_size, test_size=test_size, stratify=stratify, random_state=random_state)
        
        return maker
    
    @classmethod
    def from_folders(cls, 
                     train_dir: Union[str,Path], 
                     val_dir: Union[str,Path], 
                     test_dir: Optional[Union[str,Path]] = None) -> 'DragonDatasetImageClassification':
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
        maker.classes = list(maker.class_map.keys())
        
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

    def _split_data(self, 
                   val_size: float = 0.2, 
                   test_size: float = 0.0, 
                   stratify: bool = True, 
                   random_state: Optional[int] = None) -> None:
        """
        PRIVATE METHOD: No need to call this, both `from_folder()` and `from_folders()` handle splitting automatically.
        
        Splits the dataset into train, validation, and optional test sets.
        
        This method MUST be called if `from_folder()` was used. It has no effect if `from_folders()` was used.

        Args:
            val_size (float): Proportion of the dataset to reserve for
                              validation (e.g., 0.2 for 20%).
            test_size (float): Proportion of the dataset to reserve for
                               testing.
            stratify (bool): If True, splits are performed in a stratified
                             fashion, preserving the class distribution.
            random_state (int | None): Seed for the random number generator for reproducible splits.

        Raises:
            ValueError: If `val_size` and `test_size` sum to 1.0 or more.
        """
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return

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

    def configure_transforms(self, 
                             resize_size: int = 256, 
                             mean: Optional[tuple[float, ...]] = (0.485, 0.456, 0.406), 
                             std: Optional[tuple[float, ...]] = (0.229, 0.224, 0.225),
                             pre_transforms: Optional[list[Callable]] = None,
                             extra_train_transforms: Optional[list[Callable]] = None,
                             ## params for train transforms
                             random_horizontal_flip_probability: float = 0.5,
                             random_resize_crop_scale: tuple[float, float] = (0.08, 1.0),
                             random_resize_crop_ratio: tuple[float, float] = (3/4, 4/3),
                             random_rotation_degrees: float = 90.0
                             ) -> None:
        """
        Configures and applies the image transformations and augmentations.
        
        This method must be called AFTER data is loaded and split.
        
        It sets up two pipelines:
        1.  **Training Pipeline:** Includes random transforms for online augmentation:
            - `RandomResizedCrop(size=resize_size, scale=random_resize_crop_scale, ratio=random_resize_crop_ratio)`
            - `RandomHorizontalFlip(p=random_horizontal_flip_probability)`
            - `RandomRotation(degrees=random_rotation_degrees)` 
            - (Any `extra_train_transforms`)
            
        2.  **Validation/Test Pipeline:** A deterministic pipeline using `Resize` for consistent evaluation.
            
        Both pipelines finish with `ToTensor` and `Normalize`.

        Args:
            resize_size (int): The target size for `RandomResizedCrop` (training) and the size to resize the smallest edge of the image (validation/test).
            mean (tuple[float, ...] | None): The mean values for normalization (e.g., ImageNet mean).
            std (tuple[float, ...] | None): The standard deviation values for normalization (e.g., ImageNet std).
            extra_train_transforms (list[Callable] | None): A list of additional torchvision transforms to add to the end of the training transformations.
            pre_transforms (list[Callable] | None): An list of transforms to be applied at the very beginning of the transformations for all sets.
            random_horizontal_flip_probability (float): Probability of applying horizontal flip during training.
            random_resize_crop_scale (tuple[float, float]): Scale range for random resized crop during training.
            random_resize_crop_ratio (tuple[float, float]): Aspect ratio range for random resized crop during training.
            random_rotation_degrees (float): Maximum degrees for random rotation during training.
            
        ⚠️ WARNING: PyTorch DataLoaders require all images in a batch to have the same dimensions. 
        Since `Resize` with a single integer scales the shortest edge and maintains the aspect ratio, 
        datasets with varying aspect ratios will crash the DataLoader. If your images are of varying aspect ratios, 
        you MUST include `ResizeAspectFill` or `LetterboxResize` in the `pre_transforms` list, or handle aspect ratio preprocessing before creating the dataset.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data (or using `from_folders`). Call .split_data() first if using `from_folder`.")
            raise RuntimeError()
        
        if (mean is None and std is not None) or (mean is not None and std is None):
            _LOGGER.error(f"'mean' and 'std' must be both None or both defined, but only one was provided.")
            raise ValueError()
        
        # --- Store components for validation recipe ---
        self._val_recipe_components = {
            VisionTransformRecipeKeys.PRE_TRANSFORMS: pre_transforms or [],
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            # VisionTransformRecipeKeys.CROP_SIZE: crop_size,
        }
        
        if mean is not None and std is not None:
            self._val_recipe_components.update({
                VisionTransformRecipeKeys.MEAN: list(mean),
                VisionTransformRecipeKeys.STD: list(std)
            })
            self._has_mean_std = True
        
        base_pipeline = []
        if pre_transforms:
            base_pipeline.extend(pre_transforms)

        # Base augmentations for training
        base_train_transforms = [
            transforms.RandomResizedCrop(size=resize_size, scale=random_resize_crop_scale, ratio=random_resize_crop_ratio), # Random crops over the image replace Resize
            transforms.RandomHorizontalFlip(p=random_horizontal_flip_probability),
            transforms.RandomRotation(degrees=random_rotation_degrees)
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
            # transforms.CenterCrop(crop_size), 
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
    
    def _get_task_name(self) -> str:
        """
        Returns the task name for the transform recipe.
        """
        return MLTaskKeys.BINARY_IMAGE_CLASSIFICATION if len(self.class_map) == 2 else MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
    
    def _build_recipe_pipeline(self) -> list[dict[str, Any]]:
        components = self._val_recipe_components
        if not components:
            return []

        pipeline = []
        
        for t in components[VisionTransformRecipeKeys.PRE_TRANSFORMS]:
            t_name = t.__class__.__name__
            t_class = t.__class__
            kwargs = {}
            
            if t_name in TRANSFORM_REGISTRY:
                _LOGGER.debug(f"Found '{t_name}' in TRANSFORM_REGISTRY.")
                kwargs = getattr(t, VisionTransformRecipeKeys.KWARGS, {})
            else:
                _LOGGER.debug(f"'{t_name}' not in registry. Attempting introspection...")
                try:
                    sig = inspect.signature(t_class.__init__)
                    
                    for param in sig.parameters.values():
                        if param.name == 'self':
                            continue
                        
                        attr_name_public = param.name
                        attr_name_private = '_' + param.name
                        
                        attr_to_get = ""
                        
                        if hasattr(t, attr_name_public):
                            attr_to_get = attr_name_public
                        elif hasattr(t, attr_name_private):
                            attr_to_get = attr_name_private
                        else:
                            continue 
                        
                        kwargs[param.name] = getattr(t, attr_to_get)
                            
                    _LOGGER.debug(f"Introspection for '{t_name}' found kwargs: {kwargs}")

                except (ValueError, TypeError):
                    _LOGGER.warning(f"Could not introspect parameters for '{t_name}'. If this transform has parameters, they will not be saved.")
                    kwargs = {}

            pipeline.append({
                VisionTransformRecipeKeys.NAME: t_name,
                VisionTransformRecipeKeys.KWARGS: kwargs
            })
                
        pipeline.extend([
            {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}},
            # {VisionTransformRecipeKeys.NAME: "CenterCrop", "kwargs": {"size": components[VisionTransformRecipeKeys.CROP_SIZE]}},
            {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}}
        ])
        
        if self._has_mean_std:
            pipeline.append(
                {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                "mean": components[VisionTransformRecipeKeys.MEAN],
                "std": components[VisionTransformRecipeKeys.STD]
                }}
            )
        return pipeline
    
    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}>:\n"
        s += f"  Transforms Configured: {self._are_transforms_configured}\n"
        
        if self.class_map:
            classes = list(self.class_map.keys())
            if len(classes) > 4:
                class_str = f"[{', '.join(f'{repr(c)}' for c in classes[:4])}, ...]"
            else:
                class_str = str(classes)
            s += f"  Classes ({len(self.class_map)}): {class_str}\n"

        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0 # type: ignore
            val_len = len(self._val_dataset) if self._val_dataset else 0 # type: ignore
            test_len = len(self._test_dataset) if self._test_dataset else 0 # type: ignore
            s += f"  Datasets (Train|Val|Test): {train_len} | {val_len} | {test_len}\n"
        elif self._full_dataset:
            s += f"  Full Dataset Size: {len(self._full_dataset)} images\n"

        return s
