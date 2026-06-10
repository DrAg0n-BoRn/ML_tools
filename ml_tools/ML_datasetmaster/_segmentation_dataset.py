import random
import numpy
from typing import Union, Optional, Callable, Any
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from ..ML_vision_transformers._core_transforms import _save_recipe
from ..ML_vision_transformers._inspect_folder import inspect_folder

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys


_LOGGER = get_logger("Segmentation Dataset")


__all__ = ["DragonDatasetSegmentation"]


# --- Segmentation dataset ----
class _SegmentationDataset(Dataset):
    """
    Internal helper class to load image-mask pairs.
    
    Loads images as RGB and masks as 'L' (grayscale, 8-bit integer pixels).
    """
    def __init__(self, image_paths: list[Path], mask_paths: list[Path], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # --- Propagate 'classes' if they exist for trainer ---
        self.classes: list[str] = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # Open as PIL Images. Masks should be 'L'
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            _LOGGER.error(f"Error loading sample #{idx}: {img_path.name} / {mask_path.name}. Error: {e}")
            # Return empty tensors
            return torch.empty(3, 224, 224), torch.empty(224, 224, dtype=torch.long)
            
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask


# Internal Paired Transform Helpers
class _PairedCompose:
    """A 'Compose' for paired image/mask transforms."""
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image: Any, mask: Any) -> tuple[Any, Any]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class _PairedToTensor:
    """Converts a PIL Image pair (image, mask) to Tensors."""
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        # Use new variable names to satisfy the linter
        image_tensor = TF.to_tensor(image)
        # Convert mask to LongTensor, not float.
        # This creates a [H, W] tensor of integer class IDs.
        mask_tensor = torch.from_numpy(numpy.array(mask, dtype=numpy.int64))
        return image_tensor, mask_tensor

class _PairedNormalize:
    """Normalizes the image tensor and leaves the mask untouched."""
    def __init__(self, mean: list[float], std: list[float]):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.normalize(image)
        return image, mask

class _PairedResize:
    """Resizes an image and mask to the same size."""
    def __init__(self, size: int):
        self.size = [size, size]
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        # Use new variable names to avoid linter confusion
        resized_image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR) # type: ignore
        # Use NEAREST for mask to avoid interpolating class IDs (e.g., 1.5)
        resized_mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST) # type: ignore
        return resized_image, resized_mask # type: ignore
        
class _PairedCenterCrop:
    """Center-crops an image and mask to the same size."""
    def __init__(self, size: int):
        self.size = [size, size]
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        cropped_image = TF.center_crop(image, self.size) # type: ignore
        cropped_mask = TF.center_crop(mask, self.size) # type: ignore
        return cropped_image, cropped_mask # type: ignore

class _PairedRandomHorizontalFlip:
    """Applies the same random horizontal flip to both image and mask."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            flipped_image = TF.hflip(image) # type: ignore
            flipped_mask = TF.hflip(mask) # type: ignore
        return flipped_image, flipped_mask # type: ignore
        
class _PairedRandomResizedCrop:
    """Applies the same random resized crop to both image and mask."""
    def __init__(self, size: int, scale: tuple[float, float]=(0.08, 1.0), ratio: tuple[float, float]=(3./4., 4./3.)):
        self.size = [size, size]
        self.scale = scale
        self.ratio = ratio
        self.interpolation = TF.InterpolationMode.BILINEAR
        self.mask_interpolation = TF.InterpolationMode.NEAREST

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        # Get parameters for the random crop
        # Convert scale/ratio tuples to lists to satisfy the linter's type hint
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, list(self.scale), list(self.ratio)) # type: ignore
        
        # Apply the crop with the SAME parameters and use new variable names
        cropped_image = TF.resized_crop(image, i, j, h, w, self.size, self.interpolation) # type: ignore
        cropped_mask = TF.resized_crop(mask, i, j, h, w, self.size, self.mask_interpolation) # type: ignore
        
        return cropped_image, cropped_mask # type: ignore


# --- Segmentation Dataset ---
class DragonDatasetSegmentation:
    """
    Creates processed PyTorch datasets for segmentation from image and mask folders.

    This maker finds all matching image-mask pairs from two directories,
    splits them, and applies identical transformations (including augmentations)
    to both the image and its corresponding mask.
    
    Workflow:
    1. `maker = DragonDatasetSegmentation.from_folders(img_dir, mask_dir)`
    2. `maker.set_class_map({'background': 0, 'road': 1})`
    3. `maker.split_data(val_size=0.2)`
    4. `maker.configure_transforms(crop_size=256)`
    5. `train_ds, val_ds = maker.get_datasets()`
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    def __init__(self):
        """
        Typically not called directly. Use the class method `from_folders()` to create an instance.
        """
        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        self.class_map: dict[str, int] = {}
        
        self._is_split = False
        self._are_transforms_configured = False
        self.train_transform: Optional[Callable] = None
        self.val_transform: Optional[Callable] = None
        self._has_mean_std: bool = False

    @classmethod
    def from_folders(cls, image_dir: Union[str, Path], mask_dir: Union[str, Path]) -> 'DragonDatasetSegmentation':
        """
        Creates a maker instance by loading all matching image-mask pairs
        from two corresponding directories.
        
        This method assumes that for an image `images/img_001.png`, there
        is a corresponding mask `masks/img_001.png`.
        
        Args:
            image_dir (str | Path): Path to the directory containing input images.
            mask_dir (str | Path): Path to the directory containing segmentation masks.

        Returns:
            DragonDatasetSegmentation: A new instance with all pairs loaded.
        """
        maker = cls()
        img_path_obj = make_fullpath(image_dir, enforce="directory")
        msk_path_obj = make_fullpath(mask_dir, enforce="directory")

        # Find all images
        image_files = sorted([
            p for p in img_path_obj.glob('*.*') 
            if p.suffix.lower() in cls.IMG_EXTENSIONS
        ])
        
        if not image_files:
            _LOGGER.error(f"No images with extensions {cls.IMG_EXTENSIONS} found in {image_dir}")
            raise FileNotFoundError()

        _LOGGER.info(f"Found {len(image_files)} images. Searching for matching masks in {mask_dir}...")
        
        good_img_paths = []
        good_mask_paths = []

        for img_file in image_files:
            mask_file = None
            
            # 1. Try to find mask with the exact same name
            mask_file_primary = msk_path_obj / img_file.name
            if mask_file_primary.exists():
                mask_file = mask_file_primary
            
            # 2. If not, try to find mask with same stem + common mask extension
            if mask_file is None:
                for ext in cls.IMG_EXTENSIONS: # Masks are often .png
                    mask_file_secondary = msk_path_obj / (img_file.stem + ext)
                    if mask_file_secondary.exists():
                        mask_file = mask_file_secondary
                        break
            
            # 3. If a match is found, add the pair
            if mask_file:
                good_img_paths.append(img_file)
                good_mask_paths.append(mask_file)
            else:
                _LOGGER.warning(f"No corresponding mask found for image: {img_file.name}")
        
        if not good_img_paths:
            _LOGGER.error("No matching image-mask pairs were found.")
            raise FileNotFoundError()
            
        _LOGGER.info(f"Successfully found {len(good_img_paths)} image-mask pairs.")
        maker.image_paths = good_img_paths
        maker.mask_paths = good_mask_paths
        
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

    def set_class_map(self, class_map: dict[str, int]) -> 'DragonDatasetSegmentation':
        """
        Sets a map of class_name -> pixel value. This is used by the Trainer for clear evaluation reports.

        Args:
            class_map (dict[str, int]): A dictionary mapping the integer pixel
                value in a mask to its string name.
                Example: {'background': 0, 'road': 1, 'car': 2}
        """
        self.class_map = class_map
        _LOGGER.info(f"Class map set: {class_map}")
        return self
    
    @property
    def classes(self) -> list[str]:
        """Returns the list of class names, if set."""
        if self.class_map:
            return list(self.class_map.keys())
        return []

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   random_state: Optional[int] = 42) -> 'DragonDatasetSegmentation':
        """
        Splits the loaded image-mask pairs into train, validation, and test sets.

        Args:
            val_size (float): Proportion of the dataset to reserve for validation.
            test_size (float): Proportion of the dataset to reserve for testing.
            random_state (int | None): Seed for reproducible splits.

        Returns:
            DragonDatasetSegmentation: The same instance, now with datasets created.
        """
        if self._is_split:
            _LOGGER.warning("Data has already been split.")
            return self

        if val_size + test_size >= 1.0:
            _LOGGER.error("The sum of val_size and test_size must be less than 1.")
            raise ValueError()
        
        if not self.image_paths:
            _LOGGER.error("There is no data to split. Use .from_folders() first.")
            raise RuntimeError()
        
        indices = list(range(len(self.image_paths)))

        # Split indices
        train_indices, val_test_indices = train_test_split(
            indices, test_size=(val_size + test_size), random_state=random_state
        )
        
        # Helper to get paths from indices
        def get_paths(idx_list):
            return [self.image_paths[i] for i in idx_list], [self.mask_paths[i] for i in idx_list]

        train_imgs, train_masks = get_paths(train_indices)
        
        if test_size > 0:
            val_indices, test_indices = train_test_split(
                val_test_indices, test_size=(test_size / (val_size + test_size)), 
                random_state=random_state
            )
            val_imgs, val_masks = get_paths(val_indices)
            test_imgs, test_masks = get_paths(test_indices)
            
            self._test_dataset = _SegmentationDataset(test_imgs, test_masks, transform=None)
            self._test_dataset.classes = self.classes # type: ignore
            _LOGGER.info(f"Test set created with {len(self._test_dataset)} images.")
        else:
            val_imgs, val_masks = get_paths(val_test_indices)
        
        self._train_dataset = _SegmentationDataset(train_imgs, train_masks, transform=None)
        self._val_dataset = _SegmentationDataset(val_imgs, val_masks, transform=None)
        
        # Propagate class names to datasets for trainer
        self._train_dataset.classes = self.classes # type: ignore
        self._val_dataset.classes = self.classes # type: ignore

        self._is_split = True
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, 
                             resize_size: int = 256, 
                             crop_size: Optional[int] = 224, 
                             mean: Optional[list[float]] = [0.485, 0.456, 0.406], 
                             std: Optional[list[float]] = [0.229, 0.224, 0.225]) -> 'DragonDatasetSegmentation':
        """
        Configures and applies the image and mask transformations.
        
        This method must be called AFTER data is split.

        Args:
            resize_size (int): The size to resize the smallest edge to
                               for validation/testing.
            crop_size (int | None): The target size (square) for the final cropped image.
            mean (List[float] | None): The mean values for image normalization.
            std (List[float] | None): The std dev values for image normalization.

        Returns:
            DragonDatasetSegmentation: The same instance, with transforms applied.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        if (mean is None and std is not None) or (mean is not None and std is None):
            _LOGGER.error(f"'mean' and 'std' must be both None or both defined, but only one was provided.")
            raise ValueError()
        
        if crop_size is None:
            crop_size = resize_size
        
        # --- Store components for validation recipe ---
        self.val_recipe_components: dict[str,Any] = {
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            VisionTransformRecipeKeys.CROP_SIZE: crop_size,
        }
    
        if mean is not None and std is not None:
            self.val_recipe_components.update({
                VisionTransformRecipeKeys.MEAN: mean,
                VisionTransformRecipeKeys.STD: std
            })
            self._has_mean_std = True

        # --- Validation/Test Pipeline (Deterministic) ---
        if self._has_mean_std:
            self.val_transform = _PairedCompose([
                _PairedResize(resize_size),
                _PairedCenterCrop(crop_size),
                _PairedToTensor(),
                _PairedNormalize(mean, std) # type: ignore
            ])
            # --- Training Pipeline (Augmentation) ---
            self.train_transform = _PairedCompose([
                _PairedRandomResizedCrop(crop_size),
                _PairedRandomHorizontalFlip(p=0.5),
                _PairedToTensor(),
                _PairedNormalize(mean, std) # type: ignore
            ])
        else:
            self.val_transform = _PairedCompose([
                _PairedResize(resize_size),
                _PairedCenterCrop(crop_size),
                _PairedToTensor()
            ])
            # --- Training Pipeline (Augmentation) ---
            self.train_transform = _PairedCompose([
                _PairedRandomResizedCrop(crop_size),
                _PairedRandomHorizontalFlip(p=0.5),
                _PairedToTensor()
            ])

        # --- Apply Transforms to the Datasets ---
        self._train_dataset.transform = self.train_transform # type: ignore
        self._val_dataset.transform = self.val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.transform = self.val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Paired segmentation transforms configured and applied.")
        return self

    def get_datasets(self) -> tuple[Dataset, ...]:
        """
        Returns the final train, validation, and optional test datasets.
        
        Raises:
            RuntimeError: If called before data is split.
            RuntimeError: If called before transforms are configured.
        """
        if not self._is_split:
            _LOGGER.error("Data has not been split. Call .split_data() first.")
            raise RuntimeError()
        if not self._are_transforms_configured:
            _LOGGER.error("Transforms have not been configured. Call .configure_transforms() first.")
            raise RuntimeError()

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
        
        components = self.val_recipe_components
        
        if not components:
            _LOGGER.error(f"Error getting the transformers recipe for validation set.")
            raise ValueError()
        
        # validate path
        file_path = make_fullpath(filepath, make=True, enforce="file")
        
        # Add standard transforms
        recipe: dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "segmentation",
            VisionTransformRecipeKeys.PIPELINE: [
                {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}},
                {VisionTransformRecipeKeys.NAME: "CenterCrop", "kwargs": {"size": components[VisionTransformRecipeKeys.CROP_SIZE]}},
                {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}}
            ]
        }
        
        if self._has_mean_std:
            recipe[VisionTransformRecipeKeys.PIPELINE].append(
                {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                    "mean": components[VisionTransformRecipeKeys.MEAN],
                    "std": components[VisionTransformRecipeKeys.STD]
                }}
            )
        
        # Save the file
        _save_recipe(recipe, file_path)
        
    def images_per_dataset(self) -> str:
        """
        Get the number of images per dataset as a string.
        """
        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0
            val_len = len(self._val_dataset) if self._val_dataset else 0
            test_len = len(self._test_dataset) if self._test_dataset else 0
            return f"Train | Validation | Test: {train_len} | {val_len} | {test_len} images"
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
        s += f"  Total Image-Mask Pairs: {len(self.image_paths)}\n"
        s += f"  Split: {self._is_split}\n"
        s += f"  Transforms Configured: {self._are_transforms_configured}\n"
        
        if self.class_map:
            s += f"  Classes: {list(self.class_map.keys())}\n"

        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0
            val_len = len(self._val_dataset) if self._val_dataset else 0
            test_len = len(self._test_dataset) if self._test_dataset else 0
            s += f"  Datasets (Train|Val|Test): {train_len} | {val_len} | {test_len}\n"
            
        return s
