import torch
from torch.utils.data import Dataset, Subset
import numpy
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, List, Optional, Callable, Dict, Any
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import random
import json

from .ML_datasetmaster import _BaseMaker
from .path_manager import make_fullpath
from ._logger import _LOGGER
from ._script_info import _script_info
from .keys import VisionTransformRecipeKeys, ObjectDetectionKeys
from ._ML_vision_recipe import save_recipe
from .ML_vision_transformers import TRANSFORM_REGISTRY


__all__ = [
    "VisionDatasetMaker",
    "SegmentationDatasetMaker",
    "ObjectDetectionDatasetMaker"
]


# --- VisionDatasetMaker ---
class VisionDatasetMaker(_BaseMaker):
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
        super().__init__()
        self._full_dataset: Optional[ImageFolder] = None
        self.labels: Optional[List[int]] = None
        self.class_map: Optional[dict[str,int]] = None
        
        self._is_split = False
        self._are_transforms_configured = False
        self._val_recipe_components = None

    @classmethod
    def from_folder(cls, root_dir: str) -> 'VisionDatasetMaker':
        """
        Creates a maker instance from a single root directory of images.
        
        This method assumes a single directory (e.g., 'data/') that
        contains class subfolders (e.g., 'data/cat/', 'data/dog/').
        
        The dataset will be loaded in its entirety, and you MUST call
        `.split_data()` afterward to create train/validation/test sets.

        Args:
            root_dir (str): The path to the root directory containing
                            class subfolders.

        Returns:
            VisionDatasetMaker: A new instance with the full dataset loaded.
        """
        # Load with NO transform. We get PIL Images.
        full_dataset = ImageFolder(root=root_dir, transform=None)
        _LOGGER.info(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes.")
        
        maker = cls()
        maker._full_dataset = full_dataset
        maker.labels = [s[1] for s in full_dataset.samples]
        maker.class_map = full_dataset.class_to_idx
        return maker
    
    @classmethod
    def from_folders(cls, 
                     train_dir: str, 
                     val_dir: str, 
                     test_dir: Optional[str] = None) -> 'VisionDatasetMaker':
        """
        Creates a maker instance from separate, pre-split directories.
        
        This method is used when you already have 'train', 'val', and
        optionally 'test' folders, each containing class subfolders.
        It bypasses the need for `.split_data()`.

        Args:
            train_dir (str): Path to the training data directory.
            val_dir (str): Path to the validation data directory.
            test_dir (str, None): Path to the test data directory.

        Returns:
            VisionDatasetMaker: A new, pre-split instance.

        Raises:
            ValueError: If the classes found in train, val, or test directories are inconsistent.
        """
        _LOGGER.info("Loading data from separate directories.")
        # Load with NO transform. We get PIL Images.
        train_ds = ImageFolder(root=train_dir, transform=None)
        val_ds = ImageFolder(root=val_dir, transform=None)
        
        # Check for class consistency
        if train_ds.class_to_idx != val_ds.class_to_idx:
            _LOGGER.error("Train and validation directories have different or inconsistent classes.")
            raise ValueError()

        maker = cls()
        maker._train_dataset = train_ds
        maker._val_dataset = val_ds
        maker.class_map = train_ds.class_to_idx
        
        if test_dir:
            test_ds = ImageFolder(root=test_dir, transform=None)
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
        path_obj = make_fullpath(path)

        non_image_files = set()
        img_types = set()
        img_sizes = set()
        img_channels = set()
        img_counter = 0

        _LOGGER.info(f"Inspecting folder: {path_obj}...")
        # Use rglob to recursively find all files
        for filepath in path_obj.rglob('*'):
            if filepath.is_file():
                try:
                    # Using PIL to open is a more reliable check
                    with Image.open(filepath) as img:
                        img_types.add(img.format)
                        img_sizes.add(img.size)
                        img_channels.update(img.getbands())
                        img_counter += 1
                except (IOError, SyntaxError):
                    non_image_files.add(filepath.name)

        if non_image_files:
            _LOGGER.warning(f"Non-image or corrupted files found and ignored: {non_image_files}")

        report = (
            f"\n--- Inspection Report for '{path_obj.name}' ---\n"
            f"Total images found: {img_counter}\n"
            f"Image formats: {img_types or 'None'}\n"
            f"Image sizes (WxH): {img_sizes or 'None'}\n"
            f"Image channels (bands): {img_channels or 'None'}\n"
            f"--------------------------------------"
        )
        print(report)

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   stratify: bool = True, random_state: Optional[int] = None) -> 'VisionDatasetMaker':
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
            VisionDatasetMaker: The same instance, now with datasets split.
            
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

    def configure_transforms(self, resize_size: int = 256, crop_size: int = 224, 
                             mean: List[float] = [0.485, 0.456, 0.406], 
                             std: List[float] = [0.229, 0.224, 0.225],
                             pre_transforms: Optional[List[Callable]] = None,
                             extra_train_transforms: Optional[List[Callable]] = None) -> 'VisionDatasetMaker':
        """
        Configures and applies the image transformations and augmentations.
        
        This method must be called AFTER data is loaded and split.
        
        It sets up two pipelines:
        1.  **Training Pipeline:** Includes random augmentations like
            `RandomResizedCrop` and `RandomHorizontalFlip` (plus any
            `extra_train_transforms`) for online augmentation.
        2.  **Validation/Test Pipeline:** A deterministic pipeline using
            `Resize` and `CenterCrop` for consistent evaluation.
            
        Both pipelines finish with `ToTensor` and `Normalize`.

        Args:
            resize_size (int): The size to resize the smallest edge to
                               for validation/testing.
            crop_size (int): The target size (square) for the final
                             cropped image.
            mean (List[float]): The mean values for normalization (e.g., ImageNet mean).
            std (List[float]): The standard deviation values for normalization (e.g., ImageNet std).
            extra_train_transforms (List[Callable] | None): A list of additional torchvision transforms to add to the end of the training transformations.
            pre_transforms (List[Callable] | None): An list of transforms to be applied at the very beginning of the transformations for all sets.

        Returns:
            VisionDatasetMaker: The same instance, with transforms applied.
            
        Raises:
            RuntimeError: If called before data is split.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data (or using `from_folders`). Call .split_data() first if using `from_folder`.")
            raise RuntimeError()

        # --- Define Transform Pipelines ---
        # These now MUST include ToTensor and Normalize, as the ImageFolder was loaded with transform=None.
        
        # --- Store components for validation recipe ---
        self._val_recipe_components = {
            VisionTransformRecipeKeys.PRE_TRANSFORMS: pre_transforms or [],
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            VisionTransformRecipeKeys.CROP_SIZE: crop_size,
            VisionTransformRecipeKeys.MEAN: mean,
            VisionTransformRecipeKeys.STD: std
        }
        
        base_pipeline = []
        if pre_transforms:
            base_pipeline.extend(pre_transforms)

        # Base augmentations for training
        base_train_transforms = [
            transforms.RandomResizedCrop(crop_size), 
            transforms.RandomHorizontalFlip()
        ]
        if extra_train_transforms:
            base_train_transforms.extend(extra_train_transforms)
        
        # Final conversion and normalization
        final_transforms = [
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ]

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
        
        self._train_dataset = _DatasetTransformer(self._train_dataset, train_transform) # type: ignore
        self._val_dataset = _DatasetTransformer(self._val_dataset, val_transform) # type: ignore
        if self._test_dataset:
            self._test_dataset = _DatasetTransformer(self._test_dataset, val_transform) # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Image transforms configured and applied.")
        return self

    def get_datasets(self) -> Tuple[Dataset, ...]:
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
            return self._train_dataset, self._val_dataset, self._test_dataset
        return self._train_dataset, self._val_dataset

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

        recipe: Dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "classification",
            VisionTransformRecipeKeys.PIPELINE: []
        }
        
        components = self._val_recipe_components
        
        if not components:
            _LOGGER.error(f"Error getting the transformers recipe for validation set.")
            raise ValueError()
        
        # validate path
        file_path = make_fullpath(filepath, make=True, enforce="file")

        # 1. Handle pre_transforms
        for t in components[VisionTransformRecipeKeys.PRE_TRANSFORMS]:
            t_name = t.__class__.__name__
            if t_name in TRANSFORM_REGISTRY:
                recipe[VisionTransformRecipeKeys.PIPELINE].append({
                    VisionTransformRecipeKeys.NAME: t_name,
                    VisionTransformRecipeKeys.KWARGS: getattr(t, VisionTransformRecipeKeys.KWARGS, {})
                })
            else:
                _LOGGER.warning(f"Skipping unknown pre_transform '{t_name}' in recipe. Not in TRANSFORM_REGISTRY.")
                
        # 2. Add standard transforms
        recipe[VisionTransformRecipeKeys.PIPELINE].extend([
            {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}},
            {VisionTransformRecipeKeys.NAME: "CenterCrop", "kwargs": {"size": components[VisionTransformRecipeKeys.CROP_SIZE]}},
            {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}},
            {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                "mean": components[VisionTransformRecipeKeys.MEAN],
                "std": components[VisionTransformRecipeKeys.STD]
            }}
        ])
        
        # 3. Save the file
        save_recipe(recipe, file_path)
    

class _DatasetTransformer(Dataset):
    """
    Internal wrapper class to apply a specific transform pipeline to any
    dataset (e.g., a full ImageFolder or a Subset).
    """
    def __init__(self, dataset: Dataset, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.transform = transform
        
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


# --- Segmentation dataset ----
class _SegmentationDataset(Dataset):
    """
    Internal helper class to load image-mask pairs.
    
    Loads images as RGB and masks as 'L' (grayscale, 8-bit integer pixels).
    """
    def __init__(self, image_paths: List[Path], mask_paths: List[Path], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # --- Propagate 'classes' if they exist (for MLTrainer) ---
        self.classes: List[str] = []

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
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: Any, mask: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class _PairedToTensor:
    """Converts a PIL Image pair (image, mask) to Tensors."""
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use new variable names to satisfy the linter
        image_tensor = TF.to_tensor(image)
        # Convert mask to LongTensor, not float.
        # This creates a [H, W] tensor of integer class IDs.
        mask_tensor = torch.from_numpy(numpy.array(mask, dtype=numpy.int64))
        return image_tensor, mask_tensor

class _PairedNormalize:
    """Normalizes the image tensor and leaves the mask untouched."""
    def __init__(self, mean: List[float], std: List[float]):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.normalize(image)
        return image, mask

class _PairedResize:
    """Resizes an image and mask to the same size."""
    def __init__(self, size: int):
        self.size = [size, size]
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Use new variable names to avoid linter confusion
        resized_image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR) # type: ignore
        # Use NEAREST for mask to avoid interpolating class IDs (e.g., 1.5)
        resized_mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST) # type: ignore
        return resized_image, resized_mask # type: ignore
        
class _PairedCenterCrop:
    """Center-crops an image and mask to the same size."""
    def __init__(self, size: int):
        self.size = [size, size]
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        cropped_image = TF.center_crop(image, self.size) # type: ignore
        cropped_mask = TF.center_crop(mask, self.size) # type: ignore
        return cropped_image, cropped_mask # type: ignore

class _PairedRandomHorizontalFlip:
    """Applies the same random horizontal flip to both image and mask."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            flipped_image = TF.hflip(image) # type: ignore
            flipped_mask = TF.hflip(mask) # type: ignore
        return flipped_image, flipped_mask # type: ignore
        
class _PairedRandomResizedCrop:
    """Applies the same random resized crop to both image and mask."""
    def __init__(self, size: int, scale: Tuple[float, float]=(0.08, 1.0), ratio: Tuple[float, float]=(3./4., 4./3.)):
        self.size = [size, size]
        self.scale = scale
        self.ratio = ratio
        self.interpolation = TF.InterpolationMode.BILINEAR
        self.mask_interpolation = TF.InterpolationMode.NEAREST

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Get parameters for the random crop
        # Convert scale/ratio tuples to lists to satisfy the linter's type hint
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, list(self.scale), list(self.ratio)) # type: ignore
        
        # Apply the crop with the SAME parameters and use new variable names
        cropped_image = TF.resized_crop(image, i, j, h, w, self.size, self.interpolation) # type: ignore
        cropped_mask = TF.resized_crop(mask, i, j, h, w, self.size, self.mask_interpolation) # type: ignore
        
        return cropped_image, cropped_mask # type: ignore

# --- SegmentationDatasetMaker ---
class SegmentationDatasetMaker(_BaseMaker):
    """
    Creates processed PyTorch datasets for segmentation from image and mask folders.

    This maker finds all matching image-mask pairs from two directories,
    splits them, and applies identical transformations (including augmentations)
    to both the image and its corresponding mask.
    
    Workflow:
    1. `maker = SegmentationDatasetMaker.from_folders(img_dir, mask_dir)`
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
        super().__init__()
        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self.class_map: Dict[str, int] = {}
        
        self._is_split = False
        self._are_transforms_configured = False
        self.train_transform: Optional[Callable] = None
        self.val_transform: Optional[Callable] = None

    @classmethod
    def from_folders(cls, image_dir: Union[str, Path], mask_dir: Union[str, Path]) -> 'SegmentationDatasetMaker':
        """
        Creates a maker instance by loading all matching image-mask pairs
        from two corresponding directories.
        
        This method assumes that for an image `images/img_001.png`, there
        is a corresponding mask `masks/img_001.png`.
        
        Args:
            image_dir (str | Path): Path to the directory containing input images.
            mask_dir (str | Path): Path to the directory containing segmentation masks.

        Returns:
            SegmentationDatasetMaker: A new instance with all pairs loaded.
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
        found in the directory. Useful for checking masks.
        """
        VisionDatasetMaker.inspect_folder(path)

    def set_class_map(self, class_map: Dict[str, int]) -> 'SegmentationDatasetMaker':
        """
        Sets a map of pixel_value -> class_name. This is used by the MLTrainer for clear evaluation reports.

        Args:
            class_map (Dict[int, str]): A dictionary mapping the integer pixel
                value in a mask to its string name.
                Example: {'background': 0, 'road': 1, 'car': 2}
        """
        self.class_map = class_map
        _LOGGER.info(f"Class map set: {class_map}")
        return self
    
    @property
    def classes(self) -> List[str]:
        """Returns the list of class names, if set."""
        if self.class_map:
            return list(self.class_map.keys())
        return []

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   random_state: Optional[int] = 42) -> 'SegmentationDatasetMaker':
        """
        Splits the loaded image-mask pairs into train, validation, and test sets.

        Args:
            val_size (float): Proportion of the dataset to reserve for validation.
            test_size (float): Proportion of the dataset to reserve for testing.
            random_state (int | None): Seed for reproducible splits.

        Returns:
            SegmentationDatasetMaker: The same instance, now with datasets created.
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
        
        # Propagate class names to datasets for MLTrainer
        self._train_dataset.classes = self.classes # type: ignore
        self._val_dataset.classes = self.classes # type: ignore

        self._is_split = True
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, 
                             resize_size: int = 256, 
                             crop_size: int = 224, 
                             mean: List[float] = [0.485, 0.456, 0.406], 
                             std: List[float] = [0.229, 0.224, 0.225]) -> 'SegmentationDatasetMaker':
        """
        Configures and applies the image and mask transformations.
        
        This method must be called AFTER data is split.

        Args:
            resize_size (int): The size to resize the smallest edge to
                               for validation/testing.
            crop_size (int): The target size (square) for the final
                             cropped image.
            mean (List[float]): The mean values for image normalization.
            std (List[float]): The std dev values for image normalization.

        Returns:
            SegmentationDatasetMaker: The same instance, with transforms applied.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        # --- Store components for validation recipe ---
        self.val_recipe_components = {
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            VisionTransformRecipeKeys.CROP_SIZE: crop_size,
            VisionTransformRecipeKeys.MEAN: mean,
            VisionTransformRecipeKeys.STD: std
        }

        # --- Validation/Test Pipeline (Deterministic) ---
        self.val_transform = _PairedCompose([
            _PairedResize(resize_size),
            _PairedCenterCrop(crop_size),
            _PairedToTensor(),
            _PairedNormalize(mean, std)
        ])
        
        # --- Training Pipeline (Augmentation) ---
        self.train_transform = _PairedCompose([
            _PairedRandomResizedCrop(crop_size),
            _PairedRandomHorizontalFlip(p=0.5),
            _PairedToTensor(),
            _PairedNormalize(mean, std)
        ])

        # --- Apply Transforms to the Datasets ---
        self._train_dataset.transform = self.train_transform # type: ignore
        self._val_dataset.transform = self.val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.transform = self.val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Paired segmentation transforms configured and applied.")
        return self

    def get_datasets(self) -> Tuple[Dataset, ...]:
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
            return self._train_dataset, self._val_dataset, self._test_dataset
        return self._train_dataset, self._val_dataset
    
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
        recipe: Dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "segmentation",
            VisionTransformRecipeKeys.PIPELINE: [
                {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components["resize_size"]}},
                {VisionTransformRecipeKeys.NAME: "CenterCrop", "kwargs": {"size": components["crop_size"]}},
                {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}},
                {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                    "mean": components["mean"],
                    "std": components["std"]
                }}
            ]
        }
        
        # Save the file
        save_recipe(recipe, file_path)


# Object detection
def _od_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection.
    
    Takes a list of (image, target) tuples and zips them into two lists:
    (list_of_images, list_of_targets).
    This is required for models like Faster R-CNN, which accept a list
    of images of varying sizes.
    """
    return tuple(zip(*batch)) # type: ignore


class _ObjectDetectionDataset(Dataset):
    """
    Internal helper class to load image-annotation pairs.
    
    Loads an image as 'RGB' and parses its corresponding JSON annotation file
    to create the required target dictionary (boxes, labels).
    """
    def __init__(self, image_paths: List[Path], annotation_paths: List[Path], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
        
        # --- Propagate 'classes' if they exist (for MLTrainer) ---
        self.classes: List[str] = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]
        
        try:
            # Open image
            image = Image.open(img_path).convert("RGB")
            
            # Load and parse annotation
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            # Get boxes and labels from JSON
            boxes = ann_data[ObjectDetectionKeys.BOXES]  # Expected: [[x1, y1, x2, y2], ...]
            labels = ann_data[ObjectDetectionKeys.LABELS] # Expected: [1, 2, 1, ...]
            
            # Convert to tensors
            target: Dict[str, Any] = {}
            target[ObjectDetectionKeys.BOXES] = torch.as_tensor(boxes, dtype=torch.float32)
            target[ObjectDetectionKeys.LABELS] = torch.as_tensor(labels, dtype=torch.int64)
            
        except Exception as e:
            _LOGGER.error(f"Error loading sample #{idx}: {img_path.name} / {ann_path.name}. Error: {e}")
            # Return empty/dummy data
            return torch.empty(3, 224, 224), {ObjectDetectionKeys.BOXES: torch.empty((0, 4)), ObjectDetectionKeys.LABELS: torch.empty(0, dtype=torch.long)}

        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target

# Internal Paired Transform Helpers for Object Detection
class _OD_PairedCompose:
    """A 'Compose' for paired image/target_dict transforms."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class _OD_PairedToTensor:
    """Converts a PIL Image to Tensor, passes targets dict through."""
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return TF.to_tensor(image), target

class _OD_PairedNormalize:
    """Normalizes the image tensor and leaves the target dict untouched."""
    def __init__(self, mean: List[float], std: List[float]):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_normalized = self.normalize(image)
        return image_normalized, target

class _OD_PairedRandomHorizontalFlip:
    """Applies the same random horizontal flip to both image and targets['boxes']."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            w, h = image.size
            # Use new variable names to avoid linter confusion
            flipped_image = TF.hflip(image) # type: ignore
            
            # Flip boxes
            boxes = target[ObjectDetectionKeys.BOXES].clone() # [N, 4]
            
            # xmin' = w - xmax
            # xmax' = w - xmin
            boxes[:, 0] = w - target[ObjectDetectionKeys.BOXES][:, 2] # xmin'
            boxes[:, 2] = w - target[ObjectDetectionKeys.BOXES][:, 0] # xmax'
            target[ObjectDetectionKeys.BOXES] = boxes
            
            return flipped_image, target # type: ignore
            
        return image, target


class ObjectDetectionDatasetMaker(_BaseMaker):
    """
    Creates processed PyTorch datasets for object detection from image
    and JSON annotation folders.

    This maker finds all matching image-annotation pairs from two directories,
    splits them, and applies identical transformations (including augmentations)
    to both the image and its corresponding target dictionary.
    
    The `DragonFastRCNN` model expects a list of images and a list of targets,
    so this class provides a `collate_fn` to be used with a DataLoader.
    
    Workflow:
    1. `maker = ObjectDetectionDatasetMaker.from_folders(img_dir, ann_dir)`
    2. `maker.set_class_map({'background': 0, 'person': 1, 'car': 2})`
    3. `maker.split_data(val_size=0.2)`
    4. `maker.configure_transforms()`
    5. `train_ds, val_ds = maker.get_datasets()`
    6. `collate_fn = maker.collate_fn`
    7. `train_loader = DataLoader(train_ds, ..., collate_fn=collate_fn)`
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    def __init__(self):
        """
        Typically not called directly. Use the class method `from_folders()` to create an instance.
        """
        super().__init__()
        self.image_paths: List[Path] = []
        self.annotation_paths: List[Path] = []
        self.class_map: Dict[str, int] = {}
        
        self._is_split = False
        self._are_transforms_configured = False
        self.train_transform: Optional[Callable] = None
        self.val_transform: Optional[Callable] = None
        self._val_recipe_components: Optional[Dict[str, Any]] = None

    @classmethod
    def from_folders(cls, image_dir: Union[str, Path], annotation_dir: Union[str, Path]) -> 'ObjectDetectionDatasetMaker':
        """
        Creates a maker instance by loading all matching image-annotation pairs
        from two corresponding directories.
        
        This method assumes that for an image `images/img_001.png`, there
        is a corresponding annotation `annotations/img_001.json`.
        
        The JSON file must contain "boxes" and "labels" keys:
        `{"boxes": [[x1,y1,x2,y2], ...], "labels": [1, 2, ...]}`

        Args:
            image_dir (str | Path): Path to the directory containing input images.
            annotation_dir (str | Path): Path to the directory containing .json
                                         annotation files.

        Returns:
            ObjectDetectionDatasetMaker: A new instance with all pairs loaded.
        """
        maker = cls()
        img_path_obj = make_fullpath(image_dir, enforce="directory")
        ann_path_obj = make_fullpath(annotation_dir, enforce="directory")

        # Find all images
        image_files = sorted([
            p for p in img_path_obj.glob('*.*') 
            if p.suffix.lower() in cls.IMG_EXTENSIONS
        ])
        
        if not image_files:
            _LOGGER.error(f"No images with extensions {cls.IMG_EXTENSIONS} found in {image_dir}")
            raise FileNotFoundError()

        _LOGGER.info(f"Found {len(image_files)} images. Searching for matching .json annotations in {annotation_dir}...")
        
        good_img_paths = []
        good_ann_paths = []

        for img_file in image_files:
            # Find annotation with same stem + .json
            ann_file = ann_path_obj / (img_file.stem + ".json")
            
            if ann_file.exists():
                good_img_paths.append(img_file)
                good_ann_paths.append(ann_file)
            else:
                _LOGGER.warning(f"No corresponding .json annotation found for image: {img_file.name}")
        
        if not good_img_paths:
            _LOGGER.error("No matching image-annotation pairs were found.")
            raise FileNotFoundError()
            
        _LOGGER.info(f"Successfully found {len(good_img_paths)} image-annotation pairs.")
        maker.image_paths = good_img_paths
        maker.annotation_paths = good_ann_paths
        
        return maker

    @staticmethod
    def inspect_folder(path: Union[str, Path]):
        """
        Logs a report of the types, sizes, and channels of image files
        found in the directory.
        """
        VisionDatasetMaker.inspect_folder(path)

    def set_class_map(self, class_map: Dict[str, int]) -> 'ObjectDetectionDatasetMaker':
        """
        Sets a map of class_name -> pixel_value. This is used by the
        MLTrainer for clear evaluation reports.
        
        **Important:** For object detection models, 'background' MUST
        be included as class 0.
        Example: `{'background': 0, 'person': 1, 'car': 2}`

        Args:
            class_map (Dict[str, int]): A dictionary mapping the string name
                to its integer label.
        """
        if 'background' not in class_map or class_map['background'] != 0:
            _LOGGER.warning("Object detection class map should include 'background' mapped to 0.")
        
        self.class_map = class_map
        _LOGGER.info(f"Class map set: {class_map}")
        return self
    
    @property
    def classes(self) -> List[str]:
        """Returns the list of class names, if set."""
        if self.class_map:
            return list(self.class_map.keys())
        return []

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   random_state: Optional[int] = 42) -> 'ObjectDetectionDatasetMaker':
        """
        Splits the loaded image-annotation pairs into train, validation, and test sets.

        Args:
            val_size (float): Proportion of the dataset to reserve for validation.
            test_size (float): Proportion of the dataset to reserve for testing.
            random_state (int | None): Seed for reproducible splits.

        Returns:
            ObjectDetectionDatasetMaker: The same instance, now with datasets created.
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
            return [self.image_paths[i] for i in idx_list], [self.annotation_paths[i] for i in idx_list]

        train_imgs, train_anns = get_paths(train_indices)
        
        if test_size > 0:
            val_indices, test_indices = train_test_split(
                val_test_indices, test_size=(test_size / (val_size + test_size)), 
                random_state=random_state
            )
            val_imgs, val_anns = get_paths(val_indices)
            test_imgs, test_anns = get_paths(test_indices)
            
            self._test_dataset = _ObjectDetectionDataset(test_imgs, test_anns, transform=None)
            self._test_dataset.classes = self.classes # type: ignore
            _LOGGER.info(f"Test set created with {len(self._test_dataset)} images.")
        else:
            val_imgs, val_anns = get_paths(val_test_indices)
        
        self._train_dataset = _ObjectDetectionDataset(train_imgs, train_anns, transform=None)
        self._val_dataset = _ObjectDetectionDataset(val_imgs, val_anns, transform=None)
        
        # Propagate class names to datasets for MLTrainer
        self._train_dataset.classes = self.classes # type: ignore
        self._val_dataset.classes = self.classes # type: ignore

        self._is_split = True
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, 
                             mean: List[float] = [0.485, 0.456, 0.406], 
                             std: List[float] = [0.229, 0.224, 0.225]) -> 'ObjectDetectionDatasetMaker':
        """
        Configures and applies the image and target transformations.
        
        This method must be called AFTER data is split.
        
        For object detection models like Faster R-CNN, images are NOT
        resized or cropped, as the model handles variable input sizes.
        Transforms are limited to augmentation (flip), ToTensor, and Normalize.

        Args:
            mean (List[float]): The mean values for image normalization.
            std (List[float]): The std dev values for image normalization.

        Returns:
            ObjectDetectionDatasetMaker: The same instance, with transforms applied.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        # --- Store components for validation recipe ---
        self._val_recipe_components = {
            VisionTransformRecipeKeys.MEAN: mean,
            VisionTransformRecipeKeys.STD: std
        }

        # --- Validation/Test Pipeline (Deterministic) ---
        self.val_transform = _OD_PairedCompose([
            _OD_PairedToTensor(),
            _OD_PairedNormalize(mean, std)
        ])
        
        # --- Training Pipeline (Augmentation) ---
        self.train_transform = _OD_PairedCompose([
            _OD_PairedRandomHorizontalFlip(p=0.5),
            _OD_PairedToTensor(),
            _OD_PairedNormalize(mean, std)
        ])

        # --- Apply Transforms to the Datasets ---
        self._train_dataset.transform = self.train_transform # type: ignore
        self._val_dataset.transform = self.val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.transform = self.val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Paired object detection transforms configured and applied.")
        return self

    def get_datasets(self) -> Tuple[Dataset, ...]:
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
            return self._train_dataset, self._val_dataset, self._test_dataset
        return self._train_dataset, self._val_dataset
    
    @property
    def collate_fn(self) -> Callable:
        """
        Returns the collate function required by a DataLoader for this
        dataset. This function ensures that images and targets are
        batched as separate lists.
        """
        return _od_collate_fn
    
    def save_transform_recipe(self, filepath: Union[str, Path]) -> None:
        """
        Saves the validation transform pipeline as a JSON recipe file.
        
        For object detection, this recipe only includes ToTensor and
        Normalize, as resizing is handled by the model.

        Args:
            filepath (str | Path): The path to save the .json recipe file.
        """
        if not self._are_transforms_configured:
            _LOGGER.error("Transforms are not configured. Call .configure_transforms() first.")
            raise RuntimeError()
        
        components = self._val_recipe_components
        
        if not components:
            _LOGGER.error(f"Error getting the transformers recipe for validation set.")
            raise ValueError()
        
        # validate path
        file_path = make_fullpath(filepath, make=True, enforce="file")

        # Add standard transforms
        recipe: Dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "object_detection",
            VisionTransformRecipeKeys.PIPELINE: [
                {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}},
                {VisionTransformRecipeKeys.NAME: "Normalize", "kwargs": {
                    "mean": components["mean"],
                    "std": components["std"]
                }}
            ]
        }
        
        # Save the file
        save_recipe(recipe, file_path)


def info():
    _script_info(__all__)
