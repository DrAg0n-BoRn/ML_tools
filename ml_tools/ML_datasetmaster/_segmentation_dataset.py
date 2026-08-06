import numpy
from typing import Union, Optional, Callable, Any
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import VisionTransformRecipeKeys, MLTaskKeys

from ._base_vision_dataset import _BaseVisionDataset


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
        self.class_map: dict[str, int] = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # Open as PIL Images safely to prevent file descriptor leaks
            with open(img_path, 'rb') as f_img:
                image = Image.open(f_img).convert("RGB")
            with open(mask_path, 'rb') as f_mask:
                mask = Image.open(f_mask).convert("L")
        except Exception as e:
            _LOGGER.error(f"Error loading sample #{idx}: {img_path.name} / {mask_path.name}. Error: {e}")
            # Fallback to the next index to prevent DataLoader collate crashes
            return self.__getitem__((idx + 1) % len(self))
            
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
        self.size = [size]
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
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
        if torch.rand(1).item() < self.p:
            image = TF.hflip(image) # type: ignore
            mask = TF.hflip(mask)  # type: ignore
        return image, mask
        
class _PairedRandomResizedCrop:
    """Applies the same random resized crop to both image and mask."""
    def __init__(self, size: int, scale: tuple[float, float]=(0.5, 1.0), ratio: tuple[float, float]=(3./4., 4./3.)):
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

class _PairedResizeAspectFill:
    """Pads an image and mask to be square, matching the longest edge."""
    def __init__(self, image_fill: int = 0, mask_fill: int = 0):
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        w, h = image.size
        if w == h:
            return image, mask

        if w > h:
            top_padding = (w - h) // 2
            bottom_padding = w - h - top_padding
            padding = (0, top_padding, 0, bottom_padding)
        else: # h > w
            left_padding = (h - w) // 2
            right_padding = h - w - left_padding
            padding = (left_padding, 0, right_padding, 0)

        padded_image = TF.pad(image, padding, fill=self.image_fill) # type: ignore
        padded_mask = TF.pad(mask, padding, fill=self.mask_fill) # type: ignore
        
        return padded_image, padded_mask # type: ignore


# --- Segmentation Dataset ---
class DragonDatasetSegmentation(_BaseVisionDataset):
    """
    Creates processed PyTorch datasets for segmentation from image and mask folders.

    This maker finds all matching image-mask pairs from two directories,
    splits them, and applies identical transformations (including augmentations)
    to both the image and its corresponding mask.
    
    Workflow:
    1. `maker = DragonDatasetSegmentation.from_folders(img_dir, mask_dir)`
    2. `maker.set_class_map({'background': 0, 'road': 1})`
    3. `maker.split_data(val_size=0.2, test_size=0.1)`
    4. `maker.configure_transforms(resize_size=256, crop_size=224, mean=[...], std=[...])`
    5. `maker.save_transform_recipe('segmentation_val_recipe.json')`
    6. `maker.save_class_map('data/')`
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    def __init__(self):
        super().__init__()
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        
        self.train_transform: Optional[Callable] = None
        self.val_transform: Optional[Callable] = None

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

        _LOGGER.info(f"Found {len(image_files)} images. Searching for matching masks in '{mask_dir}'...")
        
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
            
            # 3. Validate files can be opened before adding to the pair list
            if mask_file:
                try:
                    # Lightweight check: just open and verify, don't load into memory
                    with Image.open(img_file) as img, Image.open(mask_file) as msk:
                        img.verify()
                        msk.verify()
                    good_img_paths.append(img_file)
                    good_mask_paths.append(mask_file)
                except Exception as e:
                    _LOGGER.warning(f"Skipping corrupted pair {img_file.name}: {e}")
            else:
                _LOGGER.warning(f"No corresponding mask found for image: {img_file.name}")
        
        if not good_img_paths:
            _LOGGER.error("No matching image-mask pairs were found.")
            raise FileNotFoundError()
            
        _LOGGER.info(f"Successfully found {len(good_img_paths)} image-mask pairs.")
        maker.image_paths = good_img_paths
        maker.mask_paths = good_mask_paths
        
        return maker

    def set_class_map(self, class_map: dict[str, int]) -> 'DragonDatasetSegmentation':
        """
        Sets a map of class_name -> pixel value. This is used by the Trainer for clear evaluation reports.
        
        Propagates the class names and mapping to any datasets that have already been created (train/val/test).

        Args:
            class_map (dict[str, int]): A dictionary mapping the integer pixel
                value in a mask to its string name.
                Example: {'background': 0, 'road': 1, 'car': 2}
        """
        self.class_map = class_map
        self.classes = list(class_map.keys())
        
        # Retroactively sync datasets if split_data was already called
        if self._is_split:
            if self._train_dataset: 
                self._train_dataset.classes = self.classes # type: ignore
                self._train_dataset.class_map = self.class_map # type: ignore
            if self._val_dataset: 
                self._val_dataset.classes = self.classes # type: ignore
                self._val_dataset.class_map = self.class_map # type: ignore
            if self._test_dataset: 
                self._test_dataset.classes = self.classes # type: ignore
                self._test_dataset.class_map = self.class_map # type: ignore
        
        _LOGGER.info(f"Class map set: {class_map}")
        return self

    def split_data(self, val_size: float = 0.2, 
                   test_size: float = 0.0, 
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
            self._test_dataset.classes = self.classes
            self._test_dataset.class_map = self.class_map
            _LOGGER.info(f"Test set created with {len(self._test_dataset)} images.")
        else:
            val_imgs, val_masks = get_paths(val_test_indices)
        
        self._train_dataset = _SegmentationDataset(train_imgs, train_masks, transform=None)
        self._val_dataset = _SegmentationDataset(val_imgs, val_masks, transform=None)
        
        # Propagate class names and maps to datasets for trainer
        self._train_dataset.classes = self.classes
        self._val_dataset.classes = self.classes
        self._train_dataset.class_map = self.class_map
        self._val_dataset.class_map = self.class_map

        self._is_split = True
        _LOGGER.info(f"Data split into: \n- Training: {len(self._train_dataset)} images \n- Validation: {len(self._val_dataset)} images")
        return self

    def configure_transforms(self, 
                             resize_size: int = 256, 
                             mean: Optional[tuple[float, ...]] = (0.485, 0.456, 0.406), 
                             std: Optional[tuple[float, ...]] = (0.229, 0.224, 0.225),
                             apply_paired_square_aspect: bool = False,
                             # parameters for training transforms
                             random_horizontal_flip_probability: float = 0.5,
                             random_resize_crop_scale: tuple[float, float] = (0.5, 1.0),
                             random_resize_crop_ratio: tuple[float, float] = (3/4, 4/3),
                             ) -> 'DragonDatasetSegmentation':
        """
        Configures and applies the image and mask transformations.
        
        This method must be called AFTER data is split.

        Args:
            resize_size (int): The target size for Paired-RandomResizedCrop (training) and the size to resize the smallest edge of the image (validation/test).
            mean (tuple[float] | None): The mean values for image normalization.
            std (tuple[float] | None): The std dev values for image normalization.
            apply_paired_square_aspect (bool): If True, applies a Paired-CenterCrop(resize_size) to both validation and test pipelines to enforce square inputs. 
                Does not replace advanced preprocessing for specialized aspect ratios. Use with caution.
            random_horizontal_flip_probability (float): Probability of applying horizontal flip during training.
            random_resize_crop_scale (tuple[float, float]): Scale range for random resized crop during training.
            random_resize_crop_ratio (tuple[float, float]): Aspect ratio range for random resized crop during training.

        Returns:
            DragonDatasetSegmentation: The same instance, with transforms applied.
            
        ⚠️ WARNING: PyTorch DataLoaders require all images in a batch to have the same dimensions. 
        Since `resize_size` scales the shortest edge and maintains the aspect ratio, 
        datasets with varying aspect ratios will crash the DataLoader. 
        
        Setting `apply_paired_square_aspect=True` applies a Paired-CenterCrop(resize_size) to guarantee uniform batch 
        dimensions. However, note that cropping may trim boundary context/mask data. 
        For optimal results and specialized aspect-ratio handling, advanced preprocessing should be performed on the raw data beforehand.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        if (mean is None and std is not None) or (mean is not None and std is None):
            _LOGGER.error(f"'mean' and 'std' must be both None or both defined, but only one was provided.")
            raise ValueError()
        
        
        # --- Store components for validation recipe ---
        self._val_recipe_components: dict[str,Any] = {
            VisionTransformRecipeKeys.RESIZE_SIZE: resize_size,
            "APPLY_SQUARE_ASPECT": apply_paired_square_aspect
        }
        
        mean_list: list[float] = []
        std_list: list[float] = []
        # cast to list
        if mean is not None and std is not None:
            mean_list = list(mean)
            std_list = list(std)
        
            self._val_recipe_components.update({
                VisionTransformRecipeKeys.MEAN: mean_list,
                VisionTransformRecipeKeys.STD: std_list
            })
            self._has_mean_std = True
        
        base_val_pipeline: list[Any] = [_PairedResize(resize_size)]
        if apply_paired_square_aspect:
            base_val_pipeline.append(_PairedCenterCrop(resize_size))
        
        base_train_pipeline: list[Any] = [
            _PairedRandomResizedCrop(size=resize_size, scale=random_resize_crop_scale, ratio=random_resize_crop_ratio),
            _PairedRandomHorizontalFlip(p=random_horizontal_flip_probability)
        ]
        
        final_pipeline: list[Any] = [_PairedToTensor()]
        
        if self._has_mean_std:
            final_pipeline.append(_PairedNormalize(mean_list, std_list))

        # --- Validation/Test Pipeline (Deterministic) ---
        self.val_transform = _PairedCompose([*base_val_pipeline, *final_pipeline])
        
        # --- Training Pipeline (Augmentation) ---
        self.train_transform = _PairedCompose([*base_train_pipeline, *final_pipeline])

        # --- Apply Transforms to the Datasets ---
        self._train_dataset.transform = self.train_transform # type: ignore
        self._val_dataset.transform = self.val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.transform = self.val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Paired segmentation transforms configured and applied.")
        return self
    
    def _get_task_name(self) -> str:
        return MLTaskKeys.BINARY_SEGMENTATION if len(self.classes) == 2 else MLTaskKeys.MULTICLASS_SEGMENTATION
    
    def _build_recipe_pipeline(self) -> list[dict[str, Any]]:
        components = self._val_recipe_components
        if not components:
            return []

        pipeline = [
            {VisionTransformRecipeKeys.NAME: "Resize", "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}}
        ]
        
        if components.get("APPLY_SQUARE_ASPECT"):
            pipeline.append({
                VisionTransformRecipeKeys.NAME: "CenterCrop", 
                "kwargs": {"size": components[VisionTransformRecipeKeys.RESIZE_SIZE]}
            })

        pipeline.append({VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}})
        
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
        s += f"  Total Image-Mask Pairs: {len(self.image_paths)}\n"
        s += f"  Split: {self._is_split}\n"
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
            
        return s
