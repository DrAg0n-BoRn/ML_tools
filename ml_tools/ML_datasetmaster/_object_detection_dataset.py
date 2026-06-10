import random
import json
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
from ..keys._keys import VisionTransformRecipeKeys, ObjectDetectionKeys


_LOGGER = get_logger("Object Detection Dataset")


__all__ = ["DragonDatasetObjectDetection"]


# Object detection
def _od_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
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
    def __init__(self, image_paths: list[Path], annotation_paths: list[Path], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
        
        # --- Propagate 'classes' if they exist ---
        self.classes: list[str] = []

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
            target: dict[str, Any] = {}
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
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> tuple[Any, Any]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class _OD_PairedToTensor:
    """Converts a PIL Image to Tensor, passes targets dict through."""
    def __call__(self, image: Image.Image, target: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        return TF.to_tensor(image), target

class _OD_PairedNormalize:
    """Normalizes the image tensor and leaves the target dict untouched."""
    def __init__(self, mean: list[float], std: list[float]):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, image: torch.Tensor, target: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        image_normalized = self.normalize(image)
        return image_normalized, target

class _OD_PairedRandomHorizontalFlip:
    """Applies the same random horizontal flip to both image and targets['boxes']."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
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


class DragonDatasetObjectDetection:
    """
    Creates processed PyTorch datasets for object detection from image
    and JSON annotation folders.

    This maker finds all matching image-annotation pairs from two directories,
    splits them, and applies identical transformations (including augmentations)
    to both the image and its corresponding target dictionary.
    
    The `DragonFastRCNN` model expects a list of images and a list of targets,
    so this class provides a `collate_fn` to be used with a DataLoader.
    
    Workflow:
    1. `maker = DragonDatasetObjectDetection.from_folders(img_dir, ann_dir)`
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
        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None
        self.image_paths: list[Path] = []
        self.annotation_paths: list[Path] = []
        self.class_map: dict[str, int] = {}
        
        self._is_split = False
        self._are_transforms_configured = False
        self.train_transform: Optional[Callable] = None
        self.val_transform: Optional[Callable] = None
        self._val_recipe_components: Optional[dict[str, Any]] = None
        self._has_mean_std: bool = False

    @classmethod
    def from_folders(cls, image_dir: Union[str, Path], annotation_dir: Union[str, Path]) -> 'DragonDatasetObjectDetection':
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
            DragonDatasetObjectDetection: A new instance with all pairs loaded.
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
        found in the directory and its subdirectories.
        
        This is a utility method to help diagnose potential dataset
        issues (e.g., mixed image modes, corrupted files) before loading.

        Args:
            path (str, Path): The directory path to inspect.
        """
        inspect_folder(path)

    def set_class_map(self, class_map: dict[str, int]) -> 'DragonDatasetObjectDetection':
        """
        Sets a map of class_name -> pixel_value. This is used by the
        trainer for clear evaluation reports.
        
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
    def classes(self) -> list[str]:
        """Returns the list of class names, if set."""
        if self.class_map:
            return list(self.class_map.keys())
        return []

    def split_data(self, val_size: float = 0.2, test_size: float = 0.0, 
                   random_state: Optional[int] = 42) -> 'DragonDatasetObjectDetection':
        """
        Splits the loaded image-annotation pairs into train, validation, and test sets.

        Args:
            val_size (float): Proportion of the dataset to reserve for validation.
            test_size (float): Proportion of the dataset to reserve for testing.
            random_state (int | None): Seed for reproducible splits.

        Returns:
            DragonDatasetObjectDetection: The same instance, now with datasets created.
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
                             mean: Optional[list[float]] = [0.485, 0.456, 0.406], 
                             std: Optional[list[float]] = [0.229, 0.224, 0.225]) -> 'DragonDatasetObjectDetection':
        """
        Configures and applies the image and target transformations.
        
        This method must be called AFTER data is split.
        
        For object detection models like Faster R-CNN, images are NOT
        resized or cropped, as the model handles variable input sizes.
        Transforms are limited to augmentation (flip), ToTensor, and Normalize.

        Args:
            mean (List[float] | None): The mean values for image normalization.
            std (List[float] | None): The std dev values for image normalization.

        Returns:
            DragonDatasetObjectDetection: The same instance, with transforms applied.
        """
        if not self._is_split:
            _LOGGER.error("Transforms must be configured AFTER splitting data. Call .split_data() first.")
            raise RuntimeError()
        
        if (mean is None and std is not None) or (mean is not None and std is None):
            _LOGGER.error(f"'mean' and 'std' must be both None or both defined, but only one was provided.")
            raise ValueError()
        
        if mean is not None and std is not None:
            # --- Store components for validation recipe ---
            self._val_recipe_components = {
                VisionTransformRecipeKeys.MEAN: mean,
                VisionTransformRecipeKeys.STD: std
            }
            self._has_mean_std = True
            
        if self._has_mean_std:
            # --- Validation/Test Pipeline (Deterministic) ---
            self.val_transform = _OD_PairedCompose([
                _OD_PairedToTensor(),
                _OD_PairedNormalize(mean, std) # type: ignore
            ])
            
            # --- Training Pipeline (Augmentation) ---
            self.train_transform = _OD_PairedCompose([
                _OD_PairedRandomHorizontalFlip(p=0.5),
                _OD_PairedToTensor(),
                _OD_PairedNormalize(mean, std) # type: ignore
            ])
        else:
            # --- Validation/Test Pipeline (Deterministic) ---
            self.val_transform = _OD_PairedCompose([
                _OD_PairedToTensor()
            ])
            
            # --- Training Pipeline (Augmentation) ---
            self.train_transform = _OD_PairedCompose([
                _OD_PairedRandomHorizontalFlip(p=0.5),
                _OD_PairedToTensor()
            ])

        # --- Apply Transforms to the Datasets ---
        self._train_dataset.transform = self.train_transform # type: ignore
        self._val_dataset.transform = self.val_transform # type: ignore
        if self._test_dataset:
            self._test_dataset.transform = self.val_transform # type: ignore
        
        self._are_transforms_configured = True
        _LOGGER.info("Paired object detection transforms configured and applied.")
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
        
        # validate path
        file_path = make_fullpath(filepath, make=True, enforce="file")

        # Add standard transforms
        recipe: dict[str, Any] = {
            VisionTransformRecipeKeys.TASK: "object_detection",
            VisionTransformRecipeKeys.PIPELINE: [
                {VisionTransformRecipeKeys.NAME: "ToTensor", "kwargs": {}},
            ]
        }
        
        if self._has_mean_std and components:
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
        s += f"  Total Image-Annotation Pairs: {len(self.image_paths)}\n"
        s += f"  Split: {self._is_split}\n"
        s += f"  Transforms Configured: {self._are_transforms_configured}\n"
        
        if self.class_map:
            s += f"  Classes ({len(self.class_map)}): {list(self.class_map.keys())}\n"

        if self._is_split:
            train_len = len(self._train_dataset) if self._train_dataset else 0
            val_len = len(self._val_dataset) if self._val_dataset else 0
            test_len = len(self._test_dataset) if self._test_dataset else 0
            s += f"  Datasets (Train|Val|Test): {train_len} | {val_len} | {test_len}\n"
            
        return s
