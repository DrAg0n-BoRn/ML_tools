import torch
from torch import nn
import torchvision.models as vision_models
from torchvision.models import detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import List, Dict, Any, Literal, Optional
from abc import ABC, abstractmethod

from .ML_models import _ArchitectureHandlerMixin
from ._logger import _LOGGER
from ._script_info import _script_info


__all__ = [
    "DragonResNet",
    "DragonEfficientNet",
    "DragonVGG",
    "DragonFCN",
    "DragonDeepLabv3",
    "DragonFastRCNN",
]


class _BaseVisionWrapper(nn.Module, _ArchitectureHandlerMixin, ABC):
    """
    Abstract base class for torchvision model wrappers.
    
    Handles common logic for:
    - Model instantiation (with/without pretrained weights)
    - Input layer modification (for custom in_channels)
    - Output layer modification (for custom num_classes)
    - Architecture saving/loading and representation
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 model_name: str,
                 init_with_pretrained: bool,
                 weights_enum_name: Optional[str] = None):
        super().__init__()
        
        # --- 1. Validation and Configuration ---
        if not hasattr(vision_models, model_name):
            _LOGGER.error(f"'{model_name}' is not a valid model name in torchvision.models.")
            raise ValueError()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_name = model_name
        self._pretrained_default_transforms = None

        # --- 2. Instantiate the base model ---
        if init_with_pretrained:
            weights_enum = getattr(vision_models, weights_enum_name, None) if weights_enum_name else None
            weights = weights_enum.IMAGENET1K_V1 if weights_enum else None
            
            # Save transformations for pretrained models
            if weights:
                self._pretrained_default_transforms = weights.transforms()
            
            if weights is None and init_with_pretrained:
                 _LOGGER.warning(f"Could not find modern weights for {model_name}. Using 'pretrained=True' legacy fallback.")
                 self.model = getattr(vision_models, model_name)(pretrained=True)
            else:
                 self.model = getattr(vision_models, model_name)(weights=weights)
        else:
            self.model = getattr(vision_models, model_name)(weights=None)

        # --- 3. Modify the input layer (using abstract method) ---
        if in_channels != 3:
            original_conv1 = self._get_input_layer()
            
            new_conv1 = nn.Conv2d(
                in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size, # type: ignore
                stride=original_conv1.stride, # type: ignore
                padding=original_conv1.padding, # type: ignore
                bias=(original_conv1.bias is not None)
            )
            
            # (Optional) Average original weights if starting from pretrained
            if init_with_pretrained and original_conv1.in_channels == 3:
                with torch.no_grad():
                    avg_weights = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                    new_conv1.weight[:] = avg_weights.repeat(1, in_channels, 1, 1)

            self._set_input_layer(new_conv1)

        # --- 4. Modify the output layer (using abstract method) ---
        original_fc = self._get_output_layer()
        if original_fc is None: # Handle case where layer isn't found
             _LOGGER.error(f"Model '{model_name}' has an unexpected classifier structure. Cannot replace final layer.")
             raise AttributeError("Could not find final classifier layer.")

        num_filters = original_fc.in_features
        self._set_output_layer(nn.Linear(num_filters, num_classes))

    @abstractmethod
    def _get_input_layer(self) -> nn.Conv2d:
        """Returns the first convolutional layer of the model."""
        raise NotImplementedError

    @abstractmethod
    def _set_input_layer(self, layer: nn.Conv2d):
        """Sets the first convolutional layer of the model."""
        raise NotImplementedError

    @abstractmethod
    def _get_output_layer(self) -> Optional[nn.Linear]:
        """Returns the final fully-connected layer of the model."""
        raise NotImplementedError

    @abstractmethod
    def _set_output_layer(self, layer: nn.Linear):
        """Sets the final fully-connected layer of the model."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.model(x)

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Returns the structural configuration of the model.
        The 'init_with_pretrained' flag is intentionally omitted,
        as .load() should restore the architecture, not the weights.
        """
        return {
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'model_name': self.model_name
        }

    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        return (
            f"{self.__class__.__name__}(model='{self.model_name}', "
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes})"
        )


class DragonResNet(_BaseVisionWrapper):
    """
    Image Classification
    
    A customizable wrapper for the torchvision ResNet family, compatible
    with saving/loading architecture.

    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = 'resnet50',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes for the final layer.
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the ResNet model to use (e.g., 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'). Number is the layer count.
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on ImageNet. This flag is for initialization only and is NOT saved in the architecture config.
        """
        
        weights_enum_name = getattr(vision_models, f"{model_name.upper()}_Weights", None)
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            init_with_pretrained=init_with_pretrained,
            weights_enum_name=weights_enum_name
        )

    def _get_input_layer(self) -> nn.Conv2d:
        return self.model.conv1

    def _set_input_layer(self, layer: nn.Conv2d):
        self.model.conv1 = layer

    def _get_output_layer(self) -> Optional[nn.Linear]:
        return self.model.fc

    def _set_output_layer(self, layer: nn.Linear):
        self.model.fc = layer


class DragonEfficientNet(_BaseVisionWrapper):
    """
    Image Classification
    
    A customizable wrapper for the torchvision EfficientNet family, compatible
    with saving/loading architecture.

    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: str = 'efficientnet_b0',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes for the final layer.
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the EfficientNet model to use (e.g., 'efficientnet_b0'
                through 'efficientnet_b7', or 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l').
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on
                ImageNet. This flag is for initialization only and is
                NOT saved in the architecture config. Defaults to False.
        """
        
        weights_enum_name = getattr(vision_models, f"{model_name.upper()}_Weights", None)

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            init_with_pretrained=init_with_pretrained,
            weights_enum_name=weights_enum_name
        )

    def _get_input_layer(self) -> nn.Conv2d:
        # The first conv layer in EfficientNet is model.features[0][0]
        return self.model.features[0][0]

    def _set_input_layer(self, layer: nn.Conv2d):
        self.model.features[0][0] = layer

    def _get_output_layer(self) -> Optional[nn.Linear]:
        # The classifier in EfficientNet is model.classifier[1]
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            output_layer = self.model.classifier[1]
            if isinstance(output_layer, nn.Linear):
                return output_layer
        return None

    def _set_output_layer(self, layer: nn.Linear):
        self.model.classifier[1] = layer


class DragonVGG(_BaseVisionWrapper):
    """
    Image Classification
    
    A customizable wrapper for the torchvision VGG family, compatible
    with saving/loading architecture.

    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: Literal["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"] = 'vgg16',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes for the final layer.
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the VGG model to use (e.g., 'vgg16', 'vgg16_bn').
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on
                ImageNet. This flag is for initialization only and is
                NOT saved in the architecture config. Defaults to False.
        """
        
        # Format model name to find weights enum, e.g., vgg16_bn -> VGG16_BN_Weights
        weights_enum_name = f"{model_name.replace('_bn', '_BN').upper()}_Weights"
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            init_with_pretrained=init_with_pretrained,
            weights_enum_name=weights_enum_name
        )

    def _get_input_layer(self) -> nn.Conv2d:
        # The first conv layer in VGG is model.features[0]
        return self.model.features[0]

    def _set_input_layer(self, layer: nn.Conv2d):
        self.model.features[0] = layer

    def _get_output_layer(self) -> Optional[nn.Linear]:
        # The final classifier in VGG is model.classifier[6]
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) == 7:
            output_layer = self.model.classifier[6]
            if isinstance(output_layer, nn.Linear):
                return output_layer
        return None

    def _set_output_layer(self, layer: nn.Linear):
        self.model.classifier[6] = layer


# Image segmentation
class _BaseSegmentationWrapper(nn.Module, _ArchitectureHandlerMixin, ABC):
    """
    Abstract base class for torchvision segmentation model wrappers.
    
    Handles common logic for:
    - Model instantiation (with/without pretrained weights and custom num_classes)
    - Input layer modification (for custom in_channels)
    - Forward pass dictionary unpacking (returns 'out' tensor)
    - Architecture saving/loading and representation
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 model_name: str,
                 init_with_pretrained: bool,
                 weights_enum_name: Optional[str] = None):
        super().__init__()
        
        # --- 1. Validation and Configuration ---
        if not hasattr(vision_models.segmentation, model_name):
            _LOGGER.error(f"'{model_name}' is not a valid model name in torchvision.models.segmentation.")
            raise ValueError()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_name = model_name
        self._pretrained_default_transforms = None

        # --- 2. Instantiate the base model ---
        model_kwargs = {
            'num_classes': num_classes,
            'weights': None
        }
        model_constructor = getattr(vision_models.segmentation, model_name)

        if init_with_pretrained:
            weights_enum = getattr(vision_models.segmentation, weights_enum_name, None) if weights_enum_name else None
            weights = weights_enum.DEFAULT if weights_enum else None
            
            # save pretrained model transformations
            if weights:
                self._pretrained_default_transforms = weights.transforms()
            
            if weights is None:
                 _LOGGER.warning(f"Could not find modern weights for {model_name}. Using 'pretrained=True' legacy fallback.")
                 # Legacy models used 'pretrained=True' and num_classes was separate
                 self.model = model_constructor(pretrained=True, **model_kwargs)
            else:
                 # Modern way: weights object implies pretraining
                 model_kwargs['weights'] = weights
                 self.model = model_constructor(**model_kwargs)
        else:
            self.model = model_constructor(**model_kwargs)

        # --- 3. Modify the input layer (using abstract method) ---
        if in_channels != 3:
            original_conv1 = self._get_input_layer()
            
            new_conv1 = nn.Conv2d(
                in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size, # type: ignore
                stride=original_conv1.stride, # type: ignore
                padding=original_conv1.padding, # type: ignore
                bias=(original_conv1.bias is not None)
            )
            
            # (Optional) Average original weights if starting from pretrained
            if init_with_pretrained and original_conv1.in_channels == 3:
                with torch.no_grad():
                    avg_weights = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                    new_conv1.weight[:] = avg_weights.repeat(1, in_channels, 1, 1)

            self._set_input_layer(new_conv1)

    @abstractmethod
    def _get_input_layer(self) -> nn.Conv2d:
        """Returns the first convolutional layer of the model (in the backbone)."""
        raise NotImplementedError

    @abstractmethod
    def _set_input_layer(self, layer: nn.Conv2d):
        """Sets the first convolutional layer of the model (in the backbone)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.
        Returns the 'out' tensor from the segmentation model's output dict.
        """
        output_dict = self.model(x)
        return output_dict['out'] # Key for standard torchvision seg models

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Returns the structural configuration of the model.
        The 'init_with_pretrained' flag is intentionally omitted,
        as .load() should restore the architecture, not the weights.
        """
        return {
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'model_name': self.model_name
        }

    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        return (
            f"{self.__class__.__name__}(model='{self.model_name}', "
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes})"
        )


class DragonFCN(_BaseSegmentationWrapper):
    """
    Image Segmentation
    
    A customizable wrapper for the torchvision FCN (Fully Convolutional Network)
    family, compatible with saving/loading architecture.

    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: Literal["fcn_resnet50", "fcn_resnet101"] = 'fcn_resnet50',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes (including background).
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the FCN model to use ('fcn_resnet50' or 'fcn_resnet101').
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on COCO.
                This flag is for initialization only and is NOT saved in the
                architecture config. Defaults to False.
        """
        # Format model name to find weights enum, e.g., fcn_resnet50 -> FCN_ResNet50_Weights
        weights_model_name = model_name.replace('fcn_', 'FCN_').replace('resnet', 'ResNet')
        weights_enum_name = f"{weights_model_name}_Weights"
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            init_with_pretrained=init_with_pretrained,
            weights_enum_name=weights_enum_name
        )

    def _get_input_layer(self) -> nn.Conv2d:
        # FCN models use a ResNet backbone, input layer is backbone.conv1
        return self.model.backbone.conv1

    def _set_input_layer(self, layer: nn.Conv2d):
        self.model.backbone.conv1 = layer


class DragonDeepLabv3(_BaseSegmentationWrapper):
    """
    Image Segmentation
    
    A customizable wrapper for the torchvision DeepLabv3 family, compatible
    with saving/loading architecture.

    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: Literal["deeplabv3_resnet50", "deeplabv3_resnet101"] = 'deeplabv3_resnet50',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes (including background).
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the DeepLabv3 model to use ('deeplabv3_resnet50' or 'deeplabv3_resnet101').
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on COCO.
                This flag is for initialization only and is NOT saved in the
                architecture config. Defaults to False.
        """
        
        # Format model name to find weights enum, e.g., deeplabv3_resnet50 -> DeepLabV3_ResNet50_Weights
        weights_model_name = model_name.replace('deeplabv3_', 'DeepLabV3_').replace('resnet', 'ResNet')
        weights_enum_name = f"{weights_model_name}_Weights"
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            init_with_pretrained=init_with_pretrained,
            weights_enum_name=weights_enum_name
        )

    def _get_input_layer(self) -> nn.Conv2d:
        # DeepLabv3 models use a ResNet backbone, input layer is backbone.conv1
        return self.model.backbone.conv1

    def _set_input_layer(self, layer: nn.Conv2d):
        self.model.backbone.conv1 = layer


class DragonFastRCNN(nn.Module, _ArchitectureHandlerMixin):
    """
    Object Detection
    
    A customizable wrapper for the torchvision Faster R-CNN family.
    
    This wrapper allows for customizing the model backbone, input channels,
    and the number of output classes for transfer learning.

    NOTE: Use an Object Detection compatible trainer.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 model_name: Literal["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"] = 'fasterrcnn_resnet50_fpn_v2',
                 init_with_pretrained: bool = False):
        """
        Args:
            num_classes (int):
                Number of output classes (including background).
            in_channels (int):
                Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            model_name (str):
                The name of the Faster R-CNN model to use.
            init_with_pretrained (bool):
                If True, initializes the model with weights pretrained on COCO.
                This flag is for initialization only and is NOT saved in the
                architecture config. Defaults to False.
        """
        super().__init__()
        
        # --- 1. Validation and Configuration ---
        if not hasattr(detection_models, model_name):
            _LOGGER.error(f"'{model_name}' is not a valid model name in torchvision.models.detection.")
            raise ValueError()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_name = model_name
        self._pretrained_default_transforms = None

        # --- 2. Instantiate the base model ---
        model_constructor = getattr(detection_models, model_name)
        
        # Format model name to find weights enum, e.g., fasterrcnn_resnet50_fpn_v2 -> FasterRCNN_ResNet50_FPN_V2_Weights
        weights_model_name = model_name.replace('fasterrcnn_', 'FasterRCNN_').replace('resnet', 'ResNet').replace('_fpn', '_FPN')
        weights_enum_name = f"{weights_model_name.upper()}_Weights"
        
        weights_enum = getattr(detection_models, weights_enum_name, None) if weights_enum_name else None
        weights = weights_enum.DEFAULT if weights_enum and init_with_pretrained else None
        
        if weights:
            self._pretrained_default_transforms = weights.transforms()

        self.model = model_constructor(weights=weights, weights_backbone=weights)
        
        # --- 4. Modify the output layer (Box Predictor) ---
        # Get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # --- 3. Modify the input layer (Backbone conv1) ---
        if in_channels != 3:
            original_conv1 = self.model.backbone.body.conv1
            
            new_conv1 = nn.Conv2d(
                in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size, # type: ignore
                stride=original_conv1.stride, # type: ignore
                padding=original_conv1.padding, # type: ignore
                bias=(original_conv1.bias is not None)
            )
            
            # (Optional) Average original weights if starting from pretrained
            if init_with_pretrained and original_conv1.in_channels == 3 and weights is not None:
                with torch.no_grad():
                    # Average the weights across the input channel dimension
                    avg_weights = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                    # Repeat the averaged weights for the new number of input channels
                    new_conv1.weight[:] = avg_weights.repeat(1, in_channels, 1, 1)

            self.model.backbone.body.conv1 = new_conv1

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Defines the forward pass.
        
        - In train mode, expects (images, targets) and returns a dict of losses.
        - In eval mode, expects (images) and returns a list of prediction dicts.
        """
        # The model's forward pass handles train/eval mode internally.
        return self.model(images, targets)

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Returns the structural configuration of the model.
        The 'init_with_pretrained' flag is intentionally omitted,
        as .load() should restore the architecture, not the weights.
        """
        return {
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'model_name': self.model_name
        }

    def __repr__(self) -> str:
        """Returns the developer-friendly string representation of the model."""
        return (
            f"{self.__class__.__name__}(model='{self.model_name}', "
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes})"
        )


def info():
    _script_info(__all__)
