from torch import nn
from typing import Literal

from ._base_wrapper import _BaseSegmentationWrapper


__all__ = [
    "DragonFCN",
    "DragonDeepLabv3",
]


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

