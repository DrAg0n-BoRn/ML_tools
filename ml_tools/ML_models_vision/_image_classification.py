from torch import nn
import torchvision.models as vision_models
from typing import Literal, Optional

from ._base_wrapper import _BaseVisionWrapper


__all__ = [
    "DragonResNet",
    "DragonEfficientNet",
    "DragonVGG",
]


# Image classification
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

