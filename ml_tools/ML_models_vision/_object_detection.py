import torch
from torch import nn
from torchvision.models import detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Any, Literal, Optional

from ..ML_models._base_save_load import _ArchitectureHandlerMixin

from .._core import get_logger


_LOGGER = get_logger("DragonFastRCNN")


__all__ = [
    "DragonFastRCNN",
]


# Object Detection
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

    def forward(self, images: list[torch.Tensor], targets: Optional[list[dict[str, torch.Tensor]]] = None):
        """
        Defines the forward pass.
        
        - In train mode, expects (images, targets) and returns a dict of losses.
        - In eval mode, expects (images) and returns a list of prediction dicts.
        """
        # The model's forward pass handles train/eval mode internally.
        return self.model(images, targets)

    def get_architecture_config(self) -> dict[str, Any]:
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

