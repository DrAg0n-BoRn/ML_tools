import torch
from torch import nn
import torchvision.models as vision_models
from typing import Any, Optional
from abc import ABC, abstractmethod

from ..ML_models._base_save_load import _ArchitectureHandlerMixin

from .._core import get_logger


_LOGGER = get_logger("DragonVisionModel")


__all__ = [
    "_BaseVisionWrapper",
    "_BaseSegmentationWrapper",
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

