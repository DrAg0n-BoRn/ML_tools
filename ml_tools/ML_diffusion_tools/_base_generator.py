from typing import Union, Optional, Any, Literal
import torch
from abc import ABC, abstractmethod
from pathlib import Path

from ..ML_models_diffusion import DragonAutoencoder, DragonDiT, DragonDiTGuided
from ..ML_utilities import validate_torch_device

from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("DiffusionGenerator")


__all__ = [
    "_BaseDiffusionGenerator",
]


class _BaseDiffusionGenerator(ABC):
    def __init__(self,
                 save_dir: Union[Path, str],
                 diffusion_model: Union[DragonDiT, DragonDiTGuided],
                 encoder: DragonAutoencoder,
                 device: Union[torch.device, str]):
        """Base class for diffusion generators."""
        
        self.save_root_dir = make_fullpath(save_dir, make=True, enforce="directory")
        self.diffusion_model = diffusion_model
        self.encoder = encoder

        if isinstance(device, str):
            self.device = validate_torch_device(device)
        else:
            self.device = device
        
        # Move models to the specified device
        self.diffusion_model.to(self.device)
        self.encoder.to(self.device)
        # Set models to evaluation mode
        self.diffusion_model.eval()
        self.encoder.eval()
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def plot_metrics(self, *args, **kwargs) -> Any:
        pass
    
    