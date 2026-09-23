from typing import Literal, Union, Optional
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset

from ..ML_configuration import DragonDDPConfig

from .._core import get_logger
from ..keys._keys import PyTorchLogKeys, MLTaskKeys, DDPKeys

from ._base_parallel_trainer import _BaseParallelTrainer


_LOGGER = get_logger("Parallel Vision Trainer")


__all__ = ["DragonVisionTrainerDDP"]


class DragonVisionTrainerDDP(_BaseParallelTrainer):
    """
    Distributed Data Parallel (DDP) Trainer for computer vision tasks.
    
    Supports binary and multiclass image classification and segmentation.
    Designed exclusively for distributed training; evaluation and finalization 
    should be performed using the standard DragonVisionTrainer by loading 
    the saved checkpoints.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Union[Literal["binary segmentation", 
                               "multiclass segmentation", 
                               "binary image classification", 
                               "multiclass image classification"], str],
                 optimizer: torch.optim.Optimizer, 
                 criterion: Union[nn.Module, Literal["auto"]] = "auto", 
                 ddp_config: Optional[DragonDDPConfig] = None):
        """
        Initializes the DDP trainer for computer vision tasks.
        
        Args:
            model (nn.Module): The vision model to be trained.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (Union[str, Path]): Directory to save checkpoints and logs.
            kind (Union[Literal["binary segmentation", "multiclass segmentation", "binary image classification", "multiclass image classification"], str]): The type of vision task.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            criterion (Union[nn.Module, Literal["auto"]]): The loss function for training.
            ddp_config (Optional[DragonDDPConfig]): Configuration for DDP training.
        """
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            save_dir=save_dir,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            ddp_config=ddp_config
        )
        
        if kind not in [MLTaskKeys.BINARY_SEGMENTATION,
                        MLTaskKeys.MULTICLASS_SEGMENTATION,
                        MLTaskKeys.BINARY_IMAGE_CLASSIFICATION,
                        MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
            if self.is_main_process():
                _LOGGER.error(f"'{kind}' is not a valid vision task type.")
            raise ValueError()

        self.kind = kind
        
        if criterion == "auto":
            if kind in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.BINARY_SEGMENTATION]:
                self.criterion = nn.BCEWithLogitsLoss()
            elif kind in [MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        if not isinstance(self.criterion, nn.Module):
            if self.is_main_process():
                _LOGGER.error(f"The provided criterion is not a valid PyTorch loss module: {type(self.criterion)}")
            raise TypeError()
            
        self.criterion = self.criterion.to(self.device)
    
    def _format_output_and_target(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Formats the outputs and targets strictly for the criterion."""
        # Strict type enforcement for loss functions
        if self.kind in MLTaskKeys.ALL_BINARY_TASKS:
            target = target.float()
        elif self.kind in [MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
            target = target.long()

        # Shape mismatch handling
        if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
            if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                output = output.squeeze(1)
        
        if self.kind == MLTaskKeys.BINARY_SEGMENTATION:
            if output.ndim == 4 and output.shape[1] == 1 and target.ndim == 3:
                output = output.squeeze(1)
        
        if self.kind == MLTaskKeys.MULTICLASS_SEGMENTATION:
            if target.ndim == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
                
        return output, target

    def _train_step(self) -> dict[str, float]:
        self.model: nn.Module # model is already wrapped in DDP in the base class
        self.criterion: nn.Module # loss function is a PyTorch loss module
        
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            batch_size = features.size(0)
            
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: batch_size
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(features)
            
            output, target = self._format_output_and_target(output, target)
                
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * batch_size 
            total_samples += batch_size 
            
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if total_samples == 0:
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        valid_samples_limit = self._val_samples_limit()
        
        with torch.no_grad():
            for features, target in self.validation_loader: # type: ignore
                batch_size = features.size(0)
                
                if total_samples >= valid_samples_limit:
                    continue
                    
                if total_samples + batch_size > valid_samples_limit:
                    allowed = valid_samples_limit - total_samples
                    features = features[:allowed]
                    target = target[:allowed]
                    batch_size = allowed
                
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                
                output, target = self._format_output_and_target(output, target)
                
                loss = self.criterion(output, target)
                
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Return the raw sums to the base class instead of the local average
        return {PyTorchLogKeys.VAL_LOSS: running_loss, DDPKeys.VALIDATION_SAMPLES: float(total_samples)}
