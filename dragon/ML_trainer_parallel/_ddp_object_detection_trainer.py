from typing import Optional, Union, Callable
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset

from ..ML_configuration import DragonDDPConfig

from ..keys._keys import PyTorchLogKeys, DDPKeys
from .._core import get_logger

from ._base_parallel_trainer import _BaseParallelTrainer


_LOGGER = get_logger("Parallel Detection Trainer")


__all__ = ["DragonDetectionTrainerDDP"]


class DragonDetectionTrainerDDP(_BaseParallelTrainer):
    """
    Distributed Data Parallel (DDP) Trainer for object detection tasks.
    
    Designed exclusively for distributed training; evaluation and finalization 
    should be performed using the standard DragonDetectionTrainer by loading 
    the saved checkpoints.
    """
    def __init__(self, 
                 model: nn.Module,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 collate_fn: Callable,
                 optimizer: torch.optim.Optimizer,
                 save_dir: Union[str, Path],
                 ddp_config: Optional[DragonDDPConfig] = None):
        """
        Initializes the DDP trainer for object detection tasks.
        
        Args:
            model (nn.Module): The object detection model to be trained.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            collate_fn (Callable): Custom collate function for batching.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            save_dir (Union[str, Path]): Directory to save checkpoints and logs.
            ddp_config (Optional[DragonDDPConfig]): Configuration for DDP training.
        """
        
        super().__init__(
            model=model, 
            optimizer=optimizer, 
            save_dir=save_dir, 
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            collate_fn=collate_fn,
            ddp_config=ddp_config
        )

    def _train_step(self) -> dict[str, float]:
        self.model: nn.Module # model is already wrapped in DDP in the base class
        
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        faulty_batches = 0
        for batch_idx, (images, targets) in enumerate(self.train_loader): # type: ignore
            batch_size = len(images)
            
            batch_logs: dict[str, float] = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: batch_size
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model(images, targets)
            
            if not loss_dict:
                faulty_batches += 1
                loss = sum(p.sum() for p in self.model.parameters()) * 0.0 # type: ignore
            else:
                loss: torch.Tensor = sum(l for l in loss_dict.values()) # type: ignore
            
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            total_samples += batch_size
            
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if faulty_batches > 0 and self.is_main_process():
            _LOGGER.warning(f"Encountered {faulty_batches} faulty batches that did not return a loss dict during training.")
        
        if total_samples == 0:
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self) -> dict[str, float]:
        self.model.train() # Object detection models often require train mode to return loss dicts
        
        # Prevent BatchNorm running stats and Dropout from mutating/dropping during validation
        for module in self.model.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, 
                                    torch.nn.Dropout, 
                                    torch.nn.Dropout2d, 
                                    torch.nn.Dropout3d)):
                module.eval()
        
        running_loss = 0.0
        total_samples = 0 
        
        valid_samples_limit = self._val_samples_limit()
        
        faulty_batches = 0
        with torch.no_grad():
            for images, targets in self.validation_loader: # type: ignore
                batch_size = len(images)
                
                if total_samples >= valid_samples_limit:
                    continue
                    
                if total_samples + batch_size > valid_samples_limit:
                    allowed = valid_samples_limit - total_samples
                    images = images[:allowed]
                    targets = targets[:allowed]
                    batch_size = allowed
                
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                
                if not loss_dict:
                    faulty_batches += 1
                    total_samples += batch_size
                    continue 
                
                loss: torch.Tensor = sum(l for l in loss_dict.values()) # type: ignore
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        if faulty_batches > 0 and self.is_main_process():
            _LOGGER.warning(f"Encountered {faulty_batches} faulty batches that did not return a loss dict during validation.")
        
        # Return the raw sums to the base class instead of the local average
        return {PyTorchLogKeys.VAL_LOSS: running_loss, DDPKeys.VALIDATION_SAMPLES: float(total_samples)}
