from typing import Literal, Union, Optional
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset

from ..ML_configuration import DragonDDPConfig

from ..keys._keys import PyTorchLogKeys, MLTaskKeys, DDPKeys
from .._core import get_logger

from ._base_parallel_trainer import _BaseParallelTrainer


_LOGGER = get_logger("Parallel Distribution Trainer")


__all__ = ["DragonDistributionTrainerDDP"]


class DragonDistributionTrainerDDP(_BaseParallelTrainer):
    """
    Distributed Data Parallel (DDP) Trainer for probabilistic distribution prediction tasks.
    
    Designed exclusively for distributed training; evaluation and finalization 
    should be performed using the standard DragonDistributionTrainer.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Union[Literal["regression", "multitarget regression"], str],
                 optimizer: torch.optim.Optimizer, 
                 criterion: Union[nn.Module, Literal["auto"]] = "auto", 
                 ddp_config: Optional[DragonDDPConfig] = None):
        """
        Initializes the DDP trainer for probabilistic distribution prediction tasks.
        
        Args:
            model (nn.Module): The model to be trained.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (Union[str, Path]): Directory to save checkpoints and logs.
            kind (Union[Literal["regression", "multitarget regression"], str]): The type of distribution prediction task.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            criterion (Union[nn.Module, Literal["auto"]]): The loss function for training. If "auto", it will default to `nn.GaussianNLLLoss()`. Must be compatible with `(mean, target, variance)` inputs.
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
        
        if kind not in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
            if self.is_main_process():
                _LOGGER.error(f"Invalid 'kind' argument: '{kind}'.")
            raise ValueError()

        self.kind = kind
        
        if criterion == "auto":
            self.criterion = nn.GaussianNLLLoss()
        else:
            self.criterion = criterion
            
        if not isinstance(self.criterion, nn.Module):
            if self.is_main_process():
                _LOGGER.error(f"The provided criterion is not a valid PyTorch loss module: {type(self.criterion)}")
            raise TypeError()
            
        self.criterion = self.criterion.to(self.device)
        
    def _train_step(self) -> dict[str, float]:
        self.model: nn.Module # model is already wrapped in DDP in the base class
        
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
            
            mean, var_logits = torch.tensor_split(output, 2, dim=-1)
            var = torch.nn.functional.softplus(var_logits)
            
            if self.kind == MLTaskKeys.REGRESSION:
                if mean.ndim == 2 and mean.shape[1] == 1 and target.ndim == 1:
                    mean = mean.squeeze(1)
                    var = var.squeeze(1)
                    
            loss = self.criterion(mean, target, var) # type: ignore
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
                
                mean, var_logits = torch.tensor_split(output, 2, dim=-1)
                var = torch.nn.functional.softplus(var_logits)
                
                if self.kind == MLTaskKeys.REGRESSION:
                    if mean.ndim == 2 and mean.shape[1] == 1 and target.ndim == 1:
                        mean = mean.squeeze(1)
                        var = var.squeeze(1)
                
                loss = self.criterion(mean, target, var) # type: ignore
                
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Return the raw sums to the base class instead of the local average
        return {PyTorchLogKeys.VAL_LOSS: running_loss, DDPKeys.VALIDATION_SAMPLES: float(total_samples)}
