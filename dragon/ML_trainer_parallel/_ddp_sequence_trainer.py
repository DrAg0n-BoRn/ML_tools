
from typing import Literal, Union, Optional, Any
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset

from ..ML_configuration import DragonDDPConfig

from ..keys._keys import PyTorchLogKeys, MLTaskKeys, DatasetKeys, DDPKeys
from .._core import get_logger

from ._base_parallel_trainer import _BaseParallelTrainer


_LOGGER = get_logger("DDP Sequence Trainer")


__all__ = ["DragonSequenceTrainerDDP"]


class DragonSequenceTrainerDDP(_BaseParallelTrainer):
    """
    Distributed Data Parallel (DDP) Trainer for sequence-based tasks.
    
    Supports models returning single Tensors or dictionaries of output heads.
    Designed exclusively for distributed training; evaluation and finalization 
    should be performed using the standard DragonSequenceTrainer.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Union[Literal["autoregressive-sequence-to-sequence", 
                                     "autoregressive-sequence-to-value", 
                                     "exogenous-sequence-to-sequence", 
                                     "exogenous-sequence-to-value"], str],
                 optimizer: torch.optim.Optimizer, 
                 target_types: Optional[dict[str, str]] = None,
                 criterion: Union[nn.Module, dict[str, nn.Module], Literal["auto"]] = "auto", 
                 ddp_config: Optional[DragonDDPConfig] = None):
        """
        Initializes the DDP trainer for sequence-based tasks.
        
        Args:
            model (nn.Module): The sequence model to be trained.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (Union[str, Path]): Directory to save checkpoints and logs.
            kind (Union[Literal["autoregressive-sequence-to-sequence", "autoregressive-sequence-to-value", "exogenous-sequence-to-sequence", "exogenous-sequence-to-value"], str]): The type of sequence task.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            target_types (Optional[dict[str, str]]): Optional mapping of target names to their types (continuous or categorical).
            criterion (nn.Module | dict[str, nn.Module] | "auto"): Loss function.
                - If "auto", infers MSE or CrossEntropy per target.
                - If `nn.Module`, applies the same loss to all targets.
                - If `dict[str, nn.Module]`, applies specified loss per target name.
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
        
        if kind not in MLTaskKeys.ALL_SEQUENCE_TASKS:
            if self.is_main_process():
                _LOGGER.error(f"'{kind}' is not a valid task type for this trainer.")
            raise ValueError()

        self.kind = kind        
        
        # Extract or assign target types
        self.target_types = target_types
        if self.target_types is None:
            self.target_types = self._get_dataset_attr(self.train_dataset, DatasetKeys.TARGET_TYPES)
        
        if self.target_types:
            for t_name, t_type in self.target_types.items():
                if t_type not in [DatasetKeys.TARGET_CONTINUOUS, DatasetKeys.TARGET_CATEGORICAL]:
                    if self.is_main_process():
                        _LOGGER.error(f"Invalid target type '{t_type}' for target '{t_name}'.")
                    raise ValueError()
        
        self.target_names = self._get_target_names()
        
        # Transform "auto" into explicit module(s)
        if criterion == "auto":
            if len(self.target_names) > 1:
                auto_criterion: dict[str, nn.Module] = {}
                for t_name in self.target_names:
                    is_cat = self.target_types and self.target_types.get(t_name) == DatasetKeys.TARGET_CATEGORICAL
                    auto_criterion[t_name] = nn.CrossEntropyLoss() if is_cat else nn.MSELoss()
                self.criterion = auto_criterion
            else:
                t_name = self.target_names[0]
                is_cat = self.target_types and self.target_types.get(t_name) == DatasetKeys.TARGET_CATEGORICAL
                self.criterion = nn.CrossEntropyLoss() if is_cat else nn.MSELoss()
        else:
            self.criterion = criterion
            
        if not isinstance(self.criterion, (nn.Module, dict)):
            if self.is_main_process():
                _LOGGER.error(f"Invalid criterion type: {type(self.criterion)}")
            raise TypeError()
        
        # Move criterion to device locally
        if isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion.to(self.device)
        elif isinstance(self.criterion, dict):
            self.criterion = {k: v.to(self.device) for k, v in self.criterion.items()}

    def _get_dataset_attr(self, dataset: Any, attr_name: str, default: Any = None) -> Any:
        """Helper to extract metadata safely."""
        if hasattr(dataset, attr_name):
            try:
                val = getattr(dataset, attr_name)
                if val is not None:
                    return val
            except AttributeError:
                pass
        if hasattr(dataset, "dataset"):
            return self._get_dataset_attr(dataset.dataset, attr_name, default)
        return default

    @property
    def _is_seq_to_val(self) -> bool:
        return self.kind in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE, MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE]

    @property
    def _is_seq_to_seq(self) -> bool:
        return self.kind in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE]

    def _get_target_names(self) -> list[str]:
        target_names = self._get_dataset_attr(self.train_dataset, DatasetKeys.TARGET_NAMES)
        if target_names is not None:
            return target_names
        elif hasattr(self.model, "module") and hasattr(self.model.module, "targets"):
            return getattr(self.model.module, "targets")
        elif self.target_types is not None:
            return list(self.target_types.keys())
        
        if self.is_main_process():
            _LOGGER.error("Target names could not be determined from the dataset, model, or target_types.")
        raise ValueError()

    def _compute_loss(self, outputs: Union[dict[str, torch.Tensor], torch.Tensor], targets: Union[dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        target_names = self._get_target_names()

        if isinstance(outputs, dict):
            # Anchor all outputs to the graph to satisfy DDP's find_unused_parameters=False
            total_loss = sum(out.sum() for out in outputs.values()) * 0.0
            
            for i, target_name in enumerate(target_names if target_names else list(outputs.keys())):
                if target_name not in outputs:
                    continue
                
                pred_t = outputs[target_name]
                
                if isinstance(targets, dict):
                    target_t = targets[target_name]
                elif isinstance(targets, torch.Tensor):
                    if targets.ndim == 1:
                        target_t = targets
                    elif self._is_seq_to_val:
                        target_t = targets[:, i] if targets.ndim == 2 and targets.shape[1] > i else targets
                    elif self._is_seq_to_seq:
                        target_t = targets[:, :, i] if targets.ndim == 3 and targets.shape[2] > i else targets
                else:
                    if self.is_main_process():
                        _LOGGER.error(f"Unsupported target type: {type(targets)}")
                    raise TypeError()

                is_categorical = False
                if self.target_types and target_name in self.target_types:
                    is_categorical = (self.target_types[target_name] == DatasetKeys.TARGET_CATEGORICAL)
                else:
                    is_categorical = (pred_t.ndim > target_t.ndim or target_t.dtype in [torch.int64, torch.long, torch.int32])

                if isinstance(self.criterion, dict):
                    loss_fn = self.criterion[target_name]
                else:
                    loss_fn = self.criterion # type: ignore

                if is_categorical:
                    pred_flat = pred_t.reshape(-1, pred_t.shape[-1])
                    target_flat = target_t.reshape(-1).long()
                    loss_t = loss_fn(pred_flat, target_flat) # type: ignore
                else:
                    target_t = target_t.float()
                    if pred_t.shape != target_t.shape:
                        if pred_t.ndim == target_t.ndim + 1 and pred_t.shape[-1] == 1:
                            pred_t = pred_t.squeeze(-1)
                    loss_t = loss_fn(pred_t, target_t) # type: ignore

                total_loss = total_loss + loss_t
            return total_loss # type: ignore

        else:
            if isinstance(targets, dict):
                if self.is_main_process():
                    _LOGGER.error("Output is a tensor, but targets are a dict.")
                raise TypeError()
            
            target_name = self.target_names[0] if self.target_names else None
            is_categorical = False
            
            if self.target_types and (target_name is not None) and (target_name in self.target_types):
                is_categorical = (self.target_types[target_name] == DatasetKeys.TARGET_CATEGORICAL)
            else:
                is_categorical = (outputs.ndim > targets.ndim or targets.dtype in [torch.int64, torch.long, torch.int32])

            if is_categorical:
                target = targets.long()
            else:
                target = targets.float()
                
            loss_fn = self.criterion

            if self._is_seq_to_val:
                if outputs.ndim == 2 and outputs.shape[1] == 1 and target.ndim == 1:
                    outputs = outputs.squeeze(1)
            elif self._is_seq_to_seq:
                if outputs.ndim == 3 and outputs.shape[2] == 1 and target.ndim == 2:
                    outputs = outputs.squeeze(-1)
            
            if is_categorical:
                pred_flat = outputs.reshape(-1, outputs.shape[-1])
                target_flat = target.reshape(-1)
                return loss_fn(pred_flat, target_flat) # type: ignore
            else:
                return loss_fn(outputs, target) # type: ignore

    def _train_step(self) -> dict[str, float]:
        self.model: nn.Module # model is already wrapped in DDP in the base class
        
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader):  # type: ignore
            batch_size = features.size(0)
            
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: batch_size
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features = features.to(self.device)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)
            elif isinstance(target, dict):
                target = {k: v.to(self.device) for k, v in target.items()}

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self._compute_loss(outputs, target)
            
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
            for features, target in self.validation_loader:  # type: ignore
                batch_size = features.size(0)
                
                if total_samples >= valid_samples_limit:
                    continue
                    
                if total_samples + batch_size > valid_samples_limit:
                    allowed = valid_samples_limit - total_samples
                    features = features[:allowed]
                    if isinstance(target, torch.Tensor):
                        target = target[:allowed]
                    elif isinstance(target, dict):
                        target = {k: v[:allowed] for k, v in target.items()}
                    batch_size = allowed
                
                features = features.to(self.device)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)
                elif isinstance(target, dict):
                    target = {k: v.to(self.device) for k, v in target.items()}

                outputs = self.model(features)
                loss = self._compute_loss(outputs, target)
                
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Return the raw sums to the base class instead of the local average
        return {PyTorchLogKeys.VAL_LOSS: running_loss, DDPKeys.VALIDATION_SAMPLES: float(total_samples)}
