import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal, Union

from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys
from .._core import get_logger

from ._base import _Callback
from ._early_stop import _DragonEarlyStopping
from ._scheduler import _DragonLRScheduler


_LOGGER = get_logger("Parallel Callbacks")


__all__ = [
    "DDPProgressBar",
    "DDPHistory",
    "DDPModelCheckpoint",
    "DDPPatienceEarlyStopping",
    "DDPPlateauScheduler"
]


class DDPProgressBar(_Callback):
    """
    A DDP-safe progress bar that only renders on the main process (Rank 0).
    """
    def __init__(self):
        super().__init__()
        self.epoch_bar = None
        self.batch_bar = None
        self.epochs = 0

    def on_train_begin(self, logs=None):
        if self.trainer.is_main_process(): # type: ignore
            self.epochs = self.trainer.epochs # type: ignore
            initial_epoch = self.trainer.start_epoch - 1 # type: ignore
            self.epoch_bar = tqdm(total=self.epochs, initial=initial_epoch, desc="🐲 DDP Training")

    def on_epoch_begin(self, epoch, logs=None):
        if self.trainer.is_main_process(): # type: ignore
            # DDP trainers usually have length on the dataloader representing batches per GPU
            total_batches = len(self.trainer.train_loader) # type: ignore
            self.batch_bar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{self.epochs}", leave=False)

    def on_batch_end(self, batch, logs=None):
        if self.trainer.is_main_process() and self.batch_bar: # type: ignore
            self.batch_bar.update(1)
            if logs:
                self.batch_bar.set_postfix(loss=f"{logs.get(PyTorchLogKeys.BATCH_LOSS, 0):.4f}")

    def on_epoch_end(self, epoch, logs=None):
        if self.trainer.is_main_process(): # type: ignore
            if self.batch_bar:
                self.batch_bar.close()
            if self.epoch_bar:
                self.epoch_bar.update(1)
                if logs:
                    train_loss = f"{logs.get(PyTorchLogKeys.TRAIN_LOSS, 0):.4f}"
                    val_loss = f"{logs.get(PyTorchLogKeys.VAL_LOSS, 0):.4f}"
                    self.epoch_bar.set_postfix_str(f"Train Loss: {train_loss}, Val Loss: {val_loss}")

    def on_train_end(self, logs=None):
        if self.trainer.is_main_process() and self.epoch_bar: # type: ignore
            self.epoch_bar.close()


class DDPHistory(_Callback):
    """
    A DDP-safe history tracker. Only Rank 0 maintains the history dictionary.
    """
    def on_train_begin(self, logs=None):
        if self.trainer.is_main_process(): # type: ignore
            if self.trainer.start_epoch <= 1: # type: ignore
                self.trainer.history = {} # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        if self.trainer.is_main_process(): # type: ignore
            logs = logs or {}
            for k, v in logs.items():
                self.trainer.history.setdefault(k, []).append(v) # type: ignore


class DDPModelCheckpoint(_Callback):
    """
    A DDP-safe model checkpoint. Only Rank 0 evaluates the metric and writes to disk.
    Properly accesses the underlying model inside the DDP wrapper.
    """
    def __init__(self, 
                 monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 save_three_best: bool = True, 
                 mode: Literal['min', 'max'] = 'min', 
                 verbose: int = 1):
        
        super().__init__()
        
        # Standardize monitor key
        if monitor == "Training Loss":
            self.monitor = PyTorchLogKeys.TRAIN_LOSS
        elif monitor == "Validation Loss":
            self.monitor = PyTorchLogKeys.VAL_LOSS
        elif monitor == "both":
            self.monitor = "both"
        else:
            _LOGGER.error(f"Unknown monitor key: {monitor}.")
            raise ValueError()
        
        self.save_three_best = save_three_best
        self.mode = mode
        self.verbose = verbose
        
        self.best_checkpoints = [] 
        self.recent_checkpoints = []

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def _get_metric_value(self, logs):
        """Extracts or calculates the metric value based on configuration."""
        if self.monitor == "both":
            t_loss = logs.get(PyTorchLogKeys.TRAIN_LOSS)
            v_loss = logs.get(PyTorchLogKeys.VAL_LOSS)
            if t_loss is None or v_loss is None:
                return None
            return t_loss + v_loss
        else:
            return logs.get(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        # Strictly restrict execution to the main process
        if not self.trainer.is_main_process(): # type: ignore
            return
            
        logs = logs or {}
        current_score = self._get_metric_value(logs)

        if current_score is None:
            if self.verbose > 0:
                _LOGGER.warning(f"Epoch {epoch}: Metric '{self.monitor}' not found. Skipping checkpoint.")
            return
        
        if self.monitor_op(current_score, self.best):
            self.best = current_score

        if self.save_three_best:
            self._save_top_k_checkpoints(epoch, current_score)
        else:
            self._save_rolling_checkpoints(epoch, current_score)

    def _save_checkpoint_file(self, epoch, current_score):
        save_dir: Path = self.trainer._checkpoints_directory # type: ignore
        if save_dir is None:
            return None
            
        score_str = f"{current_score:.4f}".replace('.', '_')
        filename = f"DDP-epoch{epoch}-{PyTorchCheckpointKeys.CHECKPOINT_NAME}_{score_str}.pth"
        filepath = save_dir / filename
        
        # Safely unwrap both torch.compile and DDP wrappers regardless of nesting order
        base_model = self.trainer.model # type: ignore
        while hasattr(base_model, "_orig_mod") or hasattr(base_model, "module"):
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod
            if hasattr(base_model, "module"):
                base_model = base_model.module
        
        checkpoint_data = {
            PyTorchCheckpointKeys.EPOCH: epoch,
            PyTorchCheckpointKeys.MODEL_STATE: base_model.state_dict(),
            PyTorchCheckpointKeys.OPTIMIZER_STATE: self.trainer.optimizer.state_dict(), # type: ignore
            PyTorchCheckpointKeys.BEST_SCORE: self.best,
            PyTorchCheckpointKeys.HISTORY: getattr(self.trainer, 'history', {}),
        }
        
        torch.save(checkpoint_data, filepath)
        return filepath

    def _save_top_k_checkpoints(self, epoch, current_score):
        should_save = len(self.best_checkpoints) < 3
        is_reverse = (self.mode == 'max')
        
        if not should_save:
            self.best_checkpoints.sort(key=lambda x: x['score'], reverse=is_reverse)
            worst_entry = self.best_checkpoints[-1]
            if self.monitor_op(current_score, worst_entry['score']):
                should_save = True

        if should_save:
            filepath = self._save_checkpoint_file(epoch, current_score)
            if filepath:
                self.best_checkpoints.append({'path': filepath, 'score': current_score, 'epoch': epoch})
                
                if len(self.best_checkpoints) > 3:
                    self.best_checkpoints.sort(key=lambda x: x['score'], reverse=is_reverse)
                    entry_to_delete = self.best_checkpoints.pop(-1)
                    if entry_to_delete['path'].exists():
                        entry_to_delete['path'].unlink()

    def _save_rolling_checkpoints(self, epoch, current_score):
        filepath = self._save_checkpoint_file(epoch, current_score)
        if filepath:
            self.recent_checkpoints.append(filepath)
            if len(self.recent_checkpoints) > 3:
                file_to_delete = self.recent_checkpoints.pop(0)
                if file_to_delete.exists():
                    file_to_delete.unlink()


class DDPPatienceEarlyStopping(_DragonEarlyStopping):
    """
    DDP-safe Patience Early Stopping. 
    Executes stopping logic on all ranks identically, but only Rank 0 logs to the console.
    """
    def __init__(self, 
                 monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 min_delta: float = 0.0, 
                 patience: int = 10, 
                 mode: Literal['min', 'max'] = 'min', 
                 verbose: int = 1):
        
        # Standardize monitor key
        if monitor == "Training Loss":
            std_monitor = PyTorchLogKeys.TRAIN_LOSS
        elif monitor == "Validation Loss":
            std_monitor = PyTorchLogKeys.VAL_LOSS
        elif monitor == "both":
            std_monitor = "both"
        else:
            _LOGGER.error(f"Unknown monitor key: {monitor}.")
            raise ValueError()
        
        super().__init__(std_monitor, verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.mode = mode
        
        if self.mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
            
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_train_begin(self, logs=None):
        # Only reset the early stopping state if training from scratch
        if self.trainer.start_epoch <= 1: # type: ignore
            self.wait = 0
            self.best = np.inf if self.monitor_op == np.less else -np.inf
        
    def _get_metric_value(self, logs):
        """Extracts or calculates the metric value based on configuration."""
        if self.monitor == "both":
            t_loss = logs.get(PyTorchLogKeys.TRAIN_LOSS)
            v_loss = logs.get(PyTorchLogKeys.VAL_LOSS)
            if t_loss is None or v_loss is None:
                return None
            return t_loss + v_loss
        else:
            return logs.get(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = self._get_metric_value(logs)
        
        if current is None:
            return

        if self.monitor_op == np.less:
            is_improvement = self.monitor_op(current, self.best - self.min_delta)
        else:
            is_improvement = self.monitor_op(current, self.best + self.min_delta)

        if is_improvement:
            if self.verbose > 2 and self.trainer.is_main_process(): # type: ignore
                _LOGGER.info(f"EarlyStopping: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._stop_training(epoch, f"No improvement in {self.monitor} for {self.wait} epochs.")
                
    def _stop_training(self, epoch: int, reason: str):
        """Overrides base to restrict logging to Rank 0."""
        self.stopped_epoch = epoch
        self.trainer.stop_training = True # type: ignore
        if self.verbose > 0 and self.trainer.is_main_process(): # type: ignore
            _LOGGER.info(f"Epoch {epoch}: Early stopping triggered. Reason: {reason}")


class DDPPlateauScheduler(_DragonLRScheduler):
    """
    DDP-safe ReduceLROnPlateau scheduler.
    Steps the optimizer on all ranks to keep learning rates synchronized, but only Rank 0 logs changes.
    """
    def __init__(self, 
                 monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 mode: Literal['min', 'max'] = 'min', 
                 factor: float = 0.1, 
                 patience: int = 5, 
                 threshold: float = 1e-4, 
                 threshold_mode: Literal['rel', 'abs'] = 'rel', 
                 cooldown: int = 0, 
                 min_lr: float = 0, 
                 eps: float = 1e-8, 
                 verbose: int = 1):
        super().__init__()
        
        # Standardize monitor key
        if monitor == "Training Loss":
            self.monitor = PyTorchLogKeys.TRAIN_LOSS
        elif monitor == "Validation Loss":
            self.monitor = PyTorchLogKeys.VAL_LOSS
        elif monitor == "both":
            self.monitor = "both"
        else:
            _LOGGER.error(f"Unknown monitor key: {monitor}.")
            raise ValueError()
        
        self.verbose = verbose
        
        self.config = {
            'mode': mode, 'factor': factor, 'patience': patience,
            'threshold': threshold, 'threshold_mode': threshold_mode,
            'cooldown': cooldown, 'min_lr': min_lr, 'eps': eps,
        }

    def set_trainer(self, trainer):
        super().set_trainer(trainer)
        if self.verbose > 1 and self.trainer.is_main_process(): # type: ignore
            _LOGGER.info(f"Initializing ReduceLROnPlateau monitoring '{self.monitor}'")
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.trainer.optimizer, # type: ignore
            **self.config
        )
        self.trainer.scheduler = self.scheduler # type: ignore

    def _get_metric_value(self, logs):
        """Extracts or calculates the metric value based on configuration."""
        if self.monitor == "both":
            t_loss = logs.get(PyTorchLogKeys.TRAIN_LOSS)
            v_loss = logs.get(PyTorchLogKeys.VAL_LOSS)
            if t_loss is None or v_loss is None:
                return None
            return t_loss + v_loss
        else:
            return logs.get(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metric_val = self._get_metric_value(logs)
        inner_verbose = True if self.verbose >= 1 else False
        
        if metric_val is None:
            if self.trainer.is_main_process() and self.verbose >= 1: # type: ignore
                _LOGGER.warning(f"DDPPlateauScheduler could not find metric '{self.monitor}'. Step skipped.")
            self._check_and_log_lr(epoch, logs, inner_verbose)
            return

        # Execute step on all ranks to keep optimizers synced
        self.scheduler.step(metric_val)
        self._check_and_log_lr(epoch, logs, inner_verbose)
        
    def _check_and_log_lr(self, epoch, logs, verbose: bool):
        """Overrides base to restrict logging to Rank 0 while updating history."""
        if not self.trainer.optimizer: # type: ignore
            return

        current_lr = self.trainer.optimizer.param_groups[0]['lr'] # type: ignore

        if self.previous_lr is not None and current_lr != self.previous_lr:
            if verbose and self.trainer.is_main_process(): # type: ignore
                print(f"    > Epoch {epoch}: Learning rate changed to {current_lr:.6f}")
            self.previous_lr = current_lr
        
        logs[PyTorchLogKeys.LEARNING_RATE] = current_lr
