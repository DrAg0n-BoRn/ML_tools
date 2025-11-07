import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Union, Literal, Optional
from pathlib import Path

from .path_manager import make_fullpath
from ._keys import PyTorchLogKeys, PyTorchCheckpointKeys
from ._logger import _LOGGER
from ._script_info import _script_info


__all__ = [
    "History", 
    "TqdmProgressBar",
    "DragonEarlyStopping",  
    "DragonModelCheckpoint",
    "DragonLRScheduler"
]


class _Callback:
    """
    Abstract base class used to build new callbacks.
    
    The methods of this class are automatically called by the Trainer at different
    points during training. Subclasses can override these methods to implement
    custom logic.
    """
    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer):
        """This is called by the Trainer to associate itself with the callback."""
        self.trainer = trainer

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        pass

    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch."""
        pass

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        pass


class History(_Callback):
    """
    Callback that records events into a `history` dictionary.
    
    This callback is automatically applied to every MyTrainer model.
    The `history` attribute is a dictionary mapping metric names (e.g., 'val_loss')
    to a list of metric values.
    """
    def on_train_begin(self, logs=None):
        # Clear history at the beginning of training
        self.trainer.history = {} # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            # Append new log values to the history dictionary
            self.trainer.history.setdefault(k, []).append(v) # type: ignore


class TqdmProgressBar(_Callback):
    """Callback that provides a tqdm progress bar for training."""
    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        self.epochs = self.trainer.epochs # type: ignore
        self.epoch_bar = tqdm(total=self.epochs, desc="Training Progress")

    def on_epoch_begin(self, epoch, logs=None):
        total_batches = len(self.trainer.train_loader) # type: ignore
        self.batch_bar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{self.epochs}", leave=False)

    def on_batch_end(self, batch, logs=None):
        self.batch_bar.update(1) # type: ignore
        if logs:
            self.batch_bar.set_postfix(loss=f"{logs.get(PyTorchLogKeys.BATCH_LOSS, 0):.4f}") # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        self.batch_bar.close() # type: ignore
        self.epoch_bar.update(1) # type: ignore
        if logs:
            train_loss_str = f"{logs.get(PyTorchLogKeys.TRAIN_LOSS, 0):.4f}"
            val_loss_str = f"{logs.get(PyTorchLogKeys.VAL_LOSS, 0):.4f}"
            self.epoch_bar.set_postfix_str(f"Train Loss: {train_loss_str}, Val Loss: {val_loss_str}") # type: ignore

    def on_train_end(self, logs=None):
        self.epoch_bar.close() # type: ignore


class DragonEarlyStopping(_Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(self, monitor: str=PyTorchLogKeys.VAL_LOSS, min_delta: float=0.0, patience: int=5, mode: Literal['auto', 'min', 'max']='auto', verbose: int=1):
        """
        Args:
            monitor (str): Quantity to be monitored. Defaults to 'val_loss'.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity
                        monitored has stopped decreasing; in 'max' mode it will stop when the quantity
                        monitored has stopped increasing; in 'auto' mode, the direction is automatically
                        inferred from the name of the monitored quantity.
            verbose (int): Verbosity mode.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.verbose = verbose
        
        if mode not in ['auto', 'min', 'max']:
            _LOGGER.error(f"EarlyStopping mode {mode} is unknown, choose one of ('auto', 'min', 'max')")
            raise ValueError()
        self.mode = mode

        # Determine the comparison operator based on the mode
        if self.mode == 'min':
            self.monitor_op = np.less
        elif self.mode == 'max':
            self.monitor_op = np.greater
        else: # auto mode
            if 'acc' in self.monitor.lower():
                self.monitor_op = np.greater
            else: # Default to min mode for loss or other metrics
                self.monitor_op = np.less
        
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_train_begin(self, logs=None):
        # Reset state at the beginning of training
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
                    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor) # type: ignore
        if current is None:
            return

        # Determine the comparison threshold based on the mode
        if self.monitor_op == np.less:
            # For 'min' mode, we need to be smaller than 'best' by at least 'min_delta'
            # Correct check: current < self.best - self.min_delta
            is_improvement = self.monitor_op(current, self.best - self.min_delta)
        else:
            # For 'max' mode, we need to be greater than 'best' by at least 'min_delta'
            # Correct check: current > self.best + self.min_delta
            is_improvement = self.monitor_op(current, self.best + self.min_delta)

        if is_improvement:
            if self.verbose > 1:
                _LOGGER.info(f"EarlyStopping: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.stop_training = True # type: ignore
                if self.verbose > 0:
                    _LOGGER.info(f"Epoch {epoch+1}: early stopping after {self.wait} epochs with no improvement.")


class DragonModelCheckpoint(_Callback):
    """
    Saves the model weights, optimizer state, LR scheduler state (if any), and epoch number to a directory with automated filename generation and rotation. 
    """
    def __init__(self, save_dir: Union[str,Path], monitor: str = PyTorchLogKeys.VAL_LOSS,
                 save_best_only: bool = True, mode: Literal['auto', 'min', 'max']= 'auto', verbose: int = 0):
        """
        - If `save_best_only` is True, it saves the single best model, deleting the previous best. 
        - If `save_best_only` is False, it keeps the 3 most recent checkpoints, deleting the oldest ones automatically.

        Args:
            save_dir (str): Directory where checkpoint files will be saved.
            monitor (str): Metric to monitor.
            save_best_only (bool): If true, save only the best model.
            mode (str): One of {'auto', 'min', 'max'}.
            verbose (int): Verbosity mode.
        """
        
        super().__init__()
        self.save_dir = make_fullpath(save_dir, make=True, enforce="directory")
        if not self.save_dir.is_dir():
            _LOGGER.error(f"{save_dir} is not a valid directory.")
            raise IOError()
        
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self._latest_checkpoint_path = None
        self._checkpoint_name = PyTorchCheckpointKeys.CHECKPOINT_NAME

        # State variables to be managed during training
        self.saved_checkpoints = []
        self.last_best_filepath = None

        if mode not in ['auto', 'min', 'max']:
            _LOGGER.error(f"ModelCheckpoint mode {mode} is unknown.")
            raise ValueError()
        self.mode = mode

        if self.mode == 'min':
            self.monitor_op = np.less
        elif self.mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less if 'loss' in self.monitor else np.greater
        
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_train_begin(self, logs=None):
        """Reset state when training starts."""
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.saved_checkpoints = []
        self.last_best_filepath = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.save_best_only:
            self._save_best_model(epoch, logs)
        else:
            self._save_rolling_checkpoints(epoch, logs)

    def _save_best_model(self, epoch, logs):
        """Saves a single best model and deletes the previous one."""
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            old_best_str = f"{self.best:.4f}" if self.best not in [np.inf, -np.inf] else "inf"
            
            # Create a descriptive filename
            filename = f"epoch{epoch}_{self._checkpoint_name}_{current:.4f}.pth"
            new_filepath = self.save_dir / filename
            
            if self.verbose > 0:
                _LOGGER.info(f"Epoch {epoch}: {self.monitor} improved from {old_best_str} to {current:.4f}, saving model to {new_filepath}")
            
            # Update best score *before* saving
            self.best = current

            # Create a comprehensive checkpoint dictionary
            checkpoint_data = {
                PyTorchCheckpointKeys.EPOCH: epoch,
                PyTorchCheckpointKeys.MODEL_STATE: self.trainer.model.state_dict(), # type: ignore
                PyTorchCheckpointKeys.OPTIMIZER_STATE: self.trainer.optimizer.state_dict(), # type: ignore
                PyTorchCheckpointKeys.BEST_SCORE: self.best, 
                PyTorchCheckpointKeys.HISTORY: self.trainer.history, # type: ignore
            }
            
            # Check for scheduler
            if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None: # type: ignore
                checkpoint_data[PyTorchCheckpointKeys.SCHEDULER_STATE] = self.trainer.scheduler.state_dict() # type: ignore
            
            # Save the new best model
            torch.save(checkpoint_data, new_filepath)
            self._latest_checkpoint_path = new_filepath

            # Delete the old best model file
            if self.last_best_filepath and self.last_best_filepath.exists():
                self.last_best_filepath.unlink()
            
            # Update state
            self.last_best_filepath = new_filepath

    def _save_rolling_checkpoints(self, epoch, logs):
        """Saves the latest model and keeps only the most recent ones."""
        current = logs.get(self.monitor)

        filename = f"epoch{epoch}_{self._checkpoint_name}_{current:.4f}.pth"
        filepath = self.save_dir / filename
        
        if self.verbose > 0:
            _LOGGER.info(f'Epoch {epoch}: saving model to {filepath}')

        # Create a comprehensive checkpoint dictionary
        checkpoint_data = {
            PyTorchCheckpointKeys.EPOCH: epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.trainer.model.state_dict(), # type: ignore
            PyTorchCheckpointKeys.OPTIMIZER_STATE: self.trainer.optimizer.state_dict(), # type: ignore
            PyTorchCheckpointKeys.BEST_SCORE: self.best, # Save the current best score
            PyTorchCheckpointKeys.HISTORY: self.trainer.history, # type: ignore
        }
        
        if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None: # type: ignore
            checkpoint_data[PyTorchCheckpointKeys.SCHEDULER_STATE] = self.trainer.scheduler.state_dict() # type: ignore
        
        torch.save(checkpoint_data, filepath)
        
        self._latest_checkpoint_path = filepath

        self.saved_checkpoints.append(filepath)

        # If we have more than n checkpoints, remove the oldest one
        if len(self.saved_checkpoints) > 3:
            file_to_delete = self.saved_checkpoints.pop(0)
            if file_to_delete.exists():
                if self.verbose > 0:
                    _LOGGER.info(f"  -> Deleting old checkpoint: {file_to_delete.name}")
                file_to_delete.unlink()

    @property
    def best_checkpoint_path(self):
        if self._latest_checkpoint_path:
            return self._latest_checkpoint_path
        else:
            _LOGGER.error("No checkpoint paths saved.")
            raise ValueError()


class DragonLRScheduler(_Callback):
    """
    Callback to manage a PyTorch learning rate scheduler.
    """
    def __init__(self, scheduler, monitor: Optional[str] = PyTorchLogKeys.VAL_LOSS):
        """
        This callback automatically calls the scheduler's `step()` method at the
        end of each epoch. It also logs a message when the learning rate changes.

        Args:
            scheduler: An initialized PyTorch learning rate scheduler.
            monitor (str): The metric to monitor for schedulers that require it, like `ReduceLROnPlateau`. Should match a key in the logs (e.g., 'val_loss').
        """
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.previous_lr = None
        
    def set_trainer(self, trainer):
        """This is called by the Trainer to associate itself with the callback."""
        super().set_trainer(trainer)
        # Register the scheduler with the trainer so it can be added to the checkpoint
        self.trainer.scheduler = self.scheduler # type: ignore

    def on_train_begin(self, logs=None):
        """Store the initial learning rate."""
        self.previous_lr = self.trainer.optimizer.param_groups[0]['lr'] # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        """Step the scheduler and log any change in learning rate."""
        logs = logs or {}
        
        # For schedulers that need a metric (e.g., val_loss)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.monitor is None:
                _LOGGER.error("LRScheduler needs a `monitor` metric for ReduceLROnPlateau.")
                raise ValueError()
            
            metric_val = logs.get(self.monitor) # type: ignore
            if metric_val is not None:
                self.scheduler.step(metric_val)
            else:
                _LOGGER.warning(f"LRScheduler could not find metric '{self.monitor}' in logs.")
        
        # For all other schedulers
        else:
            self.scheduler.step()
            
        # Get the current learning rate
        current_lr = self.trainer.optimizer.param_groups[0]['lr'] # type: ignore

        # Log the change if the LR was updated
        if current_lr != self.previous_lr:
            _LOGGER.info(f"Epoch {epoch}: Learning rate changed to {current_lr:.6f}")
            self.previous_lr = current_lr
        
        # --- Add LR to logs and history ---
        # Add to the logs dict for any subsequent callbacks
        logs[PyTorchLogKeys.LEARNING_RATE] = current_lr
        
        # Also add directly to the trainer's history dict
        if hasattr(self.trainer, 'history'):
            self.trainer.history.setdefault(PyTorchLogKeys.LEARNING_RATE, []).append(current_lr) # type: ignore


def info():
    _script_info(__all__)
