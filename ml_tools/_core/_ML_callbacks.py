import numpy as np
import torch
from collections import deque
from tqdm.auto import tqdm
from typing import Union, Literal, Optional
from pathlib import Path

from ._path_manager import make_fullpath
from ._keys import PyTorchLogKeys, PyTorchCheckpointKeys
from ._logger import get_logger
from ._script_info import _script_info


_LOGGER = get_logger("Callbacks")


__all__ = [
    "History", 
    "TqdmProgressBar",
    "DragonPatienceEarlyStopping",
    "DragonPrecheltEarlyStopping",
    "DragonModelCheckpoint",
    "DragonScheduler",
    "DragonReduceLROnPlateau"
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


class _DragonEarlyStopping(_Callback):
    """
    Base class for Early Stopping strategies.
    Ensures type compatibility and shared logging logic.
    """
    def __init__(self, 
                 monitor: str, 
                 verbose: int = 1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.stopped_epoch = 0

    def _stop_training(self, epoch: int, reason: str):
        """Helper to trigger the stop."""
        self.stopped_epoch = epoch
        self.trainer.stop_training = True # type: ignore
        if self.verbose > 0:
            _LOGGER.info(f"Epoch {epoch}: Early stopping triggered. Reason: {reason}")


class DragonPatienceEarlyStopping(_DragonEarlyStopping):
    """
    Standard early stopping: Tracks minimum validation loss (or other metric) with a patience counter.
    """
    def __init__(self, 
                 monitor: Literal["Training Loss", "Validation Loss"] = "Validation Loss", 
                 min_delta: float = 0.0, 
                 patience: int = 10, 
                 mode: Literal['min', 'max'] = 'min', 
                 verbose: int = 1):
        """  
        Args:
            monitor (str): Metric to monitor.
            min_delta (float): Minimum change to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
            verbose (int): Verbosity mode.
        """
        # standardize monitor key
        if monitor == "Training Loss":
            std_monitor = PyTorchLogKeys.TRAIN_LOSS
        elif monitor == "Validation Loss":
            std_monitor =  PyTorchLogKeys.VAL_LOSS
        else:
            _LOGGER.error(f"Unknown monitor key: {monitor}.")
            raise ValueError()
        
        super().__init__(std_monitor, verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.mode = mode
        
        if mode not in ['min', 'max']:
            _LOGGER.error(f"EarlyStopping mode {mode} is unknown, choose one of ('min', 'max')")
            raise ValueError()

        # Determine the comparison operator
        if self.mode == 'min':
            self.monitor_op = np.less
        elif self.mode == 'max':
            self.monitor_op = np.greater
        else:
            # raise error for unknown mode
            _LOGGER.error(f"EarlyStopping mode {mode} is unknown, choose one of ('min', 'max')")
            raise ValueError()
        
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor) # type: ignore
        if current is None:
            return

        # Check improvement
        if self.monitor_op == np.less:
            is_improvement = self.monitor_op(current, self.best - self.min_delta)
        else:
            is_improvement = self.monitor_op(current, self.best + self.min_delta)

        if is_improvement:
            if self.verbose > 1:
                _LOGGER.info(f"EarlyStopping: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._stop_training(epoch, f"No improvement in {self.monitor} for {self.wait} epochs.")


class DragonPrecheltEarlyStopping(_DragonEarlyStopping):
    """
    Implements Prechelt's 'Progress-Modified GL' criterion.
    Tracks the ratio between Generalization Loss (overfitting) and Training Progress.
    
    References:
        Prechelt, L. (1998). Early Stopping - But When?
    """
    def __init__(self, 
                 alpha: float = 0.75, 
                 k: int = 5, 
                 verbose: int = 1):
        """
        This early stopping strategy monitors both validation loss and training loss to determine the optimal stopping point.
        
        Args:
            alpha (float): The threshold for the stopping criterion.
            k (int): The window size for calculating training progress.
            verbose (int): Verbosity mode.
            
        NOTE: 
        
        - **The Strip Size (k)**:
            - `5`: The empirical "gold standard." It is long enough to smooth out batch noise but short enough to react to convergence plateaus quickly.
            - `10` to `20`: Use if the training curve is very jagged (e.g., noisy data, small batch sizes, high dropout, or Reinforcement Learning). A larger k value prevents premature stopping due to random volatility.
        - **The threshold (alpha)**:
            - `< 0.5`: Aggressive. Stops training very early.
            - `0.75` to `0.80`: Prechelt found this range to be the most robust across different datasets. It typically yields the best trade-off between generalization and training cost.
            - `1.0` to `1.2`: Useful for complex tasks (like Transformers) where training progress might dip temporarily before recovering. It risks slightly more overfitting but ensures potential is exhausted.
        """
        super().__init__(PyTorchLogKeys.VAL_LOSS, verbose)
        self.train_monitor = PyTorchLogKeys.TRAIN_LOSS
        self.alpha = alpha
        self.k = k
        
        self.best_val_loss = np.inf
        self.train_strip = deque(maxlen=k)

    def on_train_begin(self, logs=None):
        self.best_val_loss = np.inf
        self.train_strip.clear()

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor) # type: ignore
        train_loss = logs.get(self.train_monitor) # type: ignore
        
        if val_loss is None or train_loss is None:
            return

        # 1. Update Best Validation Loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # 2. Update Training Strip
        self.train_strip.append(train_loss)

        # 3. Calculate Generalization Loss (GL)
        # GL(t) = 100 * (E_val / E_opt - 1)
        # Low GL is good. High GL means we are drifting away from best val score (overfitting).
        gl = 100 * ((val_loss / self.best_val_loss) - 1)

        # 4. Calculate Progress (Pk)
        # Pk(t) = 1000 * (Sum(strip) / (k * min(strip)) - 1)
        # High Pk is good (training loss is still dropping fast). Low Pk means training has stalled.
        if len(self.train_strip) < self.k:
            # Not enough data for progress yet
            return
            
        strip_sum = sum(self.train_strip)
        strip_min = min(self.train_strip)
        
        # Avoid division by zero
        if strip_min == 0:
            pk = 0.1 # Arbitrary small number
        else:
            pk = 1000 * ((strip_sum / (self.k * strip_min)) - 1)

        # 5. The Quotient Criterion
        # Stop if GL / Pk > alpha
        # Intuition: Stop if Overfitting is high AND Progress is low.
        
        # Avoid division by zero
        if pk == 0: 
            pk = 1e-6
            
        quotient = gl / pk
        
        if self.verbose > 1:
            _LOGGER.info(f"Epoch {epoch}: GL={gl:.3f} | Pk={pk:.3f} | Quotient={quotient:.3f} (Threshold={self.alpha})")

        if quotient > self.alpha:
            self._stop_training(epoch, f"Prechelt Criterion triggered. Generalization/Progress quotient ({quotient:.3f}) > alpha ({self.alpha}).")


class DragonModelCheckpoint(_Callback):
    """
    Saves the model weights, optimizer state, LR scheduler state (if any), and epoch number to a directory with automated filename generation and rotation. 
    """
    def __init__(self, 
                 save_dir: Union[str, Path], 
                 monitor: Literal["Training Loss", "Validation Loss", "both"] = "Validation Loss",
                 save_three_best: bool = True, 
                 mode: Literal['min', 'max'] = 'min', 
                 verbose: int = 0):
        """
        Args:
            save_dir (str): Directory where checkpoint files will be saved.
            monitor (str): Metric to monitor. If "both", the sum of training loss and validation loss is used.
            save_three_best (bool): 
                - If True, keeps the top 3 best checkpoints found during training (based on metric).
                - If False, keeps the 3 most recent checkpoints (rolling window).
            mode (str): One of {'min', 'max'}.
            verbose (int): Verbosity mode.
        """
        super().__init__()
        self.save_dir = make_fullpath(save_dir, make=True, enforce="directory")
        
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
        
        self.monitor = std_monitor
        self.save_three_best = save_three_best
        self.verbose = verbose
        self._latest_checkpoint_path = None
        self._checkpoint_name = PyTorchCheckpointKeys.CHECKPOINT_NAME

        # State variables
        # stored as list of dicts: [{'path': Path, 'score': float, 'epoch': int}]
        self.best_checkpoints = [] 
        # For rolling check (save_three_best=False)
        self.recent_checkpoints = []

        if mode not in ['min', 'max']:
            _LOGGER.error(f"ModelCheckpoint mode {mode} is unknown. Use 'min' or 'max'.")
            raise ValueError()
        self.mode = mode

        # Determine comparison operator
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf

    def on_train_begin(self, logs=None):
        """Reset file tracking state when training starts.
        NOTE: Do nOT reset self.best here if it differs from the default. This allows the Trainer to restore 'best' from a checkpoint before calling train()."""
        self.best_checkpoints = []
        self.recent_checkpoints = []
        
        # Check if self.best is at default initialization value
        is_default_min = (self.mode == 'min' and self.best == np.inf)
        is_default_max = (self.mode == 'max' and self.best == -np.inf)
        
        # If it is NOT default, it means it was restored.
        if not (is_default_min or is_default_max):
            _LOGGER.debug(f"Resuming with best score: {self.best:.4f}")

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
        current_score = self._get_metric_value(logs)

        if current_score is None:
            if self.verbose > 0:
                _LOGGER.warning(f"Epoch {epoch}: Metric '{self.monitor}' not found in logs. Skipping checkpoint.")
            return
        
        # 1. Update global best score (for logging/metadata)
        if self.monitor_op(current_score, self.best):
            if self.verbose > 0:
                 # Only log explicit "improvement" if we are beating the historical best
                 old_best_str = f"{self.best:.4f}" if not np.isinf(self.best) else "inf"
                 _LOGGER.info(f"Epoch {epoch}: {self.monitor} improved from {old_best_str} to {current_score:.4f}")
            self.best = current_score

        if self.save_three_best:
            self._save_top_k_checkpoints(epoch, current_score)
        else:
            self._save_rolling_checkpoints(epoch, current_score)

    def _save_checkpoint_file(self, epoch, current_score):
        """Helper to physically save the file."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        score_str = f"{current_score:.4f}".replace('.', '_')
        filename = f"epoch{epoch}_{self._checkpoint_name}-{score_str}.pth"
        filepath = self.save_dir / filename

        # Create checkpoint dict
        checkpoint_data = {
            PyTorchCheckpointKeys.EPOCH: epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.trainer.model.state_dict(), # type: ignore
            PyTorchCheckpointKeys.OPTIMIZER_STATE: self.trainer.optimizer.state_dict(), # type: ignore
            PyTorchCheckpointKeys.BEST_SCORE: current_score,
            PyTorchCheckpointKeys.HISTORY: self.trainer.history, # type: ignore
        }
        
        if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None: # type: ignore
            checkpoint_data[PyTorchCheckpointKeys.SCHEDULER_STATE] = self.trainer.scheduler.state_dict() # type: ignore
        
        torch.save(checkpoint_data, filepath)
        self._latest_checkpoint_path = filepath
        
        return filepath

    def _save_top_k_checkpoints(self, epoch, current_score):
        """Logic for maintaining the top 3 best checkpoints."""
        
        def sort_key(item): return item['score']
        
        # Determine sort direction so that Index 0 is BEST and Index -1 is WORST
        # Min mode (lower is better): Ascending (reverse=False) -> [0.1, 0.5, 0.9] (0.1 is best)
        # Max mode (higher is better): Descending (reverse=True) -> [0.9, 0.5, 0.1] (0.9 is best)
        is_reverse = (self.mode == 'max')

        should_save = False
        
        if len(self.best_checkpoints) < 3:
            should_save = True
        else:
            # Sort current list to identify the worst (last item)
            self.best_checkpoints.sort(key=sort_key, reverse=is_reverse)
            worst_entry = self.best_checkpoints[-1]
            
            # Check if current is better than the worst in the list
            # min mode: current < worst['score']
            # max mode: current > worst['score']
            if self.monitor_op(current_score, worst_entry['score']):
                should_save = True

        if should_save:
            filepath = self._save_checkpoint_file(epoch, current_score)
            
            if self.verbose > 0:
                _LOGGER.info(f"Epoch {epoch}: {self.monitor} ({current_score:.4f}) is in top 3. Saving to {filepath.name}")

            self.best_checkpoints.append({'path': filepath, 'score': current_score, 'epoch': epoch})
            
            # Prune if > 3
            if len(self.best_checkpoints) > 3:
                # Re-sort to ensure worst is at the end
                self.best_checkpoints.sort(key=sort_key, reverse=is_reverse)
                
                # Evict the last one (Worst)
                entry_to_delete = self.best_checkpoints.pop(-1)

                if entry_to_delete['path'].exists():
                    if self.verbose > 0:
                        _LOGGER.info(f"  -> Deleting checkpoint outside top 3: {entry_to_delete['path'].name}")
                    entry_to_delete['path'].unlink()

    def _save_rolling_checkpoints(self, epoch, current_score):
        """Saves the latest model and keeps only the 3 most recent ones."""
        filepath = self._save_checkpoint_file(epoch, current_score)
        
        if self.verbose > 0:
            _LOGGER.info(f'Epoch {epoch}: saving rolling model to {filepath.name}')

        self.recent_checkpoints.append(filepath)

        # If we have more than 3 checkpoints, remove the oldest one
        if len(self.recent_checkpoints) > 3:
            file_to_delete = self.recent_checkpoints.pop(0)
            if file_to_delete.exists():
                if self.verbose > 0:
                    _LOGGER.info(f"  -> Deleting old rolling checkpoint: {file_to_delete.name}")
                file_to_delete.unlink()

    @property
    def best_checkpoint_path(self):
        # If tracking top 3, return the absolute best among them
        if self.save_three_best and self.best_checkpoints:
            def sort_key(item): return item['score']
            is_reverse = (self.mode == 'max')
            # Sort Best -> Worst
            sorted_bests = sorted(self.best_checkpoints, key=sort_key, reverse=is_reverse)
            # Index 0 is always the best based on the logic above
            return sorted_bests[0]['path']
        
        elif self._latest_checkpoint_path:
            return self._latest_checkpoint_path
        else:
            _LOGGER.error("No checkpoint paths saved.")
            raise ValueError()


class _DragonLRScheduler(_Callback):
    """
    Base class for Dragon LR Schedulers. 
    Handles common logic like logging and attaching to the trainer.
    """
    def __init__(self):
        super().__init__()
        self.scheduler = None
        self.previous_lr = None

    def set_trainer(self, trainer):
        """Associates the callback with the trainer."""
        super().set_trainer(trainer)
        # Note: Subclasses must ensure self.scheduler is set before or during this call
        # if they want to register it immediately.
        if self.scheduler:
            self.trainer.scheduler = self.scheduler # type: ignore

    def on_train_begin(self, logs=None):
        """Store the initial learning rate."""
        if not self.trainer.optimizer: # type: ignore
            _LOGGER.warning("No optimizer found in trainer. LRScheduler cannot track learning rate.")
            return
        self.previous_lr = self.trainer.optimizer.param_groups[0]['lr'] # type: ignore

    def _check_and_log_lr(self, epoch, logs, verbose: bool):
        """Helper to log LR changes and update history."""
        if not self.trainer.optimizer: # type: ignore
            return

        current_lr = self.trainer.optimizer.param_groups[0]['lr'] # type: ignore

        # Log change
        if self.previous_lr is not None and current_lr != self.previous_lr:
            if verbose:
                print(f"    > Epoch {epoch}: Learning rate changed to {current_lr:.6f}")
            self.previous_lr = current_lr
        
        # Log to dictionary
        logs[PyTorchLogKeys.LEARNING_RATE] = current_lr
        
        # Log to history
        if hasattr(self.trainer, 'history'):
            self.trainer.history.setdefault(PyTorchLogKeys.LEARNING_RATE, []).append(current_lr) # type: ignore


class DragonScheduler(_DragonLRScheduler):
    """
    Callback for standard PyTorch Learning Rate Schedulers.
    
    Compatible with: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, etc.
    
    NOT Compatible with: ReduceLROnPlateau (Use `DragonReduceLROnPlateau` instead).
    """
    def __init__(self, scheduler, verbose: bool=True):
        """
        Args:
            scheduler: An initialized PyTorch learning rate scheduler instance.
            verbose (bool): If True, logs learning rate changes to console.
        """
        super().__init__()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError(
                "DragonLRScheduler does not support 'ReduceLROnPlateau'. "
                "Please use the `DragonReduceLROnPlateau` callback instead."
            )
        self.scheduler = scheduler
        self.verbose = verbose
        
    def set_trainer(self, trainer):
        super().set_trainer(trainer)
        # Explicitly register the scheduler again to be safe
        self.trainer.scheduler = self.scheduler # type: ignore
        if self.verbose:
            _LOGGER.info(f"Registered LR Scheduler: {self.scheduler.__class__.__name__}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Standard step (no metrics needed)
        self.scheduler.step()
        
        self._check_and_log_lr(epoch, logs, self.verbose)


class DragonReduceLROnPlateau(_DragonLRScheduler):
    """
    Specific callback for `torch.optim.lr_scheduler.ReduceLROnPlateau`. Reduces learning rate when a monitored metric has stopped improving.
    
    This wrapper initializes the scheduler internally using the Trainer's optimizer, simplifying the setup process.
    """
    def __init__(self, 
                 monitor: Literal["Training Loss", "Validation Loss"] = "Validation Loss",
                 mode: Literal['min', 'max'] = 'min', 
                 factor: float = 0.1, 
                 patience: int = 5, 
                 threshold: float = 1e-4, 
                 threshold_mode: Literal['rel', 'abs'] = 'rel', 
                 cooldown: int = 0, 
                 min_lr: float = 0, 
                 eps: float = 1e-8, 
                 verbose: bool = True):
        """
        Args:
            monitor ("Training Loss", "Validation Loss"): Metric to monitor.
            mode ('min', 'max'): One of 'min', 'max'.
            factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            threshold (float): Threshold for measuring the new optimum.
            threshold_mode ('rel', 'abs'): One of 'rel', 'abs'.
            cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
            min_lr (float or list): A scalar or a list of scalars.
            eps (float): Minimal decay applied to lr.
            verbose (bool): If True, logs learning rate changes to console.
        """
        super().__init__()
        
        # Standardize monitor key
        if monitor == "Training Loss":
            std_monitor = PyTorchLogKeys.TRAIN_LOSS
        elif monitor == "Validation Loss":
            std_monitor = PyTorchLogKeys.VAL_LOSS
        else:
            _LOGGER.error(f"Unknown monitor key: {monitor}.")
            raise ValueError()
        
        self.monitor = std_monitor
        self.verbose = verbose
        
        # Config storage for delayed initialization
        self.config = {
            'mode': mode,
            'factor': factor,
            'patience': patience,
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'cooldown': cooldown,
            'min_lr': min_lr,
            'eps': eps,
        }

    def set_trainer(self, trainer):
        """
        Initializes the ReduceLROnPlateau scheduler using the trainer's optimizer and registers it.
        """
        super().set_trainer(trainer)
        
        if not hasattr(self.trainer, 'optimizer'):
            _LOGGER.error("Trainer has no optimizer. Cannot initialize ReduceLROnPlateau.")
            raise ValueError()
            
        # Initialize the actual scheduler with the optimizer
        if self.verbose:
            _LOGGER.info(f"Initializing ReduceLROnPlateau monitoring '{self.monitor}'")
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.trainer.optimizer, # type: ignore
            **self.config
        )
        
        # Register with trainer for checkpointing
        self.trainer.scheduler = self.scheduler # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        metric_val = logs.get(self.monitor)
        
        if metric_val is None:
            _LOGGER.warning(f"DragonReduceLROnPlateau could not find metric '{self.monitor}' in logs. Scheduler step skipped.")
            # Still log LR to keep history consistent
            self._check_and_log_lr(epoch, logs, self.verbose)
            return

        # Step with metric
        self.scheduler.step(metric_val)
        
        self._check_and_log_lr(epoch, logs, self.verbose)


def info():
    _script_info(__all__)
