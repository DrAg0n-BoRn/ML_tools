import torch
from typing import Literal

from ..keys._keys import PyTorchLogKeys
from .._core import get_logger

from ._base import _Callback


_LOGGER = get_logger("LR Scheduler")


__all__ = [
    "DragonScheduler",
    "DragonPlateauScheduler"
]


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


class DragonPlateauScheduler(_DragonLRScheduler):
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

