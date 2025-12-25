import numpy as np
from collections import deque
from typing import Literal

from ..keys._keys import PyTorchLogKeys
from .._core import get_logger

from ._base import _Callback


_LOGGER = get_logger("EarlyStopping")


__all__ = [
    "DragonPatienceEarlyStopping",
    "DragonPrecheltEarlyStopping",
]


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
                 window_size: int = 5, 
                 verbose: int = 1):
        """
        This early stopping strategy monitors both validation loss and training loss to determine the optimal stopping point.
        
        Args:
            alpha (float): The threshold for the stopping criterion.
            window_size (int): The window size for calculating training progress.
            verbose (int): Verbosity mode.
            
        NOTE: 
        
        - **The Window Size (k)**:
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
        self.k = window_size
        
        self.best_val_loss = np.inf
        self.train_strip = deque(maxlen=window_size)

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

