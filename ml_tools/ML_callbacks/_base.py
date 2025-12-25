from tqdm.auto import tqdm

from ..keys._keys import PyTorchLogKeys


__all__ = [
    "_Callback",
    "History", 
    "TqdmProgressBar",
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

