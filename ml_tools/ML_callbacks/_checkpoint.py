import numpy as np
import torch
from typing import Union, Literal
from pathlib import Path

from ..path_manager import make_fullpath
from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys
from .._core import get_logger

from ._base import _Callback


_LOGGER = get_logger("Checkpoint")


__all__ = [
    "DragonModelCheckpoint",
]


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

