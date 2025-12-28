from typing import Literal, Union, Optional, Any
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from abc import ABC, abstractmethod

from ..ML_callbacks._base import _Callback, History, TqdmProgressBar
from ..ML_callbacks._checkpoint import DragonModelCheckpoint
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_evaluation import plot_losses
from ..ML_utilities import inspect_pth_file

from ..path_manager import make_fullpath
from ..keys._keys import PyTorchCheckpointKeys, MagicWords
from .._core import get_logger


_LOGGER = get_logger("DragonTrainer")


__all__ = [
    "_BaseDragonTrainer",
]


class _BaseDragonTrainer(ABC):
    """
    Abstract base class for Dragon Trainers.
    
    Handles the common training loop orchestration, checkpointing, callback
    management, and device handling. Subclasses must implement the
    task-specific logic (dataloaders, train/val steps, evaluation).
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 dataloader_workers: int = 2,
                 checkpoint_callback: Optional[DragonModelCheckpoint] = None,
                 early_stopping_callback: Optional[_DragonEarlyStopping] = None,
                 lr_scheduler_callback: Optional[_DragonLRScheduler] = None,
                 extra_callbacks: Optional[list[_Callback]] = None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.device = self._validate_device(device)
        self.dataloader_workers = dataloader_workers
        
        # Callback handler
        default_callbacks = [History(), TqdmProgressBar()]
        
        self._checkpoint_callback = None
        if checkpoint_callback:
            default_callbacks.append(checkpoint_callback)
            self._checkpoint_callback = checkpoint_callback
        if early_stopping_callback:
            default_callbacks.append(early_stopping_callback)
        if lr_scheduler_callback:
            default_callbacks.append(lr_scheduler_callback)
        
        user_callbacks = extra_callbacks if extra_callbacks is not None else []
        self.callbacks = default_callbacks + user_callbacks
        self._set_trainer_on_callbacks()

        # Internal state
        self.train_loader: Optional[DataLoader] = None
        self.validation_loader: Optional[DataLoader] = None 
        self.history: dict[str, list[Any]] = {}
        self.epoch = 0
        self.epochs = 0 # Total epochs for the fit run
        self.start_epoch = 1
        self.stop_training = False
        self._batch_size = 10

    def _validate_device(self, device: str) -> torch.device:
        """Validates the selected device and returns a torch.device object."""
        device_lower = device.lower()
        if "cuda" in device_lower and not torch.cuda.is_available():
            _LOGGER.warning("CUDA not available, switching to CPU.")
            device = "cpu"
        elif device_lower == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
            device = "cpu"
        return torch.device(device)

    def _set_trainer_on_callbacks(self):
        """Gives each callback a reference to this trainer instance."""
        for callback in self.callbacks:
            callback.set_trainer(self)
    
    def _make_dataloaders(self, 
                          train_dataset: Any, 
                          validation_dataset: Any, 
                          batch_size: int, 
                          shuffle: bool,
                          collate_fn: Optional[Any] = None):
        """
        Shared logic to initialize standard DataLoaders. 
        Subclasses can call this inside their _create_dataloaders implementation.
        """
        # Ensure stability on MPS devices by setting num_workers to 0
        loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
        pin_memory = ("cuda" in self.device.type)
        
        self.train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=loader_workers, 
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        self.validation_loader = DataLoader(
            dataset=validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    def _validate_checkpoint_arg(self, model_checkpoint: Union[Path, str]) -> Union[Path, str]:
        """Validates the model_checkpoint argument."""
        if isinstance(model_checkpoint, Path):
            return make_fullpath(model_checkpoint, enforce="file")
        elif model_checkpoint in [MagicWords.BEST, MagicWords.CURRENT]:
            return model_checkpoint
        else:
            _LOGGER.error(f"'model_checkpoint' must be a Path object, or the string '{MagicWords.BEST}', or the string '{MagicWords.CURRENT}'.")
            raise ValueError()

    def _validate_save_dir(self, save_dir: Union[str, Path]) -> Path:
        """Validates and creates the save directory."""
        return make_fullpath(save_dir, make=True, enforce="directory")

    def _prepare_eval_data(self, 
                           data: Optional[Union[DataLoader, Dataset]], 
                           default_dataset: Optional[Dataset], 
                           collate_fn: Optional[Any] = None) -> tuple[DataLoader, Any]:
        """
        Prepares the DataLoader and dataset artifact source for evaluation.
        
        Returns:
            (eval_loader, dataset_for_artifacts)
        """
        eval_loader = None
        dataset_for_artifacts = None
        
        # Loader workers config
        loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
        pin_memory = (self.device.type == "cuda")

        if isinstance(data, DataLoader):
            eval_loader = data
            if hasattr(data, 'dataset'):
                dataset_for_artifacts = data.dataset
        elif isinstance(data, Dataset):
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=loader_workers,
                                     pin_memory=pin_memory,
                                     collate_fn=collate_fn)
            dataset_for_artifacts = data
        else: # data is None
            if default_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no validation dataset available in the trainer.")
                raise ValueError()
            
            eval_loader = DataLoader(default_dataset, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=loader_workers,
                                     pin_memory=pin_memory,
                                     collate_fn=collate_fn)
            dataset_for_artifacts = default_dataset

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()
            
        return eval_loader, dataset_for_artifacts

    def _save_finalized_artifact(self, 
                                 finalized_data: dict, 
                                 save_dir: Union[str, Path], 
                                 filename: str):
        """
        Handles the common logic for saving the finalized model dictionary to disk.
        """
        # handle save path
        dir_path = self._validate_save_dir(save_dir)
        full_path = dir_path / filename
        
        # checkpoint loading happens before dict creation. 
        
        torch.save(finalized_data, full_path)
        
        _LOGGER.info(f"Finalized model file saved to '{full_path}'")
        
        if full_path.is_file():
            inspect_pth_file(pth_path=full_path, save_dir=dir_path, verbose=2)
    
    def _load_checkpoint(self, path: Union[str, Path], verbose: int = 3):
        """Loads a training checkpoint to resume training."""
        p = make_fullpath(path, enforce="file")
        
        if verbose >= 2:
            _LOGGER.info(f"Loading checkpoint from '{p.name}'...")
        
        try:
            checkpoint = torch.load(p, map_location=self.device)
            
            if PyTorchCheckpointKeys.MODEL_STATE not in checkpoint or PyTorchCheckpointKeys.OPTIMIZER_STATE not in checkpoint:
                _LOGGER.error(f"Checkpoint file '{p.name}' is invalid. Missing 'model_state_dict' or 'optimizer_state_dict'.")
                raise KeyError()

            self.model.load_state_dict(checkpoint[PyTorchCheckpointKeys.MODEL_STATE])
            self.optimizer.load_state_dict(checkpoint[PyTorchCheckpointKeys.OPTIMIZER_STATE])
            self.epoch = checkpoint.get(PyTorchCheckpointKeys.EPOCH, 0)
            self.start_epoch = self.epoch + 1 # Resume on the *next* epoch
            
            # --- Load History ---
            if PyTorchCheckpointKeys.HISTORY in checkpoint:
                self.history = checkpoint[PyTorchCheckpointKeys.HISTORY]
                if verbose >= 3:
                    _LOGGER.info(f"Restored training history up to epoch {self.epoch}.")
            else:
                if verbose >= 1:
                    _LOGGER.warning("No 'history' found in checkpoint. A new history will be started.")
                self.history = {} # Ensure it's at least an empty dict
            
            # --- Scheduler State Loading Logic ---
            scheduler_state_exists = PyTorchCheckpointKeys.SCHEDULER_STATE in checkpoint
            scheduler_object_exists = self.scheduler is not None

            if scheduler_object_exists and scheduler_state_exists:
                # Case 1: Both exist. Attempt to load.
                try:
                    self.scheduler.load_state_dict(checkpoint[PyTorchCheckpointKeys.SCHEDULER_STATE]) # type: ignore
                    scheduler_name = self.scheduler.__class__.__name__
                    if verbose >= 3:
                        _LOGGER.info(f"Restored LR scheduler state for: {scheduler_name}")
                except Exception as e:
                    # Loading failed, likely a mismatch
                    scheduler_name = self.scheduler.__class__.__name__
                    _LOGGER.error(f"Failed to load scheduler state for '{scheduler_name}'. A different scheduler type might have been used.")
                    raise e

            elif scheduler_object_exists and not scheduler_state_exists:
                # Case 2: Scheduler provided, but no state in checkpoint.
                scheduler_name = self.scheduler.__class__.__name__
                if verbose >= 1:
                    _LOGGER.warning(f"'{scheduler_name}' was provided, but no scheduler state was found in the checkpoint. The scheduler will start from its initial state.")
            
            elif not scheduler_object_exists and scheduler_state_exists:
                # Case 3: State in checkpoint, but no scheduler provided.
                _LOGGER.error("Checkpoint contains an LR scheduler state, but no LRScheduler callback was provided.")
                raise ValueError()
            
            # Restore callback states
            for cb in self.callbacks:
                if isinstance(cb, DragonModelCheckpoint) and PyTorchCheckpointKeys.BEST_SCORE in checkpoint:
                    cb.best = checkpoint[PyTorchCheckpointKeys.BEST_SCORE]
                    if verbose >= 3:
                        _LOGGER.info(f"Restored {cb.__class__.__name__} 'best' score to: {cb.best:.4f}")
            
            if verbose >= 2:
                _LOGGER.info(f"Model restored to epoch {self.epoch}.")
            
        except Exception as e:
            _LOGGER.error(f"Failed to load checkpoint from '{p}': {e}")
            raise

    def fit(self, 
            save_dir: Union[str,Path],
            epochs: int = 100, 
            batch_size: int = 10, 
            shuffle: bool = True,
            resume_from_checkpoint: Optional[Union[str, Path]] = None):
        """
        Starts the training-validation process of the model.
        
        Returns the "History" callback dictionary.

        Args:
            save_dir (str | Path): Directory to save the loss plot.
            epochs (int): The total number of epochs to train for.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the training data at each epoch.
            resume_from_checkpoint (str | Path | None): Optional path to a checkpoint to resume training.
        """
        self.epochs = epochs
        self._batch_size = batch_size
        self._create_dataloaders(self._batch_size, shuffle) # type: ignore
        self.model.to(self.device)
        
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Reset stop_training flag on the trainer
        self.stop_training = False

        self._callbacks_hook('on_train_begin')
        
        if not self.train_loader:
            _LOGGER.error("Train loader is not initialized.")
            raise ValueError()
        
        if not self.validation_loader:
            _LOGGER.error("Validation loader is not initialized.")
            raise ValueError()
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.epoch = epoch
            epoch_logs: dict[str, Any] = {}
            self._callbacks_hook('on_epoch_begin', epoch, logs=epoch_logs)
            
            train_logs = self._train_step()
            epoch_logs.update(train_logs)

            val_logs = self._validation_step()
            epoch_logs.update(val_logs)
            
            self._callbacks_hook('on_epoch_end', epoch, logs=epoch_logs)
            
            # Check the early stopping flag
            if self.stop_training:
                break

        self._callbacks_hook('on_train_end')
        
        # Training History
        plot_losses(self.history, save_dir=save_dir)
        
        return self.history

    def _callbacks_hook(self, method_name: str, *args, **kwargs):
        """Calls the specified method on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)
            
    def to_cpu(self):
        """
        Moves the model to the CPU and updates the trainer's device setting.
        
        This is useful for running operations that require the CPU.
        """
        self.device = torch.device('cpu')
        self.model.to(self.device)
        _LOGGER.info("Trainer and model moved to CPU.")
    
    def to_device(self, device: str):
        """
        Moves the model to the specified device and updates the trainer's device setting.

        Args:
            device (str): The target device (e.g., 'cuda', 'mps', 'cpu').
        """
        self.device = self._validate_device(device)
        self.model.to(self.device)
        _LOGGER.info(f"Trainer and model moved to {self.device}.")
        
    def _load_model_state_wrapper(self, model_checkpoint: Union[Path, Literal['best', 'current']], verbose: int = 2):
        """
        Private helper to load the correct model state_dict based on user's choice.
        """
        if isinstance(model_checkpoint, Path):
            self._load_checkpoint(path=model_checkpoint, verbose=verbose)
        elif model_checkpoint == MagicWords.BEST and self._checkpoint_callback:
            path_to_latest = self._checkpoint_callback.best_checkpoint_path
            self._load_checkpoint(path_to_latest, verbose=verbose)
        elif model_checkpoint == MagicWords.BEST and self._checkpoint_callback is None:
            _LOGGER.error(f"'model_checkpoint' set to '{MagicWords.BEST}' but no checkpoint callback was found.")
            raise ValueError()
        elif model_checkpoint == MagicWords.CURRENT:
            pass
        else:
            _LOGGER.error(f"Unknown 'model_checkpoint' received '{model_checkpoint}'.")
            raise ValueError()
        
    # --- Abstract Methods ---
    # These must be implemented by subclasses

    @abstractmethod
    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes the DataLoaders."""
        raise NotImplementedError

    @abstractmethod
    def _train_step(self) -> dict[str, float]:
        """Runs a single training epoch."""
        raise NotImplementedError

    @abstractmethod
    def _validation_step(self) -> dict[str, float]:
        """Runs a single validation epoch."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Runs the full model evaluation."""
        raise NotImplementedError
    
    @abstractmethod
    def _evaluate(self, *args, **kwargs):
        """Internal evaluation helper."""
        raise NotImplementedError

    @abstractmethod
    def finalize_model_training(self, *args, **kwargs):
        """Saves the finalized model for inference."""
        raise NotImplementedError

