import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn

from ..ML_evaluation import plot_losses
from ..ML_configuration import DragonDDPConfig
from ..ML_callbacks._parallel_callbacks import (DDPProgressBar, 
                                                DDPHistory, 
                                                DDPModelCheckpoint,
                                                DDPPlateauScheduler,
                                                DDPPatienceEarlyStopping)

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import DragonTrainerKeys, PyTorchCheckpointKeys, DDPKeys


_LOGGER = get_logger("Parallel Trainer")


__all__ = [
    "_BaseParallelTrainer",
]


class _BaseParallelTrainer(ABC):
    """
    Abstract base class for Distributed Data Parallel (DDP) training.
    
    Handles process group setup, DistributedSampler creation, DDP model wrapping,
    metric synchronization across GPUs, and unified callback execution.
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 save_dir: Union[str, Path],
                 train_dataset: Optional[Dataset] = None,
                 validation_dataset: Optional[Dataset] = None,
                 collate_fn: Optional[Any] = None,
                 ddp_config: Optional[DragonDDPConfig] = None):
        """
        Initializes the base DDP trainer with model, optimizer, and configuration.
        
        Args:
            model (nn.Module): The PyTorch model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            save_dir (Union[str, Path]): Directory to save checkpoints and logs.
            train_dataset (Optional[Dataset]): The training dataset.
            validation_dataset (Optional[Dataset]): The validation dataset.
            collate_fn (Optional[Any]): Custom collate function for batching.
            ddp_config (Optional[DragonDDPConfig]): Configuration for DDP training.
        """
        
        # fail early if torch.distributed is not available
        if not dist.is_available():
            _LOGGER.error("torch.distributed is not available. Ensure you are running in a distributed environment.")
            raise RuntimeError()
        
        # fail early if CUDA is not available
        if not torch.cuda.is_available():
            _LOGGER.error("CUDA is not available. DDP requires CUDA-enabled GPUs.")
            raise RuntimeError()
        
        self._setup_ddp()
        
        # DDP specific attributes
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # If there is only one process, DDP is not needed
        if self.world_size < 2:
            _LOGGER.error("DDP requires at least 2 processes. 'WORLD_SIZE' must be at least 2. Ensure you are launching with torchrun.")
            raise RuntimeError()
        
        # Pin process to specific GPU
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        if self.is_main_process():
            _LOGGER.info("Converting standard BatchNorm layers to SyncBatchNorm for DDP training.")
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Move model to device and wrap in DDP
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        
        self.optimizer = optimizer
        # Move optimizer states to the device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        self.training_directory_root = make_fullpath(save_dir, make=True, enforce="directory")
        self.criterion: Optional[Union[nn.Module, dict[str, nn.Module]]] = None
        
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.collate_fn = collate_fn
        
        self.train_loader: Optional[DataLoader] = None
        self.validation_loader: Optional[DataLoader] = None 
        
        self.history: dict[str, list[Any]] = {}
        self.epoch = 0
        self.epochs = 0
        self.start_epoch = 1
        self.stop_training = False
        
        # Default Config Initialization
        if ddp_config is None:
            ddp_config = DragonDDPConfig()
            
        # Dataloader workers logic
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
            safe_max_workers = max(1, cpu_count // local_world_size)
        else:
            safe_max_workers = 2
        
        if ddp_config.dataloader_workers == -1:
            self.dataloader_workers = safe_max_workers
        else:
            self.dataloader_workers = min(ddp_config.dataloader_workers, safe_max_workers)
            if ddp_config.dataloader_workers > safe_max_workers and self.is_main_process():
                _LOGGER.warning(f"Capped dataloader_workers to {safe_max_workers} per process to prevent CPU oversubscription.")
        
        # Only rank 0 should manage checkpoints directory
        self._checkpoints_directory = None
        if self.is_main_process():
            self._checkpoints_directory = make_fullpath(
                self.training_directory_root / DragonTrainerKeys.CHECKPOINT_DIR, 
                make=True, 
                enforce="directory"
            )

        # Build Callback Handler
        # State mutators must run before Checkpointing and History recording
        self.callbacks: list[Any] = [DDPProgressBar()]
        
        if ddp_config.use_plateau_scheduler:
            self.callbacks.append(DDPPlateauScheduler(
                monitor=ddp_config.scheduler_monitor,
                mode=ddp_config.scheduler_mode, # type: ignore
                factor=ddp_config.scheduler_factor,
                patience=ddp_config.scheduler_patience,
                threshold=ddp_config.scheduler_threshold,
                threshold_mode=ddp_config.scheduler_threshold_mode, # type: ignore
                cooldown=ddp_config.scheduler_cooldown,
                min_lr=ddp_config.scheduler_min_lr,
                eps=ddp_config.scheduler_eps,
                verbose=ddp_config.scheduler_verbose
            ))
        
        if ddp_config.use_early_stopping:
            self.callbacks.append(DDPPatienceEarlyStopping(
                monitor=ddp_config.early_stopping_monitor,
                min_delta=ddp_config.early_stopping_min_delta,
                patience=ddp_config.early_stopping_patience,
                mode=ddp_config.early_stopping_mode, # type: ignore
                verbose=ddp_config.early_stopping_verbose
            ))
        
        # Last callbacks: History and Checkpointing
        # append history before checkpoint callback to ensure history is saved in the checkpoint
        self.callbacks.append(DDPHistory())
        
        self.callbacks.append(DDPModelCheckpoint(
            monitor=ddp_config.checkpoint_monitor,
            save_three_best=ddp_config.checkpoint_save_three_best,
            mode=ddp_config.checkpoint_mode, # type: ignore
            verbose=ddp_config.checkpoint_verbose
        ))

        self._set_trainer_on_callbacks()

    def _setup_ddp(self):
        """Initializes the distributed process group."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def _cleanup_ddp(self):
        """Destroys the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def is_main_process(self) -> bool:
        """Returns True if the current process is rank 0."""
        return self.global_rank == 0

    def _set_trainer_on_callbacks(self):
        """Gives each callback a reference to this trainer instance."""
        for callback in self.callbacks:
            callback.set_trainer(self)

    def _callbacks_hook(self, method_name: str, *args, **kwargs):
        """Calls the specified method on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)

    def _compile_model(self):
        """Compiles the DDP model using torch.compile if available."""
        if not hasattr(torch, 'compile'):
            if self.is_main_process():
                _LOGGER.warning("torch.compile() is not available. Skipping compilation.")
            return

        try:
            self.model = torch.compile(self.model) # type: ignore
            if self.is_main_process():
                _LOGGER.info("DDP Model successfully compiled with torch.compile().")
        except Exception as e:
            if self.is_main_process():
                _LOGGER.error(f"Failed to compile model: {e}. Proceeding uncompiled.")

    def _decompile_model(self):
        """Reverts the model back to its original DDP state if compiled."""
        if hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod # type: ignore
            if self.is_main_process():
                _LOGGER.info("Model decompiled to original DDP state.")
        else:
            if self.is_main_process():
                _LOGGER.warning("Current compiled Model has no '_orig_mod' attribute. It was not decompiled.")

    def _prepare_train_dataloader(self, 
                                  dataset: Dataset, 
                                  batch_size: int, 
                                  shuffle: bool = True,
                                  collate_fn: Optional[Any] = None) -> DataLoader:
        """Prepares the training DataLoader, dropping the last incomplete batch to protect BatchNorm."""
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.global_rank, 
            shuffle=shuffle,
            drop_last=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        return loader

    def _prepare_validation_dataloader(self, 
                                       dataset: Dataset, 
                                       batch_size: int, 
                                       collate_fn: Optional[Any] = None) -> DataLoader:
        """Prepares the validation DataLoader, preserving all samples for exact metric calculation."""
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.global_rank, 
            shuffle=False,
            drop_last=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
        return loader
        
    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes train and validation DataLoaders."""
        if self.train_dataset is None:
            if self.is_main_process():
                _LOGGER.error("train_dataset is not provided. Cannot create DataLoaders.")
            raise ValueError()
            
        self.train_loader = self._prepare_train_dataloader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
        
        if self.validation_dataset is not None:
            self.validation_loader = self._prepare_validation_dataloader(
                dataset=self.validation_dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn
            )
    
    def _val_samples_limit(self) -> int:
        """Returns the exact number of valid samples for this specific GPU rank, excluding padding."""
        if self.validation_dataset is None:
            return 0
        return len(range(self.global_rank, len(self.validation_dataset), self.world_size)) # type: ignore

    def _sync_train_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """
        Averages training metrics across all GPUs. 
        Mathematically safe because drop_last=True ensures all ranks have identical batch counts.
        """
        synced_metrics = {}
        for key in sorted(metrics.keys()):
            value = metrics[key]
            tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
            synced_metrics[key] = tensor.item()
        return synced_metrics

    def _sync_val_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """
        Sums raw metric totals and sample counts across GPUs to compute exact global averages,
        safely ignoring padded validation samples.
        """
        val_samples = metrics.pop(DDPKeys.VALIDATION_SAMPLES, 0.0)
        
        synced_metrics = {}
        for key in sorted(metrics.keys()):
            value = metrics[key]
            # Bundle the raw loss sum and the sample count into a single tensor
            local_metrics = torch.tensor([value, val_samples], dtype=torch.float32, device=self.device)
            dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
            
            global_sum = local_metrics[0].item()
            global_samples = local_metrics[1].item()
            
            if global_samples == 0:
                synced_metrics[key] = 0.0
            else:
                synced_metrics[key] = global_sum / global_samples
                
        return synced_metrics
    
    def load_checkpoint(self, checkpoint_file: Union[str, Path]) -> None:
        """
        Loads a saved checkpoint to resume training.
        
        Args:
            checkpoint_file (Union[str, Path]): Path to the checkpoint file.
        """
        checkpoint_path = make_fullpath(checkpoint_file, make=False, enforce="file")
        
        # Map location to the specific device to avoid memory spikes across GPUs
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # check if the checkpoint contains the expected keys
        required_keys = [
            PyTorchCheckpointKeys.MODEL_STATE,
            PyTorchCheckpointKeys.OPTIMIZER_STATE,
            PyTorchCheckpointKeys.EPOCH
        ]
        for key in required_keys:
            if key not in checkpoint:
                if self.is_main_process():
                    _LOGGER.error(f"Checkpoint at {checkpoint_path} is missing required key: {key}. Cannot load checkpoint.")
                raise KeyError()
        
        # Unwrap DDP and torch.compile to load the state dictionary correctly regardless of nesting order
        base_model = self.model
        while hasattr(base_model, "_orig_mod") or hasattr(base_model, "module"):
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod # type: ignore
            if hasattr(base_model, "module"):
                base_model = base_model.module # type: ignore
        base_model.load_state_dict(checkpoint[PyTorchCheckpointKeys.MODEL_STATE]) # type: ignore
        
        self.optimizer.load_state_dict(checkpoint[PyTorchCheckpointKeys.OPTIMIZER_STATE])
        # Move optimizer states to the device after loading
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        self.start_epoch = checkpoint[PyTorchCheckpointKeys.EPOCH] + 1
        self.history = checkpoint.get(PyTorchCheckpointKeys.HISTORY, {})
        
        # Restore the best score to the specific checkpoint callback only
        best_score = checkpoint.get(PyTorchCheckpointKeys.BEST_SCORE)
        if best_score is not None:
            for callback in self.callbacks:
                if isinstance(callback, DDPModelCheckpoint):
                    callback.best = best_score
        
        if self.is_main_process():
            # detailed loaded artifacts
            loaded_msg = f"Loaded checkpoint from {checkpoint_path}.\n\tResuming from epoch {self.start_epoch}."
            
            if best_score is not None:
                loaded_msg += f"\n\tBest score restored to: {best_score:.4f}."
            else:
                loaded_msg += "\n\tNo best score found in the checkpoint."
            
            if self.history:
                loaded_msg += f"\n\tTraining history restored with {len(self.history)} metrics."
            if not self.history:
                loaded_msg += "\n\tNo training history found in the checkpoint."
                
            _LOGGER.info(loaded_msg)

    def fit(self, 
            epochs: int, 
            batch_size: int, 
            shuffle: bool = True,
            use_torch_compile: bool = False):
        """
        Orchestrates the DDP training loop.
        
        Args:
            epochs (int): Total number of epochs to train.
            batch_size (int): Batch size for training and validation.
            shuffle (bool): Whether to shuffle the training dataset each epoch.
            use_torch_compile (bool): If True, attempts to compile the model with torch.compile for potential speedups.
        """
        self.epochs = epochs
        self._create_dataloaders(batch_size, shuffle)
        
        if self.train_loader is None:
            _LOGGER.error("Train loader is not initialized")
            raise ValueError()

        if use_torch_compile:
            self._compile_model()

        self.stop_training = False
        self._callbacks_hook('on_train_begin')
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.epoch = epoch
            epoch_logs: dict[str, Any] = {}
            
            # Crucial: set epoch for sampler to ensure proper shuffling across epochs
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            
            self._callbacks_hook('on_epoch_begin', epoch, logs=epoch_logs)
            
            # 1. Train and synchronize metrics
            train_logs = self._train_step()
            train_logs = self._sync_train_metrics(train_logs)
            epoch_logs.update(train_logs)
            
            # 2. Validate and synchronize exact metrics
            if self.validation_loader is not None:
                val_logs = self._validation_step()
                val_logs = self._sync_val_metrics(val_logs)
                epoch_logs.update(val_logs)
            
            # Synchronize processes before calling end-of-epoch callbacks
            dist.barrier()
            
            self._callbacks_hook('on_epoch_end', epoch, logs=epoch_logs)
            
            if self.stop_training:
                break
            
            # Prevent other ranks from advancing while Rank 0 performs heavy checkpoint disk I/O
            dist.barrier()

        self._callbacks_hook('on_train_end')
        
        if use_torch_compile:
            self._decompile_model()
            
        self._cleanup_ddp()
    
    def plot_training_history(self, skip_first_epoch: bool = True) -> dict[str, Any]:
        """
        Plots the training and validation metrics history if available.
        
        Args:
            skip_first_epoch (bool): If True, skips the first epoch in the plot to avoid skewing due to initialization effects.
            
        Returns:
            dict[str, Any]: The training history dictionary.
        """
        # Strictly restrict plotting and warnings to the main process
        if not self.is_main_process():
            return dict()
            
        # confirm that the DDP has ended and that the history is available
        if dist.is_initialized():
            _LOGGER.warning("DDP process group is still active. Ensure fit() has completed before plotting history.")
            return dict()
        
        if not self.history:
            _LOGGER.warning("No training history available to plot.")
            return dict()
        
        plot_losses(history=self.history, 
                    save_dir=self.training_directory_root,
                    skip_first_epoch=skip_first_epoch)
        
        return self.history

    @abstractmethod
    def _train_step(self) -> dict[str, float]:
        """Runs a single training epoch. Must return a dict with training logs (e.g. {'train_loss': 0.5})."""
        raise NotImplementedError

    @abstractmethod
    def _validation_step(self) -> dict[str, float]:
        """Runs a single validation epoch. Must return a dict with validation logs (e.g. {'val_loss': 0.4})."""
        raise NotImplementedError
