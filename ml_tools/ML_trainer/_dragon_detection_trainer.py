from typing import Literal, Union, Optional, Callable
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn

from ..ML_callbacks._base import _Callback
from ..ML_callbacks._checkpoint import DragonModelCheckpoint
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_evaluation import object_detection_metrics
from ..ML_configuration import FinalizeObjectDetection

from ..path_manager import make_fullpath
from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, MLTaskKeys, MagicWords, DragonTrainerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonDetectionTrainer")


__all__ = [
    "DragonDetectionTrainer",
]


# Object Detection Trainer
class DragonDetectionTrainer(_BaseDragonTrainer):
    def __init__(self, model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 collate_fn: Callable, 
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
                 dataloader_workers: int = 2):
        """
        Automates the training process of an Object Detection Model (e.g., DragonFastRCNN).
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch object detection model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The testing/validation dataset.
            collate_fn (Callable): The collate function from `ObjectDetectionDatasetMaker.collate_fn`.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading.
            checkpoint_callback (DragonModelCheckpoint | None): Callback to save the model.
            early_stopping_callback (DragonEarlyStopping | None): Callback to stop training early.
            lr_scheduler_callback (DragonLRScheduler | None): Callback to manage the LR scheduler.
            extra_callbacks (List[Callback] | None): A list of extra callbacks to use during training.
            
        ## Note:
            This trainer is specialized. It does not take a `criterion` because object detection models like Faster R-CNN return a dictionary of losses directly from their forward pass during training.
        """
        # Call the base class constructor with common parameters
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader_workers=dataloader_workers,
            checkpoint_callback=checkpoint_callback,
            early_stopping_callback=early_stopping_callback,
            lr_scheduler_callback=lr_scheduler_callback,
            extra_callbacks=extra_callbacks
        )
        
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset # <-- Renamed
        self.kind = MLTaskKeys.OBJECT_DETECTION
        self.collate_fn = collate_fn
        self.criterion = None # Criterion is handled inside the model

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes the DataLoaders with the object detection collate_fn."""
        self._make_dataloaders(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader): # type: ignore
            # images is a tuple of tensors, targets is a tuple of dicts
            batch_size = len(images)
            
            # Create a log dictionary for the batch
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: batch_size
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            # Move data to device
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            # Model returns a loss dict when in train() mode and targets are passed
            loss_dict = self.model(images, targets)
            
            if not loss_dict:
                # No losses returned, skip batch
                _LOGGER.warning(f"Model returned no losses for batch {batch_idx}. Skipping.")
                batch_logs[PyTorchLogKeys.BATCH_LOSS] = 0
                self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
                continue
            
            # Sum all losses
            loss: torch.Tensor = sum(l for l in loss_dict.values()) # type: ignore
            
            loss.backward()
            self.optimizer.step()

            # Calculate batch loss and update running loss for the epoch
            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            total_samples += batch_size # <-- Accumulate total samples
            
            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss # type: ignore
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        # Calculate loss using the correct denominator
        if total_samples == 0:
            _LOGGER.warning("No samples processed in _train_step. Returning 0 loss.")
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self):
        self.model.train() # Set to train mode even for validation loss calculation
                           # as model internals (e.g., proposals) might differ, but we still need loss_dict.
                           # use torch.no_grad() to prevent gradient updates.
        running_loss = 0.0
        total_samples = 0 
        
        with torch.no_grad():
            for images, targets in self.validation_loader: # type: ignore
                batch_size = len(images)
                
                # Move data to device
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get loss dict
                loss_dict = self.model(images, targets)
                
                if not loss_dict:
                    _LOGGER.warning("Model returned no losses during validation step. Skipping batch.")
                    continue # Skip if no losses
                
                # Sum all losses
                loss: torch.Tensor = sum(l for l in loss_dict.values()) # type: ignore
                
                running_loss += loss.item() * batch_size
                total_samples += batch_size # <-- Accumulate total samples
        
        # Calculate loss using the correct denominator
        if total_samples == 0:
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples}
        return logs
    
    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None):
        """
        Evaluates the model using object detection mAP metrics.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            model_checkpoint (Path | "best" | "current"): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'best', the best checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
            test_data (DataLoader | Dataset | None): Optional Test data to evaluate the model performance. Validation and Test metrics will be saved to subdirectories.
        """
        # Validate inputs using base helpers
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(save_dir)
        
        # Validate test data and dispatch
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
            
            validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            # Dispatch validation set
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None) # 'None' triggers use of self.test_dataset
            
            # Dispatch test set
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current", # Use 'current' state after loading checkpoint once
                           data=test_data_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None) # 'None' triggers use of self.test_dataset
    
    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 data: Optional[Union[DataLoader, Dataset]]):
        """
        Changed to a private helper method
        Evaluates the model using object detection mAP metrics.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            data (DataLoader | Dataset | None): The data to evaluate on. If None, defaults to the trainer's internal test_dataset.
            model_checkpoint ('auto' | Path | None): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'best', the best checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
        """
        # load model checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Prepare Data using Base Helper
        eval_loader, dataset_for_artifacts = self._prepare_eval_data(
            data, 
            self.validation_dataset,
            collate_fn=self.collate_fn # Important for Detection
        )
        
        # Gather all predictions and targets
        all_predictions = []
        all_targets = []
        
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        
        with torch.no_grad():
            for images, targets in eval_loader:
                # Move images to device
                images = list(img.to(self.device) for img in images)
                
                # Model returns predictions when in eval() mode
                predictions = self.model(images)
                
                # Move predictions and targets to CPU for aggregation
                cpu_preds = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
                cpu_targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
                
                all_predictions.extend(cpu_preds)
                all_targets.extend(cpu_targets)

        if not all_targets:
            _LOGGER.error("Evaluation failed: No data was processed.")
            return
        
        # Get class names from the dataset for the report
        class_names = None
        try:
            # Try to get 'classes' from ObjectDetectionDatasetMaker
            if hasattr(dataset_for_artifacts, 'classes'):
                class_names = dataset_for_artifacts.classes # type: ignore
            # Fallback for Subset
            elif hasattr(dataset_for_artifacts, 'dataset') and hasattr(dataset_for_artifacts.dataset, 'classes'): # type: ignore
                 class_names = dataset_for_artifacts.dataset.classes # type: ignore
        except AttributeError:
            _LOGGER.warning("Could not find 'classes' attribute on dataset. Per-class metrics will not be named.")
            pass # class_names is still None

        # --- Routing Logic ---
        object_detection_metrics(
            preds=all_predictions, 
            targets=all_targets, 
            save_dir=save_dir,
            class_names=class_names,
            print_output=False
        )
    
    def finalize_model_training(self, 
                                save_dir: Union[str, Path], 
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                finalize_config: FinalizeObjectDetection
                                ):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict` and the final epoch number.

        Args:
            save_dir (Union[str, Path]): The directory to save the finalized model.
            model_checkpoint (Union[Path, Literal["best", "current"]]):
                - Path: Loads the model state from a specific checkpoint file.
                - "best": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is.
            finalize_config (FinalizeObjectDetection): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if not isinstance(finalize_config, FinalizeObjectDetection):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeObjectDetection', but got {type(finalize_config).__name__}.")
            raise TypeError()
        
        # handle checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task
        }
        
        if finalize_config.class_map is not None:
            finalized_data[PyTorchCheckpointKeys.CLASS_MAP] = finalize_config.class_map
        
        # Save using base helper
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=save_dir,
            filename=finalize_config.filename
        )
