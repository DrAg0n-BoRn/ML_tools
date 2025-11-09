from typing import List, Literal, Union, Optional, Callable, Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod

from .path_manager import make_fullpath
from .ML_callbacks import _Callback, History, TqdmProgressBar, DragonModelCheckpoint, DragonEarlyStopping, DragonLRScheduler
from .ML_evaluation import classification_metrics, regression_metrics, plot_losses, shap_summary_plot, plot_attention_importance
from .ML_evaluation_multi import multi_target_regression_metrics, multi_label_classification_metrics, multi_target_shap_summary_plot
from .ML_vision_evaluation import segmentation_metrics, object_detection_metrics
from .ML_sequence_evaluation import sequence_to_sequence_metrics, sequence_to_value_metrics
from .ML_configuration import (RegressionMetricsFormat, 
                            MultiTargetRegressionMetricsFormat,
                            BinaryClassificationMetricsFormat,
                            MultiClassClassificationMetricsFormat,
                            BinaryImageClassificationMetricsFormat,
                            MultiClassImageClassificationMetricsFormat,
                            MultiLabelBinaryClassificationMetricsFormat,
                            BinarySegmentationMetricsFormat,
                            MultiClassSegmentationMetricsFormat,
                            SequenceValueMetricsFormat,
                            SequenceSequenceMetricsFormat,

                            FinalizeBinaryClassification,
                            FinalizeBinarySegmentation,
                            FinalizeBinaryImageClassification,
                            FinalizeMultiClassClassification,
                            FinalizeMultiClassImageClassification,
                            FinalizeMultiClassSegmentation,
                            FinalizeMultiLabelBinaryClassification,
                            FinalizeMultiTargetRegression,
                            FinalizeRegression,
                            FinalizeObjectDetection,
                            FinalizeSequencePrediction)

from ._script_info import _script_info
from ._keys import PyTorchLogKeys, PyTorchCheckpointKeys, DatasetKeys, MLTaskKeys, MagicWords, DragonTrainerKeys
from ._logger import _LOGGER


__all__ = [
    "DragonTrainer",
    "DragonDetectionTrainer",
    "DragonSequenceTrainer"
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
                 early_stopping_callback: Optional[DragonEarlyStopping] = None,
                 lr_scheduler_callback: Optional[DragonLRScheduler] = None,
                 extra_callbacks: Optional[List[_Callback]] = None):

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
        self.history: Dict[str, List[Any]] = {}
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

    def _load_checkpoint(self, path: Union[str, Path]):
        """Loads a training checkpoint to resume training."""
        p = make_fullpath(path, enforce="file")
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
                _LOGGER.info(f"Restored training history up to epoch {self.epoch}.")
            else:
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
                    _LOGGER.info(f"Restored LR scheduler state for: {scheduler_name}")
                except Exception as e:
                    # Loading failed, likely a mismatch
                    scheduler_name = self.scheduler.__class__.__name__
                    _LOGGER.error(f"Failed to load scheduler state for '{scheduler_name}'. A different scheduler type might have been used.")
                    raise e

            elif scheduler_object_exists and not scheduler_state_exists:
                # Case 2: Scheduler provided, but no state in checkpoint.
                scheduler_name = self.scheduler.__class__.__name__
                _LOGGER.warning(f"'{scheduler_name}' was provided, but no scheduler state was found in the checkpoint. The scheduler will start from its initial state.")
            
            elif not scheduler_object_exists and scheduler_state_exists:
                # Case 3: State in checkpoint, but no scheduler provided.
                _LOGGER.error("Checkpoint contains an LR scheduler state, but no LRScheduler callback was provided.")
                raise ValueError()
            
            # Restore callback states
            for cb in self.callbacks:
                if isinstance(cb, DragonModelCheckpoint) and PyTorchCheckpointKeys.BEST_SCORE in checkpoint:
                    cb.best = checkpoint[PyTorchCheckpointKeys.BEST_SCORE]
                    _LOGGER.info(f"Restored {cb.__class__.__name__} 'best' score to: {cb.best:.4f}")
            
            _LOGGER.info(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            
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
            epoch_logs: Dict[str, Any] = {}
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
        
    def _load_model_state_for_finalizing(self, model_checkpoint: Union[Path, Literal['latest', 'current']]):
        """
        Private helper to load the correct model state_dict based on user's choice.
        This is called by finalize_model_training() in subclasses.
        """
        if isinstance(model_checkpoint, Path):
            self._load_checkpoint(path=model_checkpoint)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback:
            path_to_latest = self._checkpoint_callback.best_checkpoint_path
            self._load_checkpoint(path_to_latest)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback is None:
            _LOGGER.error(f"'model_checkpoint' set to '{MagicWords.LATEST}' but no checkpoint callback was found.")
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
    def _train_step(self) -> Dict[str, float]:
        """Runs a single training epoch."""
        raise NotImplementedError

    @abstractmethod
    def _validation_step(self) -> Dict[str, float]:
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


# --- DragonTrainer ----
class DragonTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 kind: Literal["regression", "binary classification", "multiclass classification", 
                               "multitarget regression", "multilabel binary classification", 
                               "binary segmentation", "multiclass segmentation", "binary image classification", "multiclass image classification"],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[DragonEarlyStopping],
                 lr_scheduler_callback: Optional[DragonLRScheduler],
                 extra_callbacks: Optional[List[_Callback]] = None,
                 criterion: Union[nn.Module,Literal["auto"]] = "auto", 
                 dataloader_workers: int = 2):
        """
        Automates the training process of a PyTorch Model.
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            kind (str): Used to redirect to the correct process. 
            criterion (nn.Module | "auto"): The loss function to use. If "auto", it will be inferred from the selected task
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading.
            extra_callbacks (List[Callback] | None): A list of extra callbacks to use during training.
            
        Note:
            - For **regression** and **multi_target_regression** tasks, suggested criterions include `nn.MSELoss` or `nn.L1Loss`. The model should output as many logits as existing targets.
            
            - For **single-label, binary classification**, `nn.BCEWithLogitsLoss` is the standard choice. The model should output a single logit.
    
            - For **single-label, multi-class classification** tasks, `nn.CrossEntropyLoss` is the standard choice. The model should output as many logits as existing classes.
    
            - For **multi-label, binary classification** tasks (where each label is a 0 or 1), `nn.BCEWithLogitsLoss` is the correct choice as it treats each output as an independent binary problem. The model should output 1 logit per binary target.
        
            - For **binary segmentation** tasks, `nn.BCEWithLogitsLoss` is common. The model should output a single logit.
            
            - for **multiclass segmentation** tasks, `nn.CrossEntropyLoss` is the standard. The model should output as many logits as existing classes.
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
        
        if kind not in [MLTaskKeys.REGRESSION,
                        MLTaskKeys.BINARY_CLASSIFICATION,
                        MLTaskKeys.MULTICLASS_CLASSIFICATION,
                        MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION,
                        MLTaskKeys.MULTITARGET_REGRESSION,
                        MLTaskKeys.BINARY_SEGMENTATION,
                        MLTaskKeys.MULTICLASS_SEGMENTATION,
                        MLTaskKeys.BINARY_IMAGE_CLASSIFICATION,
                        MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
            raise ValueError(f"'{kind}' is not a valid task type.")

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        self._classification_threshold: float = 0.5
        
        # loss function
        if criterion == "auto":
            if kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                self.criterion = nn.MSELoss()
            elif kind in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
                self.criterion = nn.BCEWithLogitsLoss()
            elif kind in [MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes the DataLoaders."""
        # Ensure stability on MPS devices by setting num_workers to 0
        loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type),
            drop_last=True  # Drops the last batch if incomplete, selecting a good batch size is key.
        )
        
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type)
        )

    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            # Create a log dictionary for the batch
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(features)
            
            # --- Label Type/Shape Correction ---
            # Cast target to float for BCE-based losses
            if self.kind in MLTaskKeys.ALL_BINARY_TASKS:
                target = target.float()

            # Reshape output to match target for single-logit tasks
            if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION]:
                # If model outputs [N, 1] and target is [N], squeeze output
                if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                    output = output.squeeze(1)
            
            if self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                # If model outputs [N, 1, H, W] and target is [N, H, W], squeeze output
                if output.ndim == 4 and output.shape[1] == 1 and target.ndim == 3:
                    output = output.squeeze(1)
                
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()

            # Calculate batch loss and update running loss for the epoch
            batch_loss = loss.item()
            batch_size = features.size(0)
            running_loss += batch_loss * batch_size  # Accumulate total loss
            total_samples += batch_size # total samples
            
            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if total_samples == 0:
            _LOGGER.warning("No samples processed in a train_step. Returning 0 loss.")
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples} # type: ignore

    def _validation_step(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for features, target in self.validation_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                
                # --- Label Type/Shape Correction ---
                # Cast target to float for BCE-based losses
                if self.kind in MLTaskKeys.ALL_BINARY_TASKS:
                    target = target.float()

                # Reshape output to match target for single-logit tasks
                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION]:
                    # If model outputs [N, 1] and target is [N], squeeze output
                    if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                        output = output.squeeze(1)
                
                if self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    # If model outputs [N, 1, H, W] and target is [N, H, W], squeeze output
                    if output.ndim == 4 and output.shape[1] == 1 and target.ndim == 3:
                        output = output.squeeze(1)
                
                loss = self.criterion(output, target)
                
                running_loss += loss.item() * features.size(0)
                
        if not self.validation_loader.dataset: # type: ignore
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / len(self.validation_loader.dataset)} # type: ignore
        return logs
    
    def _predict_for_eval(self, dataloader: DataLoader):
        """
        Private method to yield model predictions batch by batch for evaluation.
        
        Yields:
            tuple: A tuple containing (y_pred_batch, y_prob_batch, y_true_batch).
                   
                - y_prob_batch is None for regression tasks.
        """
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                output = self.model(features).cpu()

                y_pred_batch = None
                y_prob_batch = None
                y_true_batch = None

                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                    y_pred_batch = output.numpy()
                    y_true_batch = target.numpy()
                    
                elif self.kind in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION]:
                    # Assumes model output is [N, 1] (a single logit)
                    # Squeeze output from [N, 1] to [N] if necessary
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.squeeze(1)
                        
                    probs_pos = torch.sigmoid(output) # Probability of positive class
                    preds = (probs_pos >= self._classification_threshold).int()
                    y_pred_batch = preds.numpy()
                    # For metrics (like ROC AUC), we often need probs for *both* classes
                    # Create an [N, 2] array: [prob_class_0, prob_class_1]
                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.stack([probs_neg, probs_pos], dim=1).numpy()
                    y_true_batch = target.numpy()

                elif self.kind in [MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
                    num_classes = output.shape[1]
                    if num_classes < 3:
                        # Optional: warn the user they are using the wrong kind
                        wrong_class = MLTaskKeys.MULTICLASS_CLASSIFICATION if self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION else MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
                        recommended_class = MLTaskKeys.BINARY_CLASSIFICATION if self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION else MLTaskKeys.BINARY_IMAGE_CLASSIFICATION
                        _LOGGER.warning(f"'{wrong_class}' kind used with {num_classes} classes. Consider using '{recommended_class}' instead.")
                    
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy()
                    y_true_batch = target.numpy()

                elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
                    probs = torch.sigmoid(output)
                    preds = (probs >= self._classification_threshold).int()
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy()
                    y_true_batch = target.numpy()
                
                elif self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    # Assumes model output is [N, 1, H, W] (logits for positive class)
                    probs_pos = torch.sigmoid(output) # Shape [N, 1, H, W]
                    preds = (probs_pos >= self._classification_threshold).int() # Shape [N, 1, H, W]
                    
                    # Squeeze preds to [N, H, W] (class indices 0 or 1)
                    y_pred_batch = preds.squeeze(1).numpy()

                    # Create [N, 2, H, W] probs for consistency
                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.cat([probs_neg, probs_pos], dim=1).numpy()

                    # Handle target shape [N, 1, H, W] -> [N, H, W]
                    if target.ndim == 4 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    y_true_batch = target.numpy()
                    
                elif self.kind == MLTaskKeys.MULTICLASS_SEGMENTATION:
                    # output shape [N, C, H, W]
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1) # shape [N, H, W]
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy() # Probs are [N, C, H, W]
                    
                    # Handle target shape [N, 1, H, W] -> [N, H, W]
                    if target.ndim == 4 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    y_true_batch = target.numpy()

                yield y_pred_batch, y_prob_batch, y_true_batch
                
    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 classification_threshold: Optional[float] = None,
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[
                        RegressionMetricsFormat, 
                        MultiTargetRegressionMetricsFormat,
                        BinaryClassificationMetricsFormat,
                        MultiClassClassificationMetricsFormat,
                        BinaryImageClassificationMetricsFormat,
                        MultiClassImageClassificationMetricsFormat,
                        MultiLabelBinaryClassificationMetricsFormat,
                        BinarySegmentationMetricsFormat,
                        MultiClassSegmentationMetricsFormat
                    ]]=None,
                 test_format_configuration: Optional[Union[
                        RegressionMetricsFormat, 
                        MultiTargetRegressionMetricsFormat,
                        BinaryClassificationMetricsFormat,
                        MultiClassClassificationMetricsFormat,
                        BinaryImageClassificationMetricsFormat,
                        MultiClassImageClassificationMetricsFormat,
                        MultiLabelBinaryClassificationMetricsFormat,
                        BinarySegmentationMetricsFormat,
                        MultiClassSegmentationMetricsFormat,
                    ]]=None):
        """
        Evaluates the model, routing to the correct evaluation function based on task `kind`.

        Args:
            model_checkpoint ('auto' | Path | None): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'latest', the latest checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
            save_dir (str | Path): Directory to save all reports and plots.
            classification_threshold (float | None): Used for tasks using a binary approach (binary classification, binary segmentation, multilabel binary classification)
            test_data (DataLoader | Dataset | None): Optional Test data to evaluate the model performance. Validation and Test metrics will be saved to subdirectories.
            val_format_configuration (object): Optional configuration for metric format output for the validation set.
            test_format_configuration (object): Optional configuration for metric format output for the test set.
        """
        # Validate model checkpoint
        if isinstance(model_checkpoint, Path):
            checkpoint_validated = make_fullpath(model_checkpoint, enforce="file")
        elif model_checkpoint in [MagicWords.LATEST, MagicWords.CURRENT]:
            checkpoint_validated = model_checkpoint
        else:
            _LOGGER.error(f"'model_checkpoint' must be a Path object, or the string '{MagicWords.LATEST}', or the string '{MagicWords.CURRENT}'.")
            raise ValueError()
        
        # Validate classification threshold
        if self.kind not in MLTaskKeys.ALL_BINARY_TASKS:
            # dummy value for tasks that do not need it
            threshold_validated = 0.5
        elif classification_threshold is None:
            # it should have been provided for binary tasks
            _LOGGER.error(f"The classification threshold must be provided for '{self.kind}'.")
            raise ValueError()
        elif classification_threshold <= 0.0 or classification_threshold >= 1.0:
            # Invalid float
            _LOGGER.error(f"A classification threshold of {classification_threshold} is invalid. Must be in the range (0.0 - 1.0).")
            raise ValueError()
        else:
            threshold_validated = classification_threshold
        
        # Validate val configuration
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (RegressionMetricsFormat, 
                                                        MultiTargetRegressionMetricsFormat,
                                                        BinaryClassificationMetricsFormat,
                                                        MultiClassClassificationMetricsFormat,
                                                        BinaryImageClassificationMetricsFormat,
                                                        MultiClassImageClassificationMetricsFormat,
                                                        MultiLabelBinaryClassificationMetricsFormat,
                                                        BinarySegmentationMetricsFormat,
                                                        MultiClassSegmentationMetricsFormat)):
                _LOGGER.error(f"Invalid 'format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
            else:
                val_configuration_validated = val_format_configuration
        else: # config is None
            val_configuration_validated = None
        
        # Validate directory
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        # Validate test data and dispatch
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
                
            validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated,
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)
            
            # Validate test configuration
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (RegressionMetricsFormat, 
                                                        MultiTargetRegressionMetricsFormat,
                                                        BinaryClassificationMetricsFormat,
                                                        MultiClassClassificationMetricsFormat,
                                                        BinaryImageClassificationMetricsFormat,
                                                        MultiClassImageClassificationMetricsFormat,
                                                        MultiLabelBinaryClassificationMetricsFormat,
                                                        BinarySegmentationMetricsFormat,
                                                        MultiClassSegmentationMetricsFormat)):
                    warning_message_type = f"Invalid test_format_configuration': '{type(test_format_configuration)}'."
                    if val_configuration_validated is not None:
                        warning_message_type += " 'val_format_configuration' will be used for the test set metrics output."
                        test_configuration_validated = val_configuration_validated
                    else:
                        warning_message_type += " Using default format."
                        test_configuration_validated = None
                    _LOGGER.warning(warning_message_type)
                else:
                    test_configuration_validated = test_format_configuration
            else: #config is None
                test_configuration_validated = None
            
            # Dispatch test set
            _LOGGER.info(f"Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           classification_threshold=threshold_validated,
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated,
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)
        
    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 classification_threshold: float,
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: Optional[Union[
                        RegressionMetricsFormat, 
                        MultiTargetRegressionMetricsFormat,
                        BinaryClassificationMetricsFormat,
                        MultiClassClassificationMetricsFormat,
                        BinaryImageClassificationMetricsFormat,
                        MultiClassImageClassificationMetricsFormat,
                        MultiLabelBinaryClassificationMetricsFormat,
                        BinarySegmentationMetricsFormat,
                        MultiClassSegmentationMetricsFormat
                    ]]=None):
        """
        Changed to a private helper function.
        """
        dataset_for_artifacts = None
        eval_loader = None
        
        # set threshold
        self._classification_threshold = classification_threshold
        
        # load model checkpoint
        if isinstance(model_checkpoint, Path):
            self._load_checkpoint(path=model_checkpoint)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback:
            path_to_latest = self._checkpoint_callback.best_checkpoint_path
            self._load_checkpoint(path_to_latest)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback is None:
            _LOGGER.error(f"'model_checkpoint' set to '{MagicWords.LATEST}' but no checkpoint callback was found.")
            raise ValueError()
        
        # Dataloader
        if isinstance(data, DataLoader):
            eval_loader = data
            # Try to get the dataset from the loader for fetching target names
            if hasattr(data, 'dataset'):
                dataset_for_artifacts = data.dataset # type: ignore
        elif isinstance(data, Dataset):
            # Create a new loader from the provided dataset
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))
            dataset_for_artifacts = data
        else: # data is None, use the trainer's default test dataset
            if self.validation_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no validation dataset available in the trainer.")
                raise ValueError()
            # Create a fresh DataLoader from the test_dataset
            eval_loader = DataLoader(self.validation_dataset, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))
            
            dataset_for_artifacts = self.validation_dataset

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()

        # print("\n--- Model Evaluation ---")

        all_preds, all_probs, all_true = [], [], []
        for y_pred_b, y_prob_b, y_true_b in self._predict_for_eval(eval_loader):
            if y_pred_b is not None: all_preds.append(y_pred_b)
            if y_prob_b is not None: all_probs.append(y_prob_b)
            if y_true_b is not None: all_true.append(y_true_b)

        if not all_true:
            _LOGGER.error("Evaluation failed: No data was processed.")
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs) if all_probs else None

        # --- Routing Logic ---
        # Single-target regression
        if self.kind == MLTaskKeys.REGRESSION:
            # Check configuration
            config = None
            if format_configuration and isinstance(format_configuration, RegressionMetricsFormat):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            regression_metrics(y_true=y_true.flatten(), 
                               y_pred=y_pred.flatten(), 
                               save_dir=save_dir,
                               config=config)
        
        # single target classification
        elif self.kind in [MLTaskKeys.BINARY_CLASSIFICATION, 
                           MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
                           MLTaskKeys.MULTICLASS_CLASSIFICATION, 
                           MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
            # get the class map if it exists
            try:
                class_map = dataset_for_artifacts.class_map # type: ignore
            except AttributeError:
                _LOGGER.warning(f"Dataset has no 'class_map' attribute. Using generics.")
                class_map = None
            else:
                if not isinstance(class_map, dict):
                    _LOGGER.warning(f"Dataset has a 'class_map' attribute, but it is not a dictionary: '{type(class_map)}'.")
                    class_map = None
            
            # Check configuration
            config = None
            if format_configuration:
                if self.kind == MLTaskKeys.BINARY_CLASSIFICATION and isinstance(format_configuration, BinaryClassificationMetricsFormat):
                    config = format_configuration
                elif self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION and isinstance(format_configuration, BinaryImageClassificationMetricsFormat):
                    config = format_configuration
                elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION and isinstance(format_configuration, MultiClassClassificationMetricsFormat):
                    config = format_configuration
                elif self.kind == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION and isinstance(format_configuration, MultiClassImageClassificationMetricsFormat):
                    config = format_configuration
                else:
                    _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
  
            classification_metrics(save_dir=save_dir,
                                   y_true=y_true,
                                   y_pred=y_pred,
                                   y_prob=y_prob,
                                   class_map=class_map,
                                   config=config)
        
        # multitarget regression
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION:
            try:
                target_names = dataset_for_artifacts.target_names # type: ignore
            except AttributeError:
                num_targets = y_true.shape[1]
                target_names = [f"target_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no 'target_names' attribute. Using generic names.")
                
            # Check configuration
            config = None
            if format_configuration and isinstance(format_configuration, MultiTargetRegressionMetricsFormat):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            multi_target_regression_metrics(y_true=y_true, 
                                            y_pred=y_pred,
                                            target_names=target_names, 
                                            save_dir=save_dir,
                                            config=config)
            
        # multi-label binary classification
        elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
            try:
                target_names = dataset_for_artifacts.target_names # type: ignore
            except AttributeError:
                num_targets = y_true.shape[1]
                target_names = [f"label_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no 'target_names' attribute. Using generic names.")
            
            if y_prob is None:
                _LOGGER.error("Evaluation for multi_label_classification requires probabilities (y_prob).")
                return
            
            # Check configuration
            config = None
            if format_configuration and isinstance(format_configuration, MultiLabelBinaryClassificationMetricsFormat):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")

            multi_label_classification_metrics(y_true=y_true,
                                               y_pred=y_pred,
                                               y_prob=y_prob,
                                               target_names=target_names,
                                               save_dir=save_dir,
                                               config=config)
        
        # Segmentation tasks
        elif self.kind in [MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
            class_names = None
            try:
                # Try to get 'classes' from VisionDatasetMaker
                if hasattr(dataset_for_artifacts, 'classes'):
                    class_names = dataset_for_artifacts.classes # type: ignore
                # Fallback for Subset
                elif hasattr(dataset_for_artifacts, 'dataset') and hasattr(dataset_for_artifacts.dataset, 'classes'): # type: ignore
                     class_names = dataset_for_artifacts.dataset.classes # type: ignore
            except AttributeError:
                pass # class_names is still None

            if class_names is None:
                try:
                    # Fallback to 'target_names'
                    class_names = dataset_for_artifacts.target_names # type: ignore
                except AttributeError:
                    # Fallback to inferring from labels
                    labels = np.unique(y_true)
                    class_names = [f"Class {i}" for i in labels]
                    _LOGGER.warning(f"Dataset has no 'classes' or 'target_names' attribute. Using generic names.")
            
            # Check configuration
            config = None
            if format_configuration and isinstance(format_configuration, (BinarySegmentationMetricsFormat, MultiClassSegmentationMetricsFormat)):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            segmentation_metrics(y_true=y_true,
                                 y_pred=y_pred,
                                 save_dir=save_dir,
                                 class_names=class_names,
                                 config=config)
    
    def explain(self,
                save_dir: Union[str,Path], 
                explain_dataset: Optional[Dataset] = None, 
                n_samples: int = 300,
                feature_names: Optional[List[str]] = None,
                target_names: Optional[List[str]] = None,
                explainer_type: Literal['deep', 'kernel'] = 'kernel'):
        """
        Explains model predictions using SHAP and saves all artifacts.

        The background data is automatically sampled from the trainer's training dataset.
        
        This method automatically routes to the appropriate SHAP summary plot
        function based on the task. If `feature_names` or `target_names` (multi-target) are not provided,
        it will attempt to extract them from the dataset.

        Args:
            explain_dataset (Dataset | None): A specific dataset to explain. 
                                                 If None, the trainer's test dataset is used.
            n_samples (int): The number of samples to use for both background and explanation.
            feature_names (list[str] | None): Feature names. If None, the names will be extracted from the Dataset and raise an error on failure.
            target_names (list[str] | None): Target names for multi-target tasks.
            save_dir (str | Path): Directory to save all SHAP artifacts.
            explainer_type (Literal['deep', 'kernel']): The explainer to use.
                - 'deep': Uses shap.DeepExplainer. Fast and efficient for PyTorch models.
                - 'kernel': Uses shap.KernelExplainer. Model-agnostic but EXTREMELY slow and memory-intensive. Use with a very low 'n_samples'< 100.
        """        
        # memory efficient helper
        def _get_random_sample(dataset: Dataset, num_samples: int):
            """
            Memory-efficiently samples data from a dataset.
            """
            if dataset is None:
                return None
            
            dataset_len = len(dataset) # type: ignore
            if dataset_len == 0:
                return None
            
            # For MPS devices, num_workers must be 0 to ensure stability
            loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
            
            # Ensure batch_size is not larger than the dataset itself
            batch_size = min(num_samples, 64, dataset_len) 
            
            loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True, # Shuffle to get random samples
                num_workers=loader_workers
            )
            
            collected_features = []
            num_collected = 0
            
            for features, _ in loader:
                collected_features.append(features)
                num_collected += features.size(0)
                if num_collected >= num_samples:
                    break # Stop once we have enough samples
            
            if not collected_features:
                return None
            
            full_data = torch.cat(collected_features, dim=0)
            
            # If we collected more than needed, trim it down
            if full_data.size(0) > num_samples:
                return full_data[:num_samples]
            
            return full_data
        
        # print(f"\n--- Preparing SHAP Data (sampling up to {n_samples} instances) ---")

        # 1. Get background data from the trainer's train_dataset
        background_data = _get_random_sample(self.train_dataset, n_samples)
        if background_data is None:
            _LOGGER.error("Trainer's train_dataset is empty or invalid. Skipping SHAP analysis.")
            return

        # 2. Determine target dataset and get explanation instances
        target_dataset = explain_dataset if explain_dataset is not None else self.validation_dataset
        instances_to_explain = _get_random_sample(target_dataset, n_samples)
        if instances_to_explain is None:
            _LOGGER.error("Explanation dataset is empty or invalid. Skipping SHAP analysis.")
            return
        
        # attempt to get feature names
        if feature_names is None:
            # _LOGGER.info("`feature_names` not provided. Attempting to extract from dataset...")
            if hasattr(target_dataset, DatasetKeys.FEATURE_NAMES):
                feature_names = target_dataset.feature_names # type: ignore
            else:
                _LOGGER.error(f"Could not extract `feature_names` from the dataset. It must be provided if the dataset object does not have a '{DatasetKeys.FEATURE_NAMES}' attribute.")
                raise ValueError()
            
        # move model to device
        self.model.to(self.device)

        # 3. Call the plotting function
        if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            shap_summary_plot(
                model=self.model,
                background_data=background_data,
                instances_to_explain=instances_to_explain,
                feature_names=feature_names,
                save_dir=save_dir,
                explainer_type=explainer_type,
                device=self.device
            )
        elif self.kind in [MLTaskKeys.MULTITARGET_REGRESSION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
            # try to get target names
            if target_names is None:
                target_names = []
                if hasattr(target_dataset, DatasetKeys.TARGET_NAMES):
                    target_names = target_dataset.target_names # type: ignore
                else:
                    # Infer number of targets from the model's output layer
                    try:
                        num_targets = self.model.output_layer.out_features # type: ignore
                        target_names = [f"target_{i}" for i in range(num_targets)] # type: ignore
                        _LOGGER.warning("Dataset has no 'target_names' attribute. Using generic names.")
                    except AttributeError:
                        _LOGGER.error("Cannot determine target names for multi-target SHAP plot. Skipping.")
                        return

            multi_target_shap_summary_plot(
                model=self.model,
                background_data=background_data,
                instances_to_explain=instances_to_explain,
                feature_names=feature_names, # type: ignore
                target_names=target_names, # type: ignore
                save_dir=save_dir,
                explainer_type=explainer_type,
                device=self.device
            )

    def _attention_helper(self, dataloader: DataLoader):
        """
        Private method to yield model attention weights batch by batch for evaluation.

        Args:
            dataloader (DataLoader): The dataloader to predict on.

        Yields:
            (torch.Tensor): Attention weights
        """
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                attention_weights = None
                
                # Get model output
                # Unpack logits and weights from the special forward method
                _output, attention_weights = self.model.forward_attention(features) # type: ignore
                
                if attention_weights is not None:
                    attention_weights = attention_weights.cpu()

                yield attention_weights
    
    def explain_attention(self, save_dir: Union[str, Path], 
                          feature_names: Optional[List[str]] = None, 
                          explain_dataset: Optional[Dataset] = None,
                          plot_n_features: int = 10):
        """
        Generates and saves a feature importance plot based on attention weights.

        This method only works for models with models with 'has_interpretable_attention'.

        Args:
            save_dir (str | Path): Directory to save the plot and summary data.
            feature_names (List[str] | None): Names for the features for plot labeling. If None, the names will be extracted from the Dataset and raise an error on failure.
            explain_dataset (Dataset, optional): A specific dataset to explain. If None, the trainer's test dataset is used.
            plot_n_features (int): Number of top features to plot.
        """
        
        # print("\n--- Attention Analysis ---")
        
        # --- Step 1: Check if the model supports this explanation ---
        if not getattr(self.model, 'has_interpretable_attention', False):
            _LOGGER.warning("Model is not compatible with interpretable attention analysis. Skipping.")
            return

        # --- Step 2: Set up the dataloader ---
        dataset_to_use = explain_dataset if explain_dataset is not None else self.validation_dataset
        if not isinstance(dataset_to_use, Dataset):
            _LOGGER.error("The explanation dataset is empty or invalid. Skipping attention analysis.")
            return
        
        # Get feature names
        if feature_names is None:
            if hasattr(dataset_to_use, DatasetKeys.FEATURE_NAMES):
                feature_names = dataset_to_use.feature_names # type: ignore
            else:
                _LOGGER.error(f"Could not extract `feature_names` from the dataset for attention plot. It must be provided if the dataset object does not have a '{DatasetKeys.FEATURE_NAMES}' attribute.")
                raise ValueError()
        
        explain_loader = DataLoader(
            dataset=dataset_to_use, batch_size=32, shuffle=False,
            num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
            pin_memory=("cuda" in self.device.type)
        )
        
        # --- Step 3: Collect weights ---
        all_weights = []
        for att_weights_b in self._attention_helper(explain_loader):
            if att_weights_b is not None:
                all_weights.append(att_weights_b)

        # --- Step 4: Call the plotting function ---
        if all_weights:
            plot_attention_importance(
                weights=all_weights,
                feature_names=feature_names,
                save_dir=save_dir,
                top_n=plot_n_features
            )
        else:
            _LOGGER.error("No attention weights were collected from the model.")
        
    def finalize_model_training(self, 
                                model_checkpoint: Union[Path, Literal['latest', 'current']],
                                save_dir: Union[str, Path], 
                                finalize_config: Union[FinalizeRegression,
                                                       FinalizeMultiTargetRegression,
                                                       FinalizeBinaryClassification,
                                                       FinalizeBinaryImageClassification,
                                                       FinalizeMultiClassClassification,
                                                       FinalizeMultiClassImageClassification,
                                                       FinalizeBinarySegmentation,
                                                       FinalizeMultiClassSegmentation,
                                                       FinalizeMultiLabelBinaryClassification]):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict`, the final epoch number, and optional configuration for the task at hand.

        Args:
            model_checkpoint (Path | "latest" | "current"):
                - Path: Loads the model state from a specific checkpoint file.
                - "latest": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is at the end of the `fit()` call.
            save_dir (str | Path): The directory to save the finalized model.
            finalize_config (object): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if self.kind == MLTaskKeys.REGRESSION and not isinstance(finalize_config, FinalizeRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeRegression', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION and not isinstance(finalize_config, FinalizeMultiTargetRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiTargetRegression', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.BINARY_CLASSIFICATION and not isinstance(finalize_config, FinalizeBinaryClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeBinaryClassification', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION and not isinstance(finalize_config, FinalizeBinaryImageClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeBinaryImageClassification', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION and not isinstance(finalize_config, FinalizeMultiClassClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiClassClassification', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION and not isinstance(finalize_config, FinalizeMultiClassImageClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiClassImageClassification', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.BINARY_SEGMENTATION and not isinstance(finalize_config, FinalizeBinarySegmentation):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeBinarySegmentation', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTICLASS_SEGMENTATION and not isinstance(finalize_config, FinalizeMultiClassSegmentation):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiClassSegmentation', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION and not isinstance(finalize_config, FinalizeMultiLabelBinaryClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiLabelBinaryClassification', but got {type(finalize_config).__name__}.")
            raise TypeError()
        
        # handle save path
        dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = dir_path / finalize_config.filename
        
        # handle checkpoint
        self._load_model_state_for_finalizing(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
        }

        # Parse config
        if finalize_config.target_name is not None:
            finalized_data[PyTorchCheckpointKeys.TARGET_NAME] = finalize_config.target_name
        if finalize_config.target_names is not None:
            finalized_data[PyTorchCheckpointKeys.TARGET_NAMES] = finalize_config.target_names
        if finalize_config.classification_threshold is not None:
            finalized_data[PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD] = finalize_config.classification_threshold
        if finalize_config.class_map is not None:
            finalized_data[PyTorchCheckpointKeys.CLASS_MAP] = finalize_config.class_map

        # Save model file
        torch.save(finalized_data, full_path)
        
        _LOGGER.info(f"Finalized model file saved to '{full_path}'")


# Object Detection Trainer
class DragonDetectionTrainer(_BaseDragonTrainer):
    def __init__(self, model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 collate_fn: Callable, optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[DragonEarlyStopping],
                 lr_scheduler_callback: Optional[DragonLRScheduler],
                 extra_callbacks: Optional[List[_Callback]] = None,
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
        # Ensure stability on MPS devices by setting num_workers to 0
        loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type), 
            collate_fn=self.collate_fn, # Use the provided collate function
            drop_last=True 
        )
        
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type),
            collate_fn=self.collate_fn # Use the provided collate function
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
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None):
        """
        Evaluates the model using object detection mAP metrics.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            model_checkpoint ('auto' | Path | None): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'latest', the latest checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
            test_data (DataLoader | Dataset | None): Optional Test data to evaluate the model performance. Validation and Test metrics will be saved to subdirectories.
        """
        # Validate model checkpoint
        if isinstance(model_checkpoint, Path):
            checkpoint_validated = make_fullpath(model_checkpoint, enforce="file")
        elif model_checkpoint in [MagicWords.LATEST, MagicWords.CURRENT]:
            checkpoint_validated = model_checkpoint
        else:
            _LOGGER.error(f"'model_checkpoint' must be a Path object, or the string '{MagicWords.LATEST}', or the string '{MagicWords.CURRENT}'.")
            raise ValueError()
        
        # Validate directory
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        # Validate test data and dispatch
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
            
            validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated,
                           data=None) # 'None' triggers use of self.test_dataset
            
            # Dispatch test set
            _LOGGER.info(f"Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current", # Use 'current' state after loading checkpoint once
                           data=test_data_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated,
                           data=None) # 'None' triggers use of self.test_dataset
    
    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 data: Optional[Union[DataLoader, Dataset]]):
        """
        Changed to a private helper method
        Evaluates the model using object detection mAP metrics.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            data (DataLoader | Dataset | None): The data to evaluate on. If None, defaults to the trainer's internal test_dataset.
            model_checkpoint ('auto' | Path | None): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'latest', the latest checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
        """
        dataset_for_artifacts = None
        eval_loader = None
        
        # load model checkpoint
        if isinstance(model_checkpoint, Path):
            self._load_checkpoint(path=model_checkpoint)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback:
            path_to_latest = self._checkpoint_callback.best_checkpoint_path
            self._load_checkpoint(path_to_latest)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback is None:
            _LOGGER.error(f"'model_checkpoint' set to '{MagicWords.LATEST}' but no checkpoint callback was found.")
            raise ValueError()

        # Dataloader
        if isinstance(data, DataLoader):
            eval_loader = data
            if hasattr(data, 'dataset'):
                dataset_for_artifacts = data.dataset # type: ignore
        elif isinstance(data, Dataset):
            # Create a new loader from the provided dataset
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"),
                                     collate_fn=self.collate_fn)
            dataset_for_artifacts = data
        else: # data is None, use the trainer's default test dataset
            if self.validation_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no test_dataset available in the trainer.")
                raise ValueError()
            # Create a fresh DataLoader from the test_dataset
            eval_loader = DataLoader(
                self.validation_dataset, 
                batch_size=self._batch_size, 
                shuffle=False, 
                num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=self.collate_fn
            )
            dataset_for_artifacts = self.validation_dataset

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()

        # print("\n--- Model Evaluation ---")

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
                                model_checkpoint: Union[Path, Literal['latest', 'current']],
                                finalize_config: FinalizeObjectDetection
                                ):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict` and the final epoch number.

        Args:
            save_dir (Union[str, Path]): The directory to save the finalized model.
            model_checkpoint (Union[Path, Literal["latest", "current"]]):
                - Path: Loads the model state from a specific checkpoint file.
                - "latest": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is at the end of the `fit()` call.
            finalize_config (FinalizeObjectDetection): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if not isinstance(finalize_config, FinalizeObjectDetection):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeObjectDetection', but got {type(finalize_config).__name__}.")
            raise TypeError()
        
        # handle save path
        dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = dir_path / finalize_config.filename
        
        # handle checkpoint
        self._load_model_state_for_finalizing(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
        }
        
        if finalize_config.class_map is not None:
            finalized_data[PyTorchCheckpointKeys.CLASS_MAP] = finalize_config.class_map
        
        torch.save(finalized_data, full_path)
        
        _LOGGER.info(f"Finalized model file saved to '{full_path}'")

# --- DragonSequenceTrainer ----
class DragonSequenceTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 kind: Literal["sequence-to-sequence", "sequence-to-value"],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[DragonEarlyStopping],
                 lr_scheduler_callback: Optional[DragonLRScheduler],
                 extra_callbacks: Optional[List[_Callback]] = None,
                 criterion: Union[nn.Module,Literal["auto"]] = "auto", 
                 dataloader_workers: int = 2):
        """
        Automates the training process of a PyTorch Sequence Model.
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            kind (str): Used to redirect to the correct process ('sequence-to-sequence' or 'sequence-to-value'). 
            criterion (nn.Module | "auto"): The loss function to use. If "auto", it will be inferred from the selected task
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading.
            extra_callbacks (List[Callback] | None): A list of extra callbacks to use during training.
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
        
        if kind not in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            raise ValueError(f"'{kind}' is not a valid task type for DragonSequenceTrainer.")

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        
        # try to validate against Dragon Sequence model
        if hasattr(self.model, "prediction_mode"):
            key_to_check: str = self.model.prediction_mode # type: ignore
            if not key_to_check == self.kind:
                _LOGGER.error(f"Trainer was set for '{self.kind}', but model architecture '{self.model}' is built for '{key_to_check}'.")
                raise RuntimeError()
        
        # loss function
        if criterion == "auto":
            # Both sequence tasks are treated as regression problems
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes the DataLoaders."""
        # Ensure stability on MPS devices by setting num_workers to 0
        loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type),
            drop_last=True  # Drops the last batch if incomplete, selecting a good batch size is key.
        )
        
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type)
        )

    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            # Create a log dictionary for the batch
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(features)
            
            # --- Label Type/Shape Correction ---
            # Ensure target is float for MSELoss
            target = target.float()

            # For seq-to-val, models might output [N, 1] but target is [N].
            if self.kind == MLTaskKeys.SEQUENCE_VALUE:
                if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                    output = output.squeeze(1)
            
            # For seq-to-seq, models might output [N, Seq, 1] but target is [N, Seq].
            elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
                if output.ndim == 3 and output.shape[2] == 1 and target.ndim == 2:
                    output = output.squeeze(-1)
            
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()

            # Calculate batch loss and update running loss for the epoch
            batch_loss = loss.item()
            batch_size = features.size(0)
            running_loss += batch_loss * batch_size  # Accumulate total loss
            total_samples += batch_size # total samples
            
            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if total_samples == 0:
            _LOGGER.warning("No samples processed in a train_step. Returning 0 loss.")
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples} # type: ignore

    def _validation_step(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for features, target in self.validation_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                
                # --- Label Type/Shape Correction ---
                target = target.float()
                
                # For seq-to-val, models might output [N, 1] but target is [N].
                if self.kind == MLTaskKeys.SEQUENCE_VALUE:
                    if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                        output = output.squeeze(1)
                        
                # For seq-to-seq, models might output [N, Seq, 1] but target is [N, Seq].
                elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
                    if output.ndim == 3 and output.shape[2] == 1 and target.ndim == 2:
                        output = output.squeeze(-1)
                
                loss = self.criterion(output, target)
                
                running_loss += loss.item() * features.size(0)
                
        if not self.validation_loader.dataset: # type: ignore
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / len(self.validation_loader.dataset)} # type: ignore
        return logs
    
    def _predict_for_eval(self, dataloader: DataLoader):
        """
        Private method to yield model predictions batch by batch for evaluation.
        
        Yields:
            tuple: A tuple containing (y_pred_batch, y_prob_batch, y_true_batch).
                   y_prob_batch is always None for sequence tasks.
        """
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                output = self.model(features).cpu()

                y_pred_batch = output.numpy()
                y_prob_batch = None # Not applicable for sequence regression
                y_true_batch = target.numpy()

                yield y_pred_batch, y_prob_batch, y_true_batch
                
    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[SequenceValueMetricsFormat, 
                                                          SequenceSequenceMetricsFormat]]=None,
                 test_format_configuration: Optional[Union[SequenceValueMetricsFormat, 
                                                           SequenceSequenceMetricsFormat]]=None):
        """
        Evaluates the model, routing to the correct evaluation function.

        Args:
            model_checkpoint ('auto' | Path | None): 
                - Path to a valid checkpoint for the model.
                - If 'latest', the latest checkpoint will be loaded.
                - If 'current', use the current state of the trained model.
            save_dir (str | Path): Directory to save all reports and plots.
            test_data (DataLoader | Dataset | None): Optional Test data.
            val_format_configuration: Optional configuration for validation metrics.
            test_format_configuration: Optional configuration for test metrics.
        """
        # Validate model checkpoint
        if isinstance(model_checkpoint, Path):
            checkpoint_validated = make_fullpath(model_checkpoint, enforce="file")
        elif model_checkpoint in [MagicWords.LATEST, MagicWords.CURRENT]:
            checkpoint_validated = model_checkpoint
        else:
            _LOGGER.error(f"'model_checkpoint' must be a Path object, or '{MagicWords.LATEST}', or '{MagicWords.CURRENT}'.")
            raise ValueError()
        
        # Validate val configuration
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (SequenceValueMetricsFormat, SequenceSequenceMetricsFormat)):
                _LOGGER.error(f"Invalid 'val_format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
        
        # Validate directory
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        # Validate test data and dispatch
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
    
            validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated,
                           data=None,
                           format_configuration=val_format_configuration)
            
            # Validate test configuration
            test_configuration_validated = None
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (SequenceValueMetricsFormat, SequenceSequenceMetricsFormat)):
                    warning_message_type = f"Invalid test_format_configuration': '{type(test_format_configuration)}'."
                    if val_format_configuration is not None:
                        warning_message_type += " 'val_format_configuration' will be used."
                        test_configuration_validated = val_format_configuration
                    else:
                        warning_message_type += " Using default format."
                    _LOGGER.warning(warning_message_type)
                else:
                    test_configuration_validated = test_format_configuration
            
            # Dispatch test set
            _LOGGER.info(f"Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated,
                           data=None,
                           format_configuration=val_format_configuration)
        
    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["latest", "current"]],
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: object):
        """
        Private evaluation helper.
        """
        eval_loader = None
        
        # load model checkpoint
        if isinstance(model_checkpoint, Path):
            self._load_checkpoint(path=model_checkpoint)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback:
            path_to_latest = self._checkpoint_callback.best_checkpoint_path
            self._load_checkpoint(path_to_latest)
        elif model_checkpoint == MagicWords.LATEST and self._checkpoint_callback is None:
            _LOGGER.error(f"'model_checkpoint' set to '{MagicWords.LATEST}' but no checkpoint callback was found.")
            raise ValueError()
        
        # Dataloader
        if isinstance(data, DataLoader):
            eval_loader = data
        elif isinstance(data, Dataset):
            # Create a new loader from the provided dataset
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))
        else: # data is None, use the trainer's default validation dataset
            if self.validation_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no validation_dataset available in the trainer.")
                raise ValueError()
            eval_loader = DataLoader(self.validation_dataset, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()

        all_preds, _, all_true = [], [], []
        for y_pred_b, y_prob_b, y_true_b in self._predict_for_eval(eval_loader):
            if y_pred_b is not None: all_preds.append(y_pred_b)
            if y_true_b is not None: all_true.append(y_true_b)

        if not all_true:
            _LOGGER.error("Evaluation failed: No data was processed.")
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)

        # --- Routing Logic ---
        if self.kind == MLTaskKeys.SEQUENCE_VALUE:
            config = None
            if format_configuration and isinstance(format_configuration, SequenceValueMetricsFormat):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong config type: Received {type(format_configuration).__name__}, expected SequenceValueMetricsFormat.")
            
            sequence_to_value_metrics(y_true=y_true, 
                                      y_pred=y_pred, 
                                      save_dir=save_dir,
                                      config=config)

        elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
            config = None
            if format_configuration and isinstance(format_configuration, SequenceSequenceMetricsFormat):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong config type: Received {type(format_configuration).__name__}, expected SequenceSequenceMetricsFormat.")

            sequence_to_sequence_metrics(y_true=y_true, 
                                         y_pred=y_pred, 
                                         save_dir=save_dir,
                                         config=config)
    
    def finalize_model_training(self, 
                                save_dir: Union[str, Path], 
                                model_checkpoint: Union[Path, Literal['latest', 'current']],
                                finalize_config: FinalizeSequencePrediction):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict` and the final epoch number.

        Args:
            save_dir (Union[str, Path]): The directory to save the finalized model.
            model_checkpoint (Union[Path, Literal["latest", "current"]]):
                - Path: Loads the model state from a specific checkpoint file.
                - "latest": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is at the end of the `fit()` call.
            finalize_config (FinalizeSequencePrediction): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if not isinstance(finalize_config, FinalizeSequencePrediction):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeSequencePrediction', but got {type(finalize_config).__name__}.")
            raise TypeError()

        # handle save path
        dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        full_path = dir_path / finalize_config.filename
        
        # handle checkpoint
        self._load_model_state_for_finalizing(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
        }
        
        if finalize_config.sequence_length is not None:
            finalized_data[PyTorchCheckpointKeys.SEQUENCE_LENGTH] = finalize_config.sequence_length
        if finalize_config.initial_sequence is not None:
            finalized_data[PyTorchCheckpointKeys.INITIAL_SEQUENCE] = finalize_config.initial_sequence
        
        torch.save(finalized_data, full_path)
        
        _LOGGER.info(f"Finalized model file saved to '{full_path}'")


def info():
    _script_info(__all__)
