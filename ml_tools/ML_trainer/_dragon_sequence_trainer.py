from typing import Literal, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np

from ..ML_callbacks._base import _Callback
from ..ML_callbacks._checkpoint import DragonModelCheckpoint
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_evaluation import sequence_to_sequence_metrics, sequence_to_value_metrics
from ..ML_evaluation_captum import captum_feature_importance
from ..ML_configuration import (FormatSequenceValueMetrics,
                            FormatSequenceSequenceMetrics,
                            
                            FinalizeSequenceSequencePrediction,
                            FinalizeSequenceValuePrediction)

from ..path_manager import make_fullpath
from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, DatasetKeys, MLTaskKeys, MagicWords, DragonTrainerKeys, ScalerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonSequenceTrainer")


__all__ = [
    "DragonSequenceTrainer"
]


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
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
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
        self._make_dataloaders(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle
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
        
        Automatically checks for 'scaler'.
        
        Yields:
            tuple: A tuple containing (y_pred_batch, y_prob_batch, y_true_batch).
                   y_prob_batch is always None for sequence tasks.
        """
        self.model.eval()
        self.model.to(self.device)
        
        # --- Check for Scaler ---
        # DragonDatasetSequence stores it as 'scaler'
        scaler = None
        if hasattr(self.train_dataset, ScalerKeys.TARGET_SCALER):
            scaler = getattr(self.train_dataset, ScalerKeys.TARGET_SCALER)
            if scaler is not None:
                _LOGGER.debug("Sequence scaler detected. Un-scaling predictions and targets.")
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                target = target.to(self.device)
                
                output = self.model(features)

                # --- Automatic Un-scaling Logic ---
                if scaler:
                    # 1. Reshape for scaler (N, 1) or (N*Seq, 1)
                    original_out_shape = output.shape
                    original_target_shape = target.shape
                    
                    # Flatten sequence dims
                    output_flat = output.reshape(-1, 1)
                    target_flat = target.reshape(-1, 1)
                    
                    # 2. Inverse Transform
                    output_flat = scaler.inverse_transform(output_flat)
                    target_flat = scaler.inverse_transform(target_flat)
                    
                    # 3. Restore
                    output = output_flat.reshape(original_out_shape)
                    target = target_flat.reshape(original_target_shape)

                # Move to CPU
                y_pred_batch = output.cpu().numpy()
                y_true_batch = target.cpu().numpy()
                y_prob_batch = None

                yield y_pred_batch, y_prob_batch, y_true_batch
                
    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[FormatSequenceValueMetrics, 
                                                          FormatSequenceSequenceMetrics]]=None,
                 test_format_configuration: Optional[Union[FormatSequenceValueMetrics, 
                                                           FormatSequenceSequenceMetrics]]=None):
        """
        Evaluates the model, routing to the correct evaluation function.

        Args:
            model_checkpoint (Path | "best" | "current"): 
                - Path to a valid checkpoint for the model.
                - If 'best', the best checkpoint will be loaded.
                - If 'current', use the current state of the trained model.
            save_dir (str | Path): Directory to save all reports and plots.
            test_data (DataLoader | Dataset | None): Optional Test data.
            val_format_configuration: Optional configuration for validation metrics.
            test_format_configuration: Optional configuration for test metrics.
        """
        # Validate inputs using base helpers
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(save_dir)
        
        # Validate val configuration
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (FormatSequenceValueMetrics, FormatSequenceSequenceMetrics)):
                _LOGGER.error(f"Invalid 'val_format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
        
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
                           data=None,
                           format_configuration=val_format_configuration)
            
            # Validate test configuration
            test_configuration_validated = None
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (FormatSequenceValueMetrics, FormatSequenceSequenceMetrics)):
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
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_format_configuration)
        
    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: object):
        """
        Private evaluation helper.
        """
        # load model checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Prepare Data using Base Helper
        eval_loader, _ = self._prepare_eval_data(data, self.validation_dataset)

        # Gather Predictions
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
            if format_configuration and isinstance(format_configuration, FormatSequenceValueMetrics):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong config type: Received {type(format_configuration).__name__}, expected SequenceValueMetricsFormat.")
            
            sequence_to_value_metrics(y_true=y_true, 
                                      y_pred=y_pred, 
                                      save_dir=save_dir,
                                      config=config)

        elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
            config = None
            if format_configuration and isinstance(format_configuration, FormatSequenceSequenceMetrics):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong config type: Received {type(format_configuration).__name__}, expected SequenceSequenceMetricsFormat.")

            sequence_to_sequence_metrics(y_true=y_true, 
                                         y_pred=y_pred, 
                                         save_dir=save_dir,
                                         config=config)
            
    def explain_captum(self,
                       save_dir: Union[str, Path],
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       feature_names: Optional[list[str]] = None,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50):
        """
        Explains sequence model predictions using Captum's Integrated Gradients.

        This method calculates global feature importance by aggregating attributions across 
        the time dimension. 
        - For **multivariate** sequences, it highlights which variables (channels) are most influential.
        - For **univariate** sequences, it attributes importance to the single signal feature.

        Args:
            save_dir (str | Path): Directory to save the importance plots and CSV reports.
            explain_dataset (Dataset | None): A specific dataset to sample from. If None, the 
                                            trainer's validation dataset is used.
            n_samples (int): The number of samples to use for the explanation (background + inputs).
            feature_names (List[str] | None): Names of the features (signals). If None, attempts to extract them from the dataset attribute.
            target_names (List[str] | None): Names of the model outputs (e.g., for Seq2Seq or Multivariate output). If None, attempts to extract them from the dataset attribute.
            n_steps (int): Number of integral approximation steps.

        Note:
            For univariate data (Shape: N, Seq_Len), the 'feature' is the signal itself. 
        """            
        dataset_to_use = explain_dataset if explain_dataset is not None else self.validation_dataset
        if dataset_to_use is None:
            _LOGGER.error("No dataset available for explanation.")
            return

        # Helper to sample data (same as DragonTrainer)
        def _get_samples(ds, n):
            loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
            data_iter = iter(loader)
            features, targets = next(data_iter)
            return features, targets

        input_data, _ = _get_samples(dataset_to_use, n_samples)
        
        if feature_names is None:
             if hasattr(dataset_to_use, DatasetKeys.FEATURE_NAMES):
                feature_names = dataset_to_use.feature_names # type: ignore
             else:
                # If retrieval fails, leave it as None. 
                _LOGGER.warning("'feature_names' not provided and not found in dataset. Generic names will be used.")
            
        if target_names is None:
            if hasattr(dataset_to_use, DatasetKeys.TARGET_NAMES):
                target_names = dataset_to_use.target_names # type: ignore
            else:
                # If retrieval fails, leave it as None. 
                _LOGGER.warning("'target_names' not provided and not found in dataset. Generic names will be used.")

        # Sequence models usually output [N, 1] (Value) or [N, Seq, 1] (Seq2Seq)
        # captum_feature_importance handles the aggregation.
        
        captum_feature_importance(
            model=self.model,
            input_data=input_data,
            feature_names=feature_names,
            save_dir=save_dir,
            target_names=target_names,
            n_steps=n_steps,
            device=self.device
        )
    
    def finalize_model_training(self, 
                                save_dir: Union[str, Path], 
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                finalize_config: Union[FinalizeSequenceSequencePrediction, FinalizeSequenceValuePrediction]):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict` and the final epoch number.

        Args:
            save_dir (Union[str, Path]): The directory to save the finalized model.
            model_checkpoint (Union[Path, Literal["best", "current"]]):
                - Path: Loads the model state from a specific checkpoint file.
                - "best": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is.
            finalize_config (FinalizeSequencePrediction): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if self.kind == MLTaskKeys.SEQUENCE_SEQUENCE and not isinstance(finalize_config, FinalizeSequenceSequencePrediction):
            _LOGGER.error(f"Received a wrong finalize configuration for task {self.kind}: {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.SEQUENCE_VALUE and not isinstance(finalize_config, FinalizeSequenceValuePrediction):
            _LOGGER.error(f"Received a wrong finalize configuration for task {self.kind}: {type(finalize_config).__name__}.")
            raise TypeError()
        
        # handle checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task
        }
        
        if finalize_config.sequence_length is not None:
            finalized_data[PyTorchCheckpointKeys.SEQUENCE_LENGTH] = finalize_config.sequence_length
        if finalize_config.initial_sequence is not None:
            finalized_data[PyTorchCheckpointKeys.INITIAL_SEQUENCE] = finalize_config.initial_sequence
        
        # Save using base helper
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=save_dir,
            filename=finalize_config.filename
        )

