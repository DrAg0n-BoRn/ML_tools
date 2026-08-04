from typing import Literal, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np

from ..ML_callbacks._base import _Callback
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_evaluation._eval_sequence import (
    sequence_to_sequence_regression_metrics, 
    sequence_to_sequence_classification_metrics
)
from ..ML_evaluation._eval_classification import classification_metrics
from ..ML_evaluation._eval_regression import regression_metrics
from ..ML_evaluation_captum import captum_sequence_feature_importance
from ..ML_scaler import DragonScaler
from ..ML_configuration import (FormatSequenceValueMetrics,
                            FormatSequenceSequenceMetrics,
                            FinalizeSequenceSequencePrediction,
                            FinalizeSequenceValuePrediction,
                            DragonCheckpointConfig)

from ..keys._keys import PyTorchLogKeys, DatasetKeys, MLTaskKeys, DragonTrainerKeys, ScalerKeys
from .._core import get_logger
from ..path_manager import sanitize_filename

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("Sequence Trainer")


__all__ = [
    "DragonSequenceTrainer"
]


# --- DragonSequenceTrainer ----
class DragonSequenceTrainer(_BaseDragonTrainer):
    """
    Trainer for sequence-based tasks (sequence-to-sequence and sequence-to-value) using PyTorch models.
    
    Supports models returning single Tensors or dictionaries of output heads (dict[str, Tensor]).
    
    Built-in Callbacks: `History`, `TqdmProgressBar`
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Union[Literal["sequence-to-sequence", "sequence-to-value"], str],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'], str],
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
                 target_types: Optional[dict[str, str]] = None,
                 criterion: Union[nn.Module, dict[str, nn.Module], Literal["auto"]] = "auto", 
                 checkpoint_config: Union[DragonCheckpointConfig, Literal["default", "No-Checkpoints"]] = "default",
                 dataloader_workers: int = 2):
        """
        Automates the training process of a PyTorch Sequence Model.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (str | Path): Root directory where all training artifacts will be saved.
            kind (str): Task type ('sequence-to-sequence' or 'sequence-to-value'). 
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            device (str): Computing device ('cpu', 'cuda', 'mps').
            early_stopping_callback: Callback for early stopping.
            lr_scheduler_callback: Callback for learning rate scheduling.
            extra_callbacks (List[Callback] | None): Additional custom callbacks.
            target_types (dict[str, str] | None): Optional mapping of target names to their types ('continuous' or 'categorical').
            criterion (nn.Module | dict | "auto"): Loss function. If "auto", infers MSE or CrossEntropy per target.
            checkpoint_config (Union[DragonCheckpointConfig, Literal["default", "No-Checkpoints"]]): Configuration for model checkpointing.
                - "default": Tracks minimization of validation loss and keeps track of the best 3 checkpoints.
                - "No-Checkpoints": No checkpoints will be saved.
                - `DragonCheckpointConfig`: Custom configuration.
            dataloader_workers (int): Subprocesses for data loading.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            dataloader_workers=dataloader_workers,
            checkpoint_config=checkpoint_config,
            early_stopping_callback=early_stopping_callback,
            lr_scheduler_callback=lr_scheduler_callback,
            extra_callbacks=extra_callbacks
        )
        
        if kind not in [MLTaskKeys.SEQUENCE_SEQUENCE, MLTaskKeys.SEQUENCE_VALUE]:
            raise ValueError(f"'{kind}' is not a valid task type for DragonSequenceTrainer.")

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        self.criterion = criterion
        
        # Validate against Dragon Sequence model architecture mode if present
        if hasattr(self.model, "prediction_mode"):
            key_to_check: str = getattr(self.model, "prediction_mode")
            if key_to_check != self.kind:
                _LOGGER.error(f"Trainer set for '{self.kind}', but model architecture is built for '{key_to_check}'.")
                raise RuntimeError()
        
        # <-- Extract or assign target types -->
        self.target_types = target_types
        if self.target_types is None:
            self.target_types = self._get_dataset_attr(self.train_dataset, DatasetKeys.TARGET_TYPES)
            if not self.target_types:
                _LOGGER.warning(f"No '{DatasetKeys.TARGET_TYPES}' provided or found in the dataset")
        

    def _get_target_names(self) -> list[str]:
        """Helper to extract target variable names from the dataset or model."""
        target_names = self._get_dataset_attr(self.train_dataset, DatasetKeys.TARGET_NAMES)
        if target_names is not None:
            return target_names
        elif hasattr(self.model, "targets"):
            return getattr(self.model, "targets")
        return []

    def _compute_loss(self, outputs: Union[dict[str, torch.Tensor], torch.Tensor], targets: Union[dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Computes total loss across single-head or multi-head sequence outputs.
        Handles dictionary outputs seamlessly.
        """
        target_names = self._get_target_names()

        # --- Case A: Multi-Head Output (Dictionary) ---
        if isinstance(outputs, dict):
            total_loss = torch.tensor(0.0, device=self.device)
            
            for i, target_name in enumerate(target_names if target_names else list(outputs.keys())):
                if target_name not in outputs:
                    continue
                
                pred_t = outputs[target_name]
                
                # Extract ground truth target for this head
                if isinstance(targets, dict):
                    target_t = targets[target_name]
                elif isinstance(targets, torch.Tensor):
                    if targets.ndim == 1:
                        target_t = targets
                    elif self.kind == MLTaskKeys.SEQUENCE_VALUE:
                        target_t = targets[:, i] if targets.ndim == 2 and targets.shape[1] > i else targets
                    else:  # SEQUENCE_SEQUENCE
                        target_t = targets[:, :, i] if targets.ndim == 3 and targets.shape[2] > i else targets
                else:
                    raise TypeError(f"Unsupported target type: {type(targets)}")

                # Determine criterion for target head
                if isinstance(self.criterion, dict):
                    loss_fn = self.criterion.get(target_name, nn.MSELoss())
                elif self.criterion == "auto":
                    # <-- Use exact target_types if available, else fallback -->
                    is_categorical = False
                    if self.target_types and target_name in self.target_types:
                        is_categorical = (self.target_types[target_name] == DatasetKeys.TARGET_CATEGORICAL)
                    else:
                        # Fallback heuristic: If prediction tensor has more dimensions than target or target is integer type, treat as categorical
                        is_categorical = (pred_t.ndim > target_t.ndim or target_t.dtype in [torch.int64, torch.long, torch.int32])

                    if is_categorical:
                        loss_fn = nn.CrossEntropyLoss()
                    else:
                        loss_fn = nn.MSELoss()
                elif isinstance(self.criterion, nn.Module):
                    loss_fn = self.criterion
                else:
                    _LOGGER.error(f"Invalid criterion type: {type(self.criterion)}")
                    raise TypeError()

                # Compute loss with shape alignment
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    pred_flat = pred_t.reshape(-1, pred_t.shape[-1])
                    target_flat = target_t.reshape(-1).long()
                    loss_t = loss_fn(pred_flat, target_flat)
                else:
                    target_t = target_t.float()
                    if pred_t.shape != target_t.shape:
                        if pred_t.ndim == target_t.ndim + 1 and pred_t.shape[-1] == 1:
                            pred_t = pred_t.squeeze(-1)
                    loss_t = loss_fn(pred_t, target_t)

                total_loss = total_loss + loss_t

            return total_loss

        # --- Case B: Single Tensor Output ---
        else:
            if isinstance(targets, dict):
                _LOGGER.error("Model returned a single tensor output, but targets are provided as a dictionary. Cannot determine mapping.")
                raise TypeError()
            
            target = targets.float()
            
            # Shape corrections
            if self.kind == MLTaskKeys.SEQUENCE_VALUE:
                if outputs.ndim == 2 and outputs.shape[1] == 1 and target.ndim == 1:
                    outputs = outputs.squeeze(1)
            elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
                if outputs.ndim == 3 and outputs.shape[2] == 1 and target.ndim == 2:
                    outputs = outputs.squeeze(-1)

            loss_fn = nn.MSELoss() if self.criterion == "auto" else self.criterion  # type: ignore
            
            if not isinstance(loss_fn, nn.Module):
                _LOGGER.error(f"Invalid criterion type: {type(self.criterion)} for single-head output. Must be nn.Module or 'auto'.")
                raise TypeError()
            
            return loss_fn(outputs, target)

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        """Initializes the DataLoaders."""
        self._make_dataloaders(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def _train_step(self) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader):  # type: ignore
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features = features.to(self.device)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)
            elif isinstance(target, dict):
                target = {k: v.to(self.device) for k, v in target.items()}

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self._compute_loss(outputs, target)
            
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            batch_size = features.size(0)
            running_loss += batch_loss * batch_size
            total_samples += batch_size
            
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if total_samples == 0:
            _LOGGER.warning("No samples processed in train_step. Returning 0 loss.")
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for features, target in self.validation_loader:  # type: ignore
                features = features.to(self.device)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)
                elif isinstance(target, dict):
                    target = {k: v.to(self.device) for k, v in target.items()}

                outputs = self.model(features)
                loss = self._compute_loss(outputs, target)
                
                batch_size = features.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
        if total_samples == 0:
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        return {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples}

    def _predict_for_eval(self, dataloader: DataLoader):
        """
        Yields predictions batch by batch for evaluation as dictionaries.
        Isolates continuous variables for inverse scaling to prevent corrupting categorical indices.
        """
        self.model.eval()
        self.model.to(self.device)
        
        scaler: Optional[DragonScaler] = None
        scaler = self._get_dataset_attr(self.train_dataset, ScalerKeys.TARGET_SCALER)
        if scaler is None:
            scaler = self._get_dataset_attr(self.train_dataset, "scaler")
            
        target_names = self._get_target_names()

        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                outputs = self.model(features)

                # --- Standardize to Dictionaries ---
                if not isinstance(outputs, dict):
                    outputs = {target_names[0]: outputs}
                if not isinstance(target, dict):
                    target = {target_names[0]: target}

                y_pred_batch = {}
                y_true_batch = {}

                continuous_preds = []
                continuous_trues = []
                continuous_keys = []

                # --- Isolate Categorical vs Continuous ---
                for t in target_names:
                    if t not in outputs or t not in target:
                        continue
                        
                    p = outputs[t]
                    t_val = target[t].to(self.device)
                    
                    is_categorical = (self.target_types and self.target_types.get(t) == DatasetKeys.TARGET_CATEGORICAL)

                    if is_categorical:
                        if p.ndim > (2 if self.kind == MLTaskKeys.SEQUENCE_SEQUENCE else 1):
                            p = p.argmax(dim=-1)
                        y_pred_batch[t] = p.cpu().numpy()
                        y_true_batch[t] = t_val.cpu().numpy()
                    else:
                        continuous_keys.append(t)
                        continuous_preds.append(p)
                        continuous_trues.append(t_val)

                # --- Safe Inverse Scaling for Continuous Targets ---
                if scaler and continuous_keys:
                    # Stack continuous features to match scaler's expected shape
                    p_cont = torch.stack(continuous_preds, dim=-1) if len(continuous_preds) > 1 else continuous_preds[0].unsqueeze(-1)
                    t_cont = torch.stack(continuous_trues, dim=-1) if len(continuous_trues) > 1 else continuous_trues[0].unsqueeze(-1)

                    orig_p_shape = p_cont.shape
                    orig_t_shape = t_cont.shape
                    
                    p_flat = p_cont.reshape(-1, len(continuous_keys))
                    t_flat = t_cont.reshape(-1, len(continuous_keys))
                    
                    p_unscaled = scaler.inverse_transform(p_flat)
                    t_unscaled = scaler.inverse_transform(t_flat)
                    
                    p_cont = torch.as_tensor(p_unscaled, device=self.device).reshape(orig_p_shape)
                    t_cont = torch.as_tensor(t_unscaled, device=self.device).reshape(orig_t_shape)

                    for i, t in enumerate(continuous_keys):
                        y_pred_batch[t] = p_cont[..., i].cpu().numpy()
                        y_true_batch[t] = t_cont[..., i].cpu().numpy()
                else:
                    for i, t in enumerate(continuous_keys):
                        y_pred_batch[t] = continuous_preds[i].cpu().numpy()
                        y_true_batch[t] = continuous_trues[i].cpu().numpy()

                yield y_pred_batch, None, y_true_batch

    def evaluate(self, 
                 model_checkpoint: Union[Path, str, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[FormatSequenceValueMetrics, FormatSequenceSequenceMetrics]] = None,
                 test_format_configuration: Optional[Union[FormatSequenceValueMetrics, FormatSequenceSequenceMetrics]] = None):
        """
        Evaluates the model and logs metrics.
        
        Args:
            model_checkpoint (Path | str | "best" | "current"): Checkpoint to evaluate. Can be a path to a .pth file, or "best"/"current" for the best or most recent checkpoint.
            test_data (DataLoader | Dataset | None): Optional test dataset. If None, only validation metrics are computed.
            val_format_configuration (FormatSequenceValueMetrics | FormatSequenceSequenceMetrics | None): Optional configuration for formatting validation metrics.
            test_format_configuration (FormatSequenceValueMetrics | FormatSequenceSequenceMetrics | None): Optional configuration for formatting test metrics. If None, validation configuration is reused.
        """
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(self.training_directory_root)
        validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
        
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (FormatSequenceValueMetrics, FormatSequenceSequenceMetrics)):
                _LOGGER.error(f"Invalid 'val_format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
        
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
    
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_format_configuration)
            
            test_configuration_validated = test_format_configuration or val_format_configuration
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data,
                           format_configuration=test_configuration_validated)
        else:
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_format_configuration)

    def _evaluate(self, 
                  save_dir: Union[str, Path], 
                  model_checkpoint: Union[Path, Literal["best", "current"]],
                  data: Optional[Union[DataLoader, Dataset]],
                  format_configuration: Optional[Union[FormatSequenceValueMetrics, FormatSequenceSequenceMetrics]] = None):
        """Private evaluation helper. Routes to task-specific evaluation modules per target."""
        self._load_model_state_wrapper(model_checkpoint)
        eval_loader, _ = self._prepare_eval_data(data, self.validation_dataset)
        save_dir_path = self._validate_save_dir(save_dir)

        target_names = self._get_target_names()
        all_preds = {t: [] for t in target_names}
        all_true = {t: [] for t in target_names}

        processed_any = False
        for y_pred_dict, _, y_true_dict in self._predict_for_eval(eval_loader):
            processed_any = True
            for t in target_names:
                if t in y_pred_dict:
                    all_preds[t].append(y_pred_dict[t])
                    all_true[t].append(y_true_dict[t])

        if not processed_any:
            _LOGGER.error("Evaluation failed: No data was processed.")
            return

        # Iterate per-target to apply specific classification or regression metrics
        for t in target_names:
            if not all_preds[t]:
                continue
                
            y_pred = np.concatenate(all_preds[t], axis=0)
            y_true = np.concatenate(all_true[t], axis=0)
            
            is_categorical = (self.target_types and self.target_types.get(t) == DatasetKeys.TARGET_CATEGORICAL)
            target_save_dir = save_dir_path / sanitize_filename(t)

            # Route evaluation using the formalized composite configuration classes
            if self.kind == MLTaskKeys.SEQUENCE_VALUE:
                if is_categorical:
                    classification_metrics(
                        y_true=y_true, y_pred=y_pred, save_dir=target_save_dir, 
                        config=format_configuration.classification_config if format_configuration else None
                    )
                else:
                    regression_metrics(
                        y_true=y_true, y_pred=y_pred, save_dir=target_save_dir, 
                        config=format_configuration.regression_config if format_configuration else None
                    )

            elif self.kind == MLTaskKeys.SEQUENCE_SEQUENCE:
                # Pass the full configuration to seq-to-seq metrics to retain per-step plot styling.
                # The seq-to-seq functions will then route the sub-configs to the overall metric reports.
                if is_categorical:
                    sequence_to_sequence_classification_metrics(
                        y_true=y_true, y_pred=y_pred, save_dir=target_save_dir, 
                        config=format_configuration
                    )
                else:
                    sequence_to_sequence_regression_metrics(
                        y_true=y_true, y_pred=y_pred, save_dir=target_save_dir, 
                        config=format_configuration
                    )

    def explain_captum(self,
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       feature_names: Optional[list[str]] = None,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50,
                       verbose: int = 0):
        """
        Explains sequence model predictions using Captum's Integrated Gradients.
        Automatically wraps multi-head dictionary models into continuous PyTorch modules.
        
        Args:
            explain_dataset (Dataset | None): Dataset to use for explanation. Defaults to validation dataset.
            n_samples (int): Number of random samples to use for explanation.
            feature_names (list[str] | None): Optional feature names for plotting. If None, attempts to extract from dataset.
            target_names (list[str] | None): Optional target names for plotting. If None, attempts to extract from dataset or model.
            n_steps (int): Number of steps for Integrated Gradients approximation.
            verbose (int): Verbosity level for Captum output.
        """
        dataset_to_use = explain_dataset if explain_dataset is not None else self.validation_dataset
        if dataset_to_use is None:
            _LOGGER.error("No dataset available for explanation.")
            return
        
        captum_save_dir = self._validate_save_dir(self.training_directory_root / DragonTrainerKeys.CAPTUM_DIR)

        def _get_samples(ds, n):
            loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
            data_iter = iter(loader)
            features, targets = next(data_iter)
            return features, targets

        input_data, _ = _get_samples(dataset_to_use, n_samples)
        
        if feature_names is None:
            feature_names = self._get_dataset_attr(dataset_to_use, DatasetKeys.FEATURE_NAMES)
            if feature_names is None:
                _LOGGER.warning("'feature_names' not provided or found. Generic names will be used.")

        if target_names is None:
            target_names = self._get_target_names()

        # Internal Wrapper to turn multi-head dict models into single-tensor modules for Captum
        class _CaptumDictWrapper(nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner_model = inner_model

            def forward(self, x):
                out = self.inner_model(x)
                if isinstance(out, dict):
                    # Concatenate outputs across target heads
                    tensors = []
                    for v in out.values():
                        if v.ndim > 2 and v.shape[-1] == 1:
                            v = v.squeeze(-1)
                        tensors.append(v)
                    stacked = torch.stack(tensors, dim=-1) if tensors else list(out.values())[0]
                else:
                    stacked = out
                    
                # If the output is Sequence-to-Sequence (Batch, Seq_Len, Targets), 
                # average across the sequence dimension so Captum only deals with a 2D (Batch, Targets) space.
                if stacked.ndim == 3:
                    stacked = stacked.mean(dim=1)
                    
                return stacked

        wrapped_model = _CaptumDictWrapper(self.model)

        captum_sequence_feature_importance(
            model=wrapped_model,
            input_data=input_data,
            feature_names=feature_names,
            save_dir=captum_save_dir,
            target_names=target_names,
            n_steps=n_steps,
            device=self.device,
            verbose=verbose
        )
    
    def finalize_model_training(self, 
                                finalize_config: Union[FinalizeSequenceSequencePrediction, FinalizeSequenceValuePrediction]):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        Uses the current model state and training metadata to create a standardized finalized artifact.

        Args:
            finalize_config (FinalizeSequenceSequencePrediction | FinalizeSequenceValuePrediction): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if self.kind == MLTaskKeys.SEQUENCE_SEQUENCE and not isinstance(finalize_config, FinalizeSequenceSequencePrediction):
            _LOGGER.error(f"Received a wrong finalize configuration for task {self.kind}: '{type(finalize_config).__name__}'.")
            raise TypeError()
        elif self.kind == MLTaskKeys.SEQUENCE_VALUE and not isinstance(finalize_config, FinalizeSequenceValuePrediction):
            _LOGGER.error(f"Received a wrong finalize configuration for task {self.kind}: '{type(finalize_config).__name__}'.")
            raise TypeError()
        
        self._save_finalized_artifact(finalize_config=finalize_config)

