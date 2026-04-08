from typing import Union, Optional, Literal
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..ML_models_diffusion._autoencoder import DragonAutoencoder
from ..ML_configuration._finalize import FinalizeAutoencoder
from ..ML_configuration._metrics import FormatAutoencoderMetrics
from ..ML_evaluation import autoencoder_metrics
from ..ML_callbacks._base import _Callback
from ..ML_callbacks._checkpoint import DragonModelCheckpoint
from ..ML_callbacks._early_stop import _DragonEarlyStopping, DragonPrecheltEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler

from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, MLTaskKeys, DragonTrainerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonAutoencoderTrainer")


__all__ = [
    "DragonAutoencoderTrainer",
]


class DragonAutoencoderTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: DragonAutoencoder, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
                 dataloader_workers: int = 2):
        """
        Automates the unsupervised training process of a DragonAutoencoder.
        
        Args:
            model (DragonAutoencoder): The autoencoder model to be trained.
            train_dataset (Dataset): The dataset to use for training. Should yield either (features) or (features, target) tuples, but only the features will be used for training since this is an unsupervised task.
            validation_dataset (Dataset): The dataset to use for validation during training. Should have the same format as train_dataset.
            optimizer (torch.optim.Optimizer): The optimizer to use for training the model.
            device (Union[Literal['cuda', 'mps', 'cpu'],str]): The device to train on.
            checkpoint_callback (Optional[DragonModelCheckpoint]): A callback to save model checkpoints during training. Can be None to disable checkpointing.
            early_stopping_callback (Optional[_DragonEarlyStopping]): A callback to perform early stopping based on a chosen metric. Can be None to disable early stopping. Must work with the uncertainty weighting loss technique, loss values can drop below zero.
            lr_scheduler_callback (Optional[_DragonLRScheduler]): A callback to adjust the learning rate during training. Can be None to disable learning rate scheduling.
            extra_callbacks (Optional[list[_Callback]]): A list of any additional callbacks to use during training.
            dataloader_workers (int): The number of worker processes to use for data loading.
        """
        # Block incompatible params
        if isinstance(early_stopping_callback, DragonPrecheltEarlyStopping):
            _LOGGER.error("DragonPrecheltEarlyStopping is incompatible because the uncertainty weighting loss can drop below zero.")
            raise TypeError()
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader_workers=dataloader_workers,
            checkpoint_callback=checkpoint_callback,
            early_stopping_callback=early_stopping_callback,
            lr_scheduler_callback=lr_scheduler_callback,
            extra_callbacks=extra_callbacks)
        
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = MLTaskKeys.AUTOENCODER

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
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
        
        for batch_idx, batch in enumerate(self.train_loader): # type: ignore
            # Handle dataset yielding either (features, target) or just features
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            features = features.to(self.device)
            
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            self.optimizer.zero_grad()
            
            # Encode & Decode
            tokens = self.model(features)
            num_reconstructed, cat_logits = self.model._decode(tokens) # type: ignore
            
            # Internal Loss with Uncertainty Weighting
            loss = _compute_autoencoder_loss(features, num_reconstructed, cat_logits, self.model) # type: ignore
            
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            batch_size = features.size(0)
            running_loss += batch_loss * batch_size
            total_samples += batch_size
            
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
            
        if total_samples == 0:
            _LOGGER.warning("No samples processed in a train_step. Returning 0 loss.")
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.validation_loader: # type: ignore
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                features = features.to(self.device)
                
                tokens = self.model(features)
                num_reconstructed, cat_logits = self.model._decode(tokens) # type: ignore
                loss = _compute_autoencoder_loss(features, num_reconstructed, cat_logits, self.model) # type: ignore
                
                running_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
        if total_samples == 0:
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples} # type: ignore
        return logs

    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[FormatAutoencoderMetrics] = None,
                 test_format_configuration: Optional[FormatAutoencoderMetrics] = None):
        """
        Evaluates the autoencoder's reconstruction performance.
        
        Args:
            save_dir (Union[str, Path]): Directory where evaluation metrics and artifacts will be saved.
            model_checkpoint (Union[Path, Literal["best", "current"]]): Which checkpoint to load for evaluation. Can be a specific .pth file path or "best"/"current" to use the corresponding checkpoint from training.
            test_data (Optional[Union[DataLoader, Dataset]]): Optional test dataset to evaluate on after validation. If None, only validation evaluation will be performed.
            val_format_configuration (Optional[FormatAutoencoderMetrics]): Configuration for formatting validation metrics and artifacts. If None, default formatting will be applied.
            test_format_configuration (Optional[FormatAutoencoderMetrics]): Configuration for formatting test metrics and artifacts. If None, default formatting will be applied.
        """
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(save_dir)
        
        # Validate val_format_configuration
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, FormatAutoencoderMetrics):
                _LOGGER.error(f"Invalid type for 'val_format_configuration': '{type(val_format_configuration)}'. Expected 'FormatAutoencoderMetrics' or None.")
                raise ValueError()
            else:
                validated_val_format_config = val_format_configuration
        else:
            validated_val_format_config = None

        # Validate test_format_configuration
        if test_format_configuration is not None:
            if not isinstance(test_format_configuration, FormatAutoencoderMetrics):
                _LOGGER.error(f"Invalid type for 'test_format_configuration': '{type(test_format_configuration)}'. Expected 'FormatAutoencoderMetrics' or None.")
                raise ValueError()
            else:
                validated_test_format_config = test_format_configuration
        else:
            validated_test_format_config = None

        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            
            validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=validated_val_format_config)
            
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{test_metrics_path.name}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data,
                           format_configuration=validated_test_format_config)
        else:
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=validated_val_format_config)

    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: Optional[FormatAutoencoderMetrics]):
        
        self._load_model_state_wrapper(model_checkpoint)
        eval_loader, dataset_for_artifacts = self._prepare_eval_data(data, self.validation_dataset)
        
        self.model.eval()
        self.model.to(self.device)
        
        num_indices: list[int] = self.model.numerical_indices # type: ignore
        cat_indices: list[int] = self.model.categorical_indices # type: ignore
        
        all_num_true, all_num_pred = [], []
        # List of lists for categoricals: [feature_idx][batch_items]
        all_cat_true = [[] for _ in cat_indices]
        all_cat_pred = [[] for _ in cat_indices]
        all_cat_prob = [[] for _ in cat_indices]

        with torch.no_grad():
            for batch in eval_loader:
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                features = features.to(self.device)
                
                tokens = self.model(features)
                num_reconstructed, cat_logits = self.model._decode(tokens) # type: ignore
                
                # Store numerical reconstructions
                if num_indices:
                    all_num_true.append(features[:, num_indices].cpu().numpy())
                    all_num_pred.append(num_reconstructed.cpu().numpy())
                    
                # Store categorical reconstructions
                if cat_indices:
                    for i, idx in enumerate(cat_indices):
                        true_cat = features[:, idx].cpu().long().numpy()
                        logits = cat_logits[i]
                        probs = torch.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        
                        all_cat_true[i].append(true_cat)
                        all_cat_pred[i].append(preds.cpu().numpy())
                        all_cat_prob[i].append(probs.cpu().numpy())
        
        # --- Compile Data for Autoencoder Metrics ---
        y_true_num = np.concatenate(all_num_true, axis=0) if num_indices else None
        y_pred_num = np.concatenate(all_num_pred, axis=0) if num_indices else None
        num_target_names: Optional[list[str]] = [self.model.schema.feature_names[i] for i in num_indices] if num_indices else None # type: ignore

        cat_true_list = [np.concatenate(all_cat_true[i], axis=0) for i in range(len(cat_indices))] if cat_indices else None
        cat_pred_list = [np.concatenate(all_cat_pred[i], axis=0) for i in range(len(cat_indices))] if cat_indices else None
        cat_prob_list = [np.concatenate(all_cat_prob[i], axis=0) for i in range(len(cat_indices))] if cat_indices else None
        
        cat_target_names = None
        cat_class_maps = None
        
        if cat_indices:
            cat_target_names = []
            cat_class_maps = []
            for idx in cat_indices:
                feat_name = self.model.schema.feature_names[idx] # type: ignore
                cat_target_names.append(feat_name)
                
                class_map: Optional[dict[str, int]] = None
                if self.model.schema.categorical_mappings and feat_name in self.model.schema.categorical_mappings: # type: ignore
                    class_map = self.model.schema.categorical_mappings[feat_name] # type: ignore
                cat_class_maps.append(class_map)

        # _LOGGER.info(f"Generating autoencoder reconstruction metrics...")
        
        autoencoder_metrics(
            y_true_num=y_true_num,
            y_pred_num=y_pred_num,
            num_target_names=num_target_names,
            cat_true_list=cat_true_list,
            cat_pred_list=cat_pred_list,
            cat_prob_list=cat_prob_list,
            cat_target_names=cat_target_names,
            cat_class_maps=cat_class_maps,
            save_dir=save_dir,
            config=format_configuration
        )

    def finalize_model_training(self, 
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                save_dir: Union[str, Path],
                                finalize_config: FinalizeAutoencoder):
        """
        Saves a finalized, inference-ready model state to a .pth file.
        
        Args:
            model_checkpoint (Union[Path, Literal['best', 'current']]): Which checkpoint to load for finalization. Can be a specific .pth file path or "best"/"current" to use the corresponding checkpoint from training.
            save_dir (Union[str, Path]): Directory where the finalized model state will be saved.
            finalize_config (FinalizeAutoencoder): Configuration object containing metadata about the training run and instructions for finalization.
        """
        self._load_model_state_wrapper(model_checkpoint)
        
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task,
        }
        
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=save_dir,
            filename=finalize_config.filename
        )





# loss helper
def _compute_autoencoder_loss(
    x: torch.Tensor, 
    num_reconstructed: torch.Tensor, 
    cat_logits: list[torch.Tensor], 
    embedder: DragonAutoencoder
) -> torch.Tensor:
    """
    Computes the reconstruction loss for the DragonAutoencoder using homoscedastic uncertainty weighting to dynamically balance multi-modal objectives.
    
    Source Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (https://arxiv.org/abs/1705.07115)
    """
    loss = torch.tensor(0.0, device=x.device)
    
    # 1. Numerical Loss (MSE) with Uncertainty Weighting
    if embedder.numerical_indices:
        x_numerical = x[:, embedder.numerical_indices].float()
        mse_loss = F.mse_loss(num_reconstructed, x_numerical)
        
        # L_num = 0.5 * exp(-log_var) * MSE + 0.5 * log_var
        num_loss_weighted = 0.5 * torch.exp(-embedder.log_var_num) * mse_loss + 0.5 * embedder.log_var_num # type: ignore
        loss = loss + num_loss_weighted[0]
        
    # 2. Categorical Loss (Cross-Entropy) with Uncertainty Weighting
    if embedder.categorical_indices:
        x_categorical = x[:, embedder.categorical_indices].long()
        ce_loss = torch.tensor(0.0, device=x.device)
        for i, logits in enumerate(cat_logits):
            targets = x_categorical[:, i]
            ce_loss = ce_loss + F.cross_entropy(logits, targets)
            
        # Normalize by the number of categorical features to ensure a stable starting scale
        ce_loss = ce_loss / len(embedder.categorical_indices)
        
        # L_cat = exp(-log_var) * CE + 0.5 * log_var
        cat_loss_weighted = torch.exp(-embedder.log_var_cat) * ce_loss + 0.5 * embedder.log_var_cat # type: ignore
        loss = loss + cat_loss_weighted[0]
    
    # Edge case safeguard (should not happen in practice)
    if not embedder.numerical_indices and not embedder.categorical_indices:
        loss.requires_grad = True
        
    return loss
