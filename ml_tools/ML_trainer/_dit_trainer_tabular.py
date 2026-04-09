from typing import Union, Optional, Literal
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..ML_callbacks._base import _Callback
from ..ML_callbacks._checkpoint import DragonModelCheckpoint
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_evaluation._dit_metrics import dit_generation_metrics
from ..ML_configuration import FormatTabularDiffusionMetrics, FinalizeTabularDiffusion
from ..ML_models_diffusion import DragonAutoencoder, DragonDiT, DragonDiTGuided

from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, MLTaskKeys, DragonTrainerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonTabularDiTTrainer")


__all__ = [
    "DragonTabularDiTTrainer",
]


class DragonTabularDiTTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: Union[DragonDiT, DragonDiTGuided], 
                 token_embedder: DragonAutoencoder,
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'], str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint] = None,
                 early_stopping_callback: Optional[_DragonEarlyStopping] = None,
                 lr_scheduler_callback: Optional[_DragonLRScheduler] = None,
                 extra_callbacks: Optional[list[_Callback]] = None,
                 dataloader_workers: int = 2,
                 cfg_dropout_rate: float = 0.15):
        """
        Trainer class specifically designed for training DiT models on tabular data. 
        It handles the unique training loop of DiT, including the preparation of noisy inputs and time steps, and supports both guided and unguided training modes.
        
        Args:
            model (Union[DragonDiT, DragonDiTGuided]): The DiT model to be trained. Can be either the standard unguided DiT or the guided version that incorporates target information.
            token_embedder (DragonAutoencoder): A pretrained autoencoder used to embed tabular features into a continuous latent space suitable for diffusion modeling. The embedder's weights are frozen during DiT training.
            train_dataset (Dataset): The training dataset containing tabular data. Each sample should be a tensor of shape (num_features,) or a tuple (features, target) if using the guided DiT.
            validation_dataset (Dataset): The validation dataset for evaluating model performance during training. Should have the same format as the training dataset.
            save_dir (Union[str, Path]): The root directory where all training artifacts (checkpoints, metrics, plots) will be saved. Subdirectories will be automatically created for organization.
            optimizer (torch.optim.Optimizer): The optimizer used for training the DiT model.
            device (Union[Literal['cuda', 'mps', 'cpu'], str]): The device on which to train the model.
            checkpoint_callback (Optional[DragonModelCheckpoint]): Optional callback for saving model checkpoints during training.
            early_stopping_callback (Optional[_DragonEarlyStopping]): Optional callback for early stopping based on chosen metric performance.
            lr_scheduler_callback (Optional[_DragonLRScheduler]): Optional callback for learning rate scheduling during training.
            extra_callbacks (Optional[list[_Callback]]): Optional list of additional callbacks to integrate into the training loop.
            dataloader_workers (int): Number of worker processes for data loading.
            cfg_dropout_rate (float): The dropout rate for the guided DiT model, which randomly drops the conditioning information during training to improve robustness. Recommended between 0.1 and 0.3.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader_workers=dataloader_workers,
            checkpoint_callback=checkpoint_callback,
            early_stopping_callback=early_stopping_callback,
            lr_scheduler_callback=lr_scheduler_callback,
            extra_callbacks=extra_callbacks,
            save_dir=save_dir
        )
        self.token_embedder = token_embedder
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.cfg_dropout_rate = cfg_dropout_rate
        
        self.is_guided = isinstance(model, DragonDiTGuided)
        self.kind = MLTaskKeys.DIFFUSION
        
        # Ensure token embedder weights are frozen during DiT training
        self.token_embedder.to(self.device)
        self.token_embedder.eval()
        self.token_embedder.requires_grad_(False)

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
        self._make_dataloaders(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def _prepare_flow_matching_batch(self, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_1: The embedded ground truth tokens [batch_size, num_features, embed_dim]
        """
        batch_size, num_features, embed_dim = x_1.shape
        # 1. Sample noise x_0 from standard normal distribution with the same shape as x_1
        x_0 = torch.randn_like(x_1, device=self.device)
        
        # 2. Sample time t uniformly between 0 and 1
        # Multiplying by a value slightly larger than 1 and clamping ensures that exactly 1.0 is mathematically possible and included in the training distribution.
        t = torch.rand(batch_size, 1, 1, device=self.device) * 1.001
        t = t.clamp(0.0, 1.0)
        
        # 3. Compute x_t (interpolated state) and the target velocity
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        
        # shapes: 
        # x_t [batch_size, num_features, embed_dim], 
        # t [batch_size, 1, 1], 
        # target_velocity [batch_size, num_features, embed_dim]
        return x_t, t, target_velocity

    def _train_step(self) -> dict[str, float]:
        self.model.train()
        # Explicitly enforce embedder eval mode
        self.token_embedder.eval()
        
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader): # type: ignore
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            features = features.to(self.device)
            
            targets = None
            if self.is_guided and isinstance(batch, (tuple, list)) and len(batch) > 1:
                targets = batch[1].to(self.device)
                if targets.ndim == 1:
                    targets = targets.view(-1, 1)
            
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)
            
            # Encode raw data into tokens without tracking gradients
            with torch.no_grad():
                tokens = self.token_embedder(features)
                
            # Prepare the flow matching batch by sampling noise, time steps, and computing the target velocity
            x_t, t, target_velocity = self._prepare_flow_matching_batch(tokens)

            self.optimizer.zero_grad()
            
            # Forward pass through the model. If guided, apply dropout to the conditioning information.
            if self.is_guided and targets is not None:
                # During training, randomly drop the conditioning information with a certain probability to improve robustness. 
                # This forces the model to learn to predict the velocity both with and without the target information.
                drop_mask = (torch.rand(features.size(0), 1, device=self.device) < self.cfg_dropout_rate)
                predicted_velocity = self.model(x_t, t, targets, drop_mask)
            else:
                predicted_velocity = self.model(x_t, t)
            
            # Compute MSE loss between predicted velocity and target velocity
            loss = F.mse_loss(predicted_velocity, target_velocity)
            
            # Backpropagation and optimization step
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

    def _validation_step(self) -> dict[str, float]:
        self.model.eval()
        self.token_embedder.eval()
        
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.validation_loader: # type: ignore
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                features = features.to(self.device)
                
                targets = None
                if self.is_guided and isinstance(batch, (tuple, list)) and len(batch) > 1:
                    targets = batch[1].to(self.device)
                    if targets.ndim == 1:
                        targets = targets.view(-1, 1)

                tokens = self.token_embedder(features)
                x_t, t, target_velocity = self._prepare_flow_matching_batch(tokens)

                if self.is_guided and targets is not None:
                    # Do not drop condition during validation
                    drop_mask = torch.zeros(features.size(0), 1, device=self.device, dtype=torch.bool)
                    predicted_velocity = self.model(x_t, t, targets, drop_mask)
                else:
                    predicted_velocity = self.model(x_t, t)

                loss = F.mse_loss(predicted_velocity, target_velocity)
                
                batch_size = features.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
        if total_samples == 0:
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        return {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples}

    def evaluate(self, 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[FormatTabularDiffusionMetrics] = None,
                 test_format_configuration: Optional[FormatTabularDiffusionMetrics] = None):
        """
        Evaluates the diffusion model by comparing generated distributions against the real distributions.
        
        Args:
            model_checkpoint (Union[Path, Literal["best", "current"]]): Which checkpoint to load for evaluation. Can be a specific .pth file path or "best"/"current" to use the corresponding checkpoint from training.
            test_data (Optional[Union[DataLoader, Dataset]]): Optional test dataset to evaluate on after validation. If None, only validation evaluation will be performed.
            val_format_configuration (Optional[FormatTabularDiffusionMetrics]): Configuration for formatting validation metrics.
            test_format_configuration (Optional[FormatTabularDiffusionMetrics]): Configuration for formatting test metrics.
        """
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(self.training_directory_root)
        
        validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR

        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_format_configuration)
            
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{test_metrics_path.name}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data,
                           format_configuration=test_format_configuration)
        else:
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_format_configuration)

    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: Optional[FormatTabularDiffusionMetrics]):
        
        self._load_model_state_wrapper(model_checkpoint)
        eval_loader, _ = self._prepare_eval_data(data, self.validation_dataset)
        
        self.model.eval()
        self.model.to(self.device)
        self.token_embedder.eval()
        self.token_embedder.to(self.device)
        
        num_indices: list[int] = self.token_embedder.numerical_indices # type: ignore
        cat_indices: list[int] = self.token_embedder.categorical_indices # type: ignore
        
        # total number of features (sequence length).
        num_features: int = self.token_embedder.num_features # type: ignore 
        
        all_real_features = []
        all_real_targets = []
        
        # 1. Gather all real data
        with torch.no_grad():
            for batch in eval_loader:
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                all_real_features.append(features.cpu())
                
                if self.is_guided and isinstance(batch, (tuple, list)) and len(batch) > 1:
                    targets = batch[1]
                    if targets.ndim == 1:
                        targets = targets.view(-1, 1)
                    all_real_targets.append(targets.cpu())

        real_features_tensor = torch.cat(all_real_features, dim=0)
        total_samples = real_features_tensor.shape[0]
        
        all_targets_tensor = torch.cat(all_real_targets, dim=0) if all_real_targets else None

        # 2. Generate matching amount of synthetic tokens via Heun's ODE
        gen_tokens_list = []
        batch_size = self._batch_size
        num_steps = 20
        
        # _LOGGER.info(f"Generating {total_samples} synthetic samples to compare against real distribution...")
        
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                curr_batch_size = end_idx - i
                
                x_t = torch.randn(curr_batch_size, num_features, self.model.embed_dim, device=self.device) # type: ignore
                t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
                
                curr_targets = None
                if self.is_guided and all_targets_tensor is not None:
                    curr_targets = all_targets_tensor[i:end_idx].to(self.device)
                
                for step in range(num_steps):
                    t_val = t_steps[step]
                    t_next = t_steps[step + 1]
                    dt = t_next - t_val
                    
                    t_tensor = torch.full((curr_batch_size, 1, 1), t_val.item(), device=self.device)
                    
                    if self.is_guided and curr_targets is not None:
                        # Full conditional generation (mask = False)
                        mask = torch.zeros((curr_batch_size, 1), device=self.device, dtype=torch.bool)
                        v_pred = self.model(x_t, t_tensor, curr_targets, mask)
                    else:
                        v_pred = self.model(x_t, t_tensor)
                        
                    x_euler = x_t + v_pred * dt
                    
                    if step == num_steps - 1:
                        x_t = x_euler
                        break
                        
                    t_next_tensor = torch.full((curr_batch_size, 1, 1), t_next.item(), device=self.device)
                    if self.is_guided and curr_targets is not None:
                        v_pred_next = self.model(x_euler, t_next_tensor, curr_targets, mask)
                    else:
                        v_pred_next = self.model(x_euler, t_next_tensor)
                        
                    v_heun = 0.5 * (v_pred + v_pred_next)
                    x_t = x_t + v_heun * dt
                    
                gen_tokens_list.append(x_t)
                
        gen_tokens_tensor = torch.cat(gen_tokens_list, dim=0)

        # 3. Decode Tokens & Inverse Transform Numerical Features
        with torch.no_grad():
            gen_num_reconstructed, gen_cat_logits = self.token_embedder._decode_to_raw_tensors(gen_tokens_tensor)
            
            if num_indices and self.token_embedder.scaler is not None:
                # Scale back real data (which is kept as raw features)
                real_inv = self.token_embedder.scaler.inverse_transform(real_features_tensor.to(self.device))
                real_num = real_inv[:, num_indices].cpu().numpy()
            else:
                real_num = real_features_tensor[:, num_indices].numpy() if num_indices else None
                
            gen_num = gen_num_reconstructed.cpu().numpy() if num_indices else None

        # 4. Process Categorical Features
        real_cat_list = []
        gen_cat_list = []
        if cat_indices:
            for idx in cat_indices:
                real_cat_list.append(real_features_tensor[:, idx].numpy())
                
            for logits in gen_cat_logits:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                gen_cat_list.append(preds.cpu().numpy())

        # 5. Extract Schema Information
        schema = self.token_embedder.schema
        num_names = [schema.feature_names[i] for i in num_indices] if num_indices else None
        cat_names = [schema.feature_names[i] for i in cat_indices] if cat_indices else None
        
        cat_class_maps = []
        if cat_indices and cat_names is not None:
            for name in cat_names:
                if schema.categorical_mappings and name in schema.categorical_mappings:
                    cat_class_maps.append(schema.categorical_mappings[name])
                else:
                    cat_class_maps.append(None)

        # _LOGGER.info("Calculating diffusion distribution metrics for tabular data...")
        dit_generation_metrics(
            real_num=real_num,
            gen_num=gen_num,
            num_target_names=num_names,
            real_cat_list=real_cat_list, # type: ignore
            gen_cat_list=gen_cat_list, # type: ignore
            cat_target_names=cat_names,
            cat_class_maps=cat_class_maps,
            save_dir=save_dir,
            config=format_configuration
        )

    def finalize_model_training(self, 
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                finalize_config: FinalizeTabularDiffusion):
        """
        Saves a finalized, inference-ready DiT model state to a .pth file.
        
        Args:
            model_checkpoint (Union[Path, Literal['best', 'current']]): Which checkpoint to load for finalization. Can be a specific .pth file path or "best"/"current" to use the corresponding checkpoint from training.
            finalize_config (FinalizeTabularDiffusion): Configuration object containing metadata about the training run and instructions for finalization.
        """
        self._load_model_state_wrapper(model_checkpoint)
        
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task,
        }
        
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=self.training_directory_root,
            filename=finalize_config.filename
        )
        
    #override device changing methods
    def to_cpu(self):
        """Moves the trainer, model, and token embedder to the CPU."""
        super().to_cpu()
        self.token_embedder.to(self.device)
        
    def to_device(self, device: str):
        """Moves the trainer, model, and token embedder to the specified device."""
        super().to_device(device)
        self.token_embedder.to(self.device)
