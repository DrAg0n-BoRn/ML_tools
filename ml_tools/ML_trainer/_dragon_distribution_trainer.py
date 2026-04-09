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
from ..ML_configuration import FormatRegressionMetrics, FormatMultiTargetRegressionMetrics, FinalizeRegression, FinalizeMultiTargetRegression
from ..ML_evaluation import regression_metrics, multi_target_regression_metrics, distribution_metrics, multi_target_distribution_metrics
from ..ML_evaluation_captum import captum_feature_importance

from ..keys._keys import PyTorchLogKeys, MLTaskKeys, DragonTrainerKeys, ScalerKeys, PyTorchCheckpointKeys, DatasetKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonDistributionTrainer")


__all__ = [
    "DragonDistributionTrainer",
]


class DragonDistributionTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Literal["regression", "multitarget regression"],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'], str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint] = None,
                 early_stopping_callback: Optional[_DragonEarlyStopping] = None,
                 lr_scheduler_callback: Optional[_DragonLRScheduler] = None,
                 extra_callbacks: Optional[list[_Callback]] = None,
                 criterion: Union[nn.Module, Literal["auto"]] = "auto", 
                 dataloader_workers: int = 2):
        """
        Automates the training process of a PyTorch Model for probabilistic distribution prediction.
        
        Built-in Callbacks: `History`, `TqdmProgressBar`
        
        Args:
            model (nn.Module): The PyTorch model to train. It must output a single concatenated tensor 
                of shape (batch_size, 2 * num_targets). The first half of the outputs represents the 
                mean, and the second half represents the variance logits.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (str | Path): The root directory where all training artifacts (checkpoints, metrics, plots) will be saved. Subdirectories will be automatically created.
            kind (str): Used to redirect to the correct process. Must be either 'regression' or 'multitarget regression'.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            checkpoint_callback (DragonModelCheckpoint | None): Callback for saving model checkpoints.
            early_stopping_callback (DragonEarlyStopping | None): Callback to halt training based on validation.
            lr_scheduler_callback (DragonLRScheduler | None): Callback to adjust the learning rate.
            extra_callbacks (list[Callback] | None): A list of extra callbacks to use during training.
            criterion (nn.Module | "auto"): The loss function to use. If "auto", it will default to `nn.GaussianNLLLoss()`. Must be compatible with `(mean, target, variance)` inputs.
            dataloader_workers (int): Subprocesses for data loading.
            
        Note:
            - The trainer automatically applies `torch.nn.functional.softplus` to the variance half of the 
              model's output to ensure strictly positive variance during training and evaluation. The model 
              should simply output raw, unconstrained linear logits for the variance half.

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
            save_dir=save_dir)
        
        if kind not in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
            _LOGGER.error(f"Invalid 'kind' argument: '{kind}'. Must be either {MLTaskKeys.REGRESSION} or {MLTaskKeys.MULTITARGET_REGRESSION}.")
            raise ValueError()

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        
        if criterion == "auto":
            # Default PyTorch Gaussian Negative Log Likelihood Loss
            self.criterion = nn.GaussianNLLLoss()
        else:
            self.criterion = criterion

    def _create_dataloaders(self, batch_size: int, shuffle: bool):
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
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(features)
            
            # --- Distribution Output Splitting ---
            # Assume first half is mean, second half is variance logits
            mean, var_logits = torch.tensor_split(output, 2, dim=-1)
            
            # Enforce positive variance
            var = torch.nn.functional.softplus(var_logits)
            
            # Reshape output to match target for single-target regression
            if self.kind == MLTaskKeys.REGRESSION:
                if mean.ndim == 2 and mean.shape[1] == 1 and target.ndim == 1:
                    mean = mean.squeeze(1)
                    var = var.squeeze(1)
                    
            # GaussianNLLLoss takes (input, target, var) where input is the mean
            loss = self.criterion(mean, target, var)
            
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
        running_loss = 0.0
        
        with torch.no_grad():
            for features, target in self.validation_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                
                # --- Distribution Output Splitting ---
                mean, var_logits = torch.tensor_split(output, 2, dim=-1)
                var = torch.nn.functional.softplus(var_logits)
                
                if self.kind == MLTaskKeys.REGRESSION:
                    if mean.ndim == 2 and mean.shape[1] == 1 and target.ndim == 1:
                        mean = mean.squeeze(1)
                        var = var.squeeze(1)
                
                loss = self.criterion(mean, target, var)
                running_loss += loss.item() * features.size(0)
                
        if not self.validation_loader.dataset: # type: ignore
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / len(self.validation_loader.dataset)} # type: ignore
        return logs
    
    def _predict_for_eval(self, dataloader: DataLoader):
        """
        Yields predictions batch by batch.
        For distribution tasks, it yields (mean, variance, true_target).
        """
        self.model.eval()
        self.model.to(self.device)
        
        target_scaler = None
        if hasattr(self.train_dataset, ScalerKeys.TARGET_SCALER):
            target_scaler = getattr(self.train_dataset, ScalerKeys.TARGET_SCALER)
            if target_scaler is not None:
                 _LOGGER.debug("Target scaler detected. Un-scaling predictions (mean/variance) and targets.")
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                target = target.to(self.device)
                
                output = self.model(features)
                
                # Split and constrain variance
                mean, var_logits = torch.tensor_split(output, 2, dim=-1)
                var = torch.nn.functional.softplus(var_logits)

                if target_scaler is not None and target_scaler.std_ is not None:
                    # 1. Un-scale Mean and Target using DragonScaler (operates on tensors natively)
                    mean_unscaled = target_scaler.inverse_transform(mean)
                    target_unscaled = target_scaler.inverse_transform(target)
                    
                    # 2. Un-scale Variance
                    # Variance scales by the square of the standard deviation.
                    std_tensor = target_scaler.std_.to(self.device)
                    scale_factor_sq = (std_tensor + 1e-8) ** 2
                    var_unscaled = var * scale_factor_sq
                    
                    yield mean_unscaled.cpu().numpy(), var_unscaled.cpu().numpy(), target_unscaled.cpu().numpy()

                else:
                    yield mean.cpu().numpy(), var.cpu().numpy(), target.cpu().numpy()

    def evaluate(self, 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics
                    ]]=None,
                 test_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics
                    ]]=None):
        """
        Evaluates the probabilistic distribution model and generates comprehensive reports and plots.
        
        This evaluation computes both standard point-prediction regression metrics (RMSE, MAE) 
        using the predicted mean, as well as distribution-specific metrics (Gaussian NLL, 95% PICP, 
        and 95% MPIW) utilizing both the predicted mean and variance. 

        Args:
            model_checkpoint (Path | "best" | "current"): 
                - Path to a valid `.pth` checkpoint file.
                - "best": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's in-memory state as-is.
            test_data (DataLoader | Dataset | None): Optional holdout test data to evaluate the model's 
                performance. If provided, metrics for both validation and test sets will be saved into 
                separate subdirectories.
            val_format_configuration (FormatRegressionMetrics | FormatMultiTargetRegressionMetrics | None): 
                Optional configuration to customize the appearance of the validation set plots (colors, fonts, etc.).
            test_format_configuration (FormatRegressionMetrics | FormatMultiTargetRegressionMetrics | None): 
                Optional configuration to customize the appearance of the test set plots.
        """
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(self.training_directory_root)
        
        validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
        
        # Validate val configuration
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (FormatRegressionMetrics, FormatMultiTargetRegressionMetrics)):
                _LOGGER.error(f"Invalid 'format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
            else:
                val_configuration_validated = val_format_configuration
        else:
            val_configuration_validated = None
        
        # Validate test data and dispatch
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
                
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_configuration_validated)
            
            # Validate test configuration
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (FormatRegressionMetrics, FormatMultiTargetRegressionMetrics)):
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
            else:
                test_configuration_validated = None
            
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{test_metrics_path.name}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           data=None,
                           format_configuration=val_configuration_validated)

    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics
                    ]]=None):
        
        self._load_model_state_wrapper(model_checkpoint)
        eval_loader, dataset_for_artifacts = self._prepare_eval_data(data, self.validation_dataset)
        
        all_means, all_vars, all_true = [], [], []
        for mean_b, var_b, true_b in self._predict_for_eval(eval_loader):
            all_means.append(mean_b)
            all_vars.append(var_b)
            all_true.append(true_b)

        if not all_true:
            _LOGGER.error("Evaluation failed: No data was processed.")
            return

        mean_pred = np.concatenate(all_means)
        var_pred = np.concatenate(all_vars)
        y_true = np.concatenate(all_true)

        # from ..ML_evaluation._distribution import distribution_metrics, multi_target_distribution_metrics
        
        if self.kind == MLTaskKeys.REGRESSION:
            config = format_configuration if isinstance(format_configuration, FormatRegressionMetrics) else None
            
            # 1. Standard Metrics (treating the mean as the point prediction)
            regression_metrics(y_true=y_true.flatten(), 
                               y_pred=mean_pred.flatten(), 
                               save_dir=save_dir,
                               config=config)
            
            # 2. Distribution Specific Metrics
            distribution_metrics(y_true=y_true.flatten(),
                                 mean_pred=mean_pred.flatten(),
                                 var_pred=var_pred.flatten(),
                                 save_dir=save_dir,
                                 config=config)
                               
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION:
            try:
                target_names = dataset_for_artifacts.target_names # type: ignore
            except AttributeError:
                num_targets = y_true.shape[1]
                target_names = [f"target_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no 'target_names' attribute. Using generic names.")
                
            config = format_configuration if isinstance(format_configuration, FormatMultiTargetRegressionMetrics) else None
            
            # 1. Standard Metrics
            multi_target_regression_metrics(y_true=y_true, 
                                            y_pred=mean_pred,
                                            target_names=target_names, 
                                            save_dir=save_dir,
                                            config=config)
            
            # 2. Distribution Specific Metrics
            multi_target_distribution_metrics(y_true=y_true,
                                              mean_pred=mean_pred,
                                              var_pred=var_pred,
                                              target_names=target_names,
                                              save_dir=save_dir,
                                              config=config)
    
    def explain_captum(self,
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       feature_names: Optional[list[str]] = None,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50,
                       verbose: int = 0):
        """
        Explains the model's MEAN predictions using Captum's Integrated Gradients.
        
        Args:
            explain_dataset (Dataset | None): Dataset to sample from. Defaults to validation set.
            n_samples (int): Number of samples to evaluate.
            feature_names (list[str] | None): Feature names. Required for Tabular tasks.
            target_names (list[str] | None): Names for the model outputs.
            n_steps (int): Number of interpolation steps.
        """
        dataset_to_use = explain_dataset if explain_dataset is not None else self.validation_dataset
        if dataset_to_use is None:
            _LOGGER.error("No dataset available for explanation.")
            return
        
        #set subdirectory for Captum explanations
        captum_save_dir = self._validate_save_dir(self.training_directory_root / DragonTrainerKeys.CAPTUM_DIR)

        # Efficient sampling helper
        def _get_samples(ds, n):
            loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
            data_iter = iter(loader)
            features, targets = next(data_iter)
            return features, targets

        input_data, _ = _get_samples(dataset_to_use, n_samples)
        
        # Get Feature Names
        if feature_names is None:
            if hasattr(dataset_to_use, DatasetKeys.FEATURE_NAMES):
                feature_names = dataset_to_use.feature_names # type: ignore
            else:
                _LOGGER.error(f"Could not extract `feature_names`. It must be provided if the dataset does not have it.")
                raise ValueError()

        # Handle Target Names
        if target_names is None:
            if hasattr(dataset_to_use, DatasetKeys.TARGET_NAMES):
                target_names = dataset_to_use.target_names # type: ignore
            elif self.kind == MLTaskKeys.REGRESSION:
                target_names = ["Output"]

        # --- Wrap the model to isolate the MEAN for Captum ---
        class _MeanWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                out = self.model(x)
                # Split and return only the mean predictions
                mean, _ = torch.tensor_split(out, 2, dim=-1)
                
                # Squeeze to 1D if it's a single target to match standard Captum expectations
                if mean.ndim == 2 and mean.shape[1] == 1:
                    mean = mean.squeeze(1)
                    
                return mean
                
        wrapped_model = _MeanWrapper(self.model)

        captum_feature_importance(
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
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                finalize_config: Union[FinalizeRegression, FinalizeMultiTargetRegression]):
        """
        Saves a finalized, "inference-ready" model state to a .pth file.

        This method saves the model's `state_dict`, the final epoch number, and optional configuration for the task at hand.

        Args:
            model_checkpoint (Path | "best" | "current"):
                - Path: Loads the model state from a specific checkpoint file.
                - "best": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is.
            finalize_config (object): A data class instance specific to the ML task containing task-specific metadata required for inference.
        """
        if self.kind == MLTaskKeys.REGRESSION and not isinstance(finalize_config, FinalizeRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeRegression', but got {type(finalize_config).__name__}.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION and not isinstance(finalize_config, FinalizeMultiTargetRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiTargetRegression', but got {type(finalize_config).__name__}.")
            raise TypeError()
                
        # Handle checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Create finalized data
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task
        }

        # Parse config
        if finalize_config.target_name is not None:
            finalized_data[PyTorchCheckpointKeys.TARGET_NAME] = finalize_config.target_name
        if finalize_config.target_names is not None:
            finalized_data[PyTorchCheckpointKeys.TARGET_NAMES] = finalize_config.target_names

        # Save model file using base helper
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=self.training_directory_root,
            filename=finalize_config.filename
        )
