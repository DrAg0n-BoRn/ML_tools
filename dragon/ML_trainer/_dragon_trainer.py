from typing import Literal, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np

from ..ML_callbacks._base import _Callback
from ..ML_callbacks._early_stop import _DragonEarlyStopping
from ..ML_callbacks._scheduler import _DragonLRScheduler
from ..ML_scaler import DragonScaler
from ..ML_evaluation import classification_metrics, regression_metrics, shap_summary_plot
from ..ML_evaluation import multi_target_regression_metrics, multi_label_classification_metrics
from ..ML_evaluation_captum import captum_feature_importance
from ..ML_configuration import (FormatRegressionMetrics, 
                            FormatMultiTargetRegressionMetrics,
                            FormatBinaryClassificationMetrics,
                            FormatMultiClassClassificationMetrics,
                            FormatMultiLabelBinaryClassificationMetrics,
                            FinalizeBinaryClassification,
                            FinalizeMultiClassClassification,
                            FinalizeMultiLabelBinaryClassification,
                            FinalizeMultiTargetRegression,
                            FinalizeRegression,
                            DragonCheckpointConfig)

from ..keys._keys import PyTorchLogKeys, DatasetKeys, MLTaskKeys, DragonTrainerKeys, ScalerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("Dragon Trainer")


__all__ = [
    "DragonTrainer",
]


# --- DragonTrainer ----
class DragonTrainer(_BaseDragonTrainer):
    """
    Automates the training process of a PyTorch Model for tabular and general machine learning tasks.
    
    This trainer specifically supports single and multi-target regression, 
    as well as binary, multiclass, and multi-label classification tasks.
    
    Built-in Callbacks: `History`, `TqdmProgressBar`.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Union[Literal["regression", 
                               "binary classification", 
                               "multiclass classification", 
                               "multitarget regression", 
                               "multilabel binary classification"], str],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
                 criterion: Union[nn.Module,Literal["auto"]] = "auto", 
                 checkpoint_config: Union[DragonCheckpointConfig, Literal["default", "No-Checkpoints"]] = "default",
                 dataloader_workers: int = 2):
        """
        Initializes the DragonTrainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (Union[str, Path]): Root directory where all training artifacts (checkpoints, metrics, plots) will be saved.
            kind (str): The specific general ML task to perform. Must be one of the supported task string literals.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device (Union[Literal['cuda', 'mps', 'cpu'], str]): The device to run training on.
            early_stopping_callback (Optional[_DragonEarlyStopping]): Callback to stop training early if metric stops improving.
            lr_scheduler_callback (Optional[_DragonLRScheduler]): Callback for learning rate scheduling.
            extra_callbacks (Optional[list[_Callback]]): Additional custom callbacks to apply during training.
            criterion (Union[nn.Module, Literal["auto"]]): The loss function. If "auto", it is inferred from the `kind` parameter.
            checkpoint_config (Union[DragonCheckpointConfig, Literal["default", "No-Checkpoints"]]): Configuration for model checkpointing.
                - "default": Tracks minimization of validation loss and keeps track of the best 3 checkpoints.
                - "No-Checkpoints": No checkpoints will be saved.
                - `DragonCheckpointConfig`: Custom configuration.
            dataloader_workers (int): Number of subprocesses to use for data loading.
            
        <br>

        ### **Note about Loss Function (Criterion):**
            - **Regression & Multi-Target Regression:** Use `nn.MSELoss`, `nn.L1Loss`, or `nn.HuberLoss`. The model must output exactly as many raw logits as there are continuous targets (e.g., `[N, 1]` for single, `[N, T]` for multi-target).
            - **Binary Classification (Single-Label):** `nn.BCEWithLogitsLoss` is standard. The model should output a single unnormalized logit per sample (`[N, 1]`).
            - **Multi-Class Classification:** `nn.CrossEntropyLoss` is the standard choice. It expects raw, unnormalized logits for each class (`[N, C]`).
            - **Multi-Label Binary Classification:** `nn.BCEWithLogitsLoss` is correct as it treats each class as an independent Bernoulli trial. The model should output one logit per binary target (`[N, C]`).
            
            - *Important:* PyTorch's `BCEWithLogitsLoss` and `CrossEntropyLoss` apply `Sigmoid` and `Softmax` internally for numerical stability. Ensure the model's final layer **does not** include an activation function if using these criterions.
        """
        # Call the base class constructor with common parameters
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
        
        if kind not in [MLTaskKeys.REGRESSION,
                        MLTaskKeys.BINARY_CLASSIFICATION,
                        MLTaskKeys.MULTICLASS_CLASSIFICATION,
                        MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION,
                        MLTaskKeys.MULTITARGET_REGRESSION]:
            raise ValueError(f"'{kind}' is not a valid task type for DragonTrainer.")

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        self._classification_threshold: float = 0.5
        
        if criterion == "auto":
            if kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                self.criterion = nn.MSELoss()
            elif kind in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
                self.criterion = nn.BCEWithLogitsLoss()
            elif kind == MLTaskKeys.MULTICLASS_CLASSIFICATION:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # criterion should be an instance of nn.Module
        if not isinstance(self.criterion, nn.Module):
            _LOGGER.error(f"The provided criterion is not a valid PyTorch loss module: {type(self.criterion)}")
            raise TypeError()
        self.criterion: nn.Module
    
        self._move_criterion_to_device()
        
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
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(features)
            
            if self.kind in MLTaskKeys.ALL_BINARY_TASKS:
                target = target.float()

            if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION]:
                if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                    output = output.squeeze(1)
                
            loss = self.criterion(output, target)
            
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
            for features, target in self.validation_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                
                if self.kind in MLTaskKeys.ALL_BINARY_TASKS:
                    target = target.float()

                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION]:
                    if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                        output = output.squeeze(1)
                
                loss = self.criterion(output, target)
                
                running_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
        if not self.validation_loader.dataset: # type: ignore
            _LOGGER.warning("No samples processed in _validation_step. Returning 0 loss.")
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples}
        return logs
    
    def _predict_for_eval(self, dataloader: DataLoader):
        self.model.eval()
        self.model.to(self.device)
        
        target_scaler: Optional[DragonScaler] = None
        if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
            target_scaler = self._get_dataset_attr(self.train_dataset, ScalerKeys.TARGET_SCALER)
            if target_scaler is not None:
                _LOGGER.debug("Target scaler detected. Un-scaling predictions and targets for metric calculation.")
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                target = target.to(self.device) 
                
                output = self.model(features)

                y_pred_batch = None
                y_prob_batch = None
                y_true_batch = None

                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                    if target_scaler:
                        original_out_shape = output.shape
                        original_target_shape = target.shape
                        
                        if output.ndim == 1: output = output.reshape(-1, 1)
                        if target.ndim == 1: target = target.reshape(-1, 1)
                            
                        output = target_scaler.inverse_transform(output)
                        target = target_scaler.inverse_transform(target)
                        
                        if len(original_out_shape) == 1: output = output.flatten()
                        if len(original_target_shape) == 1: target = target.flatten()

                    y_pred_batch = output.cpu().numpy()
                    y_true_batch = target.cpu().numpy()
                    
                elif self.kind == MLTaskKeys.BINARY_CLASSIFICATION:
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.squeeze(1)
                        
                    probs_pos = torch.sigmoid(output) 
                    preds = (probs_pos >= self._classification_threshold).int()
                    y_pred_batch = preds.cpu().numpy()
                    
                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.stack([probs_neg, probs_pos], dim=1).cpu().numpy()
                    y_true_batch = target.cpu().numpy()

                elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_pred_batch = preds.cpu().numpy()
                    y_prob_batch = probs.cpu().numpy()
                    y_true_batch = target.cpu().numpy()

                elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
                    probs = torch.sigmoid(output)
                    preds = (probs >= self._classification_threshold).int()
                    y_pred_batch = preds.cpu().numpy()
                    y_prob_batch = probs.cpu().numpy()
                    y_true_batch = target.cpu().numpy()
                
                yield y_pred_batch, y_prob_batch, y_true_batch
                
    def evaluate(self, 
                 model_checkpoint: Union[Path, str, Literal["best", "current"]],
                 classification_threshold: Optional[float] = None,
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics,
                        FormatBinaryClassificationMetrics,
                        FormatMultiClassClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics
                    ]]=None,
                 test_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics,
                        FormatBinaryClassificationMetrics,
                        FormatMultiClassClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics
                    ]]=None):
        """
        Evaluates the trained model and generates task-specific evaluation metrics.

        Args:
            model_checkpoint (Union[Path, str, Literal["best", "current"]]): The specific checkpoint state to load before evaluating.
            classification_threshold (Optional[float]): The threshold used for calculating binary classification metrics.
            test_data (Optional[Union[DataLoader, Dataset]]): An optional test dataset to evaluate model performance completely separated from validation.
            val_format_configuration (Optional[object]): Formatting configuration object for validation metric outputs.
            test_format_configuration (Optional[object]): Formatting configuration object for test metric outputs.
        """
        # Validate inputs using base helpers
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(self.training_directory_root)
        
        validation_metrics_path = save_path / DragonTrainerKeys.VALIDATION_METRICS_DIR
        
        if self.kind not in MLTaskKeys.ALL_BINARY_TASKS:
            threshold_validated = 0.5
        elif classification_threshold is None:
            _LOGGER.error(f"The classification threshold must be provided for '{self.kind}'.")
            raise ValueError()
        elif classification_threshold <= 0.0 or classification_threshold >= 1.0:
            _LOGGER.error(f"A classification threshold of {classification_threshold} is invalid. Must be in the range (0.0 - 1.0).")
            raise ValueError()
        else:
            threshold_validated = classification_threshold
        
        if val_format_configuration is not None:
            if not isinstance(val_format_configuration, (FormatRegressionMetrics, 
                                                        FormatMultiTargetRegressionMetrics,
                                                        FormatBinaryClassificationMetrics,
                                                        FormatMultiClassClassificationMetrics,
                                                        FormatMultiLabelBinaryClassificationMetrics)):
                _LOGGER.error(f"Invalid 'format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
            else:
                val_configuration_validated = val_format_configuration
        else:
            val_configuration_validated = None
        
        if test_data is not None:
            if not isinstance(test_data, (DataLoader, Dataset)):
                _LOGGER.error(f"Invalid type for 'test_data': '{type(test_data)}'.")
                raise ValueError()
            test_data_validated = test_data
                
            test_metrics_path = save_path / DragonTrainerKeys.TEST_METRICS_DIR
            
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)
            
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (FormatRegressionMetrics, 
                                                        FormatMultiTargetRegressionMetrics,
                                                        FormatBinaryClassificationMetrics,
                                                        FormatMultiClassClassificationMetrics,
                                                        FormatMultiLabelBinaryClassificationMetrics)):
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
            
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           classification_threshold=threshold_validated,
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)

    def _evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 classification_threshold: float,
                 data: Optional[Union[DataLoader, Dataset]],
                 format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics,
                        FormatBinaryClassificationMetrics,
                        FormatMultiClassClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics
                    ]]=None):
        
        self._classification_threshold = classification_threshold
        self._load_model_state_wrapper(model_checkpoint)
        
        eval_loader, dataset_for_artifacts = self._prepare_eval_data(data, self.validation_dataset)
        
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

        if self.kind == MLTaskKeys.REGRESSION:
            config = None
            if format_configuration and isinstance(format_configuration, FormatRegressionMetrics):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            regression_metrics(y_true=y_true.flatten(), 
                               y_pred=y_pred.flatten(), 
                               save_dir=save_dir,
                               config=config)
        
        elif self.kind in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            class_map = self._get_dataset_attr(dataset_for_artifacts, DatasetKeys.CLASS_MAP)
            
            # Fallback to building class_map from classes list if class_map is missing
            if class_map is None:
                classes = self._get_dataset_attr(dataset_for_artifacts, DatasetKeys.CLASSES)
                if classes and isinstance(classes, list):
                    class_map = {name: idx for idx, name in enumerate(classes)}
            
            if class_map is None:
                _LOGGER.warning("Dataset has no 'class_map' or 'classes' attribute. Using generics.")
            elif not isinstance(class_map, dict):
                _LOGGER.warning(f"Extracted 'class_map' is not a dictionary: '{type(class_map)}'. Using generics.")
                class_map = None
            
            config = None
            if format_configuration:
                if self.kind == MLTaskKeys.BINARY_CLASSIFICATION and isinstance(format_configuration, FormatBinaryClassificationMetrics):
                    config = format_configuration
                elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION and isinstance(format_configuration, FormatMultiClassClassificationMetrics):
                    config = format_configuration
                else:
                    _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
  
            classification_metrics(save_dir=save_dir,
                                   y_true=y_true,
                                   y_pred=y_pred,
                                   y_prob=y_prob,
                                   class_map=class_map,
                                   config=config)
        
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION:
            
            target_names = self._get_dataset_attr(dataset_for_artifacts, DatasetKeys.TARGET_NAMES)
            
            if target_names is None:
                num_targets = y_true.shape[1]
                target_names = [f"target_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no '{DatasetKeys.TARGET_NAMES}' attribute. Using generic names.")
            
            config = None
            if format_configuration and isinstance(format_configuration, FormatMultiTargetRegressionMetrics):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            multi_target_regression_metrics(y_true=y_true, 
                                            y_pred=y_pred,
                                            target_names=target_names, 
                                            save_dir=save_dir,
                                            config=config)
            
        elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION:
            
            target_names = self._get_dataset_attr(dataset_for_artifacts, DatasetKeys.TARGET_NAMES)
            
            if target_names is None:
                num_targets = y_true.shape[1]
                target_names = [f"label_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no '{DatasetKeys.TARGET_NAMES}' attribute. Using generic names.")
            
            if y_prob is None:
                _LOGGER.error("Evaluation for multi_label_classification requires probabilities (y_prob).")
                return
            
            config = None
            if format_configuration and isinstance(format_configuration, FormatMultiLabelBinaryClassificationMetrics):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")

            multi_label_classification_metrics(y_true=y_true,
                                               y_pred=y_pred,
                                               y_prob=y_prob,
                                               target_names=target_names,
                                               save_dir=save_dir,
                                               config=config)
    
    def explain_shap(self,
                explain_dataset: Optional[Dataset] = None, 
                n_samples: int = 300,
                feature_names: Optional[list[str]] = None,
                target_names: Optional[list[str]] = None,
                explainer_type: Literal['deep', 'kernel'] = 'kernel'):
        """
        Explains model predictions using SHAP and saves all artifacts.
        
        NOTE: SHAP support is limited to single-target tasks (Regression, Binary Classification, and Multiclass Classification).

        The background data is automatically sampled from the trainer's training dataset.
        Support is generally limited to single-target regression and standard classification tasks.

        Args:
            explain_dataset (Optional[Dataset]): A specific dataset to explain. Defaults to the internal validation set if None.
            n_samples (int): The number of samples to use for both background and explanation.
            feature_names (Optional[list[str]]): Feature names. If None, it will be extracted from the Dataset.
            target_names (Optional[list[str]]): Target names for multi-target tasks.
            explainer_type (Literal['deep', 'kernel']): The explainer to use ('deep' for efficiency, 'kernel' for model-agnostic but slower explanation).
        """
        # --- 1. Compatibility Guard ---
        valid_shap_tasks = [
            MLTaskKeys.REGRESSION, 
            MLTaskKeys.BINARY_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_CLASSIFICATION
        ]
        
        if self.kind not in valid_shap_tasks:
            _LOGGER.warning(f"SHAP explanation is deprecated for task '{self.kind}' due to instability. Please use 'explain_captum()' instead.")
            return
        
        def _get_random_sample(dataset: Dataset, num_samples: int):
            if dataset is None:
                return None
            
            dataset_len = len(dataset) # type: ignore
            if dataset_len == 0:
                return None
            
            loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
            batch_size = min(num_samples, 64, dataset_len) 
            
            loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=loader_workers
            )
            
            collected_features = []
            num_collected = 0
            
            for features, _ in loader:
                collected_features.append(features)
                num_collected += features.size(0)
                if num_collected >= num_samples:
                    break 
            
            if not collected_features:
                return None
            
            full_data = torch.cat(collected_features, dim=0)
            
            if full_data.size(0) > num_samples:
                return full_data[:num_samples]
            
            return full_data
        
        background_data = _get_random_sample(self.train_dataset, n_samples)
        if background_data is None:
            _LOGGER.error("Trainer's train_dataset is empty or invalid. Skipping SHAP analysis.")
            return

        target_dataset = explain_dataset if explain_dataset is not None else self.validation_dataset
        instances_to_explain = _get_random_sample(target_dataset, n_samples)
        if instances_to_explain is None:
            _LOGGER.error("Explanation dataset is empty or invalid. Skipping SHAP analysis.")
            return
        
        if feature_names is None:
            
            feature_names = self._get_dataset_attr(target_dataset, DatasetKeys.FEATURE_NAMES)
            
            if feature_names is None:
                _LOGGER.error(f"Could not extract `feature_names` from the dataset. It must be provided if the dataset object does not have a '{DatasetKeys.FEATURE_NAMES}' attribute.")
                raise ValueError()
            
        self.model.to(self.device)
        shap_save_dir = self._validate_save_dir(self.training_directory_root / DragonTrainerKeys.SHAP_DIR)

        shap_summary_plot(
            model=self.model,
            background_data=background_data,
            instances_to_explain=instances_to_explain,
            feature_names=feature_names,
            save_dir=shap_save_dir,
            explainer_type=explainer_type,
            device=self.device
        )


    def explain_captum(self,
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       feature_names: Optional[list[str]] = None,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50,
                       verbose: int = 0):
        """
        Explains model predictions using Captum's Integrated Gradients to generate feature importance scores and bar charts.
        
        Args:
            explain_dataset (Optional[Dataset]): Dataset to sample from. Defaults to the internal validation set if None.
            n_samples (int): Number of samples to evaluate.
            feature_names (Optional[list[str]]): Feature names. Required for tabular tasks; attempts to extract from dataset attributes if None.
            target_names (Optional[list[str]]): Names for the model outputs or classes.
            n_steps (int): Number of interpolation steps.
            verbose (int): Verbosity level for logging operations.
        """         
        # 2. Prepare Data
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
                _LOGGER.error(f"Could not extract '{DatasetKeys.FEATURE_NAMES}'. It must be provided if the dataset does not have it.")
                raise ValueError()

        if target_names is None:
            # 1. Prioritize 'classes' or 'class_map' for classification tasks
            if self.kind in [MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.BINARY_CLASSIFICATION]:
                classes = self._get_dataset_attr(dataset_to_use, DatasetKeys.CLASSES)
                class_map = self._get_dataset_attr(dataset_to_use, DatasetKeys.CLASS_MAP)
                
                if classes:
                    target_names = classes
                elif class_map and isinstance(class_map, dict):
                    sorted_items = sorted(class_map.items(), key=lambda item: item[1])
                    target_names = [k for k, v in sorted_items]
            
            # 2. Use target column names for Regression/Multi-target tasks, or as a fallback
            if target_names is None:
                target_names = self._get_dataset_attr(dataset_to_use, DatasetKeys.TARGET_NAMES)

            # 3. Absolute fallback
            if target_names is None:
                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION]:
                    target_names = ["Output"]
                
        captum_feature_importance(
            model=self.model,
            input_data=input_data,
            feature_names=feature_names,
            save_dir=captum_save_dir,
            target_names=target_names,
            n_steps=n_steps,
            device=self.device,
            verbose=verbose
        )
        
    def finalize_model_training(self,
                                finalize_config: Union[FinalizeRegression,
                                                       FinalizeMultiTargetRegression,
                                                       FinalizeBinaryClassification,
                                                       FinalizeMultiClassClassification,
                                                       FinalizeMultiLabelBinaryClassification]):
        """
        Saves a finalized, inference-ready model state to a .pth file alongside relevant task metadata.

        Uses the current model state and the provided task-specific configuration to create a finalized artifact.

        Args:
            finalize_config (object): Task-specific data class instance containing metadata required for running inference later.
        """
        if self.kind == MLTaskKeys.REGRESSION and not isinstance(finalize_config, FinalizeRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeRegression', but got '{type(finalize_config).__name__}'.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTITARGET_REGRESSION and not isinstance(finalize_config, FinalizeMultiTargetRegression):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiTargetRegression', but got '{type(finalize_config).__name__}'.")
            raise TypeError()
        elif self.kind == MLTaskKeys.BINARY_CLASSIFICATION and not isinstance(finalize_config, FinalizeBinaryClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeBinaryClassification', but got '{type(finalize_config).__name__}'.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION and not isinstance(finalize_config, FinalizeMultiClassClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiClassClassification', but got '{type(finalize_config).__name__}'.")
            raise TypeError()
        elif self.kind == MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION and not isinstance(finalize_config, FinalizeMultiLabelBinaryClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeMultiLabelBinaryClassification', but got '{type(finalize_config).__name__}'.")
            raise TypeError()
        
        self._save_finalized_artifact(finalize_config=finalize_config)
