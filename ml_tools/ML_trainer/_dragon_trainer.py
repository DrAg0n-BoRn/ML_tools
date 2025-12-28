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
from ..ML_evaluation import classification_metrics, regression_metrics, shap_summary_plot, plot_attention_importance
from ..ML_evaluation import multi_target_regression_metrics, multi_label_classification_metrics, multi_target_shap_summary_plot
from ..ML_evaluation import segmentation_metrics
from ..ML_evaluation_captum import captum_feature_importance, captum_segmentation_heatmap, captum_image_heatmap
from ..ML_configuration import (FormatRegressionMetrics, 
                            FormatMultiTargetRegressionMetrics,
                            FormatBinaryClassificationMetrics,
                            FormatMultiClassClassificationMetrics,
                            FormatBinaryImageClassificationMetrics,
                            FormatMultiClassImageClassificationMetrics,
                            FormatMultiLabelBinaryClassificationMetrics,
                            FormatBinarySegmentationMetrics,
                            FormatMultiClassSegmentationMetrics,

                            FinalizeBinaryClassification,
                            FinalizeBinarySegmentation,
                            FinalizeBinaryImageClassification,
                            FinalizeMultiClassClassification,
                            FinalizeMultiClassImageClassification,
                            FinalizeMultiClassSegmentation,
                            FinalizeMultiLabelBinaryClassification,
                            FinalizeMultiTargetRegression,
                            FinalizeRegression)

from ..path_manager import make_fullpath
from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, DatasetKeys, MLTaskKeys, MagicWords, DragonTrainerKeys, ScalerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer


_LOGGER = get_logger("DragonTrainer")


__all__ = [
    "DragonTrainer",
]


# --- DragonTrainer ----
class DragonTrainer(_BaseDragonTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 kind: Literal["regression", 
                               "binary classification", 
                               "multiclass classification", 
                               "multitarget regression", 
                               "multilabel binary classification", 
                               "binary segmentation", 
                               "multiclass segmentation", 
                               "binary image classification", 
                               "multiclass image classification"],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
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
        
        Automatically detects if `target_scaler` is present in the training dataset
        and applies inverse transformation for Regression tasks.
        
        Yields:
            tuple: A tuple containing (y_pred_batch, y_prob_batch, y_true_batch).
                   
                - y_prob_batch is None for regression tasks.
        """
        self.model.eval()
        self.model.to(self.device)
        
        # --- Check for Target Scaler (for Regression Un-scaling) ---
        target_scaler = None
        if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
            # Try to get the scaler from the dataset attached to the trainer
            if hasattr(self.train_dataset, ScalerKeys.TARGET_SCALER):
                target_scaler = getattr(self.train_dataset, ScalerKeys.TARGET_SCALER)
                if target_scaler is not None:
                     _LOGGER.debug("Target scaler detected. Un-scaling predictions and targets for metric calculation.")
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                # Keep target on device initially for potential un-scaling
                target = target.to(self.device) 
                
                output = self.model(features)

                y_pred_batch = None
                y_prob_batch = None
                y_true_batch = None

                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]:
                    
                    # --- Automatic Un-scaling Logic ---
                    if target_scaler:
                        # 1. Reshape output/target if flattened (common in single regression)
                        # Scaler expects [N, Features]
                        original_out_shape = output.shape
                        original_target_shape = target.shape
                        
                        if output.ndim == 1: output = output.reshape(-1, 1)
                        if target.ndim == 1: target = target.reshape(-1, 1)
                            
                        # 2. Apply Inverse Transform
                        output = target_scaler.inverse_transform(output)
                        target = target_scaler.inverse_transform(target)
                        
                        # 3. Restore shapes (optional, but good for consistency)
                        if len(original_out_shape) == 1: output = output.flatten()
                        if len(original_target_shape) == 1: target = target.flatten()

                    y_pred_batch = output.cpu().numpy()
                    y_true_batch = target.cpu().numpy()
                    
                elif self.kind in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION]:
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.squeeze(1)
                        
                    probs_pos = torch.sigmoid(output) 
                    preds = (probs_pos >= self._classification_threshold).int()
                    y_pred_batch = preds.cpu().numpy()
                    
                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.stack([probs_neg, probs_pos], dim=1).cpu().numpy()
                    y_true_batch = target.cpu().numpy()

                elif self.kind in [MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
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
                
                elif self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    probs_pos = torch.sigmoid(output) 
                    preds = (probs_pos >= self._classification_threshold).int() 
                    y_pred_batch = preds.squeeze(1).cpu().numpy()

                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.cat([probs_neg, probs_pos], dim=1).cpu().numpy()

                    if target.ndim == 4 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    y_true_batch = target.cpu().numpy()
                    
                elif self.kind == MLTaskKeys.MULTICLASS_SEGMENTATION:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1) 
                    y_pred_batch = preds.cpu().numpy()
                    y_prob_batch = probs.cpu().numpy() 
                    
                    if target.ndim == 4 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    y_true_batch = target.cpu().numpy()

                yield y_pred_batch, y_prob_batch, y_true_batch
                
    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 classification_threshold: Optional[float] = None,
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics,
                        FormatBinaryClassificationMetrics,
                        FormatMultiClassClassificationMetrics,
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics
                    ]]=None,
                 test_format_configuration: Optional[Union[
                        FormatRegressionMetrics, 
                        FormatMultiTargetRegressionMetrics,
                        FormatBinaryClassificationMetrics,
                        FormatMultiClassClassificationMetrics,
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics,
                    ]]=None):
        """
        Evaluates the model, routing to the correct evaluation function based on task `kind`.

        Args:
            model_checkpoint (Path | "best" | "current"): 
                - Path to a valid checkpoint for the model. The state of the trained model will be overwritten in place.
                - If 'best', the best checkpoint will be loaded if a DragonModelCheckpoint was provided. The state of the trained model will be overwritten in place.
                - If 'current', use the current state of the trained model up the latest trained epoch.
            save_dir (str | Path): Directory to save all reports and plots.
            classification_threshold (float | None): Used for tasks using a binary approach (binary classification, binary segmentation, multilabel binary classification)
            test_data (DataLoader | Dataset | None): Optional Test data to evaluate the model performance. Validation and Test metrics will be saved to subdirectories.
            val_format_configuration (object): Optional configuration for metric format output for the validation set.
            test_format_configuration (object): Optional configuration for metric format output for the test set.
        """
        # Validate inputs using base helpers
        checkpoint_validated = self._validate_checkpoint_arg(model_checkpoint)
        save_path = self._validate_save_dir(save_dir)
        
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
            if not isinstance(val_format_configuration, (FormatRegressionMetrics, 
                                                        FormatMultiTargetRegressionMetrics,
                                                        FormatBinaryClassificationMetrics,
                                                        FormatMultiClassClassificationMetrics,
                                                        FormatBinaryImageClassificationMetrics,
                                                        FormatMultiClassImageClassificationMetrics,
                                                        FormatMultiLabelBinaryClassificationMetrics,
                                                        FormatBinarySegmentationMetrics,
                                                        FormatMultiClassSegmentationMetrics)):
                _LOGGER.error(f"Invalid 'format_configuration': '{type(val_format_configuration)}'.")
                raise ValueError()
            else:
                val_configuration_validated = val_format_configuration
        else: # config is None
            val_configuration_validated = None
        
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
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)
            
            # Validate test configuration
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (FormatRegressionMetrics, 
                                                        FormatMultiTargetRegressionMetrics,
                                                        FormatBinaryClassificationMetrics,
                                                        FormatMultiClassClassificationMetrics,
                                                        FormatBinaryImageClassificationMetrics,
                                                        FormatMultiClassImageClassificationMetrics,
                                                        FormatMultiLabelBinaryClassificationMetrics,
                                                        FormatBinarySegmentationMetrics,
                                                        FormatMultiClassSegmentationMetrics)):
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
            _LOGGER.info(f"🔎 Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           classification_threshold=threshold_validated,
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            # Dispatch validation set
            _LOGGER.info(f"🔎 Evaluating on validation dataset. Metrics will be saved to '{save_path.name}'")
            self._evaluate(save_dir=save_path,
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
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatMultiLabelBinaryClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics
                    ]]=None):
        """
        Changed to a private helper function.
        """
        # set threshold
        self._classification_threshold = classification_threshold
        
        # load model checkpoint
        self._load_model_state_wrapper(model_checkpoint)
        
        # Prepare Data using Base Helper
        eval_loader, dataset_for_artifacts = self._prepare_eval_data(data, self.validation_dataset)
        
        # Gather Predictions
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
            if format_configuration and isinstance(format_configuration, FormatRegressionMetrics):
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
                if self.kind == MLTaskKeys.BINARY_CLASSIFICATION and isinstance(format_configuration, FormatBinaryClassificationMetrics):
                    config = format_configuration
                elif self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION and isinstance(format_configuration, FormatBinaryImageClassificationMetrics):
                    config = format_configuration
                elif self.kind == MLTaskKeys.MULTICLASS_CLASSIFICATION and isinstance(format_configuration, FormatMultiClassClassificationMetrics):
                    config = format_configuration
                elif self.kind == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION and isinstance(format_configuration, FormatMultiClassImageClassificationMetrics):
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
            if format_configuration and isinstance(format_configuration, FormatMultiTargetRegressionMetrics):
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
            if format_configuration and isinstance(format_configuration, (FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics)):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            segmentation_metrics(y_true=y_true,
                                 y_pred=y_pred,
                                 save_dir=save_dir,
                                 class_names=class_names,
                                 config=config)
    
    def explain_shap(self,
                save_dir: Union[str,Path], 
                explain_dataset: Optional[Dataset] = None, 
                n_samples: int = 300,
                feature_names: Optional[list[str]] = None,
                target_names: Optional[list[str]] = None,
                explainer_type: Literal['deep', 'kernel'] = 'kernel'):
        """
        Explains model predictions using SHAP and saves all artifacts.
        
        NOTE: SHAP support is limited to single-target tasks (Regression, Binary/Multiclass Classification).
        For complex tasks (Multi-target, Multi-label, Sequences, Images), please use `explain_captum()`.

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
        # --- 1. Compatibility Guard ---
        valid_shap_tasks = [
            MLTaskKeys.REGRESSION, 
            MLTaskKeys.BINARY_CLASSIFICATION, 
            MLTaskKeys.MULTICLASS_CLASSIFICATION
        ]
        
        if self.kind not in valid_shap_tasks:
            _LOGGER.warning(f"SHAP explanation is deprecated for task '{self.kind}' due to instability. Please use 'explain_captum()' instead.")
            return
        
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
        # DEPRECATED: Multi-target SHAP support is unstable; recommend Captum instead.
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

    def explain_captum(self,
                       save_dir: Union[str, Path],
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       feature_names: Optional[list[str]] = None,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50):
        """
        Explains model predictions using Captum's Integrated Gradients.
        
        - **Tabular/Classification:** Generates Feature Importance Bar Charts.
        - **Segmentation:** Generates Spatial Heatmaps for each class.
        
        Args:
            save_dir (str | Path): Directory to save artifacts.
            explain_dataset (Dataset | None): Dataset to sample from. Defaults to validation set.
            n_samples (int): Number of samples to evaluate.
            feature_names (list[str] | None): Feature names. 
                - Required for Tabular tasks.
                - Ignored/Optional for Image tasks (defaults to Channel names).
            target_names (list[str] | None): Names for the model outputs (or Class names).
                - If None, attempts to extract from dataset attributes (`target_names`, `classes`, or `class_map`).
                - If extraction fails, generates generic names (e.g. "Output_0").
            n_steps (int): Number of interpolation steps.
        """         
        # 2. Prepare Data
        dataset_to_use = explain_dataset if explain_dataset is not None else self.validation_dataset
        if dataset_to_use is None:
            _LOGGER.error("No dataset available for explanation.")
            return

        # Efficient sampling helper
        def _get_samples(ds, n):
            # Use num_workers=0 for stability during ad-hoc sampling
            loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
            data_iter = iter(loader)
            features, targets = next(data_iter)
            return features, targets

        input_data, _ = _get_samples(dataset_to_use, n_samples)
        
        # 3. Get Feature Names (Only if NOT segmentation AND NOT image classification)
        # Image tasks generally don't have explicit feature names; Captum will default to "Channel_X"
        is_segmentation = self.kind in [MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTICLASS_SEGMENTATION]
        is_image_classification = self.kind in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]
        
        if feature_names is None and not is_segmentation and not is_image_classification:
            if hasattr(dataset_to_use, DatasetKeys.FEATURE_NAMES):
                feature_names = dataset_to_use.feature_names # type: ignore
            else:
                _LOGGER.error(f"Could not extract `feature_names`. It must be provided if the dataset does not have it.")
                raise ValueError()

        # 4. Handle Target Names (or Class Names)
        if target_names is None:
            # A. Try dataset attributes first
            if hasattr(dataset_to_use, DatasetKeys.TARGET_NAMES):
                target_names = dataset_to_use.target_names # type: ignore
            elif hasattr(dataset_to_use, "classes"): 
                 target_names = dataset_to_use.classes # type: ignore
            elif hasattr(dataset_to_use, "class_map") and isinstance(dataset_to_use.class_map, dict): # type: ignore
                 # Sort by value (index) to ensure correct order: {name: index} -> [name_at_0, name_at_1...]
                 sorted_items = sorted(dataset_to_use.class_map.items(), key=lambda item: item[1]) # type: ignore
                 target_names = [k for k, v in sorted_items]

            # B. Infer based on task
            if target_names is None:
                if self.kind in [MLTaskKeys.REGRESSION, MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.BINARY_IMAGE_CLASSIFICATION]:
                    target_names = ["Output"]
                elif self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    target_names = ["Foreground"]
                
                # For multiclass/multitarget without names, leave it None and let the evaluation function generate generics.

        # 5. Dispatch based on Task
        if is_segmentation:
            # lower n_steps for segmentation to save memory
            if n_steps > 30:
                n_steps = 30
                _LOGGER.warning(f"Segmentation task detected: Reducing Captum n_steps to {n_steps} to prevent OOM. If you encounter OOM errors, consider lowering this further.")
            
            captum_segmentation_heatmap(
                model=self.model,
                input_data=input_data,
                save_dir=save_dir,
                target_names=target_names, # Can be None, helper handles it
                n_steps=n_steps,
                device=self.device
            )
        
        elif is_image_classification:
            captum_image_heatmap(
                model=self.model,
                input_data=input_data,
                save_dir=save_dir,
                target_names=target_names,
                n_steps=n_steps,
                device=self.device
            )
            
        else:
            # Standard Tabular/Image Classification
            captum_feature_importance(
                model=self.model,
                input_data=input_data,
                feature_names=feature_names,
                save_dir=save_dir,
                target_names=target_names,
                n_steps=n_steps,
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
                          feature_names: Optional[list[str]] = None, 
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
                                model_checkpoint: Union[Path, Literal['best', 'current']],
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
            model_checkpoint (Path | "best" | "current"):
                - Path: Loads the model state from a specific checkpoint file.
                - "best": Loads the best model state saved by the `DragonModelCheckpoint` callback.
                - "current": Uses the model's state as it is.
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
                
        # handle checkpoint
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
        if finalize_config.classification_threshold is not None:
            finalized_data[PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD] = finalize_config.classification_threshold
        if finalize_config.class_map is not None:
            finalized_data[PyTorchCheckpointKeys.CLASS_MAP] = finalize_config.class_map

        # Save model file using base helper
        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=save_dir,
            filename=finalize_config.filename
        )

