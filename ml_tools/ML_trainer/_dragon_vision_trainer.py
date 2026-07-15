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
from ..ML_evaluation import classification_metrics, segmentation_metrics
from ..ML_evaluation_captum import captum_segmentation_heatmap, captum_image_heatmap
from ..ML_configuration import (
    FormatBinaryImageClassificationMetrics,
    FormatMultiClassImageClassificationMetrics,
    FormatBinarySegmentationMetrics,
    FormatMultiClassSegmentationMetrics,
    FinalizeBinaryImageClassification,
    FinalizeMultiClassImageClassification,
    FinalizeBinarySegmentation,
    FinalizeMultiClassSegmentation
)

from ..keys._keys import PyTorchLogKeys, PyTorchCheckpointKeys, DatasetKeys, MLTaskKeys, DragonTrainerKeys
from .._core import get_logger

from ._base_trainer import _BaseDragonTrainer

_LOGGER = get_logger("Vision Trainer")

__all__ = [
    "DragonVisionTrainer",
]

class DragonVisionTrainer(_BaseDragonTrainer):
    """
    Automates the training process of a PyTorch Model for computer vision tasks.
    
    This trainer specifically supports binary and multiclass image classification, 
    as well as binary and multiclass image segmentation. 
    
    Built-in Callbacks: `History`, `TqdmProgressBar`.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 validation_dataset: Dataset, 
                 save_dir: Union[str, Path],
                 kind: Literal["binary segmentation", 
                               "multiclass segmentation", 
                               "binary image classification", 
                               "multiclass image classification"],
                 optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'], str], 
                 checkpoint_callback: Optional[DragonModelCheckpoint],
                 early_stopping_callback: Optional[_DragonEarlyStopping],
                 lr_scheduler_callback: Optional[_DragonLRScheduler],
                 extra_callbacks: Optional[list[_Callback]] = None,
                 criterion: Union[nn.Module, Literal["auto"]] = "auto", 
                 dataloader_workers: int = 2):
        """
        A trainer class for automating the training of PyTorch models on computer vision tasks.

        Args:
            model (nn.Module): The PyTorch vision model to train.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            save_dir (Union[str, Path]): Root directory where all training artifacts (checkpoints, metrics, plots) will be saved.
            kind (str): The specific vision task to perform. Must be one of the supported vision task string literals.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device (Union[Literal['cuda', 'mps', 'cpu'], str]): The device to run training on.
            checkpoint_callback (Optional[DragonModelCheckpoint]): Callback to handle saving model checkpoints.
            early_stopping_callback (Optional[_DragonEarlyStopping]): Callback to stop training early if metric stops improving.
            lr_scheduler_callback (Optional[_DragonLRScheduler]): Callback for learning rate scheduling.
            extra_callbacks (Optional[list[_Callback]]): Additional custom callbacks to apply during training.
            criterion (Union[nn.Module, Literal["auto"]]): The loss function. If "auto", it is inferred from the `kind` parameter.
            dataloader_workers (int): Number of subprocesses to use for data loading.
            
        <br>

        ### **Note about Loss Function (Criterion):**
            - **Binary Image Classification:** `nn.BCEWithLogitsLoss` is standard. The model should output a single unnormalized logit per image (`[N, 1]`).
            - **Multi-Class Image Classification:** `nn.CrossEntropyLoss` is the standard choice. It expects raw, unnormalized logits for each class (`[N, C]`).
            - **Binary Segmentation:** `nn.BCEWithLogitsLoss` is commonly used. The model should output a single unnormalized logit mask per image (`[N, 1, H, W]`).
            - **Multi-Class Segmentation:** `nn.CrossEntropyLoss` is standard. The model should output raw logits for each class per pixel (`[N, C, H, W]`).
            - *Important:* PyTorch's `BCEWithLogitsLoss` and `CrossEntropyLoss` apply `Sigmoid` and `Softmax` internally for numerical stability. Ensure the model's final layer **does not** include an activation function if using these criterions.
        """
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            save_dir=save_dir,
            dataloader_workers=dataloader_workers,
            checkpoint_callback=checkpoint_callback,
            early_stopping_callback=early_stopping_callback,
            lr_scheduler_callback=lr_scheduler_callback,
            extra_callbacks=extra_callbacks
        )
        
        if kind not in [MLTaskKeys.BINARY_SEGMENTATION,
                        MLTaskKeys.MULTICLASS_SEGMENTATION,
                        MLTaskKeys.BINARY_IMAGE_CLASSIFICATION,
                        MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
            raise ValueError(f"'{kind}' is not a valid vision task type.")

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.kind = kind
        self._classification_threshold: float = 0.5
        
        if criterion == "auto":
            if kind in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.BINARY_SEGMENTATION]:
                self.criterion = nn.BCEWithLogitsLoss()
            elif kind in [MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

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

            if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                    output = output.squeeze(1)
            
            if self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                if output.ndim == 4 and output.shape[1] == 1 and target.ndim == 3:
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

                if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                    if output.ndim == 2 and output.shape[1] == 1 and target.ndim == 1:
                        output = output.squeeze(1)
                
                if self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    if output.ndim == 4 and output.shape[1] == 1 and target.ndim == 3:
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
        
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                target = target.to(self.device) 
                
                output = self.model(features)

                y_pred_batch = None
                y_prob_batch = None
                y_true_batch = None
                
                if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.squeeze(1)
                        
                    probs_pos = torch.sigmoid(output) 
                    preds = (probs_pos >= self._classification_threshold).int()
                    y_pred_batch = preds.cpu().numpy()
                    
                    probs_neg = 1.0 - probs_pos
                    y_prob_batch = torch.stack([probs_neg, probs_pos], dim=1).cpu().numpy()
                    y_true_batch = target.cpu().numpy()

                elif self.kind == MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
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
                 model_checkpoint: Union[Path, Literal["best", "current"]],
                 classification_threshold: Optional[float] = None,
                 test_data: Optional[Union[DataLoader, Dataset]] = None,
                 val_format_configuration: Optional[Union[
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics
                    ]]=None,
                 test_format_configuration: Optional[Union[
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics
                    ]]=None):
        """
        Evaluates the vision model and generates task-specific evaluation metrics.

        Args:
            model_checkpoint (Union[Path, Literal["best", "current"]]): The specific checkpoint state to load before evaluating.
            classification_threshold (Optional[float]): The threshold used for calculating binary classification/segmentation metrics.
            test_data (Optional[Union[DataLoader, Dataset]]): An optional test dataset to evaluate model performance completely separated from validation.
            val_format_configuration (Optional[object]): Formatting configuration object for validation metric outputs.
            test_format_configuration (Optional[object]): Formatting configuration object for test metric outputs.
        """
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
            if not isinstance(val_format_configuration, (FormatBinaryImageClassificationMetrics,
                                                         FormatMultiClassImageClassificationMetrics,
                                                         FormatBinarySegmentationMetrics,
                                                         FormatMultiClassSegmentationMetrics)):
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
            
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{DragonTrainerKeys.VALIDATION_METRICS_DIR}'")
            self._evaluate(save_dir=validation_metrics_path,
                           model_checkpoint=checkpoint_validated, # type: ignore
                           classification_threshold=threshold_validated,
                           data=None,
                           format_configuration=val_configuration_validated)
            
            if test_format_configuration is not None:
                if not isinstance(test_format_configuration, (FormatBinaryImageClassificationMetrics,
                                                              FormatMultiClassImageClassificationMetrics,
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
            else: 
                test_configuration_validated = None
            
            _LOGGER.info(f"Evaluating on test dataset. Metrics will be saved to '{DragonTrainerKeys.TEST_METRICS_DIR}'")
            self._evaluate(save_dir=test_metrics_path,
                           model_checkpoint="current",
                           classification_threshold=threshold_validated,
                           data=test_data_validated,
                           format_configuration=test_configuration_validated)
        else:
            _LOGGER.info(f"Evaluating on validation dataset. Metrics will be saved to '{validation_metrics_path.name}'")
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
                        FormatBinaryImageClassificationMetrics,
                        FormatMultiClassImageClassificationMetrics,
                        FormatBinarySegmentationMetrics,
                        FormatMultiClassSegmentationMetrics
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

        if self.kind in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, 
                         MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]:
            try:
                class_map = dataset_for_artifacts.class_map # type: ignore
            except AttributeError:
                _LOGGER.warning(f"Dataset has no 'class_map' attribute. Using generics.")
                class_map = None
            else:
                if not isinstance(class_map, dict):
                    _LOGGER.warning(f"Dataset has a 'class_map' attribute, but it is not a dictionary: '{type(class_map)}'.")
                    class_map = None
            
            config = None
            if format_configuration:
                if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION and isinstance(format_configuration, FormatBinaryImageClassificationMetrics):
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
        
        elif self.kind in [MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTICLASS_SEGMENTATION]:
            try:
                class_map = dataset_for_artifacts.class_map
            except AttributeError:
                _LOGGER.warning("Dataset has no 'class_map' attribute. Using generics.")
                class_map = None
            else:
                if not isinstance(class_map, dict):
                    _LOGGER.warning(f"Dataset has a 'class_map' attribute, but it is not a dictionary: '{type(class_map)}'.")
                    class_map = None
            
            config = None
            if format_configuration and isinstance(format_configuration, (FormatBinarySegmentationMetrics, FormatMultiClassSegmentationMetrics)):
                config = format_configuration
            elif format_configuration:
                _LOGGER.warning(f"Wrong configuration type: Received '{type(format_configuration).__name__}'.")
            
            segmentation_metrics(y_true=y_true,
                                 y_pred=y_pred,
                                 save_dir=save_dir,
                                 class_map=class_map,
                                 config=config)

    def explain_captum(self,
                       explain_dataset: Optional[Dataset] = None,
                       n_samples: int = 100,
                       target_names: Optional[list[str]] = None,
                       n_steps: int = 50,
                       verbose: int = 0):
        """
        Explains model predictions using Captum's Integrated Gradients.
        
        For image classification tasks, it generates Image Heatmaps. 
        For segmentation tasks, it generates Spatial Heatmaps for each class.

        Args:
            explain_dataset (Optional[Dataset]): Dataset to sample from. Defaults to the internal validation set if None.
            n_samples (int): Number of samples to evaluate and generate heatmaps for.
            target_names (Optional[list[str]]): Class names for the output. Attempts to extract from the dataset if None.
            n_steps (int): Number of interpolation steps.
            verbose (int): Verbosity level for logging operations.
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
        
        is_segmentation = self.kind in [MLTaskKeys.BINARY_SEGMENTATION, MLTaskKeys.MULTICLASS_SEGMENTATION]
        is_image_classification = self.kind in [MLTaskKeys.BINARY_IMAGE_CLASSIFICATION, MLTaskKeys.MULTICLASS_IMAGE_CLASSIFICATION]
        
        if target_names is None:
            if hasattr(dataset_to_use, DatasetKeys.TARGET_NAMES):
                target_names = dataset_to_use.target_names # type: ignore
            elif hasattr(dataset_to_use, "classes"): 
                 target_names = dataset_to_use.classes # type: ignore
            elif hasattr(dataset_to_use, "class_map") and isinstance(dataset_to_use.class_map, dict): # type: ignore
                 sorted_items = sorted(dataset_to_use.class_map.items(), key=lambda item: item[1]) # type: ignore
                 target_names = [k for k, v in sorted_items]

            if target_names is None:
                if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION:
                    target_names = ["Output"]
                elif self.kind == MLTaskKeys.BINARY_SEGMENTATION:
                    target_names = ["Foreground"]

        if is_segmentation:
            if n_steps > 30:
                n_steps = 30
                _LOGGER.warning(f"Segmentation task detected: Reducing Captum n_steps to {n_steps} to prevent OOM. If you encounter OOM errors, consider lowering this further.")
            
            captum_segmentation_heatmap(
                model=self.model,
                input_data=input_data,
                save_dir=captum_save_dir,
                target_names=target_names,
                n_steps=n_steps,
                device=self.device
            )
        
        elif is_image_classification:
            captum_image_heatmap(
                model=self.model,
                input_data=input_data,
                save_dir=captum_save_dir,
                target_names=target_names,
                n_steps=n_steps,
                device=self.device
            )
        
    def finalize_model_training(self, 
                                model_checkpoint: Union[Path, Literal['best', 'current']],
                                finalize_config: Union[FinalizeBinaryImageClassification,
                                                       FinalizeMultiClassImageClassification,
                                                       FinalizeBinarySegmentation,
                                                       FinalizeMultiClassSegmentation]):
        """
        Saves a finalized, inference-ready model state to a .pth file alongside relevant task metadata.

        Args:
            model_checkpoint (Union[Path, Literal['best', 'current']]): The checkpoint to load and finalize.
            finalize_config (Union[object]): Task-specific data class instance containing metadata required for running inference later.
        """
        if self.kind == MLTaskKeys.BINARY_IMAGE_CLASSIFICATION and not isinstance(finalize_config, FinalizeBinaryImageClassification):
            _LOGGER.error(f"For task {self.kind}, expected finalize_config of type 'FinalizeBinaryImageClassification', but got {type(finalize_config).__name__}.")
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
                
        self._load_model_state_wrapper(model_checkpoint)
        
        finalized_data = {
            PyTorchCheckpointKeys.EPOCH: self.epoch,
            PyTorchCheckpointKeys.MODEL_STATE: self.model.state_dict(),
            PyTorchCheckpointKeys.TASK: finalize_config.task
        }

        if getattr(finalize_config, "classification_threshold", None) is not None:
            finalized_data[PyTorchCheckpointKeys.CLASSIFICATION_THRESHOLD] = finalize_config.classification_threshold
        if getattr(finalize_config, "class_map", None) is not None:
            finalized_data[PyTorchCheckpointKeys.CLASS_MAP] = finalize_config.class_map

        self._save_finalized_artifact(
            finalized_data=finalized_data,
            save_dir=self.training_directory_root,
            filename=finalize_config.filename
        )
