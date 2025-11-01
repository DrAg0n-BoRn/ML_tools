from typing import List, Literal, Union, Optional, Callable, Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np

from .ML_callbacks import Callback, History, TqdmProgressBar, ModelCheckpoint
from .ML_evaluation import classification_metrics, regression_metrics, plot_losses, shap_summary_plot, plot_attention_importance
from .ML_evaluation_multi import multi_target_regression_metrics, multi_label_classification_metrics, multi_target_shap_summary_plot
from ._script_info import _script_info
from .keys import PyTorchLogKeys, PyTorchCheckpointKeys, DatasetKeys
from ._logger import _LOGGER
from .path_manager import make_fullpath
from .ML_vision_evaluation import segmentation_metrics, object_detection_metrics


__all__ = [
    "MLTrainer",
    "ObjectDetectionTrainer"
]


class MLTrainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, 
                 kind: Literal["regression", "classification", "multi_target_regression", "multi_label_classification", "segmentation"],
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], dataloader_workers: int = 2, callbacks: Optional[List[Callback]] = None):
        """
        Automates the training process of a PyTorch Model.
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing/validation dataset.
            kind (str): Can be 'regression', 'classification', 'multi_target_regression', 'multi_label_classification', or 'segmentation'.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading.
            callbacks (List[Callback] | None): A list of callbacks to use during training.
            
        Note:
            - For **regression** and **multi_target_regression** tasks, suggested criterions include `nn.MSELoss` or `nn.L1Loss`.
    
            - For **single-label, multi-class classification** tasks, `nn.CrossEntropyLoss` is the standard choice.
    
            - For **multi-label, binary classification** tasks (where each label is a 0 or 1), `nn.BCEWithLogitsLoss` is the correct choice as it treats each output as an independent binary problem.
        
            - For **segmentation** tasks, `nn.CrossEntropyLoss` (for multi-class) or `nn.BCEWithLogitsLoss` (for binary) are common.
        """
        if kind not in ["regression", "classification", "multi_target_regression", "multi_label_classification", "segmentation"]:
            raise ValueError(f"'{kind}' is not a valid task type.")

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.kind = kind
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = None
        self.device = self._validate_device(device)
        self.dataloader_workers = dataloader_workers
        
        # Callback handler - History and TqdmProgressBar are added by default
        default_callbacks = [History(), TqdmProgressBar()]
        user_callbacks = callbacks if callbacks is not None else []
        self.callbacks = default_callbacks + user_callbacks
        self._set_trainer_on_callbacks()

        # Internal state
        self.train_loader = None
        self.test_loader = None
        self.history = {}
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
        
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type)
        )
        
    def _load_checkpoint(self, path: Union[str, Path]):
        """Loads a training checkpoint to resume training."""
        p = make_fullpath(path, enforce="file")
        _LOGGER.info(f"Loading checkpoint from '{p.name}' to resume training...")
        
        try:
            checkpoint = torch.load(p, map_location=self.device)
            
            if PyTorchCheckpointKeys.MODEL_STATE not in checkpoint or PyTorchCheckpointKeys.OPTIMIZER_STATE not in checkpoint:
                _LOGGER.error(f"Checkpoint file '{p.name}' is invalid. Missing 'model_state_dict' or 'optimizer_state_dict'.")
                raise KeyError()

            self.model.load_state_dict(checkpoint[PyTorchCheckpointKeys.MODEL_STATE])
            self.optimizer.load_state_dict(checkpoint[PyTorchCheckpointKeys.OPTIMIZER_STATE])
            self.start_epoch = checkpoint.get(PyTorchCheckpointKeys.EPOCH, 0) + 1 # Resume on the *next* epoch
            
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
                if isinstance(cb, ModelCheckpoint) and PyTorchCheckpointKeys.BEST_SCORE in checkpoint:
                    cb.best = checkpoint[PyTorchCheckpointKeys.BEST_SCORE]
                    _LOGGER.info(f"Restored {cb.__class__.__name__} 'best' score to: {cb.best:.4f}")
            
            _LOGGER.info(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            
        except Exception as e:
            _LOGGER.error(f"Failed to load checkpoint from '{p}': {e}")
            raise

    def fit(self, 
            epochs: int = 10, 
            batch_size: int = 10, 
            shuffle: bool = True,
            resume_from_checkpoint: Optional[Union[str, Path]] = None):
        """
        Starts the training-validation process of the model.
        
        Returns the "History" callback dictionary.

        Args:
            epochs (int): The total number of epochs to train for.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the training data at each epoch.
            resume_from_checkpoint (str | Path | None): Optional path to a checkpoint to resume training.
            
        Note:
            For regression tasks using `nn.MSELoss` or `nn.L1Loss`, the trainer
            automatically aligns the model's output tensor with the target tensor's
            shape using `output.view_as(target)`. This handles the common case
            where a model outputs a shape of `[batch_size, 1]` and the target has a
            shape of `[batch_size]`.
        """
        self.epochs = epochs
        self._batch_size = batch_size
        self._create_dataloaders(self._batch_size, shuffle)
        self.model.to(self.device)
        
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Reset stop_training flag on the trainer
        self.stop_training = False

        self._callbacks_hook('on_train_begin')
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.epoch = epoch
            epoch_logs = {}
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
        return self.history
    
    def _train_step(self):
        self.model.train()
        running_loss = 0.0
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
            
            # Apply shape correction only for single-target regression
            if self.kind == "regression":
                output = output.view_as(target)
                
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()

            # Calculate batch loss and update running loss for the epoch
            batch_loss = loss.item()
            running_loss += batch_loss * features.size(0)
            
            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / len(self.train_loader.dataset)} # type: ignore

    def _validation_step(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for features, target in self.test_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                
                output = self.model(features)
                # Apply shape correction only for single-target regression
                if self.kind == "regression":
                    output = output.view_as(target)
                
                loss = self.criterion(output, target)
                
                running_loss += loss.item() * features.size(0)
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / len(self.test_loader.dataset)} # type: ignore
        return logs
    
    def _predict_for_eval(self, dataloader: DataLoader, classification_threshold: float = 0.5):
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

                if self.kind in ["regression", "multi_target_regression"]:
                    y_pred_batch = output.numpy()
                    y_true_batch = target.numpy()

                elif self.kind == "classification":
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy()
                    y_true_batch = target.numpy()

                elif self.kind == "multi_label_classification":
                    probs = torch.sigmoid(output)
                    preds = (probs >= classification_threshold).int()
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy()
                    y_true_batch = target.numpy()
                    
                elif self.kind == "segmentation":
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

    def evaluate(self, save_dir: Union[str, Path], data: Optional[Union[DataLoader, Dataset]] = None, classification_threshold: float = 0.5):
        """
        Evaluates the model, routing to the correct evaluation function based on task `kind`.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            data (DataLoader | Dataset | None): The data to evaluate on. If None, defaults to the trainer's internal test_dataset.
            classification_threshold (float): Probability threshold for multi-label tasks.
        """
        dataset_for_names = None
        eval_loader = None

        if isinstance(data, DataLoader):
            eval_loader = data
            # Try to get the dataset from the loader for fetching target names
            if hasattr(data, 'dataset'):
                dataset_for_names = data.dataset
        elif isinstance(data, Dataset):
            # Create a new loader from the provided dataset
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))
            dataset_for_names = data
        else: # data is None, use the trainer's default test dataset
            if self.test_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no test_dataset available in the trainer.")
                raise ValueError()
            # Create a fresh DataLoader from the test_dataset
            eval_loader = DataLoader(self.test_dataset, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"))
            
            dataset_for_names = self.test_dataset

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()

        print("\n--- Model Evaluation ---")

        all_preds, all_probs, all_true = [], [], []
        for y_pred_b, y_prob_b, y_true_b in self._predict_for_eval(eval_loader, classification_threshold):
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
        if self.kind == "regression":
            regression_metrics(y_true.flatten(), y_pred.flatten(), save_dir)

        elif self.kind == "classification":
            classification_metrics(save_dir, y_true, y_pred, y_prob)

        elif self.kind == "multi_target_regression":
            try:
                target_names = dataset_for_names.target_names # type: ignore
            except AttributeError:
                num_targets = y_true.shape[1]
                target_names = [f"target_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no 'target_names' attribute. Using generic names.")
            multi_target_regression_metrics(y_true, y_pred, target_names, save_dir)

        elif self.kind == "multi_label_classification":
            try:
                target_names = dataset_for_names.target_names # type: ignore
            except AttributeError:
                num_targets = y_true.shape[1]
                target_names = [f"label_{i}" for i in range(num_targets)]
                _LOGGER.warning(f"Dataset has no 'target_names' attribute. Using generic names.")
            
            if y_prob is None:
                _LOGGER.error("Evaluation for multi_label_classification requires probabilities (y_prob).")
                return
            multi_label_classification_metrics(y_true, y_prob, target_names, save_dir, classification_threshold)
            
        elif self.kind == "segmentation":
            class_names = None
            try:
                # Try to get 'classes' from VisionDatasetMaker
                if hasattr(dataset_for_names, 'classes'):
                    class_names = dataset_for_names.classes # type: ignore
                # Fallback for Subset
                elif hasattr(dataset_for_names, 'dataset') and hasattr(dataset_for_names.dataset, 'classes'): # type: ignore
                     class_names = dataset_for_names.dataset.classes # type: ignore
            except AttributeError:
                pass # class_names is still None

            if class_names is None:
                try:
                    # Fallback to 'target_names'
                    class_names = dataset_for_names.target_names # type: ignore
                except AttributeError:
                    # Fallback to inferring from labels
                    labels = np.unique(y_true)
                    class_names = [f"Class {i}" for i in labels]
                    _LOGGER.warning(f"Dataset has no 'classes' or 'target_names' attribute. Using generic names.")
            
            segmentation_metrics(y_true, y_pred, save_dir, class_names=class_names)
        
        print("\n--- Training History ---")
        plot_losses(self.history, save_dir=save_dir)
    
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
        # Internal helper to create a dataloader and get a random sample
        def _get_random_sample(dataset: Dataset, num_samples: int):
            if dataset is None:
                return None
            
            # For MPS devices, num_workers must be 0 to ensure stability
            loader_workers = 0 if self.device.type == 'mps' else self.dataloader_workers
            
            loader = DataLoader(
                dataset, 
                batch_size=64,
                shuffle=False,
                num_workers=loader_workers
            )
            
            all_features = [features for features, _ in loader]
            if not all_features:
                return None
            
            full_data = torch.cat(all_features, dim=0)
            
            if num_samples >= full_data.size(0):
                return full_data
            
            rand_indices = torch.randperm(full_data.size(0))[:num_samples]
            return full_data[rand_indices]

        print(f"\n--- Preparing SHAP Data (sampling up to {n_samples} instances) ---")

        # 1. Get background data from the trainer's train_dataset
        background_data = _get_random_sample(self.train_dataset, n_samples)
        if background_data is None:
            _LOGGER.error("Trainer's train_dataset is empty or invalid. Skipping SHAP analysis.")
            return

        # 2. Determine target dataset and get explanation instances
        target_dataset = explain_dataset if explain_dataset is not None else self.test_dataset
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
        if self.kind in ["regression", "classification"]:
            shap_summary_plot(
                model=self.model,
                background_data=background_data,
                instances_to_explain=instances_to_explain,
                feature_names=feature_names,
                save_dir=save_dir,
                explainer_type=explainer_type,
                device=self.device
            )
        elif self.kind in ["multi_target_regression", "multi_label_classification"]:
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
        
        print("\n--- Attention Analysis ---")
        
        # --- Step 1: Check if the model supports this explanation ---
        if not getattr(self.model, 'has_interpretable_attention', False):
            _LOGGER.warning(
                "Model is not flagged for interpretable attention analysis. Skipping. This is the correct behavior for models like MultiHeadAttentionMLP."
            )
            return

        # --- Step 2: Set up the dataloader ---
        dataset_to_use = explain_dataset if explain_dataset is not None else self.test_dataset
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


# Object Detection Trainer
class ObjectDetectionTrainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, 
                 collate_fn: Callable, optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], dataloader_workers: int = 2, callbacks: Optional[List[Callback]] = None):
        """
        Automates the training process of an Object Detection Model (e.g., DragonFastRCNN).
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch object detection model to train.
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing/validation dataset.
            collate_fn (Callable): The collate function from `ObjectDetectionDatasetMaker.collate_fn`.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading.
            callbacks (List[Callback] | None): A list of callbacks to use during training.
            
        ## Note:
            This trainer is specialized. It does not take a `criterion` because object detection models like Faster R-CNN return a dictionary of losses directly from their forward pass during training.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.kind = "object_detection"
        self.collate_fn = collate_fn
        self.criterion = None # Criterion is handled inside the model
        self.optimizer = optimizer
        self.scheduler = None
        self.device = self._validate_device(device)
        self.dataloader_workers = dataloader_workers
        
        # Callback handler - History and TqdmProgressBar are added by default
        default_callbacks = [History(), TqdmProgressBar()]
        user_callbacks = callbacks if callbacks is not None else []
        self.callbacks = default_callbacks + user_callbacks
        self._set_trainer_on_callbacks()

        # Internal state
        self.train_loader = None
        self.test_loader = None
        self.history = {}
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
            collate_fn=self.collate_fn # Use the provided collate function
        )
        
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=("cuda" in self.device.type),
            collate_fn=self.collate_fn # Use the provided collate function
        )
        
    def _load_checkpoint(self, path: Union[str, Path]):
        """Loads a training checkpoint to resume training."""
        p = make_fullpath(path, enforce="file")
        _LOGGER.info(f"Loading checkpoint from '{p.name}' to resume training...")
        
        try:
            checkpoint = torch.load(p, map_location=self.device)
            
            if PyTorchCheckpointKeys.MODEL_STATE not in checkpoint or PyTorchCheckpointKeys.OPTIMIZER_STATE not in checkpoint:
                _LOGGER.error(f"Checkpoint file '{p.name}' is invalid. Missing 'model_state_dict' or 'optimizer_state_dict'.")
                raise KeyError()

            self.model.load_state_dict(checkpoint[PyTorchCheckpointKeys.MODEL_STATE])
            self.optimizer.load_state_dict(checkpoint[PyTorchCheckpointKeys.OPTIMIZER_STATE])
            self.start_epoch = checkpoint.get(PyTorchCheckpointKeys.EPOCH, 0) + 1 # Resume on the *next* epoch
            
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
                if isinstance(cb, ModelCheckpoint) and PyTorchCheckpointKeys.BEST_SCORE in checkpoint:
                    cb.best = checkpoint[PyTorchCheckpointKeys.BEST_SCORE]
                    _LOGGER.info(f"Restored {cb.__class__.__name__} 'best' score to: {cb.best:.4f}")
            
            _LOGGER.info(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            
        except Exception as e:
            _LOGGER.error(f"Failed to load checkpoint from '{p}': {e}")
            raise

    def fit(self, 
            epochs: int = 10, 
            batch_size: int = 10, 
            shuffle: bool = True,
            resume_from_checkpoint: Optional[Union[str, Path]] = None):
        """
        Starts the training-validation process of the model.
        
        Returns the "History" callback dictionary.

        Args:
            epochs (int): The total number of epochs to train for.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the training data at each epoch.
            resume_from_checkpoint (str | Path | None): Optional path to a checkpoint to resume training.
        """
        self.epochs = epochs
        self._batch_size = batch_size
        self._create_dataloaders(self._batch_size, shuffle)
        self.model.to(self.device)
        
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Reset stop_training flag on the trainer
        self.stop_training = False

        self._callbacks_hook('on_train_begin')
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.epoch = epoch
            epoch_logs = {}
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
        return self.history
    
    def _train_step(self):
        self.model.train()
        running_loss = 0.0
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
            
            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss # type: ignore
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / len(self.train_loader.dataset)} # type: ignore

    def _validation_step(self):
        self.model.train() # Set to train mode even for validation loss calculation
                           # as model internals (e.g., proposals) might differ,
                           # but we still need loss_dict.
                           # We use torch.no_grad() to prevent gradient updates.
        running_loss = 0.0
        with torch.no_grad():
            for images, targets in self.test_loader: # type: ignore
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
        
        logs = {PyTorchLogKeys.VAL_LOSS: running_loss / len(self.test_loader.dataset)} # type: ignore
        return logs

    def evaluate(self, save_dir: Union[str, Path], data: Optional[Union[DataLoader, Dataset]] = None):
        """
        Evaluates the model using object detection mAP metrics.

        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            data (DataLoader | Dataset | None): The data to evaluate on. If None, defaults to the trainer's internal test_dataset.
        """
        dataset_for_names = None
        eval_loader = None

        if isinstance(data, DataLoader):
            eval_loader = data
            if hasattr(data, 'dataset'):
                dataset_for_names = data.dataset
        elif isinstance(data, Dataset):
            # Create a new loader from the provided dataset
            eval_loader = DataLoader(data, 
                                     batch_size=self._batch_size, 
                                     shuffle=False, 
                                     num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                                     pin_memory=(self.device.type == "cuda"),
                                     collate_fn=self.collate_fn)
            dataset_for_names = data
        else: # data is None, use the trainer's default test dataset
            if self.test_dataset is None:
                _LOGGER.error("Cannot evaluate. No data provided and no test_dataset available in the trainer.")
                raise ValueError()
            # Create a fresh DataLoader from the test_dataset
            eval_loader = DataLoader(
                self.test_dataset, 
                batch_size=self._batch_size, 
                shuffle=False, 
                num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=self.collate_fn
            )
            dataset_for_names = self.test_dataset

        if eval_loader is None:
            _LOGGER.error("Cannot evaluate. No valid data was provided or found.")
            raise ValueError()

        print("\n--- Model Evaluation ---")

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
            if hasattr(dataset_for_names, 'classes'):
                class_names = dataset_for_names.classes # type: ignore
            # Fallback for Subset
            elif hasattr(dataset_for_names, 'dataset') and hasattr(dataset_for_names.dataset, 'classes'): # type: ignore
                 class_names = dataset_for_names.dataset.classes # type: ignore
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
        
        print("\n--- Training History ---")
        plot_losses(self.history, save_dir=save_dir)
    
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


def info():
    _script_info(__all__)
