from typing import List, Literal, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np

from .ML_callbacks import Callback, History, TqdmProgressBar
from .ML_evaluation import classification_metrics, regression_metrics, plot_losses, shap_summary_plot
from .utilities import _script_info
from .keys import LogKeys
from .logger import _LOGGER


__all__ = [
    "MyTrainer"
]


class MyTrainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, 
                 kind: Literal["regression", "classification"],
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 device: Union[Literal['cuda', 'mps', 'cpu'],str], dataloader_workers: int = 2, callbacks: Optional[List[Callback]] = None):
        """
        Automates the training process of a PyTorch Model.
        
        Built-in Callbacks: `History`, `TqdmProgressBar`

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing/validation dataset.
            kind (str): The type of task, 'regression' or 'classification'.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): The device to run training on ('cpu', 'cuda', 'mps').
            dataloader_workers (int): Subprocesses for data loading. Defaults to 2.
            callbacks (List[Callback] | None): A list of callbacks to use during training.
            
        Note:
            For **regression** tasks, suggested criterions include `nn.MSELoss` or `nn.L1Loss`.
            
            For **classification** tasks, `nn.CrossEntropyLoss` (multi-class) or `nn.BCEWithLogitsLoss` (binary) are common choices.
        """
        if kind not in ["regression", "classification"]:
            raise TypeError("Kind must be 'regression' or 'classification'.")

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.kind = kind
        self.criterion = criterion
        self.optimizer = optimizer
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
        self.stop_training = False

    def _validate_device(self, device: str) -> torch.device:
        """Validates the selected device and returns a torch.device object."""
        device_lower = device.lower()
        if "cuda" in device_lower and not torch.cuda.is_available():
            _LOGGER.warning("⚠️ CUDA not available, switching to CPU.")
            device = "cpu"
        elif device_lower == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("⚠️ Apple Metal Performance Shaders (MPS) not available, switching to CPU.")
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
            pin_memory=(self.device.type == "cuda")
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=loader_workers, 
            pin_memory=(self.device.type == "cuda")
        )

    def fit(self, epochs: int = 10, batch_size: int = 32, shuffle: bool = True):
        """
        Starts the training-validation process of the model.

        Args:
            epochs (int): The total number of epochs to train for.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the training data at each epoch.
        """
        self.epochs = epochs
        self._create_dataloaders(batch_size, shuffle)
        self.model.to(self.device)
        
        # Reset stop_training flag on the trainer
        self.stop_training = False

        self.callbacks_hook('on_train_begin')
        
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            epoch_logs = {}
            self.callbacks_hook('on_epoch_begin', epoch, logs=epoch_logs)

            train_logs = self._train_step()
            epoch_logs.update(train_logs)

            val_logs = self._validation_step()
            epoch_logs.update(val_logs)
            
            self.callbacks_hook('on_epoch_end', epoch, logs=epoch_logs)
            
            # Check the early stopping flag
            if self.stop_training:
                break

        self.callbacks_hook('on_train_end')
        return self.history
    
    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        # Enumerate to get batch index
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            # Create a log dictionary for the batch
            batch_logs = {
                LogKeys.BATCH_INDEX: batch_idx, 
                LogKeys.BATCH_SIZE: features.size(0)
            }
            self.callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(features)
            if isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                output = output.view_as(target)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Calculate batch loss and update running loss for the epoch
            batch_loss = loss.item()
            running_loss += batch_loss * features.size(0)

            # Add the batch loss to the logs and call the end-of-batch hook
            batch_logs[LogKeys.BATCH_LOSS] = batch_loss
            self.callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)

        # Return the average loss for the entire epoch
        return {LogKeys.TRAIN_LOSS: running_loss / len(self.train_loader.dataset)} # type: ignore

    def _validation_step(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for features, target in self.test_loader: # type: ignore
                features, target = features.to(self.device), target.to(self.device)
                output = self.model(features)
                if isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                    output = output.view_as(target)
                loss = self.criterion(output, target)
                running_loss += loss.item() * features.size(0)
        logs = {LogKeys.VAL_LOSS: running_loss / len(self.test_loader.dataset)} # type: ignore
        return logs
    
    def predict(self, dataloader: DataLoader):
        """
        Yields model predictions batch by batch, avoids loading all predictions into memory at once.

        Args:
            dataloader (DataLoader): The dataloader to predict on.

        Yields:
            tuple: A tuple containing (y_pred_batch, y_prob_batch, y_true_batch).
                   y_prob_batch is None for regression tasks.
        """
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for features, target in dataloader:
                features = features.to(self.device)
                output = self.model(features).cpu()
                y_true_batch = target.numpy()

                if self.kind == "classification":
                    probs = nn.functional.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_pred_batch = preds.numpy()
                    y_prob_batch = probs.numpy()
                else:
                    y_pred_batch = output.numpy()
                    y_prob_batch = None
                
                yield y_pred_batch, y_prob_batch, y_true_batch
    
    def evaluate(self, data: Optional[Union[DataLoader, Dataset]] = None, save_dir: Optional[Union[str,Path]] = None):
        """
        Evaluates the model on the given data.

        Args:
            data (DataLoader | Dataset | None ): The data to evaluate on.
                Can be a DataLoader or a Dataset. If None, defaults to the trainer's internal test_dataset.
            save_dir (str | Path | None): Directory to save all reports and plots. If None, metrics are shown but not saved.
        """
        eval_loader = None
        if isinstance(data, DataLoader):
            eval_loader = data
        else:
            # Determine which dataset to use (the one passed in, or the default test_dataset)
            dataset_to_use = data if data is not None else self.test_dataset
            if not isinstance(dataset_to_use, Dataset):
                raise ValueError("Cannot evaluate. No valid DataLoader or Dataset was provided, "
                                 "and no test_dataset is available in the trainer.")

            # Create a new DataLoader from the dataset
            eval_loader = DataLoader(
                dataset=dataset_to_use,
                batch_size=32,  # A sensible default for evaluation
                shuffle=False,
                num_workers=0 if self.device.type == 'mps' else self.dataloader_workers,
                pin_memory=(self.device.type == "cuda")
            )
            
        print("\n--- Model Evaluation ---")

        # Collect results from the predict generator
        all_preds, all_probs, all_true = [], [], []
        for y_pred_b, y_prob_b, y_true_b in self.predict(eval_loader):
            all_preds.append(y_pred_b)
            if y_prob_b is not None:
                all_probs.append(y_prob_b)
            all_true.append(y_true_b)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs) if self.kind == "classification" else None

        if self.kind == "classification":
            classification_metrics(y_true, y_pred, y_prob, save_dir=save_dir)
        else:
            regression_metrics(y_true.flatten(), y_pred.flatten(), save_dir=save_dir)

        print("\n--- Training History ---")
        plot_losses(self.history, save_dir=save_dir)
    
    def explain(self, explain_dataset: Optional[Dataset] = None, n_samples: int = 100, 
                feature_names: Optional[List[str]] = None, save_dir: Optional[str] = None):
        """
        Explains model predictions using SHAP and saves all artifacts.

        The background data is automatically sampled from the trainer's training dataset.

        Args:
            explain_dataset (Dataset, optional): A specific dataset to explain. 
                                                 If None, the trainer's test dataset is used.
            n_samples (int): The number of samples to use for both background and explanation.
            feature_names (List[str], optional): Names for the features.
            save_dir (str, optional): Directory to save all SHAP artifacts.
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
            print("Warning: Trainer's train_dataset is empty or invalid. Skipping SHAP analysis.")
            return

        # 2. Determine target dataset and get explanation instances
        target_dataset = explain_dataset if explain_dataset is not None else self.test_dataset
        instances_to_explain = _get_random_sample(target_dataset, n_samples)
        if instances_to_explain is None:
            print("Warning: Explanation dataset is empty or invalid. Skipping SHAP analysis.")
            return

        # 3. Call the plotting function
        shap_summary_plot(
            model=self.model,
            background_data=background_data,
            instances_to_explain=instances_to_explain,
            feature_names=feature_names,
            save_dir=save_dir
        )
    

    def callbacks_hook(self, method_name: str, *args, **kwargs):
        """Calls the specified method on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)

def info():
    _script_info(__all__)
