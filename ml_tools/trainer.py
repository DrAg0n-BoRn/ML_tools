import time
import numpy
from typing import Literal
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, r2_score, median_absolute_error
from .utilities import _script_info


__all__ = [
    "MyTrainer"
]


class MyTrainer():
    def __init__(self, model, train_dataset: Dataset, test_dataset: Dataset, kind: Literal["regression", "classification"], 
                 criterion=None , shuffle: bool=True, batch_size: float=3, device: Literal["cpu", "cuda", "mps"]='cpu', learn_rate: float=0.001, dataloader_workers: int=2):
        """
        Automates the training process of a PyTorch Model using Adam optimization by default (`self.optimizer`).
        
        `kind`: Will be used to compute and display metrics after training is complete.
        
        `shuffle`: Whether to shuffle dataset batches at every epoch. Default is True.
        
        `criterion`: Loss function. If 'None', defaults to `nn.NLLLoss` for classification or `nn.MSELoss` for regression.
        
        `batch_size` Represents the fraction of the original dataset size to be used per batch. If an integer is passed, use that many samples, instead. Default is 3 samples at a time. 
        
        `learn_rate` Model learning rate. Default is 0.001.
        
        `dataloader_workers` Subprocesses to use for data loading. Default is 2.
        """
        # Validate kind
        if kind not in ["regression", "classification"]:
            raise TypeError("Kind must be 'regression' or 'classification'.")
        # Validate batch size
        batch_error = "Batch must a float in range [0.01, 1) or an integer."
        if isinstance(batch_size, (float, int)):
            if (1.00 > batch_size >= 0.01):
                train_batch = int(len(train_dataset) * batch_size)
                test_batch = int(len(test_dataset) * batch_size)
            elif batch_size > len(train_dataset) or batch_size > len(test_dataset):
                raise ValueError(batch_error + " Size is greater than dataset size.")
            elif batch_size >= 1:
                train_batch = int(batch_size)
                test_batch = int(batch_size)
            else:
                raise ValueError(batch_error)
        else:
            raise TypeError(batch_error)
        # Validate device
        if device == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available, switching to CPU.")
                device = "cpu"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                print("MPS not available, switching to CPU.")
                device = "cpu"
        # Validate criterion
        if criterion is None:
            if kind == "regression":
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.NLLLoss()
        else:
            self.criterion = criterion
        # Validate dataloader workers
        if not isinstance(dataloader_workers, int):
            raise TypeError("Dataloader workers must be an integer value.")
        
        # Check last layer in the model, implementation pending
        # last_layer_name, last_layer = next(reversed(model._modules.items()))
        # if isinstance(last_layer, nn.Linear):
        #     pass
        
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=shuffle, num_workers=dataloader_workers, pin_memory=True if device=="cuda" else False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=shuffle, num_workers=dataloader_workers, pin_memory=True if device=="cuda" else False)
        self.kind = kind
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learn_rate)


    def auto_train(self, epochs: int=200, patience: int=3, cmap: Literal["viridis", "Blues", "Greens", "Reds", "plasma", "coolwarm"]="Blues", 
                   roc: bool=False, **model_params):
        """
        Start training-validation process of the model. 
        
        `patience` is the number of consecutive times the Validation Loss is allowed to increase before early-stopping the training process.
        
        `cmap` Color map to use for the confusion matrix.
        
        `model_params` Keywords parameters specific to the model, if any.
        
        `roc` Whether to display the Receiver Operating Characteristic (ROC) Curve, for binary classification only.
        """
        metric_name = "accuracy" if self.kind == "classification" else "RMSE"
        previous_val_loss = None
        epoch_tracker = 0
        warnings = 0
        feedback = None
        val_losses = list()
        train_losses = list()
        
        # Validate inputs
        if isinstance(epochs, int):
            if epochs < 1:
                print("Invalid number of epochs")
                return None
        else:
            print("Invalid number of epochs")
            return None
        
        if isinstance(patience, int):
            if patience < 0:
                print("Invalid value for patience")
                return None
        else:
            print("Invalid value for patience")
            return None
        
        if cmap not in ["viridis", "Blues", "Greens", "Reds", "plasma", "coolwarm"]:
            print("Invalid cmap code, 'coolwarm' selected by default")
            cmap = "coolwarm"
        
        # Time training
        start_time = time.time()
        
        for epoch in range(1, epochs+1):
            # Train model
            self.model.train()
            current_train_loss = 0
            # Keep track of predictions and true labels on the last epoch to use later on scikit-learn
            predictions_list = list()  
            true_labels_list = list()
            probabilities_list = list()
            
            for features, target in self.train_loader:
                # features, targets to device
                features = features.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(features, **model_params)
                # check shapes
                # print(features.shape, target.shape, output.shape)
                # For Binary Cross Entropy
                if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    target = target.to(torch.float32)
                elif isinstance(self.criterion, (nn.MSELoss)):
                    output = output.view_as(target)
                train_loss = self.criterion(output, target)
                # Cumulative loss for current epoch on all batches
                current_train_loss += train_loss.item()
                # Backpropagation
                train_loss.backward()
                self.optimizer.step()
                
            # Average Train Loss per sample
            current_train_loss /= len(self.train_loader.dataset)
            train_losses.append(current_train_loss)
            
            # Evaluate
            self.model.eval()
            current_val_loss = 0
            correct = 0
            with torch.no_grad():
                for features, target in self.test_loader:
                # features, targets to device
                    features = features.to(self.device)
                    target = target.to(self.device)
                    output = self.model(features, **model_params)
                    # Save true labels for current batch (in case random shuffle was used)
                    true_labels_list.append(target.view(-1,1).cpu().numpy())
                    # For Binary Cross Entropy
                    if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        target = target.to(torch.float32)
                    elif isinstance(self.criterion, (nn.MSELoss)):
                        output = output.view_as(target)
                    current_val_loss += self.criterion(output, target).item()
                    # Save predictions of current batch, get accuracy
                    if self.kind == "classification":
                        predictions_list.append(output.argmax(dim=1).view(-1,1).cpu().numpy())
                        correct += output.argmax(dim=1).eq(target).sum().item()
                        if roc:
                            probabilities_local = nn.functional.softmax(output, dim=1)
                            probabilities_list.append(probabilities_local.cpu().numpy())
                    else:   # Regression
                        predictions_list.append(output.view(-1,1).cpu().numpy())                        
                        
            # Average Validation Loss per sample
            current_val_loss /= len(self.test_loader.dataset)
            val_losses.append(current_val_loss)
            
            # Concatenate all predictions and true labels
            predictions = numpy.concatenate(predictions_list, axis=0)
            true_labels = numpy.concatenate(true_labels_list, axis=0)
            if roc:
                probabilities = numpy.concatenate(probabilities_list, axis=0)
            
            # Accuracy / RMSE
            if self.kind == "classification":
                accuracy = correct / len(self.test_loader.dataset)
                accuracy = str(round(100*accuracy, ndigits=1)) + "%"
            else: # Regression
                accuracy = numpy.sqrt(mean_squared_error(y_true=true_labels, y_pred=predictions))
                accuracy = str(round(accuracy, ndigits=4))
            
            # Print details
            details_format = f'epoch {epoch:2}:    training loss: {current_train_loss:6.4f}    validation loss: {current_val_loss:6.4f}    {metric_name}: {accuracy}'
            if (epoch % max(1, int(0.05*epochs)) == 0) or epoch in [1, 3, 5]:
                print(details_format)
            
            # Compare validation loss per epoch
            # First run
            if previous_val_loss is None:
                previous_val_loss = current_val_loss
            # If validation loss is increasing or the same (not improving) use patience
            elif current_val_loss >= previous_val_loss:
                if epoch == epoch_tracker + 1:
                    warnings += 1
                else: 
                    warnings = 1
                epoch_tracker = epoch
            # If validation loss decreased
            else:
                warnings = 0
                
            # If patience is exhausted
            if warnings == patience:
                feedback = f"👁️ Validation Loss has increased {patience} consecutive times."
                break
            
            # Training must continue for another epoch
            previous_val_loss = current_val_loss

        # if all epochs have been completed
        else:
            feedback = "Training has been completed without any early-stopping criteria."
        
        # Print feedback message
        print('\n', details_format)
        print(feedback, f"\n")
        
        # Show elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Elapsed time:  {minutes:.0f} minutes  {seconds:2.0f} seconds  {epoch} epochs")
        
        # Plot losses
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4), dpi=150, sharey=False)
        
        ax1.plot(range(2, epoch+1), train_losses[1:])
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Average loss per sample")

        ax2.plot(range(2, epoch+1), val_losses[1:])
        ax2.set_title("Validation Loss")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Average loss per sample")
        
        plt.tight_layout()
        plt.show()
        
        # Metrics
        # Display metrics
        if self.kind == "regression":            
            rmse = numpy.sqrt(mean_squared_error(y_true=true_labels, y_pred=predictions))
            r2 = r2_score(y_true=true_labels, y_pred=predictions)
            medae = median_absolute_error(y_true=true_labels, y_pred=predictions)
            print(f"Root Mean Squared Error (RMSE): {rmse:6.4f}    (range 0 to \u221E)")
            print(f"Median Absolute Error (MedAE): {medae:6.4f}    (range: 0 to \u221E)")
            print(f"Coefficient of Determination (R2 Score): {r2:4.2f}    (range: -\u221E to 1)\n")
            
        elif self.kind == "classification":
            print(classification_report(y_true=true_labels, y_pred=predictions))
            ConfusionMatrixDisplay.from_predictions(y_true=true_labels, y_pred=predictions, cmap=cmap)
            
            # ROC curve & Area under the curve
            if roc:
                false_positives, true_positives, thresholds = roc_curve(y_true=true_labels, y_score=probabilities[:,1])
                area_under_curve = roc_auc_score(y_true=true_labels, y_score=probabilities[:,1])
                
                plt.figure(figsize=(4,4))
                plt.plot(false_positives, true_positives)
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.show()
                
                print(f"Area under the curve score: {area_under_curve:4.2f}")
        else:
            print("Error encountered while retrieving 'model.kind' attribute.")
    
    
    def rnn_forecast(self, sequence: torch.Tensor, steps: int):
        """
        Runs a sequential forecast for a RNN, where each new prediction is obtained by feeding the previous prediction.
        
        The input tensor representing a sequence must be of shape `(sequence length, number of features)` with normalized values (if needed).

        Args:
            `sequence`: Last subsequence of the sequence.
            
            `steps`: Number of future time steps to predict.

        Returns: Numpy array of predictions.
        """
        self.model.eval()
        with torch.no_grad():
            # send sequence to device
            sequence = sequence.to(self.device)
            # Make a dummy list in memory
            sequences = [torch.zeros_like(sequence, device=self.device, requires_grad=False) for _ in range(steps)]
            sequences[0] = sequence
            # Store predictions
            predictions = list()
            # Get predictions
            for i in range(steps):
                in_seq = sequences[i]
                output = self.model(in_seq)
                # Last timestamp
                output = output[-1].view(1,-1)
                # Save prediction
                # Check if it is a single feature, get value
                if output.shape[1] == 1:
                    predictions.append(output.item())
                # Else, return a list of lists
                else:
                    predictions.append(output.squeeze().cpu().tolist())
                # Create next sequence
                if i < steps-1:
                    current_seq = sequences[i]
                    new_seq = torch.concatenate([current_seq[1:], output], dim=0).to(self.device)
                    sequences[i+1] = new_seq
        
        # Cast to array and return
        predictions = numpy.array(predictions)
        return predictions


def info():
    _script_info(__all__)
