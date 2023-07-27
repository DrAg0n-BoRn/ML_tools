import torch
from torch import nn
from typing import Literal, Union
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, ConfusionMatrixDisplay
import numpy


class MyNeuralNetwork(nn.Module):
    def __init__(self, in_features: int, out_targets: int, hidden_layers: list[int]=[40,80,40], drop_out: float=0.2) -> None:
        """
        Creates a Neural Network used for Regression or Classification tasks.
        
        `out_targets` Is the number of expected output classes for classification; or `1` for regression.
        
        `hidden_layers` takes a list of integers. Each position represents a hidden layer and its number of neurons. 
        
        * One rule of thumb is to choose a number of hidden neurons between the size of the input layer and the size of the output layer. 
        * Another rule suggests that the number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. 
        * Another rule suggests that the number of hidden neurons should be less than twice the size of the input layer.
        
        `drop_out` represents the probability of neurons to be set to '0' during the training process of each layer. Range [0.0, 1.0).
        """
        super().__init__()
        
        # Validate inputs and outputs
        if isinstance(in_features, int) and isinstance(out_targets, int):
            if in_features < 1 or out_targets < 1:
                raise ValueError("Inputs or Outputs must be an integer value.")
        else:
            raise TypeError("Inputs or Outputs must be an integer value.")
        
        # Validate layers
        if isinstance(hidden_layers, list):
            for number in hidden_layers:
                if not isinstance(number, int):
                    raise TypeError("Number of neurons per hidden layer must be an integer value.")
        else:
            raise TypeError("hidden_layers must be a list of integer values.")
        
        # Validate dropout
        if isinstance(drop_out, float):
            if 1.0 > drop_out >= 0.0:
                pass
            else:
                raise TypeError("drop_out must be a float value greater than or equal to 0 and less than 1.")
        elif drop_out == 0:
            pass
        else:
            raise TypeError("drop_out must be a float value greater than or equal to 0 and less than 1.")
        
        
        # Create layers        
        layers = list()
        for neurons in hidden_layers:
            layers.append(nn.Linear(in_features=in_features, out_features=neurons))
            layers.append(nn.BatchNorm1d(num_features=neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))
            in_features = neurons    
        # Append output layer
        layers.append(nn.Linear(in_features=in_features, out_features=out_targets))
        
        # Check for classification or regression output
        if out_targets > 1:
            # layers.append(nn.Sigmoid())
            layers.append(nn.Softmax(dim=1))
        
        # Create a container for layers
        self._layers = nn.Sequential(*layers)
    
    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._layers(X)
        return X


class MyConvolutionalNetwork(nn.Module):
    def __init__(self, outputs: int, color_channels: int=3, img_size: int=250, drop_out: float=0.2):
        """
        Create a basic Convolutional Neural Network with two convolution layers. A (2x2) pooling layer is used after each convolution.

        Args:
            outputs (int): Number of output classes (1 for regression).
            color_channels (int, optional): Color channels. Default is '3' for RGB images.
            img_size (int, optional): Width and Height of image samples, must be square images. Default is '300'.
            drop_out (float, optional): Neuron drop out probability. Default is '0.2'.
        """
        super().__init__()
        
        # Validate outputs number
        integer_error = " must be an integer greater than 0."
        if isinstance(outputs, int):
            if outputs < 1:
                raise ValueError("Outputs" + integer_error)
        else:
            raise TypeError("Outputs" + integer_error)
        # Validate color channels
        if isinstance(color_channels, int):
            if color_channels < 1:
                raise ValueError("Color Channels" + integer_error)
        else:
            raise TypeError("Color Channels" + integer_error)
        # Validate image size
        if isinstance(img_size, int):
            if img_size < 1:
                raise ValueError("Image size" + integer_error)
        else:
            raise TypeError("Image size" + integer_error)        
        # Validate drop out
        if isinstance(drop_out, float):
            if 1.0 > drop_out >= 0.0:
                pass
            else:
                raise TypeError("Drop out must be a float value greater than or equal to 0 and less than 1.")
        elif drop_out == 0:
            pass
        else:
            raise TypeError("Drop out must be a float value greater than or equal to 0 and less than 1.")
        
        # 2 convolutions, 2 pooling layers
        self._cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=color_channels, out_channels=(color_channels * 3), kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            nn.Conv2d(in_channels=(color_channels * 3), out_channels=(color_channels * 5), kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=(2,2))
        )
        # Calculate output features
        flat_features = int(int((int((img_size + 2 - (5-1))/2) - (3-1))/2)**2) * (color_channels * 5)
        
        # Make a standard ANN
        ann = MyNeuralNetwork(in_features=flat_features, hidden_layers=[int(flat_features*0.7), int(flat_features*0.3), int(flat_features*0.05), int(flat_features*0.001)], 
                              out_targets=outputs, drop_out=drop_out)
        self._ann_layers = ann._layers
        
        # Join CNN and ANN
        self._structure = nn.Sequential(self._cnn_layers, nn.Flatten(), self._ann_layers)
        
    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._structure(X)
        return X
    
    
class MyTrainer():
    def __init__(self, model, criterion, optimizer, train_dataset: Dataset, test_dataset: Dataset, kind: Literal["regression", "classification"], shuffle: bool=True, batch_percentage: float=0.1):
        """
        Automates the training process of a PyTorch Model.
        
        `kind`: Will be used to compute and display metrics after training is complete.
        
        `shuffle`: Whether to shuffle dataset batches at every epoch. Default is True.
        
        `batch_percentage` Represents the fraction of the original dataset size to be used per batch. Default is 10%. 
        """
        # Validate kind
        if kind not in ["regression", "classification"]:
            raise TypeError("Kind must be 'regression' or 'classification'.")
        # Validate batch size
        if isinstance(batch_percentage, float):
            if (1.00 > batch_percentage >= 0.01):
                train_batch = int(len(train_dataset) * batch_percentage)
                test_batch = int(len(test_dataset) * batch_percentage)
            else:
                raise ValueError("batch_size must a float value in range (1.00, 0.01]")
        else:
            raise TypeError("batch_size must a float value in range (1.00, 0.01]")
            
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=shuffle)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=shuffle)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.kind = kind


    def auto_train(self, epochs: int=200, patience: int=4, **model_params):
        """
        Start training-validation process of the model. 
        
        `patience` is the number of consecutive times the Validation Loss is allowed to increase before early-stopping the training process.
        
        `model_params` Keywords parameters specific for the model, if any.
        """
        previous_val_loss = None
        epoch_tracker = 0
        warnings = 0
        feedback = None
        losses = list()
        # Time training
        start_time = time.time()
        
        for epoch in range(1, epochs+1):
            # Train model
            self.model.train()
            current_train_loss = 0
            # Keep track of predictions and true labels on the last epoch to use later on scikit-learn
            predictions_list = list()  
            true_labels_list = list()
            
            for batch_index, (features, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(features, **model_params)
                # print("output type:", output.dtype, " requires grad? ", output.requires_grad)
                # print("target type:", target.dtype, " requires grad? ", target.requires_grad)
                # Get the predicted output for classification
                if self.kind == "classification":
                    output = output.argmax(dim=1)
                    output = output.to(torch.float32)
                    output.requires_grad = True
                # For Binary Cross Entropy
                if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    target = target.to(torch.float32)
                train_loss = self.criterion(output, target)
                # Cumulative loss for current epoch on all batches
                current_train_loss += train_loss.item()
                # Backpropagation
                train_loss.backward()
                self.optimizer.step()
                
            # Average Train Loss per sample
            current_train_loss /= len(self.train_loader.dataset)
            
            # Evaluate
            self.model.eval()
            current_val_loss = 0
            correct = 0
            with torch.no_grad():
                for features, target in self.test_loader:
                    output = self.model(features, **model_params)
                    # Save true labels for current batch (in case random shuffle was used)
                    true_labels_list.append(target.view(-1,1).numpy())
                    # Get the predicted output for classification
                    if self.kind == "classification":
                        output = output.argmax(dim=1)
                        output = output.to(torch.float32)
                    # For Binary Cross Entropy
                    if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        target = target.to(torch.float32)
                    current_val_loss += self.criterion(output, target).item()
                    # Save predictions of current batch
                    predictions_list.append(output.view(-1,1).numpy())
                    # Compare (equality) the target and the predicted target, use dimensional compatibility if needed. 
                    # this results in a tensor of booleans, sum up all Trues and return the value as a scalar.
                    correct += output.eq(target.view_as(output)).sum().item()
                    
                # Average Validation Loss per sample
                current_val_loss /= len(self.test_loader.dataset)
                losses.append(current_val_loss)
                # Accuracy
                accuracy = correct / len(self.test_loader.dataset)
            
            # Print details
            details_format = f'epoch: {epoch:4}    training loss: {current_train_loss:6.4f}    validation loss: {current_val_loss:6.4f}    accuracy: {100*accuracy:5.2f}%'
            if (epoch % int(0.05*epochs) == 0) or epoch in [1, 3, 5]:
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
                # Stop if the current validation loss is less than 0.05% of the previous loss
                # if previous_val_loss - current_val_loss < (previous_val_loss * 0.0005):
                #     feedback = f"Current Validation Loss ({current_val_loss:.5f}) is less than 0.05% of the previous Loss ({previous_val_loss:.5f}), training complete."
                #     break
                
            # If patience is exhausted
            if warnings == patience:
                feedback = f"Validation Loss has increased {patience} consecutive times. Check for possible overfitting or modify the patience value."
                break
            
            # Training must continue for another epoch
            previous_val_loss = current_val_loss

        # if all epochs have been completed
        else:
            feedback = "Training has been completed without reaching any early-stopping criteria."
        
        # Print feedback message
        print('\n', details_format)
        print(feedback, f"\n")
        
        # Show elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Elapsed time:  {minutes:.0f} minutes  {seconds:2.0f} seconds  {epoch} epochs")
        
        # Plot
        plt.figure(figsize=(4,4))
        plt.plot(range(1, epoch+1), losses)
        plt.title("Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss per sample")
        plt.show()
        
        # Metrics
        # Concatenate all predictions and true labels
        predictions = numpy.concatenate(predictions_list, axis=0)
        true_labels = numpy.concatenate(true_labels_list, axis=0)
        # Display metrics
        if self.kind == "regression":            
            rmse = numpy.sqrt(mean_squared_error(y_true=true_labels, y_pred=predictions))
            print(f"Root Mean Squared Error: {rmse:.2f}")
        elif self.kind == "classification":
            print(classification_report(y_true=true_labels, y_pred=predictions))
            ConfusionMatrixDisplay.from_predictions(y_true=true_labels, y_pred=predictions)
        else:
            print("Error encountered while retrieving 'model.kind' attribute, no metrics processed.")
            
        # print("Predictions", predictions.shape, predictions[:15])
        # print("True labels", true_labels.shape, true_labels[:15])

