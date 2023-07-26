import torch
from torch import nn
from typing import Literal, Union
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, ConfusionMatrixDisplay
import numpy


class MyNeuralNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_layers: list[int]=[40,80,40], kind: Union[Literal["regression", "binary_classification"], int]="regression", drop_out: float=0.2) -> None:
        """
        Creates a Neural Network used for Regression or Classification tasks.
        
        `kind` can be set to "regression" (default) or "binary_classification".
        If multi-class is required, pass the number of output classes to `kind`.
        
        `hidden_layers` takes a list of integers. Each position represents a hidden layer and its number of neurons. 
        
        * One rule of thumb is to choose a number of hidden neurons between the size of the input layer and the size of the output layer. 
        * Another rule suggests that the number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. 
        * Another rule suggests that the number of hidden neurons should be less than twice the size of the input layer.
        
        `drop_out` represents the probability of neurons to be set to '0' during the training process of each layer. Range [0.0, 1.0).
        """
        super().__init__()
        
        # Validate kind
        if kind == "regression":
            code = "r"
            self.kind = "regression"
        elif kind == "binary_classification":
            code = "c"
            self.kind = "classification"
        elif isinstance(kind, int) and kind > 1:
            code = kind
            self.kind = "classification"
        else:
            raise TypeError("kind must be 'regression', 'binary_classification' or an integer for multiclass classification.")
        
        # Validate inputs
        if not isinstance(num_features, int):
            raise TypeError("The number of input features (num_features) must be defined as an integer value.")
        
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
            layers.append(nn.Linear(in_features=num_features, out_features=neurons))
            layers.append(nn.BatchNorm1d(num_features=neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))
            num_features = neurons
        # Append output layer
        layers.append(nn.Linear(in_features=num_features, out_features=1 if code in ["r", "c"] else code))
        
        # Check for classification output
        if code == "c":
            layers.append(nn.Sigmoid())
        elif isinstance(code, int):
            layers.append(nn.Softmax(dim=1))
        
        # Create a container for layers
        self._layers = nn.Sequential(*layers)
    
    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._layers(X)
        return X


class MyConvolutionalNetwork(nn.Module):
    def __init__(self, outputs: int, color_channels: int=3, img_size: int=300, drop_out: float=0.2):
        """
        Create a basic Convolutional Neural Network with two convolution layers. A (2x2) pooling layer is used after each convolution.

        Args:
            outputs (int): Number of output classes (1 means regression).
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
        ann = MyNeuralNetwork(num_features=flat_features, hidden_layers=[int(flat_features*0.7), int(flat_features*0.3), int(flat_features*0.05), int(flat_features*0.001)], 
                              kind=outputs if outputs > 1 else "regression", drop_out=drop_out)
        self._ann_layers = ann._layers
        
        # Join CNN and ANN
        self._structure = nn.Sequential(self._cnn_layers, nn.Flatten(), self._ann_layers)
        
         # inherit kind
        self.kind = "CNN " + ann.kind

    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._structure(X)
        return X
    
    
class MyTrainer():
    def __init__(self, model, criterion, optimizer, train_dataset: Dataset, test_dataset: Dataset, kind: Literal["regression", "classification"], shuffle: bool=True, batch_percentage: Union[float, None]=0.1):
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
        if batch_percentage is None or batch_percentage == 1:
            train_batch = batch_percentage
            test_batch = batch_percentage
        elif isinstance(batch_percentage, float) and (1.00 >= batch_percentage >= 0.01):
            train_batch = int(len(train_dataset) * batch_percentage)
            test_batch = int(len(test_dataset) * batch_percentage)
        else:
            raise TypeError("batch_size must be None or a float value between 1.00 and 0.01")
            
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=shuffle)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=shuffle)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.kind = kind


    def auto_train(self, epochs: int=200, patience: int=3, **model_params):
        """
        Start training-validation process of the model. 
        
        `patience` is the number of consecutive times the Validation Loss is allowed to increase before early-stopping the training process.
        
        `model_params` Keywords parameters specific for the model, if any.
        """
        previous_val_loss = None
        previous_train_loss = None
        epoch_tracker = 0
        warnings = 0
        feedback = None
        losses = list()
        # Time training
        start_time = time.time()
        # Keep track of predictions and true labels to use later on scikit-learn
        predictions_list = list()
        true_labels_list = list()
        
        for epoch in range(1, epochs+1):
            # Train model
            self.model.train()
            current_train_loss = 0
            for batch_index, (features, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(features, **model_params)
                # For Binary Cross Entropy
                if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    target = target.view(-1, 1).type(torch.float32)
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
                    true_labels_list.append(target.squeeze().numpy())
                    # For Binary Cross Entropy
                    if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        target = target.view(-1, 1).type(torch.float32)
                    current_val_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    # Save predictions of current batch
                    predictions_list.append(pred.squeeze().numpy())
                    # Compare (equality) the target and the predicted target, use dimensional compatibility. 
                    # this results in a tensor of booleans, sum up all Trues and return the value as a scalar.
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
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
                
            # If patience is exhausted
            if warnings == patience:
                feedback = f"Validation Loss has increased {patience} consecutive times. Check for possible overfitting or modify the patience value."
                break
            
            # Compare training loss per epoch
            # First run
            if previous_train_loss is None:
                previous_train_loss = current_train_loss
            # If training loss has not improved
            elif current_train_loss > previous_train_loss:
                pass
            # Stop if the current training loss is less than 0.1% of the previous training loss
            elif previous_train_loss - current_train_loss < (previous_train_loss * 0.001):
                feedback = f"Current Train Loss ({current_train_loss:.5f}) is less than 0.1% of the previous Train Loss ({previous_train_loss:.5f}), training complete."
                break
            
            # Training must continue for another epoch
            previous_val_loss = current_val_loss
            previous_train_loss = current_train_loss

        # if all epochs have been completed
        else:
            feedback = "Training has been completed without reaching any early-stopping criteria. Consider modifying the learning rate or using more epochs."
        
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

