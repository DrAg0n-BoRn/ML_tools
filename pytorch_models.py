import torch
from torch import nn
from typing import Literal
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, ConfusionMatrixDisplay
import numpy


class MyNeuralNetwork(nn.Module):
    def __init__(self, in_features: int, out_targets: int, hidden_layers: list[int]=[40,80,40], drop_out: float=0.2) -> None:
        """
        Creates a basic Neural Network.
        
        * For Regression the last layer is Linear. 
        * For Classification the last layer is Logarithmic Softmax.
        
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
            layers.append(nn.LogSoftmax(dim=1))
        
        # Create a container for layers
        self._layers = nn.Sequential(*layers)
    
    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._layers(X)
        return X


class MyConvolutionalNetwork(nn.Module):
    def __init__(self, outputs: int, color_channels: int=3, img_size: int=256, drop_out: float=0.2):
        """
        Create a basic Convolutional Neural Network with two convolution layers with a pooling layer after each convolution.

        Args:
            `outputs`: Number of output classes (1 for regression).
            
            `color_channels`: Color channels. Default is 3 (RGB).
            
            `img_size`: Width and Height of image samples, must be square images. Default is 200.
            
            `drop_out`: Neuron drop out probability. Default is 20%.
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
            nn.Conv2d(in_channels=color_channels, out_channels=(color_channels * 2), kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=(4,4)),
            nn.Conv2d(in_channels=(color_channels * 2), out_channels=(color_channels * 3), kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=(2,2))
        )
        # Calculate output features
        flat_features = int(int((int((img_size + 2 - (5-1))//4) - (3-1))//2)**2) * (color_channels * 3)
        
        # Make a standard ANN
        ann = MyNeuralNetwork(in_features=flat_features, hidden_layers=[int(flat_features*0.5), int(flat_features*0.2), int(flat_features*0.005)], 
                              out_targets=outputs, drop_out=drop_out)
        self._ann_layers = ann._layers
        
        # Join CNN and ANN
        self._structure = nn.Sequential(self._cnn_layers, nn.Flatten(), self._ann_layers)
        
        # Send to CUDA if available
        # if torch.cuda.is_available():
        #     self.to('cuda')
        
    # Override forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._structure(X)
        return X


class MyLSTMNetwork(nn.Module):
    def __init__(self, features: int=1, hidden_size: int=100, recurrent_layers: int=1, dropout: float=0, reset_memory: bool=False, **kwargs):
        """
        Create a simple Recurrent Neural Network to predict 1 time step into the future of sequential data.
        
        The sequence should be a 2D tensor with shape (sequence_length, number_of_features).

        Args:
            * `features`: Number of features representing the sequence. Defaults to 1.
            * `hidden_size`: Hidden size of the LSTM model. Defaults to 100.
            * `recurrent_layers`: Number of recurrent layers to use. Defaults to 1.
            * `dropout`: Probability of dropping out neurons in each recurrent layer, except the last layer. Defaults to 0.
            * `reset_memory`: Reset the initial hidden state and cell state for the recurrent layers at every epoch. Defaults to False.
            * `kwargs`: Create custom attributes for the model.
            
        Custom forward() parameters:
            * `batch_size=1` (int): batch size for the LSTM net.
            * `return_last_timestamp=False` (bool): Return only the value at `output[-1]`
        """
        # validate input size
        if not isinstance(features, int):
            raise TypeError("Input size must be an integer value.")
        # validate hidden size
        if not isinstance(hidden_size, int):
            raise TypeError("Hidden size must be an integer value.")
        # validate layers
        if not isinstance(recurrent_layers, int):
            raise TypeError("Number of recurrent layers must be an integer value.")
        # validate dropout
        if isinstance(dropout, (float, int)):
            if 0 <= dropout < 1:
                pass
            else:
                raise ValueError("Dropout must be a float in range [0.0, 1.0)")
        else:
            raise TypeError("Dropout must be a float in range [0.0, 1.0)")
        
        super().__init__()
        
        # Initialize memory
        self._reset = reset_memory
        self._memory = None
        
        # hidden size and features shape
        self._hidden = hidden_size
        self._features = features
        
        # RNN
        self._lstm = nn.LSTM(input_size=features, hidden_size=self._hidden, num_layers=recurrent_layers, dropout=dropout)
        
        # Fully connected layer
        self._ann = nn.Linear(in_features=self._hidden, out_features=features)
        
        # Parse extra parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        
    def forward(self, seq: torch.Tensor, batch_size: int=1, return_last_timestamp: bool=False) -> torch.Tensor:
        # reset memory
        if self._reset:
            self._memory = None
        # reshape sequence to feed RNN
        seq = seq.view(-1, batch_size, self._features)
        # Pass sequence through RNN
        seq, self._memory = self._lstm(seq, self._memory)
        # Detach hidden state and cell state to prevent backpropagation error
        self._memory = tuple(m.detach() for m in self._memory)
        # Reshape outputs
        seq = seq.view(-1, self._hidden)
        # Pass sequence through fully connected layer
        output = self._ann(seq)
        # Return prediction of 1 time step in the future
        if return_last_timestamp:
            return output[-1].view(1,-1) #last item as a tensor.
        else:
            return output

    
class MyTrainer():
    def __init__(self, model, train_dataset: Dataset, test_dataset: Dataset, kind: Literal["regression", "classification"], 
                 criterion=None , shuffle: bool=True, batch_size: float=0.1, device: Literal["cpu", "cuda", "mps"]='cpu', learn_rate: float=0.001, dataloader_workers: int=2):
        """
        Automates the training process of a PyTorch Model using Adam optimization by default (`self.optimizer`).
        
        `kind`: Will be used to compute and display metrics after training is complete.
        
        `shuffle`: Whether to shuffle dataset batches at every epoch. Default is True.
        
        `criterion`: Loss function. If 'None', defaults to `nn.NLLLoss` for classification or `nn.MSELoss` for regression.
        
        `batch_size` Represents the fraction of the original dataset size to be used per batch. If an integer is passed, use that many samples, instead. Default is 10%. 
        
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


    def auto_train(self, epochs: int=200, patience: int=3, **model_params):
        """
        Start training-validation process of the model. 
        
        `patience` is the number of consecutive times the Validation Loss is allowed to increase before early-stopping the training process.
        
        `model_params` Keywords parameters specific for the model, if any.
        """
        metric_name = "accuracy" if self.kind == "classification" else "RMSE"
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
                    target = target.view_as(output)
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
                        target = target.view_as(output)
                    current_val_loss += self.criterion(output, target).item()
                    # Save predictions of current batch, get accuracy
                    if self.kind == "classification":
                        predictions_list.append(output.argmax(dim=1).view(-1,1).cpu().numpy())
                        correct += output.argmax(dim=1).eq(target).sum().item()
                    else:   # Regression
                        predictions_list.append(output.view(-1,1).cpu().numpy())                        
                        
            # Average Validation Loss per sample
            current_val_loss /= len(self.test_loader.dataset)
            losses.append(current_val_loss)
            
            # Concatenate all predictions and true labels
            predictions = numpy.concatenate(predictions_list, axis=0)
            true_labels = numpy.concatenate(true_labels_list, axis=0)
            
            # Accuracy / RMSE
            if self.kind == "classification":
                accuracy = correct / len(self.test_loader.dataset)
                accuracy = str(round(100*accuracy, ndigits=1)) + "%"
            else: # Regression
                accuracy = numpy.sqrt(mean_squared_error(y_true=true_labels, y_pred=predictions))
                accuracy = str(round(accuracy, ndigits=4))
            
            # Print details
            details_format = f'epoch: {epoch:4}    training loss: {current_train_loss:6.4f}    validation loss: {current_val_loss:6.4f}    {metric_name}: {accuracy}'
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
        
        # Plot
        plt.figure(figsize=(4,4))
        plt.plot(range(1, epoch+1), losses)
        plt.title("Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss per sample")
        plt.show()
        
        # Metrics
        # Display metrics
        if self.kind == "regression":            
            rmse = numpy.sqrt(mean_squared_error(y_true=true_labels, y_pred=predictions))
            print(f"Root Mean Squared Error: {rmse:.2f}\n")
        elif self.kind == "classification":
            print(classification_report(y_true=true_labels, y_pred=predictions))
            ConfusionMatrixDisplay.from_predictions(y_true=true_labels, y_pred=predictions)
        else:
            print("Error encountered while retrieving 'model.kind' attribute.")


    def forecast(self, data_points: list[torch.Tensor]):
        """
        Returns a forecast for n data points. 
        
        Each data point must be a tensor and have the same shape and normalization expected by the model.

        Args:
            `data_points`: list of data points as tensors.

        Returns: List of predicted values.
        """
        self.model.eval()
        results = list()
        with torch.no_grad():
            for data_point in data_points:
                data_point.to(self.device)
                output = self.model(data_point)
                if self.kind == "classification":
                    results.append(output.argmax(dim=1).squeeze().item())
                else:
                    results.append(output.squeeze().item())
        
        return results
    
    
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
                