import torch
from torch import nn
from .utilities import _script_info


__all__ = [
    "MyNeuralNetwork",
    "MyLSTMNetwork"
]


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


class _MyConvolutionalNetwork(nn.Module):
    def __init__(self, outputs: int, color_channels: int=3, img_size: int=256, drop_out: float=0.2):
        """
        - EDUCATIONAL PURPOSES ONLY, not optimized and requires lots of memory.
        
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


def info():
    _script_info(__all__)
