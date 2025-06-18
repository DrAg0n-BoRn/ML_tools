import torch 
from torch.utils.data import Dataset, TensorDataset
from torch import nn
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal, Union
from imblearn.combine import SMOTETomek
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from .utilities import _script_info


__all__ = [
    "DatasetMaker",
    "PytorchDataset",
    "make_vision_dataset",
    "SequenceDataset",
]


class DatasetMaker():
    def __init__(self, *, pandas_df: pandas.DataFrame, label_col: str, cat_features: Union[list[str], None]=None, 
                 cat_method: Union[Literal["one-hot", "embed"], None]="one-hot", test_size: float=0.2, random_state: Union[int, None]=None, 
                 normalize: Union[Literal["standard", "minmax"], None]="standard", cast_labels: bool=True, balance: bool=False, **kwargs):
        """
        Create Train-Test datasets from a Pandas DataFrame. Four datasets will be created: 
        
            1. Features Train
            2. Features Test
            3. Labels Train
            4. Labels Test
            
        Use the method `to_pytorch()` to quickly get Train and Test PytorchDataset objects.
        
        `label_col` Specify the name of the label column. If label encoding is required (str -> int) set `cast_labels=True` (default). 
        A dictionary will be created with the label mapping {code: original_name}.
        
        `cat_features` List of column names to perform embedding or one-hot-encoding of categorical features. 
        Any categorical column not in the list will not be returned. 
        If `None` (default), columns containing categorical data will be inferred from dtypes: object, string and category, if any. 
        
        `cat_method` can be set to: 
        
            * `'one-hot'` (default) to perform One-Hot-Encoding using Pandas "get_dummies".
            * `'embed'` to perform Embedding using PyTorch "nn.Embedding".
            * `None` all data will be considered to be continuous.
        
        `normalize` if not None, continuous features will be normalized using Scikit-Learn's StandardScaler or MinMaxScaler.
        
        If `balance=True` attempts to balance the minority class(es) in the training data using Imbalanced-Learn's `SMOTETomek` algorithm.
        
        `**kwargs` Pass any additional keyword parameters to `pandas.get_dummies()` or `torch.nn.Embedding()`. 
            i.e. pandas `drop_first=False`.
        """
        
        # Validate dataframe
        if not isinstance(pandas_df, pandas.DataFrame):
            raise TypeError("pandas_df must be a pandas.DataFrame object.")
        # Validate label column
        if not isinstance(label_col, (str, list)):
            raise TypeError("label_col must be a string or list of strings.")
        # Validate categorical features
        if not (isinstance(cat_features, list) or cat_features is None):
            raise TypeError("cat_features must be a list of strings or None.")
        if cat_method not in ["one-hot", "embed", None]:
            raise TypeError("cat_method must be 'one-hot', 'embed' or None.")
        # Validate test size
        if not isinstance(test_size, (float, int)):
            raise TypeError("test_size must be a float in the range 0.0 to 1.0")
        if not (1.0 >= test_size >= 0.0):
            raise ValueError("test_size must be a float in the range 0.0 to 1.0")
        # Validate random state
        if not (isinstance(random_state, int) or random_state is None):
            raise TypeError("random_state must be an integer or None.")
        # validate normalize
        if not (normalize in ["standard", "minmax"] or normalize is None):
            raise TypeError("normalize must be 'standard', 'minmax' or None.")
        # Validate cast labels
        if not isinstance(cast_labels, bool):
            raise TypeError("cast_labels must be either True or False.")
        
        # Start-o
        self._labels = pandas_df[label_col]
        pandas_df = pandas_df.drop(columns=label_col)
        # Set None parameters
        self._categorical = None
        self._continuous = None
        self.labels_train = None
        self.labels_test = None
        self.labels_map = None
        self.features_test = None
        self.features_train = None
        
        # Find categorical
        cat_columns = list()
        if cat_method is not None:
            if cat_features is None:
                # find categorical columns from Object, String or Category dtypes automatically
                for column_ in pandas_df.columns:
                    if pandas_df[column_].dtype == object or pandas_df[column_].dtype == 'string' or pandas_df[column_].dtype.name == 'category':
                        cat_columns.append(column_)
            else:
                cat_columns = cat_features
                
        # Handle categorical data if required
        if len(cat_columns) > 0:
            # Set continuous/categorical data if categorical detected
            self._continuous = pandas_df.drop(columns=cat_columns)
            self._categorical = pandas_df[cat_columns].copy()
            
            # Perform one-hot-encoding
            if cat_method == "one-hot":
                for col_ in cat_columns:
                    self._categorical[col_] = self._categorical[col_].astype("category")
                self._categorical = pandas.get_dummies(data=self._categorical, dtype=numpy.int32, **kwargs)
            # Perform embedding
            else:
                self._categorical = self.embed_categorical(cat_df=self._categorical, random_state=random_state, **kwargs)
                
            # Something went wrong?
            if self._categorical.empty:
                raise AttributeError("Categorical data couldn't be processed")
        else:
            # Assume all data is continuous
            if not pandas_df.empty:
                self._continuous = pandas_df
        
        # Map labels
        if cast_labels:
            labels_ = self._labels.astype("category")
            # Get mapping
            self.labels_map = {key: value for key, value in enumerate(labels_.cat.categories)}
            self._labels = labels_.cat.codes
        
        # Train-Test splits
        if self._continuous is not None and self._categorical is not None:
            continuous_train, continuous_test, categorical_train, categorical_test, self.labels_train, self.labels_test = train_test_split(self._continuous, 
                                                                                                                                           self._categorical, 
                                                                                                                                           self._labels, 
                                                                                                                                           test_size=test_size, 
                                                                                                                                           random_state=random_state)
        elif self._categorical is None:
            continuous_train, continuous_test, self.labels_train, self.labels_test = train_test_split(self._continuous, self._labels, 
                                                                                                      test_size=test_size, random_state=random_state)
        elif self._continuous is None:
            categorical_train, categorical_test, self.labels_train, self.labels_test = train_test_split(self._categorical, self._labels, 
                                                                                                        test_size=test_size, random_state=random_state)

        # Normalize continuous features
        if normalize is not None and self._continuous is not None:
            continuous_train, continuous_test = self.normalize_continuous(train_set=continuous_train, test_set=continuous_test, method=normalize)
        
        # Merge continuous and categorical
        if self._categorical is not None and self._continuous is not None:
            self.features_train = pandas.concat(objs=[continuous_train, categorical_train], axis=1)
            self.features_test = pandas.concat(objs=[continuous_test, categorical_test], axis=1)
        elif self._continuous is not None:
            self.features_train = continuous_train
            self.features_test = continuous_test
        elif self._categorical is not None:
            self.features_train = categorical_train
            self.features_test = categorical_test
            
        # Balance train dataset
        if balance and self.features_train is not None and self.labels_train is not None:
            self.features_train, self.labels_train = self.balance_classes(train_features=self.features_train, train_labels=self.labels_train)
            
    def to_pytorch(self):
        """
        Convert the train and test features and labels to Pytorch Datasets with default dtypes.
        
        Returns: Tuple(Train Dataset, Test Dataset)
        """
        train = None
        test = None
        # Train set
        if self.labels_train is not None and self.features_train is not None:
            train = PytorchDataset(features=self.features_train, labels=self.labels_train)
        # Test set
        if self.labels_test is not None and self.features_test is not None:
            test = PytorchDataset(features=self.features_test, labels=self.labels_test)
        
        return train, test
            
    @staticmethod
    def embed_categorical(cat_df: pandas.DataFrame, random_state: Union[int, None]=None, **kwargs) -> pandas.DataFrame:
        """
        Takes a DataFrame object containing categorical data only.
        
        Calculates embedding dimensions for each categorical feature. Using `(Number_of_categories + 1) // 2` up to a maximum value of 50.
        
        Applies embedding using PyTorch and returns a Pandas Dataframe with embedded features.
        """
        df = cat_df.copy()
        embedded_tensors = list()
        columns = list()
        for col in df.columns:
            df[col] = df[col].astype("category")
            # Get number of categories
            size: int = df[col].cat.categories.size
            # Embedding dimension
            embedding_dim: int = min(50, (size+1)//2)
            # Create instance of Embedding tensor using half the value for embedding dimensions
            with torch.no_grad():
                if random_state:
                    torch.manual_seed(random_state)
                embedder = nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim, **kwargs)
                # Embed column of features and store tensor
                embedded_tensors.append(embedder(torch.LongTensor(df[col].cat.codes.copy().values)))
            # Preserve column names for embedded features
            for i in range(1, embedding_dim+1):
                columns.append(f"{col}_{i}")
            
        # Concatenate tensors
        with torch.no_grad():
            tensor = torch.cat(tensors=embedded_tensors, dim=1)
            # Convert to dataframe
        return pandas.DataFrame(data=tensor.numpy(), columns=columns)

    @staticmethod
    def normalize_continuous(train_set: Union[numpy.ndarray, pandas.DataFrame, pandas.Series], test_set: Union[numpy.ndarray, pandas.DataFrame, pandas.Series],
                             method: Literal["standard", "minmax"]="standard"):
        """
        Takes a train and a test dataset, then returns the standardized datasets as a tuple (train, test).
        
        `method`: Standardization by the mean and variance or MinMax Normalization.
        
        The transformer is fitted on the training set, so there is no data-leak of the test set.
        
        Output type is the same as Input type: nD-array, DataFrame or Series.
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Normalization method must be 'standard' or 'minmax'.")
        
        X_train = scaler.fit_transform(train_set)
        X_test = scaler.transform(test_set)
        
        if isinstance(train_set, pandas.DataFrame):
            train_indexes = train_set.index
            test_indexes = test_set.index
            cols = train_set.columns
            X_train = pandas.DataFrame(data=X_train, index=train_indexes, columns=cols)
            X_test = pandas.DataFrame(data=X_test, index=test_indexes, columns=cols)
        elif isinstance(train_set, pandas.Series):
            train_indexes = train_set.index
            test_indexes = test_set.index
            X_train = pandas.Series(data=X_train, index=train_indexes)
            X_test = pandas.Series(data=X_test, index=test_indexes)
        else:
            pass
        
        return X_train, X_test
    
    @staticmethod
    def balance_classes(train_features, train_labels, **kwargs):
        """
        Attempts to balance the minority class(es) using Imbalanced-Learn's `SMOTETomek` algorithm.
        """
        resampler = SMOTETomek(**kwargs)
        X, y = resampler.fit_resample(X=train_features, y=train_labels)
        
        return X, y


class PytorchDataset(Dataset):
    def __init__(self, features: Union[numpy.ndarray, pandas.Series, pandas.DataFrame], labels: Union[numpy.ndarray, pandas.Series, pandas.DataFrame], 
                 features_dtype: torch.dtype=torch.float32, labels_dtype: torch.dtype=torch.int64, balance: bool=False) -> None:
        """
        Make a PyTorch dataset of Features and Labels casted to Tensors.
        
        Defaults: `float32` for features and `int64` for labels.
        
        If `balance=True` attempts to balance the minority class(es) using Imbalanced-Learn's `SMOTETomek` algorithm.
        Note: Only Train-Data should be balanced.
        """
        # Validate features
        if not isinstance(features, (pandas.DataFrame, pandas.Series, numpy.ndarray)):
            raise TypeError("features must be a numpy.ndarray, pandas.Series or pandas.DataFrame")
        # Validate labels
        if not isinstance(labels, (pandas.DataFrame, pandas.Series, numpy.ndarray)):
            raise TypeError("labels must be a numpy.ndarray, pandas.Series or pandas.DataFrame")
        
        # Balance classes
        if balance:
            features, labels = self.balance_classes(train_features=features, train_labels=labels)
    
        # Cast features
        if isinstance(features, numpy.ndarray):
            self.features = torch.tensor(features, dtype=features_dtype)
        else:
            self.features = torch.tensor(features.values, dtype=features_dtype)
        
        # Cast labels 
        if isinstance(labels, numpy.ndarray):
            self.labels = torch.tensor(labels, dtype=labels_dtype)
        else:
            self.labels = torch.tensor(labels.values, dtype=labels_dtype)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    @staticmethod
    def balance_classes(train_features, train_labels, **kwargs):
        """
        Attempts to balance the minority class(es) using Imbalanced-Learn's `SMOTETomek` algorithm.
        """
        resampler = SMOTETomek(**kwargs)
        X, y = resampler.fit_resample(X=train_features, y=train_labels)
        
        return X, y


def make_vision_dataset(inputs: Union[list[Image.Image], numpy.ndarray, str], labels: Union[list[int], numpy.ndarray, None], resize: int=256, 
                        transform: Union[transforms.Compose, None]=None, test_set: bool=False):
    """
    Make a Torchvision Dataset of images to be used in a Convolutional Neural Network. 
    
    If no transform object is given, Images will undergo the following transformations by default: `RandomHorizontalFlip`, `RandomRotation`, 
    `Resize`, `CenterCrop`, `ToTensor`, `Normalize`. Except if 'test_set=True'.

    Args:
        `inputs`: List of PIL Image objects | Numpy array of image arrays | Path to root directory containing subdirectories that classify image files.
        
        `labels`: List of integer values | Numpy array of labels. Labels size must match `inputs` size. 
        If a path to a directory is given, then `labels` must be None.
        
        `transform`: Custom transformations to use. If None, use default transformations. 
        
        `test_set`: Flip, rotation and center-crop transformations will not be applied.

    Returns:
        `Dataset`: Either a `TensorDataset` or `ImageFolder` instance, depending on the method used. 
        Data dimensions: (samples, color channels, height, width).
    """
    # Validate inputs
    if not isinstance(inputs, (list, numpy.ndarray, str)):
        raise TypeError("Inputs must be one of the following:\n\ta) List of PIL Image objects.\n\tb) Numpy array of 2D or 3D arrays.\
                        \n\tc) Directory path to image files.")
    # Validate labels
    if not (isinstance(labels, (list, numpy.ndarray)) or labels is None):
        raise TypeError("Inputs must be one of the following:\n\ta) List of labels (integers).\n\tb) Numpy array of 2D or 3D arrays.\
                        \n\tc) None if inputs path is given.\nLabels size must match Inputs size.")
    # Validate resize shape
    if not isinstance(resize, int):
        raise TypeError("Resize must be an integer value for a square image of shape (W, H).")
    # Validate transform
    if isinstance(transform, transforms.Compose):
        pass
    elif transform is None:
        if test_set:
            transform = transforms.Compose([
                transforms.Resize(size=(resize,resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.Resize(size=(int(resize*1.2),int(resize*1.2))),
                transforms.CenterCrop(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        raise TypeError("Transform must be a `torchvision.transforms.Compose` object or None to use a default transform.")
    
    # Start-o
    dataset = None
    
    # CASE A: input is a path to image files, Labels is None
    if labels is None:
        if isinstance(inputs, str):
            dataset = ImageFolder(root=inputs, transform=transform)
        else:
            raise TypeError("Labels must be None if 'path' to inputs is provided. Labels will be inferred from subdirectory names in 'path'.")
    # CASE B: input is Numpy array or a list of PIL Images. Labels is Numpy array or List of integers    
    elif not isinstance(inputs, str):
        # Transform labels to tensor
        labels_ = torch.tensor(labels, dtype=torch.int64)
        
        # Transform each image to tensor
        transformed = list()
        for img_ in inputs:
            transformed.append(transform(img_))  
        # Stack image tensors
        features_ = torch.stack(transformed, dim=0)
        
        # Make a dataset with images and labels
        dataset = TensorDataset(features_, labels_)
    else:
        raise TypeError("Labels must be None if 'path' to inputs is provided. Labels will be inferred from subdirectory names in 'path'.")
    
    return dataset


class SequenceDataset():
    def __init__(self, data: Union[pandas.DataFrame, pandas.Series, numpy.ndarray], sequence_size: int, last_seq_test: bool=True, 
                 seq_labels: bool=True, normalize: Union[Literal["standard", "minmax"], None]="minmax"):
        """
        Make train/test datasets from a single timestamp sequence.
        
        Create an object containing 2 PyTorchDataset objects to be used in a Recurrent Neural Network: 
        
            1. Train Dataset
            2. Test Dataset
        
        To plot call the static method `plot()`.
        
        If normalization is used, an scaler object will be stored. 
        The scaler object can be used to invert normalization on a Tensor/Array using the method `self.denormalize()`.

        Args:
            * `data`: Pandas Dataframe with 2 columns [datetime, sequence] | 1-column Dataframe or Series sequence, where index is the datetime.
            * `sequence_size (int)`: Length of each subsequence that will be used for training.
            * `last_seq_test (bool)`: Last sequence will be used as test_set, if false a dummy test set will be returned. Default is True.
            * `seq_labels (bool)`: Labels will be returned as sequences, if false return single values for 1 future timestamp.
            * `normalize`: Whether to normalize ('minmax'), standardize ('standard') or ignore (None). Default is 'minmax'.
        """
        # Validate data
        if not isinstance(data, (pandas.Series, pandas.DataFrame, numpy.ndarray)):
            raise TypeError("Data must be pandas dataframe, pandas series or numpy array.")
        # Validate window size
        if not isinstance(sequence_size, int):
            raise TypeError("Sequence size must be an integer.")
        elif len(data) % sequence_size != 0:
            raise ValueError(f"data with length {len(data)} is not divisible in sequences of {sequence_size} values.")
        # Validate test sequence
        if not isinstance(last_seq_test, bool):
            raise TypeError("Last sequence treated as Test-set must be True or False.")
        # validate normalize
        if not (normalize in ["standard", "minmax"] or normalize is None):
            raise TypeError("normalize must be 'standard', 'minmax' or None.")
        
        # Handle data -> array
        self.time_axis = None
        if isinstance(data, pandas.DataFrame):
            if len(data.columns) == 2:
                self.sequence = data[data.columns[1]].values.astype("float")
                self.time_axis = data[data.columns[0]].values
            elif len(data.columns) == 1:
                self.sequence = data[data.columns[0]].values.astype("float")
                self.time_axis = data.index.values
            else:
                raise ValueError("Dataframe contains more than 2 columns.")
        elif isinstance(data, pandas.Series):
            self.sequence = data.values.astype("float")
            self.time_axis = data.index.values
        else:
            self.sequence = data.astype("float")
            
        # Save last sequence
        self._last_sequence = self.sequence[-sequence_size:]
        
        # Last sequence as test
        train_sequence = self.sequence
        test_sequence = None
        if last_seq_test:
            test_sequence = self.sequence[-(sequence_size*2):]
            train_sequence = self.sequence[:-sequence_size]
        
        # Normalize values
        norm_train_sequence = train_sequence
        norm_test_sequence = test_sequence
        if normalize is not None:
            # Define scaler
            if normalize == "standard":
                self.scaler = StandardScaler()
            elif normalize == "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1,1))
            # Scale and transform training set + reshape
            self.scaler.fit(train_sequence.reshape(-1,1))
            norm_train_sequence = self.scaler.transform(train_sequence.reshape(-1,1))
            norm_train_sequence = norm_train_sequence.reshape(-1)
            # Scale test if it exists + reshape
            if last_seq_test:
                norm_test_sequence = self.scaler.transform(test_sequence.reshape(-1,1))
                norm_test_sequence = norm_test_sequence.reshape(-1)
        
        # Divide train sequence into subsequences
        train_features_list = list()
        train_labels_list = list()
        train_size = len(norm_train_sequence)
        for i in range(train_size - sequence_size - 1):
            subsequence = norm_train_sequence[i:sequence_size + i]
            train_features_list.append(subsequence.reshape(1,-1))
            # Labels as sequence
            if seq_labels:
                label = norm_train_sequence[i + 1:sequence_size + i + 1]
                train_labels_list.append(label.reshape(1,-1))
            # Single value label
            else:
                label = norm_train_sequence[sequence_size + i + 1]
                train_labels_list.append(label)
            
        # Divide test sequence into subsequences
        if last_seq_test:
            test_features_list = list()
            test_labels_list = list()
            test_size = len(norm_test_sequence)
            for i in range(test_size - sequence_size - 1):
                subsequence = norm_test_sequence[i:sequence_size + i]
                test_features_list.append(subsequence.reshape(1,-1))
                # Labels as sequence
                if seq_labels:
                    label = norm_test_sequence[i + 1:sequence_size + i + 1]
                    test_labels_list.append(label.reshape(1,-1))
                # Single value label
                else:
                    label = norm_test_sequence[sequence_size + i + 1]
                    test_labels_list.append(label)
            
        # Create training arrays then cast to pytorch dataset
        train_features = numpy.concatenate(train_features_list, axis=0)
        # Check if labels are a sequence
        if seq_labels:
            train_labels = numpy.concatenate(train_labels_list, axis=0)
        else:
            train_labels = numpy.array(train_labels_list).reshape(-1,1)
        self.train_dataset = PytorchDataset(features=train_features, labels=train_labels, labels_dtype=torch.float32)
        
        # Create test arrays then cast to pytorch dataset
        if last_seq_test:
            test_features = numpy.concatenate(test_features_list, axis=0)
            # Check if labels are a sequence
            if seq_labels:
                test_labels = numpy.concatenate(test_labels_list, axis=0)
            else:
                test_labels = numpy.array(test_labels_list).reshape(-1,1)
            self.test_dataset = PytorchDataset(features=test_features, labels=test_labels, labels_dtype=torch.float32)
        else:
            self.test_dataset = PytorchDataset(features=numpy.ones(shape=(10, sequence_size)), labels=numpy.ones(shape=(10,1)), labels_dtype=torch.float32)
        
        # Attempt to plot the sequence
        if self.time_axis is not None:
            try:
                self.plot(self.time_axis, self.sequence)
            except:
                print("Plot failed, try it manually to find the problem.")

    @staticmethod   
    def plot(x_axis, y_axis, x_pred=None, y_pred=None):
        """
        Plot Time-values (X) Vs Data-values (Y).
        """
        plt.figure(figsize=(12,5))
        plt.title('Sequence')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(x_axis, y_axis)
        if x_pred is not None and y_pred is not None:
            plt.plot(x_pred, y_pred)
        plt.show()
        
    def denormalize(self, input: Union[torch.Tensor, numpy.ndarray]) -> numpy.ndarray:
        """
        Applies the inverse transformation of the object's stored scaler to a tensor or array.

        Args:
            `input`: Tensor/Array predicted using the current sequence.

        Returns: numpy.ndarray with default index.
        """
        if isinstance(input, torch.Tensor):
            with torch.no_grad():
                array = input.numpy().reshape(-1,1)
        elif isinstance(input, numpy.ndarray):
            array = input.reshape(-1,1)
        else:
            raise TypeError("Input must be a Pytorch tensor or Numpy array.")
        return self.scaler.inverse_transform(array)
    
    def get_last_sequence(self, normalize: bool=True, to_tensor: bool=True):
        """
        Returns the last subsequence of the sequence.

        Args:
            `normalize`: Normalize using the object's stored scaler. Defaults to True.
            
            `to_tensor`: Cast to Pytorch tensor. Defaults to True.

        Returns: numpy.ndarray or torch.Tensor
        """
        last_seq = self._last_sequence.reshape(-1,1)
        if normalize:
            last_seq = self.scaler.transform(last_seq)
        if to_tensor:
            last_seq = torch.Tensor(last_seq)
        return last_seq
        
    def __len__(self):
        return f"Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}"


def info():
    _script_info(__all__)
