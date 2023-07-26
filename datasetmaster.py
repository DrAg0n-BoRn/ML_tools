import torch 
from torch.utils.data import Dataset, TensorDataset
from torch import nn
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Literal, Union
from imblearn.combine import SMOTETomek
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms


class DatasetMaker():
    def __init__(self, *, pandas_df: pandas.DataFrame, label_col: str, cat_features: Union[list[str], None]=None, cat_method: Union[Literal["one-hot", "embed"], None]="one-hot", 
                 test_size: float=0.2, random_state: Union[int, None]=None, normalize: bool=True, cast_labels: bool=True, balance: bool=False, **kwargs):
        """
        Create Train-Test datasets from a Pandas DataFrame. Four datasets will be created: 
        
            1. Features Train
            2. Features Test
            3. Labels Train
            4. Labels Test
        
        `label_col` Specify the name of the label column. If label encoding is required (str -> int) set `cast_labels=True` (default). 
        A dictionary will also be created with the label mapping; original category names will be the dictionary keys.
        
        `cat_features` List of column names to perform embedding or one-hot-encoding of categorical features. Any categorical column not in the list will not be returned. 
        If `None` (default), columns containing categorical data will be inferred from dtypes: object, string and category, if any. 
        
        `cat_method` can be set to: 
        
            * `'one-hot'` (default) to perform One-Hot-Encoding using Pandas "get_dummies".
            * `'embed'` to perform Embedding using PyTorch "nn.Embedding".
            * `None` all data will be considered to be continuous.
        
        If `normalize=True` (default) continuous features will be standardized using Scikit-Learn's StandardScaler.
        
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
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be either True or False.")
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
        
        # find categorical columns from Object, String or Category dtypes
        cat_columns = list()
        if cat_method is not None:
            for column_ in pandas_df.columns:
                if pandas_df[column_].dtype == object or pandas_df[column_].dtype == 'string' or pandas_df[column_].dtype.name == 'category':
                    cat_columns.append(column_)
                
        # Set continuous data
        if not pandas_df.empty:
            self._continuous = pandas_df
        
        # Handle categorical data if required
        if len(cat_columns) > 0:
            # Modify continuous data if categorical detected
            self._continuous = self._continuous.drop(columns=cat_columns)
            # Use columns inferred to be categorical
            if cat_features is None:
                to_cast = cat_columns
            # Or get categorical columns passed as argument
            else:
                to_cast = cat_features
        
            # Perform one-hot-encoding
            if cat_method == "one-hot":
                self._categorical = pandas.get_dummies(data=pandas_df[to_cast], dtype=numpy.int32, **kwargs)
            # Perform embedding
            else:
                self._categorical = self.embed_categorical(cat_df=pandas_df[to_cast], random_state=random_state, **kwargs)
                
            # Something went wrong?
            if self._categorical.empty:
                raise AttributeError("Categorical data couldn't be processed")
        
        # Map labels
        if cast_labels:
            labels_ = self._labels.astype("category")
            # Get mapping
            self.labels_map = {key: value for value, key in enumerate(labels_.cat.categories)}
            self._labels = labels_.cat.codes
        
        # Train-Test splits
        if self._continuous is not None and self._categorical is not None:
            continuous_train, continuous_test, categorical_train, categorical_test, self.labels_train, self.labels_test = train_test_split(self._continuous, self._categorical, self._labels, test_size=test_size, random_state=random_state)
        elif self._categorical is None:
            continuous_train, continuous_test, self.labels_train, self.labels_test = train_test_split(self._continuous, self._labels, test_size=test_size, random_state=random_state)
        elif self._continuous is None:
            categorical_train, categorical_test, self.labels_train, self.labels_test = train_test_split(self._categorical, self._labels, test_size=test_size, random_state=random_state)

        # Normalize continuous features
        if normalize and self._continuous is not None:
            continuous_train, continuous_test = self.normalize_continuous(train_set=continuous_train, test_set=continuous_test)
        
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
        if balance and self.features_train and self.labels_train:
            self.features_train, self.labels_train = self.balance_classes(train_features=self.features_train, train_labels=self.labels_train)
            
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
    def normalize_continuous(train_set: Union[numpy.ndarray, pandas.DataFrame, pandas.Series], test_set: Union[numpy.ndarray, pandas.DataFrame, pandas.Series]):
        """
        Takes a train and a test dataset, then returns the standardized datasets as a tuple (train, test).
        
        The transformer is fitted on the training set, so there is no data-leak of the test set.
        
        Output type is the same as Input type: nD-array, DataFrame or Series.
        """
        scaler = StandardScaler()
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


def make_vision_dataset(inputs: Union[list[Image.Image], numpy.ndarray, str], labels: Union[list[int], numpy.ndarray, None], resize: int=300, transform: Union[transforms.Compose, None]=None):
    """
    Make a Torchvision Dataset of images to be used in a Convolutional Neural Network. 
    
    If no transform object is given, Images will undergo the following transformations by default: `RandomHorizontalFlip`, `RandomRotation`, `Resize(300,300)`, `CenterCrop`, `ToTensor`, `Normalize`.

    Args:
        `inputs`: List of PIL Image objects | Numpy array of image arrays | Path to root directory containing subdirectories that classify image files.
        
        `labels`: List of integer values | Numpy array of labels. Labels size must match `inputs` size. If a path to a directory is given, then `labels` must be None.
        
        `transform`: Custom transformations to use. If None, use default transformations.

    Returns:
        `Dataset`: Either a `TensorDataset` or `ImageFolder` instance, depending on the method used. 
        Data dimensions: (samples, [color channels], height, width).
    """
    
    # Validate inputs
    if not isinstance(inputs, (list, numpy.ndarray, str)):
        raise TypeError("Inputs must be one of the following:\n\ta) List of PIL Image objects.\n\tb) Numpy array of 2D or 3D arrays.\n\tc) Directory path to image files.")
    # Validate labels
    if not (isinstance(labels, (list, numpy.ndarray)) or labels is None):
        raise TypeError("Inputs must be one of the following:\n\ta) List of labels (integers).\n\tb) Numpy array of 2D or 3D arrays.\n\tc) None if inputs path is given.\nLabels size must match Inputs size.")
    # Validate resize shape
    if not isinstance(resize, int):
        raise TypeError("Resize must be an integer value for a square image of shape (W, H).")
    # Validate transform
    if isinstance(transform, transforms.Compose):
        pass
    elif transform is None:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.Resize(size=(resize,resize)),
            transforms.CenterCrop(size=int(resize * 0.8)),
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
