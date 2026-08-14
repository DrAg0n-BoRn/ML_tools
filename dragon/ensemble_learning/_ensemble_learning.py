import pandas as pd
import numpy as np

from pathlib import Path
from typing import Literal, Union, Optional

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.base import clone

from ..utilities import yield_dataframes_from_dir, train_dataset_yielder
from ..serde import serialize_object_filename
from ..ensemble_evaluation import (evaluate_model_classification,
                                  plot_roc_curve,
                                  plot_precision_recall_curve,
                                  plot_calibration_curve,
                                  evaluate_model_regression,
                                  get_shap_values,
                                  plot_learning_curves)

from ..path_manager import sanitize_filename, make_fullpath
from ..keys._keys import EnsembleKeys
from .._core import get_logger

import warnings # Ignore warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


_LOGGER = get_logger("Ensemble Learning")


__all__ = [
    "RegressionTreeModels",
    "ClassificationTreeModels",
    "dataset_pipeline",
    "train_test_pipeline",
    "run_ensemble_pipeline",
]

## Type aliases
HandleImbalanceStrategy = Literal[
    "ADASYN", "SMOTE", "RAND_OVERSAMPLE", "RAND_UNDERSAMPLE", "by_model", None
]

###### 1. Initialize Models ######
class RegressionTreeModels:
    """
    A factory class for creating and configuring multiple gradient boosting regression models
    with unified hyperparameters. This includes XGBoost and LightGBM.
    
    Use the `__call__`, `()` method.

    Parameters
    ----------
    random_state : int
        Seed used by the random number generator.

    learning_rate : float [0.001 - 0.300]
        Boosting learning rate (shrinkage).
    
    L1_regularization : float [0.0 - 10.0]
        L1 regularization term (alpha). Might drive to sparsity.

    L2_regularization : float [0.0 - 10.0]
        L2 regularization term (lambda).

    n_estimators : int [100 - 3000]
        Number of boosting iterations for XGBoost and LightGBM.

    max_depth : int [3 - 15]
        Maximum depth of individual trees. Controls model complexity; high values may overfit.

    subsample : float [0.5 - 1.0]
        Fraction of rows per tree; used to prevent overfitting.

    colsample_bytree : float [0.3 - 1.0]
        Fraction of features per tree; useful for regularization (used by XGBoost and LightGBM).

    min_child_weight : float [0.1 - 10.0]
        Minimum sum of instance weight (hessian) needed in a child; larger values make the algorithm more conservative (used in XGBoost).

    gamma : float [0.0 - 5.0]
        Minimum loss reduction required to make a further partition on a leaf node; higher = more regularization (used in XGBoost).

    num_leaves : int [20 - 200]
        Maximum number of leaves in one tree; should be less than 2^(max_depth); larger = more complex (used in LightGBM).

    min_data_in_leaf : int [10 - 100]
        Minimum number of data points in a leaf; increasing may prevent overfitting (used in LightGBM).
    """
    def __init__(self, 
             random_state: int = 101,
             learning_rate: float = 0.005,
             L1_regularization: float = 1.0,
             L2_regularization: float = 1.0,
             n_estimators: int = 1000,
             max_depth: int = 8,
             subsample: float = 0.8,
             colsample_bytree: float = 0.8,
             min_child_weight: float = 3.0,
             gamma: float = 1.0,
             num_leaves: int = 31,
             min_data_in_leaf: int = 40):
        
        # General config
        self.random_state = random_state
        self.lr = learning_rate
        self.L1 = L1_regularization
        self.L2 = L2_regularization

        # Shared tree structure
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        # XGBoost specific
        self.min_child_weight = min_child_weight
        self.gamma = gamma

        # LightGBM specific
        num_leaves = min(num_leaves, 2 ** (max_depth - 1))
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf

    def __call__(self) -> dict[str, object]:
        """
        Returns a dictionary with new instances of:
            - "XGBoost": XGBRegressor
            - "LightGBM": LGBMRegressor
        """
        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            reg_alpha=self.L1,
            reg_lambda=self.L2,
            eval_metric='rmse',
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            tree_method='hist',
            grow_policy='lossguide'
        )

        # LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=-1,
            reg_alpha=self.L1,
            reg_lambda=self.L2,
            boosting_type='gbdt',
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf
        )

        return {
            "XGBoost": xgb_model,
            "LightGBM": lgb_model
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}(n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.lr}, L1={self.L1}, L2={self.L2}"


class ClassificationTreeModels:
    """
    A factory class for creating and configuring multiple gradient boosting classification models
    with unified hyperparameters. This includes XGBoost and LightGBM.
    
    Use the `__call__`, `()` method.

    Parameters
    ----------
    random_state : int
        Seed used by the random number generator to ensure reproducibility.

    learning_rate : float [0.001 - 0.300]
        Boosting learning rate (shrinkage factor). 

    L1_regularization : float [0.0 - 10.0]
        L1 regularization term (alpha), might drive to sparsity.

    L2_regularization : float [0.0 - 10.0]
        L2 regularization term (lambda).

    n_estimators : int [100 - 3000]
        Number of boosting rounds for XGBoost and LightGBM.

    max_depth : int [3 - 15]
        Maximum depth of individual trees in the ensemble. Controls model complexity; high values may overfit.

    subsample : float [0.5 - 1.0]
        Fraction of samples to use when fitting base learners; used to prevent overfitting.

    colsample_bytree : float [0.3 - 1.0]
        Fraction of features per tree; useful for regularization (used by XGBoost and LightGBM).

    min_child_weight : float [0.1 - 10.0]
        Minimum sum of instance weight (Hessian) in a child node; larger values make the algorithm more conservative (used in XGBoost).

    gamma : float [0.0 - 5.0]
        Minimum loss reduction required to make a further partition; higher = more regularization (used in XGBoost).

    num_leaves : int [20 - 200]
        Maximum number of leaves in one tree. Should be less than 2^(max_depth); larger = more complex (used in LightGBM).

    min_data_in_leaf : int [10 -100]
        Minimum number of samples required in a leaf; increasing may prevent overfitting (used in LightGBM).

    Attributes
    ----------
    use_model_balance : bool
        Indicates whether to apply class balancing strategies internally. Can be overridden at runtime via the `__call__` method.
    """
    def __init__(self,
             random_state: int = 101,
             learning_rate: float = 0.005,
             L1_regularization: float = 1.0,
             L2_regularization: float = 1.0,
             n_estimators: int = 1000,
             max_depth: int = 8,
             subsample: float = 0.8,
             colsample_bytree: float = 0.8,
             min_child_weight: float = 3.0,
             gamma: float = 1.0,
             num_leaves: int = 31,
             min_data_in_leaf: int = 40):
        
        # General config
        self.random_state = random_state
        self.lr = learning_rate
        self.L1 = L1_regularization
        self.L2 = L2_regularization
        
        # To be set by the pipeline
        self.use_model_balance: bool = True

        # Shared tree structure
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        # XGBoost specific
        self.min_child_weight = min_child_weight
        self.gamma = gamma

        # LightGBM specific
        num_leaves = min(num_leaves, 2 ** (max_depth - 1))
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf

    def __call__(self, use_model_balance: Optional[bool]=None) -> dict[str, object]:
        """
        Returns a dictionary with new instances of:
            - "XGBoost": XGBClassifier
            - "LightGBM": LGBMClassifier
        """
        if use_model_balance is not None:
            self.use_model_balance = use_model_balance
        
        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            reg_alpha=self.L1,
            reg_lambda=self.L2,
            eval_metric='aucpr',
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            tree_method='hist',
            grow_policy='lossguide',
            scale_pos_weight=8.0 if self.use_model_balance else 1.0
        )

        # LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=-1,
            reg_alpha=self.L1,
            reg_lambda=self.L2,
            boosting_type='gbdt' if self.use_model_balance else 'goss',
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            class_weight='balanced' if self.use_model_balance else None
        )

        return {
            "XGBoost": xgb_model,
            "LightGBM": lgb_model
        }
        
    def __str__(self):
        return f"{self.__class__.__name__}(n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.lr}, L1={self.L1}, L2={self.L2}"


###### 2. Process Dataset ######
# function to split data into train and test
def _split_data(features, target, test_size, random_state, task):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state, 
                                                        stratify=target if task=="classification" else None)   
    return X_train, X_test, y_train, y_test

# Over-sample minority class (Positive cases) and return several single target datasets (Classification)
def _resample(X_train: np.ndarray, y_train: pd.Series, 
              strategy: HandleImbalanceStrategy, random_state):
    ''' 
    Oversample minority class or undersample majority class.
    
    Returns a Tuple `(Features: nD-Array, Target: 1D-array)`
    '''
    if strategy == 'SMOTE':
        resample_algorithm = SMOTE(random_state=random_state, k_neighbors=3)
    elif strategy == 'RAND_OVERSAMPLE':
        resample_algorithm = RandomOverSampler(random_state=random_state)
    elif strategy == 'RAND_UNDERSAMPLE':
        resample_algorithm = RandomUnderSampler(random_state=random_state)
    elif strategy == 'ADASYN':
        resample_algorithm = ADASYN(random_state=random_state, n_neighbors=3)
    else:
        _LOGGER.error(f"Invalid resampling strategy: {strategy}")
        raise ValueError()
    
    X_res, y_res, *_ = resample_algorithm.fit_resample(X_train, y_train)
    return X_res, y_res

# DATASET PIPELINE
def dataset_pipeline(df_features: pd.DataFrame, df_target: pd.Series, task: Literal["classification", "regression"],
                     resample_strategy: HandleImbalanceStrategy,
                     test_size: float=0.2, debug: bool=False, random_state: int=101):
    ''' 
    1. Make Train/Test splits
    2. Oversample imbalanced classes (classification)
    
    Return a processed Tuple: (X_train, y_train, X_test, y_test)
    
    `(nD-array, 1D-array, nD-array, Series)`
    '''
    #DEBUG
    if debug:
        _LOGGER.info(f"Split Dataframes Shapes - Features DF: {df_features.shape}, Target DF: {df_target.shape}")
        unique_values = df_target.unique()  # Get unique values for the target column
        _LOGGER.info(f"\tUnique values for '{df_target.name}': {unique_values}")
    
    #Train test split
    X_train, X_test, y_train, y_test = _split_data(features=df_features, target=df_target, test_size=test_size, random_state=random_state, task=task)
    
    #DEBUG
    if debug:
        _LOGGER.info(f"Shapes after train test split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
 
    # Resample
    if resample_strategy is None or resample_strategy == "by_model" or task == "regression":
        X_train_oversampled, y_train_oversampled = X_train, y_train
    else:
        X_train_oversampled, y_train_oversampled = _resample(X_train=X_train, y_train=y_train, strategy=resample_strategy, random_state=random_state)
    
    #DEBUG
    if debug:
        _LOGGER.info(f"Shapes after resampling - X_train: {X_train_oversampled.shape}, y_train: {y_train_oversampled.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train_oversampled, y_train_oversampled, X_test, y_test

###### 3. Train and Evaluation ######
# Trainer function
def _train_model(model, train_features, train_target):
    model.fit(train_features, train_target)
    return model

# handle local directories
def _local_directories(model_name: str, dataset_id: str, save_dir: Union[str,Path]):
    save_path = make_fullpath(save_dir, make=True)
    
    dataset_dir = save_path / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = dataset_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
        
    return model_dir

# save model
def _save_model(trained_model, model_name: str, target_name:str, feature_names: list[str], save_directory: Union[str,Path]):
    #Sanitize filenames to save
    sanitized_target_name = sanitize_filename(target_name)
    filename = f"{model_name}_{sanitized_target_name}"
    to_save = {EnsembleKeys.MODEL: trained_model, 
               EnsembleKeys.FEATURES: feature_names,
               EnsembleKeys.TARGET: target_name}

    serialize_object_filename(obj=to_save, save_dir=save_directory, filename=filename, verbose=False, raise_on_error=True)


# TRAIN EVALUATE PIPELINE
def train_test_pipeline(model, model_name: str, dataset_id: str, task: Literal["classification", "regression"],
             train_features: np.ndarray, train_target: np.ndarray,
             test_features: np.ndarray, test_target: np.ndarray,
             feature_names: list[str], target_name: str,
             save_dir: Union[str,Path],
             debug: bool=False, save_model: bool=False,
             generate_learning_curves: bool = False):
    ''' 
    1. Train model.
    2. Evaluate model.
    3. SHAP values.
    4. [Optional] Plot learning curves.
    
    Returns: Tuple(Trained model, Test-set Predictions)
    '''
    print(f"\tTraining model: {model_name} for Target: {target_name}...")
    trained_model = _train_model(model=model, train_features=train_features, train_target=train_target)
    if debug:
        _LOGGER.info(f"Trained model object: {type(trained_model)}")
    local_save_directory = _local_directories(model_name=model_name, dataset_id=dataset_id, save_dir=save_dir)
    
    if save_model:
        _save_model(trained_model=trained_model, model_name=model_name, 
                    target_name=target_name, feature_names=feature_names, 
                    save_directory=local_save_directory)
    
    # EVALUATION
    if task == "classification":
        y_pred = evaluate_model_classification(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_name=target_name)
        plot_roc_curve(true_labels=test_target,
                       probabilities_or_model=trained_model, model_name=model_name, 
                       target_name=target_name, save_directory=local_save_directory, 
                       input_features=test_features)
        plot_precision_recall_curve(true_labels=test_target,
                                    probabilities_or_model=trained_model, model_name=model_name,
                                    target_name=target_name, save_directory=local_save_directory,
                                    input_features=test_features)
        plot_calibration_curve(model=trained_model, model_name=model_name,
                               save_dir=local_save_directory,
                               x_test=test_features, y_test=test_target,
                               target_name=target_name)
    elif task == "regression":
        y_pred = evaluate_model_regression(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_name=target_name)
    else:
        _LOGGER.error(f"Unrecognized task '{task}' for model training,")
        raise ValueError()
    if debug:
        _LOGGER.info(f"Predicted vector: {type(y_pred)} with shape: {y_pred.shape}")
    
    get_shap_values(model=trained_model, model_name=model_name, save_dir=local_save_directory,
                    features_to_explain=train_features, feature_names=feature_names, target_name=target_name, task=task)
    
    if generate_learning_curves:
        # Note: We use a *clone* of the initial model object to ensure we don't use the already trained one.
        # The learning_curve function handles the fitting internally.
        initial_model_instance = clone(model)

        plot_learning_curves(estimator=initial_model_instance, X=train_features, y=train_target, 
                            task=task, model_name=model_name, target_name=target_name,
                            save_directory=local_save_directory)
    
    return trained_model, y_pred

###### 4. Execution ######
def run_ensemble_pipeline(datasets_dir: Union[str,Path], save_dir: Union[str,Path], target_columns: list[str], model_object: Union[RegressionTreeModels, ClassificationTreeModels],
         handle_classification_imbalance: HandleImbalanceStrategy=None, save_model: bool=True,
         test_size: float=0.2, debug:bool=False, generate_learning_curves: bool = False):
    #Check models
    if isinstance(model_object, RegressionTreeModels):
        task = "regression"
    elif isinstance(model_object, ClassificationTreeModels):
        task = "classification"
        if handle_classification_imbalance is None:
            _LOGGER.warning("No method to handle classification class imbalance has been selected. Datasets are assumed to be balanced.")
        elif handle_classification_imbalance == "by_model":
            model_object.use_model_balance = True
        else:
            model_object.use_model_balance = False
    else:
        _LOGGER.error(f"Unrecognized model {type(model_object)}")
        raise TypeError()
    
    #Check paths
    datasets_path = make_fullpath(datasets_dir)
    save_path = make_fullpath(save_dir, make=True)
    
    _LOGGER.info("🏁 Training starting...")
    #Yield imputed dataset
    for dataframe, dataframe_name in yield_dataframes_from_dir(datasets_path):
        #Yield features dataframe and target dataframe
        for df_features, df_target, feature_names, target_name in train_dataset_yielder(df=dataframe, target_cols=target_columns):
            #Dataset pipeline
            X_train, y_train, X_test, y_test = dataset_pipeline(df_features=df_features, df_target=df_target, task=task,
                                                                resample_strategy=handle_classification_imbalance,
                                                                test_size=test_size, debug=debug, random_state=model_object.random_state)
            #Get models
            models_dict = model_object()
            #Train models
            for model_name, model in models_dict.items():
                train_test_pipeline(model=model, model_name=model_name, dataset_id=dataframe_name, task=task,
                                    train_features=X_train, train_target=y_train, # type: ignore
                                    test_features=X_test, test_target=y_test,
                                    feature_names=feature_names,target_name=target_name,
                                    debug=debug, save_dir=save_path, save_model=save_model,
                                    generate_learning_curves=generate_learning_curves)

    _LOGGER.info("Training and evaluation complete.")

