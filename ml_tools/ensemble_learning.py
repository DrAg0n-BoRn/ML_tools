import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import rcdefaults

from pathlib import Path
from typing import Literal, Union, Optional, Iterator, Tuple, Dict, Any, List

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, roc_auc_score
import shap

from .utilities import yield_dataframes_from_dir, sanitize_filename, _script_info, serialize_object, make_fullpath, list_files_by_extension, deserialize_object
from .keys import ModelSaveKeys
from .logger import _LOGGER

import warnings # Ignore warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


__all__ = [
    "dataset_yielder",
    "RegressionTreeModels",
    "ClassificationTreeModels",
    "dataset_pipeline",
    "evaluate_model_classification",
    "plot_roc_curve",
    "evaluate_model_regression",
    "get_shap_values",
    "train_test_pipeline",
    "run_ensemble_pipeline",
    "InferenceHandler",
    "model_report"
]

## Type aliases
HandleImbalanceStrategy = Literal[
    "ADASYN", "SMOTE", "RAND_OVERSAMPLE", "RAND_UNDERSAMPLE", "by_model", None
]

TaskType = Literal[
    "classification", "regression"
]

###### 1. Dataset Loader ######
def dataset_yielder(
    df: pd.DataFrame,
    target_cols: list[str]
) -> Iterator[Tuple[pd.DataFrame, pd.Series, list[str], str]]:
    """ 
    Yields one tuple at a time:
        (features_dataframe, target_series, feature_names, target_name)

    Skips any target columns not found in the DataFrame.
    """
    # Determine which target columns actually exist in the DataFrame
    valid_targets = [col for col in target_cols if col in df.columns]

    # Features = all columns excluding valid target columns
    df_features = df.drop(columns=valid_targets)
    feature_names = df_features.columns.to_list()

    for target_col in valid_targets:
        df_target = df[target_col]
        yield (df_features, df_target, feature_names, target_col)


###### 2. Initialize Models ######
class RegressionTreeModels:
    """
    A factory class for creating and configuring multiple gradient boosting regression models
    with unified hyperparameters. This includes XGBoost, LightGBM, and HistGradientBoostingRegressor.
    
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

    min_samples_leaf : int [10 - 100]
        Minimum samples per leaf; higher = less overfitting (used in HistGB).

    max_iter : int [100 - 2000]
        Maximum number of iterations (used in HistGB).

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
                 min_samples_leaf: int = 50,
                 max_iter: int = 1000,
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

        # HistGB specific
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf

    def __call__(self) -> dict[str, object]:
        """
        Returns a dictionary with new instances of:
            - "XGBoost": XGBRegressor
            - "LightGBM": LGBMRegressor
            - "HistGB": HistGradientBoostingRegressor
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

        # HistGradientBoosting Regressor
        hist_model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            l2_regularization=self.L2,
            scoring='neg_mean_squared_error',
            early_stopping=True,
            validation_fraction=0.1
        )

        return {
            "XGBoost": xgb_model,
            "LightGBM": lgb_model,
            "HistGB": hist_model
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}(n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.lr}, L1={self.L1}, L2={self.L2}"


class ClassificationTreeModels:
    """
    A factory class for creating and configuring multiple gradient boosting classification models
    with unified hyperparameters. This includes: XGBoost, LightGBM, and HistGradientBoostingClassifier.
    
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

    min_samples_leaf : int [10 - 100]
        Minimum number of samples required to be at a leaf node; higher = less overfitting (used in HistGB).

    max_iter : int [100 - 2000]
        Maximum number of boosting iteration (used in HistGB).

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
                 min_samples_leaf: int = 50,
                 max_iter: int = 1000,
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

        # HistGB specific
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf

    def __call__(self, use_model_balance: Optional[bool]=None) -> dict[str, object]:
        """
        Returns a dictionary with new instances of:
            - "XGBoost": XGBClassifier
            - "LightGBM": LGBMClassifier
            - "HistGB": HistGradientBoostingClassifier
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

        # HistGradientBoosting Classifier
        hist_model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            l2_regularization=self.L2,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight='balanced' if self.use_model_balance else None,
            scoring='balanced_accuracy' if self.use_model_balance else 'loss'
        )

        return {
            "XGBoost": xgb_model,
            "LightGBM": lgb_model,
            "HistGB": hist_model
        }
        
    def __str__(self):
        return f"{self.__class__.__name__}(n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.lr}, L1={self.L1}, L2={self.L2}"


###### 3. Process Dataset ######
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
        raise ValueError(f"Invalid resampling strategy: {strategy}")
    
    X_res, y_res, *_ = resample_algorithm.fit_resample(X_train, y_train)
    return X_res, y_res

# DATASET PIPELINE
def dataset_pipeline(df_features: pd.DataFrame, df_target: pd.Series, task: TaskType,
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

###### 4. Train and Evaluation ######
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
    to_save = {ModelSaveKeys.MODEL: trained_model, 
               ModelSaveKeys.FEATURES: feature_names,
               ModelSaveKeys.TARGET: target_name}

    serialize_object(obj=to_save, save_dir=save_directory, filename=filename, verbose=False, raise_on_error=True)

# function to evaluate the model and save metrics (Classification)
def evaluate_model_classification(
    model,
    model_name: str,
    save_dir: Union[str,Path],
    x_test_scaled: np.ndarray,
    single_y_test: np.ndarray,
    target_name: str,
    figsize: tuple = (10, 8),
    base_fontsize: int = 24,
    cmap: Colormap = plt.cm.Blues # type: ignore
) -> np.ndarray:
    """
    Evaluates a classification model, saves the classification report and confusion matrix plot.

    Parameters:
        model: Trained classifier with .predict() method
        model_name: Identifier for the model
        save_dir: Directory where results are saved
        x_test_scaled: Feature matrix for test set
        single_y_test: True targets
        target_name: Target name
        figsize: Size of the confusion matrix figure (width, height)
        fontsize: Font size used for title, axis labels and ticks
        cmap: Color map for the confusion matrix. Examples include:
            - plt.cm.Blues (default)
            - plt.cm.Greens
            - plt.cm.Oranges
            - plt.cm.Purples
            - plt.cm.Reds
            - plt.cm.cividis
            - plt.cm.inferno

    Returns:
        y_pred: Predicted class labels
    """
    save_path = make_fullpath(save_dir, make=True)

    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(single_y_test, y_pred)

    report = classification_report(
        single_y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=False
    )

    # Save text report
    sanitized_target_name = sanitize_filename(target_name)
    report_path = save_path / f"Classification_Report_{sanitized_target_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"{model_name} - {target_name}\t\tAccuracy: {accuracy:.2f}\n")
        f.write("Classification Report:\n")
        f.write(report) # type: ignore

    # Create confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=single_y_test,
        y_pred=y_pred,
        display_labels=["Negative", "Positive"],
        cmap=cmap,
        normalize="true",
        ax=ax
    )

    ax.set_title(f"{model_name} - {target_name}", fontsize=base_fontsize)
    ax.tick_params(axis='both', labelsize=base_fontsize)
    ax.set_xlabel("Predicted label", fontsize=base_fontsize)
    ax.set_ylabel("True label", fontsize=base_fontsize)
    
    # Turn off gridlines
    ax.grid(False)
    
    # Manually update font size of cell texts
    for text in ax.texts:
        text.set_fontsize(base_fontsize+4)

    fig.tight_layout()
    fig_path = save_path / f"Confusion_Matrix_{sanitized_target_name}.svg"
    fig.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return y_pred

#Function to save ROC and ROC AUC (Classification)
def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities_or_model: Union[np.ndarray, xgb.XGBClassifier, lgb.LGBMClassifier, object],
    model_name: str,
    target_name: str,
    save_directory: Union[str,Path],
    color: str = "darkorange",
    figure_size: tuple = (10, 10),
    linewidth: int = 2,
    base_fontsize: int = 24,
    input_features: Optional[np.ndarray] = None,
) -> plt.Figure: # type: ignore
    """
    Plots the ROC curve and computes AUC for binary classification. Positive class is assumed to be in the second column of the probabilities array.
    
    Parameters:
        true_labels: np.ndarray of shape (n_samples,), ground truth binary labels (0 or 1).
        probabilities_or_model: either predicted probabilities (ndarray), or a trained model with attribute `.predict_proba()`.
        target_name: str, Target name.
        save_directory: str or Path, path to directory where figure is saved.
        color: color of the ROC curve. Accepts any valid Matplotlib color specification. Examples:
            - Named colors: "darkorange", "blue", "red", "green", "black"
            - Hex codes: "#1f77b4", "#ff7f0e"
            - RGB tuples: (0.2, 0.4, 0.6)
            - Colormap value: plt.cm.viridis(0.6)
        figure_size: Tuple for figure size (width, height).
        linewidth: int, width of the plotted ROC line.
        title_fontsize: int, font size of the title.
        label_fontsize: int, font size for axes labels.
        input_features: np.ndarray of shape (n_samples, n_features), required if a model is passed.

    Returns:
        fig: matplotlib Figure object
    """

    # Determine predicted probabilities
    if isinstance(probabilities_or_model, np.ndarray):
        # Input is already probabilities
        if probabilities_or_model.ndim == 2:
            y_score = probabilities_or_model[:, 1]
        else:
            y_score = probabilities_or_model
            
    elif hasattr(probabilities_or_model, "predict_proba"):
        if input_features is None:
            raise ValueError("input_features must be provided when using a classifier.")

        try:
            classes = probabilities_or_model.classes_ # type: ignore
            positive_class_index = list(classes).index(1)
        except (AttributeError, ValueError):
            positive_class_index = 1

        y_score = probabilities_or_model.predict_proba(input_features)[:, positive_class_index] # type: ignore

    else:
        raise TypeError("Unsupported type for 'probabilities_or_model'. Must be a NumPy array or a model with support for '.predict_proba()'.")

    # ROC and AUC
    fpr, tpr, _ = roc_curve(true_labels, y_score)
    auc_score = roc_auc_score(true_labels, y_score)

    # Plot
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(fpr, tpr, color=color, lw=linewidth, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    ax.set_title(f"{model_name} - {target_name}", fontsize=base_fontsize)
    ax.set_xlabel("False Positive Rate", fontsize=base_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=base_fontsize)
    ax.tick_params(axis='both', labelsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=base_fontsize)
    ax.grid(True)

    # Save figure
    save_path = make_fullpath(save_directory, make=True)
    sanitized_target_name = sanitize_filename(target_name)
    full_save_path = save_path / f"ROC_{sanitized_target_name}.svg"
    fig.savefig(full_save_path, bbox_inches="tight", format="svg")

    return fig


# function to evaluate the model and save metrics (Regression)
def evaluate_model_regression(model, model_name: str, 
                               save_dir: Union[str,Path],
                               x_test_scaled: np.ndarray, single_y_test: np.ndarray, 
                               target_name: str,
                               figure_size: tuple = (12, 8),
                               alpha_transparency: float = 0.5,
                               base_fontsize: int = 24):
    # Generate predictions
    y_pred = model.predict(x_test_scaled)
    
    # Calculate regression metrics
    mae = mean_absolute_error(single_y_test, y_pred)
    mse = mean_squared_error(single_y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(single_y_test, y_pred)
    
    # Create formatted report
    sanitized_target_name = sanitize_filename(target_name)
    save_path = make_fullpath(save_dir, make=True)
    report_path = save_path / f"Regression_Report_{sanitized_target_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"{model_name} - Regression Performance for '{target_name}'\n\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")

    # Generate and save residual plot
    residuals = single_y_test - y_pred
    plt.figure(figsize=figure_size)
    plt.scatter(y_pred, residuals, alpha=alpha_transparency)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values", fontsize=base_fontsize)
    plt.ylabel("Residuals", fontsize=base_fontsize)
    plt.title(f"{model_name} - Residual Plot for {target_name}", fontsize=base_fontsize)
    plt.grid(True)
    plt.tight_layout()
    residual_path = save_path / f"Residual_Plot_{sanitized_target_name}.svg"
    plt.savefig(residual_path, bbox_inches='tight', format="svg")
    plt.close()
    
    # Create true vs predicted values plot
    plt.figure(figsize=figure_size)
    plt.scatter(single_y_test, y_pred, alpha=alpha_transparency)
    plt.plot([single_y_test.min(), single_y_test.max()], 
             [single_y_test.min(), single_y_test.max()], 
             'k--', lw=2)
    plt.xlabel('True Values', fontsize=base_fontsize)
    plt.ylabel('Predictions', fontsize=base_fontsize)
    plt.title(f"{model_name} - True vs Predicted for {target_name}", fontsize=base_fontsize)
    plt.grid(True)
    plot_path = save_path / f"Regression_Plot_{sanitized_target_name}.svg"
    plt.savefig(plot_path, bbox_inches='tight', format="svg")
    plt.close()

    return y_pred


# Get SHAP values
def get_shap_values(
    model,
    model_name: str,
    save_dir: Union[str, Path],
    features_to_explain: np.ndarray,
    feature_names: list[str],
    target_name: str,
    task: Literal["classification", "regression"],
    max_display_features: int = 10,
    figsize: tuple = (16, 20),
    base_fontsize: int = 38,
):
    """
    Universal SHAP explainer for regression and classification.
        * Use `X_train` (or a subsample of it) to see how the model explains the data it was trained on.
        
	    * Use `X_test` (or a hold-out set) to see how the model explains unseen data.
     
	    * Use the entire dataset to get the global view. 
 
    Parameters:
        task: 'regression' or 'classification'.
        features_to_explain: Should match the model's training data format, including scaling.
        save_dir: Directory to save visualizations.
    """
    sanitized_target_name = sanitize_filename(target_name)
    global_save_path = make_fullpath(save_dir, make=True)
    
    def _apply_plot_style():
        styles = ['seaborn', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8', 'default']
        for style in styles:
            if style in plt.style.available or style == 'default':
                plt.style.use(style)
                break

    def _configure_rcparams():
        plt.rc('font', size=base_fontsize)
        plt.rc('axes', titlesize=base_fontsize)
        plt.rc('axes', labelsize=base_fontsize)
        plt.rc('xtick', labelsize=base_fontsize)
        plt.rc('ytick', labelsize=base_fontsize + 2)
        plt.rc('legend', fontsize=base_fontsize)
        plt.rc('figure', titlesize=base_fontsize)

    def _create_shap_plot(shap_values, features, save_path: Path, plot_type: str, title: str):
        _apply_plot_style()
        _configure_rcparams()
        plt.figure(figsize=figsize)

        shap.summary_plot(
            shap_values=shap_values,
            features=features,
            feature_names=feature_names,
            plot_type=plot_type,
            show=False,
            plot_size=figsize,
            max_display=max_display_features,
            alpha=0.7,
            # color='viridis'
        )

        ax = plt.gca()
        ax.set_xlabel("SHAP Value Impact", fontsize=base_fontsize + 2, weight='bold', labelpad=20)
        plt.title(title, fontsize=base_fontsize + 2, pad=20, weight='bold')

        for tick in ax.get_xticklabels():
            tick.set_fontsize(base_fontsize)
            tick.set_rotation(30)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(base_fontsize + 2)

        if plot_type == "dot":
            cb = plt.gcf().axes[-1]
            cb.set_ylabel("", size=1)
            cb.tick_params(labelsize=base_fontsize - 2)

        plt.savefig(save_path, bbox_inches='tight', facecolor='white', format="svg")
        plt.close()
        rcdefaults()

    def _plot_for_classification(shap_values, class_names):
        is_multiclass = isinstance(shap_values, list) and len(shap_values) > 1

        if is_multiclass:
            for class_shap, class_name in zip(shap_values, class_names):
                for plot_type in ["bar", "dot"]:
                    _create_shap_plot(
                        shap_values=class_shap,
                        features=features_to_explain,
                        save_path=global_save_path / f"SHAP_{sanitized_target_name}_Class{class_name}_{plot_type}.svg",
                        plot_type=plot_type,
                        title=f"{model_name} - {target_name} (Class {class_name})"
                    )
        else:
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
            for plot_type in ["bar", "dot"]:
                _create_shap_plot(
                    shap_values=values,
                    features=features_to_explain,
                    save_path=global_save_path / f"SHAP_{sanitized_target_name}_{plot_type}.svg",
                    plot_type=plot_type,
                    title=f"{model_name} - {target_name}"
                )

    def _plot_for_regression(shap_values):
        for plot_type in ["bar", "dot"]:
            _create_shap_plot(
                shap_values=shap_values,
                features=features_to_explain,
                save_path=global_save_path / f"SHAP_{sanitized_target_name}_{plot_type}.svg",
                plot_type=plot_type,
                title=f"{model_name} - {target_name}"
            )
    #START_O

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_to_explain)

    if task == 'classification':
        try:
            class_names = model.classes_ if hasattr(model, 'classes_') else list(range(len(shap_values)))
        except Exception:
            class_names = list(range(len(shap_values)))
        _plot_for_classification(shap_values, class_names)
    else:
        _plot_for_regression(shap_values)


# TRAIN TEST PIPELINE
def train_test_pipeline(model, model_name: str, dataset_id: str, task: TaskType,
             train_features: np.ndarray, train_target: np.ndarray,
             test_features: np.ndarray, test_target: np.ndarray,
             feature_names: list[str], target_name: str,
             save_dir: Union[str,Path],
             debug: bool=False, save_model: bool=False):
    ''' 
    1. Train model.
    2. Evaluate model.
    3. SHAP values.
    
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
        
    if task == "classification":
        y_pred = evaluate_model_classification(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_name=target_name)
        plot_roc_curve(true_labels=test_target,
                       probabilities_or_model=trained_model, model_name=model_name, 
                       target_name=target_name, save_directory=local_save_directory, 
                       input_features=test_features)
    elif task == "regression":
        y_pred = evaluate_model_regression(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_name=target_name)
    else:
        raise ValueError(f"Unrecognized task '{task}' for model training,")
    if debug:
        _LOGGER.info(f"Predicted vector: {type(y_pred)} with shape: {y_pred.shape}")
    
    get_shap_values(model=trained_model, model_name=model_name, save_dir=local_save_directory,
                    features_to_explain=train_features, feature_names=feature_names, target_name=target_name, task=task)
    
    return trained_model, y_pred

###### 5. Execution ######
def run_ensemble_pipeline(datasets_dir: Union[str,Path], save_dir: Union[str,Path], target_columns: list[str], model_object: Union[RegressionTreeModels, ClassificationTreeModels],
         handle_classification_imbalance: HandleImbalanceStrategy=None, save_model: bool=False,
         test_size: float=0.2, debug:bool=False):
    #Check models
    if isinstance(model_object, RegressionTreeModels):
        task = "regression"
    elif isinstance(model_object, ClassificationTreeModels):
        task = "classification"
        if handle_classification_imbalance is None:
            _LOGGER.warning("⚠️ No method to handle classification class imbalance has been selected. Datasets are assumed to be balanced.")
        elif handle_classification_imbalance == "by_model":
            model_object.use_model_balance = True
        else:
            model_object.use_model_balance = False
    else:
        raise TypeError(f"Unrecognized model {type(model_object)}")
    
    #Check paths
    datasets_path = make_fullpath(datasets_dir)
    save_path = make_fullpath(save_dir, make=True)
    
    _LOGGER.info("🏁 Training starting...")
    #Yield imputed dataset
    for dataframe, dataframe_name in yield_dataframes_from_dir(datasets_path):
        #Yield features dataframe and target dataframe
        for df_features, df_target, feature_names, target_name in dataset_yielder(df=dataframe, target_cols=target_columns):
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
                                    debug=debug, save_dir=save_path, save_model=save_model)

    _LOGGER.info("✅ Training and evaluation complete.")


###### 6. Inference ######
class InferenceHandler:
    """
    Handles loading ensemble models and performing inference for either regression or classification tasks.
    """
    def __init__(self, 
                 models_dir: Union[str,Path], 
                 task: TaskType,
                 verbose: bool=True) -> None:
        """
        Initializes the handler by loading all models from a directory.

        Args:
            models_dir (Path): The directory containing the saved .joblib model files.
            task ("regression" | "classification"): The type of task the models perform.
        """
        self.models: Dict[str, Any] = dict()
        self.task: str = task
        self.verbose = verbose
        self._feature_names: Optional[List[str]] = None
        
        model_files = list_files_by_extension(directory=models_dir, extension="joblib")
        
        for fname, fpath in model_files.items():
            try:
                full_object: dict
                full_object = deserialize_object(filepath=fpath, 
                                                 verbose=self.verbose, 
                                                 raise_on_error=True) # type: ignore
                
                model: Any = full_object[ModelSaveKeys.MODEL]
                target_name: str = full_object[ModelSaveKeys.TARGET]
                feature_names_list: List[str] = full_object[ModelSaveKeys.FEATURES]
                
                # Check that feature names match
                if self._feature_names is None:
                    # Store the feature names from the first model loaded.
                    self._feature_names = feature_names_list
                elif self._feature_names != feature_names_list:
                    # Add a warning if subsequent models have different feature names.
                    _LOGGER.warning(f"⚠️ Mismatched feature names in {fname}. Using feature order from the first model loaded.")
                
                self.models[target_name] = model
                if self.verbose:
                    _LOGGER.info(f"✅ Loaded model for target: {target_name}")

            except Exception as e:
                _LOGGER.warning(f"⚠️ Failed to load or parse {fname}: {e}")
                
    @property
    def feature_names(self) -> List[str]:
        """
        Getter for the list of feature names the models expect.
        Returns an empty list if no models were loaded.
        """
        return self._feature_names if self._feature_names is not None else []
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predicts on a single feature vector.

        Args:
            features (np.ndarray): A 1D or 2D NumPy array for a single sample.

        Returns:
            Dict[str, Any]: A dictionary where keys are target names.
                - For regression: The value is the single predicted float.
                - For classification: The value is another dictionary {'label': ..., 'probabilities': ...}.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[0] != 1:
            raise ValueError("The predict() method is for a single sample. Use predict_batch() for multiple samples.")

        results: Dict[str, Any] = dict()
        for target_name, model in self.models.items():
            if self.task == "regression":
                prediction = model.predict(features)
                results[target_name] = prediction.item()
            else: # Classification
                label = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                results[target_name] = {"label": label, "probabilities": probabilities}
        
        if self.verbose:
            _LOGGER.info("✅ Inference process complete.")
        return results

    def predict_batch(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predicts on a batch of feature vectors.

        Args:
            features (np.ndarray): A 2D NumPy array where each row is a sample.

        Returns:
            Dict[str, Any]: A dictionary where keys are target names.
                - For regression: The value is a NumPy array of predictions.
                - For classification: The value is another dictionary {'labels': ..., 'probabilities': ...}.
        """
        if features.ndim != 2:
            raise ValueError("Input for batch prediction must be a 2D array.")

        results: Dict[str, Any] = dict()
        for target_name, model in self.models.items():
            if self.task == "regression":
                results[target_name] = model.predict(features)
            else: # Classification
                labels = model.predict(features)
                probabilities = model.predict_proba(features)
                results[target_name] = {"labels": labels, "probabilities": probabilities}
                
        if self.verbose:
            _LOGGER.info("✅ Inference process complete.")

        return results


###### 7. Save Model info report ######
def model_report(
        model_path: Union[str,Path],
        output_dir: Optional[Union[str,Path]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
    """
    Deserializes a model and generates a summary report.

    This function loads a serialized model object (joblib), prints a summary to the
    console (if verbose), and saves a detailed JSON report.

    Args:
        model_path (str): The path to the serialized model file.
        output_dir (str, optional): Directory to save the JSON report.
            If None, it defaults to the same directory as the model file.
        verbose (bool, optional): If True, prints summary information
            to the console. Defaults to True.

    Returns:
        (Dict[str, Any]): A dictionary containing the model metadata.

    Raises:
        FileNotFoundError: If the model_path does not exist.
        KeyError: If the deserialized object is missing required keys from `ModelSaveKeys`.
    """
    # 1. Convert to Path object
    model_p = make_fullpath(model_path)

    # --- 2. Deserialize and Extract Info ---
    try:
        full_object: dict = deserialize_object(model_p) # type: ignore
        model = full_object[ModelSaveKeys.MODEL]
        target = full_object[ModelSaveKeys.TARGET]
        features = full_object[ModelSaveKeys.FEATURES]
    except FileNotFoundError:
        _LOGGER.error(f"❌ Model file not found at '{model_p}'")
        raise
    except (KeyError, TypeError) as e:
        _LOGGER.error(
            f"❌ The serialized object is missing required keys '{ModelSaveKeys.MODEL}', '{ModelSaveKeys.TARGET}', '{ModelSaveKeys.FEATURES}'"
        )
        raise e

    # --- 3. Print Summary to Console (if verbose) ---
    if verbose:
        print("\n--- 📝 Model Summary ---")
        print(f"Source File:    {model_p.name}")
        print(f"Model Type:     {type(model).__name__}")
        print(f"Target:         {target}")
        print(f"Feature Count:  {len(features)}")
        print("-----------------------")

    # --- 4. Generate JSON Report ---
    report_data = {
        "source_file": model_p.name,
        "model_type": str(type(model)),
        "target_name": target,
        "feature_count": len(features),
        "feature_names": features
    }

    # Determine output path
    output_p = make_fullpath(output_dir, make=True) if output_dir else model_p.parent
    json_filename = model_p.stem + "_info.json"
    json_filepath = output_p / json_filename    

    try:
        with open(json_filepath, 'w') as f:
            json.dump(report_data, f, indent=4)
        if verbose:
            _LOGGER.info(f"✅ JSON report saved to: '{json_filepath}'")
    except PermissionError:
        _LOGGER.error(f"❌ Permission denied to write JSON report at '{json_filepath}'")

    # --- 5. Return the extracted data ---
    return report_data


def info():
    _script_info(__all__)
