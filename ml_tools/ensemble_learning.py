import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import rcdefaults

import os
from typing import Literal, Union, Optional
import joblib

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, roc_auc_score
import shap

import warnings # Ignore warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


###### 1. Dataset Loader ######
#Load imputed datasets as a generator
def yield_imputed_dataframe(datasets_dir: str):
    '''
    Yields a tuple `(dataframe, dataframe_name)`
    '''
    dataset_filenames = [dataset for dataset in os.listdir(datasets_dir) if dataset.endswith(".csv")]
    if not dataset_filenames:
        raise IOError(f"No imputed datasets have been found at {datasets_dir}")
    
    for dataset_filename in dataset_filenames:
        full_path = os.path.join(datasets_dir, dataset_filename)
        df = pd.read_csv(full_path)
        #remove extension
        filename = os.path.splitext(os.path.basename(dataset_filename))[0]
        print(f"Working on dataset: {filename}")
        yield (df, filename)

#Split a dataset into features and targets datasets
def dataset_yielder(df: pd.DataFrame, target_cols: list[str]):
    ''' 
    Yields one Tuple at a time: `(df_features, df_target, feature_names, target_name)`
    '''
    df_features = df.drop(columns=target_cols)
    feature_names = df_features.columns.to_list()
    
    for target_col in target_cols:
        df_target = df[target_col]
        yield (df_features, df_target, feature_names, target_col)

###### 2. Initialize Models ######
def get_models(task: Literal["classification", "regression"], random_state: int=101, is_balanced: bool = True, 
              L1_regularization: float = 1.0, L2_regularization: float = 1.0, learning_rate: float=0.005) -> dict:
    ''' 
    Returns a dictionary `{Model_Name: Model}` with new instances of models.
    Valid tasks: "classification" or "regression".
    
    Classification Models:
        - "XGBoost" - XGBClassifier
        - "LightGBM" - LGBMClassifier
        - "HistGB" - HistGradientBoostingClassifier
    Regression Models:
        - "XGBoost" - XGBRegressor
        - "LightGBM" - LGBMRegressor
        - "HistGB" - HistGradientBoostingRegressor
        
    For classification only: Set `is_balanced=False` for imbalanced datasets.
    
    Increase L1 and L2 if model is overfitting
    '''
    
    # Model initialization logic
    if task not in ["classification", "regression"]:
        raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'regression'.")

    models = {}

    # Common parameters
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'reg_alpha': L1_regularization,
        'reg_lambda': L2_regularization,
    }

    lgbm_params = {
        'n_estimators': 200,
        'learning_rate': learning_rate,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'verbose': -1,
        'reg_alpha': L1_regularization,
        'reg_lambda': L2_regularization,
    }

    hist_params = {
        'max_iter': 200,
        'learning_rate': learning_rate,
        'max_depth': 5,
        'min_samples_leaf': 30,
        'random_state': random_state,
        'l2_regularization': L2_regularization,
    }

    # XGB Model
    if task == "classification":
        xgb_params.update({
            'scale_pos_weight': 1 if is_balanced else 8,
            'eval_metric': 'aucpr'
        })
        models["XGBoost"] = xgb.XGBClassifier(**xgb_params)
    else:
        xgb_params.update({'eval_metric': 'rmse'})
        models["XGBoost"] = xgb.XGBRegressor(**xgb_params)

    # LGBM Model
    if task == "classification":
        lgbm_params.update({
            'class_weight': None if is_balanced else 'balanced',
            'boosting_type': 'goss' if is_balanced else 'dart',
        })
        models["LightGBM"] = lgb.LGBMClassifier(**lgbm_params)
    else:
        lgbm_params['boosting_type'] = 'dart'
        models["LightGBM"] = lgb.LGBMRegressor(**lgbm_params)

    # HistGB Model
    if task == "classification":
        hist_params.update({
            'class_weight': None if is_balanced else 'balanced',
            'scoring': 'loss' if is_balanced else 'balanced_accuracy',
        })
        models["HistGB"] = HistGradientBoostingClassifier(**hist_params)
    else:
        hist_params['scoring'] = 'neg_mean_squared_error'
        models["HistGB"] = HistGradientBoostingRegressor(**hist_params)

    return models

###### 3. Process Dataset ######
# function to split data into train and test
def _split_data(features, target, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)   
    return X_train, X_test, y_train, y_test

# function to standardize the data
def _standardize_data(train_features, test_features, scaler_code):
    if scaler_code == "standard":
        scaler = StandardScaler()
    elif scaler_code == "minmax":
        scaler = MinMaxScaler()
    elif scaler_code == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Unrecognized scaler {scaler_code}")
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    return train_scaled, test_scaled, scaler

# Over-sample minority class (Positive cases) and return several single target datasets (Classification)
def _resample(X_train_scaled: np.ndarray, y_train: pd.Series, 
              strategy: Literal[r"ADASYN", r'SMOTE', r'RANDOM', r'UNDERSAMPLE'], random_state):
    ''' 
    Oversample minority class or undersample majority class.
    
    Returns a Tuple `(Features: nD-Array, Target: 1D-array)`
    '''
    if strategy == 'SMOTE':
        resample_algorithm = SMOTE(random_state=random_state, k_neighbors=3)
    elif strategy == 'RANDOM':
        resample_algorithm = RandomOverSampler(random_state=random_state)
    elif strategy == 'UNDERSAMPLE':
        resample_algorithm = RandomUnderSampler(random_state=random_state)
    elif strategy == 'ADASYN':
        resample_algorithm = ADASYN(random_state=random_state, n_neighbors=3)
    else:
        raise ValueError(f"Invalid resampling strategy: {strategy}")
    
    X_res, y_res = resample_algorithm.fit_resample(X_train_scaled, y_train)
    return X_res, y_res

# DATASET PIPELINE
def dataset_pipeline(df_features: pd.DataFrame, df_target: pd.Series, task: Literal["classification", "regression"],
                     resample_strategy: Union[Literal[r"ADASYN", r'SMOTE', r'RANDOM', r'UNDERSAMPLE'], None], scaler: Literal["standard", "minmax", "maxabs"],
                     test_size: float=0.2, debug: bool=False, random_state: int=101):
    ''' 
    1. Make Train/Test splits
    2. Standardize Train and Test Features
    3. Oversample imbalanced classes (classification)
    
    Return a processed Tuple: (X_train, y_train, X_test, y_test, Scaler)
    
    `(nD-array, 1D-array, nD-array, Series, Scaler)`
    '''
    #DEBUG
    if debug:
        print(f"Split Dataframes Shapes - Features DF: {df_features.shape}, Target DF: {df_target.shape}")
        unique_values = df_target.unique()  # Get unique values for the target column
        print(f"\tUnique values for '{df_target.name}': {unique_values}")
    
    #Train test split
    X_train, X_test, y_train, y_test = _split_data(features=df_features, target=df_target, test_size=test_size, random_state=random_state)
    
    #DEBUG
    if debug:
        print(f"Shapes after train test split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Standardize    
    X_train_scaled, X_test_scaled, scaler_object = _standardize_data(train_features=X_train, test_features=X_test, scaler_code=scaler)
    
    #DEBUG
    if debug:
        print(f"Shapes after scaling features - X_train: {X_train_scaled.shape}, y_train: {y_train.shape}, X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")
 
    # Scale
    if resample_strategy is None or task == "regression":
        X_train_oversampled, y_train_oversampled = X_train_scaled, y_train
    else:
        X_train_oversampled, y_train_oversampled = _resample(X_train_scaled=X_train_scaled, y_train=y_train, strategy=resample_strategy, random_state=random_state)
    
    #DEBUG
    if debug:
        print(f"Shapes after resampling - X_train: {X_train_oversampled.shape}, y_train: {y_train_oversampled.shape}, X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    
    return X_train_oversampled, y_train_oversampled, X_test_scaled, y_test, scaler_object

###### 4. Train and Evaluation ######
# Trainer function
def _train_model(model, train_features, train_target):
    model.fit(train_features, train_target)
    return model

# handle local directories
def _local_directories(model_name: str, dataset_id: str, save_dir: str):
    dataset_dir = os.path.join(save_dir, dataset_id)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    
    model_dir = os.path.join(dataset_dir, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    return model_dir

# save model
def _save_model(trained_model, model_name: str, target_name:str, feature_names: list[str], save_directory: str, scaler_object: Union[StandardScaler, MinMaxScaler, MaxAbsScaler]):
    full_path = os.path.join(save_directory, f"{model_name}_{target_name}.joblib")
    joblib.dump({'model': trained_model, 'scaler':scaler_object, 'feature_names': feature_names, 'target_name':target_name}, full_path)

# function to evaluate the model and save metrics (Classification)
def evaluate_model_classification(
    model,
    model_name: str,
    save_dir: str,
    x_test_scaled: np.ndarray,
    single_y_test: np.ndarray,
    target_id: str,
    figsize: tuple = (10, 8),
    title_fontsize: int = 24,
    label_fontsize: int = 24,
    cmap: Colormap = plt.cm.Blues # type: ignore
) -> np.ndarray:
    """
    Evaluates a classification model, saves the classification report and confusion matrix plot.

    Parameters:
        model: Trained classifier with .predict() method
        model_name: Identifier for the model
        save_dir: Directory where results are saved
        x_test_scaled: Feature matrix for test set
        single_y_test: True binary labels
        target_id: Suffix for naming output files
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
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(single_y_test, y_pred)

    report = classification_report(
        single_y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=False
    )

    # Save text report
    report_path = os.path.join(save_dir, f"Classification_Report_{target_id}.txt")
    with open(report_path, "w") as f:
        f.write(f"{model_name} - {target_id}\t\tAccuracy: {accuracy:.2f}\n")
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

    ax.set_title(f"{model_name} - {target_id}", fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=label_fontsize)
    ax.set_xlabel("Predicted label", fontsize=label_fontsize)
    ax.set_ylabel("True label", fontsize=label_fontsize)
    
    # Turn off gridlines
    ax.grid(False)
    
    # Manually update font size of cell texts
    for text in ax.texts:
        text.set_fontsize(title_fontsize+4)

    fig.tight_layout()
    fig_path = os.path.join(save_dir, f"Confusion_Matrix_{target_id}.svg")
    fig.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return y_pred

#Function to save ROC and ROC AUC (Classification)
def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities_or_model: Union[np.ndarray, xgb.XGBClassifier, lgb.LGBMClassifier, object],
    model_name: str,
    target_name: str,
    save_directory: str,
    color: str = "darkorange",
    figure_size: tuple = (10, 10),
    linewidth: int = 2,
    title_fontsize: int = 24,
    label_fontsize: int = 24,
    input_features: Optional[np.ndarray] = None,
) -> plt.Figure: # type: ignore
    """
    Plots the ROC curve and computes AUC for binary classification. Positive class is assumed to be in the second column of the probabilities array.
    
    Parameters:
        true_labels: np.ndarray of shape (n_samples,), ground truth binary labels (0 or 1).
        probabilities_or_model: either predicted probabilities (ndarray), or a trained model with attribute `.predict_proba()`.
        target_name: str, used for figure title and filename.
        save_directory: str, path to directory where figure is saved.
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

    ax.set_title(f"{model_name} - {target_name}", fontsize=title_fontsize)
    ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=label_fontsize)
    ax.legend(loc="lower right", fontsize=label_fontsize)
    ax.grid(True)

    # Save figure
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, f"ROC_{target_name}.svg")
    fig.savefig(save_path, bbox_inches="tight", format="svg")

    return fig

# function to evaluate the model and save metrics (Regression)
def evaluate_model_regression(model, model_name: str, 
                               save_dir: str,
                               x_test_scaled: np.ndarray, single_y_test: np.ndarray, 
                               target_id: str,
                               figure_size: tuple = (12, 8),
                               alpha_transparency: float = 0.5,
                               title_fontsize: int = 24,
                               normal_fontsize: int = 24):
    # Generate predictions
    y_pred = model.predict(x_test_scaled)
    
    # Calculate regression metrics
    mae = mean_absolute_error(single_y_test, y_pred)
    mse = mean_squared_error(single_y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(single_y_test, y_pred)
    
    # Create formatted report
    report_path = os.path.join(save_dir, f"Regression_Report_{target_id}.txt")
    with open(report_path, "w") as f:
        f.write(f"{model_name} - {target_id} Regression Performance\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")

    # Generate and save residual plot
    residuals = single_y_test - y_pred
    plt.figure(figsize=figure_size)
    plt.scatter(y_pred, residuals, alpha=alpha_transparency)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values", fontsize=normal_fontsize)
    plt.ylabel("Residuals", fontsize=normal_fontsize)
    plt.title(f"{model_name} - Residual Plot for {target_id}", fontsize=title_fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Residual_Plot_{target_id}.svg"), bbox_inches='tight', format="svg")
    plt.close()
    
    # Create true vs predicted values plot
    plt.figure(figsize=figure_size)
    plt.scatter(single_y_test, y_pred, alpha=alpha_transparency)
    plt.plot([single_y_test.min(), single_y_test.max()], 
             [single_y_test.min(), single_y_test.max()], 
             'k--', lw=2)
    plt.xlabel('True Values', fontsize=normal_fontsize)
    plt.ylabel('Predictions', fontsize=normal_fontsize)
    plt.title(f"{model_name} - True vs Predicted for {target_id}", fontsize=title_fontsize)
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"Regression_Plot_{target_id}.svg")
    plt.savefig(plot_path, bbox_inches='tight', format="svg")
    plt.close()

    return y_pred

# Get SHAP values
def get_shap_values(model, model_name: str, 
                   save_dir: str,
                   features_to_explain: np.ndarray, 
                   feature_names: list[str], 
                   target_id: str,
                   task: Literal["classification", "regression"],
                   max_display_features: int=8,
                   figsize: tuple=(14, 20),
                   title_fontsize: int=38,
                   label_fontsize: int=38,
                   plot_type: Literal["bar", "dot"] = "dot"
                   ):
    """
    Universal SHAP explainer for regression and classification.
    - Use `X_train` (or a subsample of it) to see how the model explains the data it was trained on.
	- Use `X_test` (or a hold-out set) to see how the model explains unseen data.
	- Use the entire dataset to get the global view. 
 
    Parameters:
    - 'task': 'regression' or 'classification'
    - 'features_to_explain': Should match the model's training data format, including scaling.
    - 'save_dir': Directory to save visualizations
    """
    def _create_shap_plot(shap_values, features, feature_names, 
                         full_save_path: str, plot_type: str, 
                         title: str):
        """Helper function to create and save SHAP plots"""
        # Set style
        preferred_styles = ['seaborn', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8', 'default']
        for style in preferred_styles:
            if style in plt.style.available or style == 'default':
                plt.style.use(style)
                break
        
        plt.figure(figsize=figsize)
        
        #set rc parameters for better readability
        plt.rc('font', size=label_fontsize)
        plt.rc('axes', titlesize=title_fontsize)
        plt.rc('axes', labelsize=label_fontsize)
        plt.rc('xtick', labelsize=label_fontsize)
        plt.rc('ytick', labelsize=label_fontsize)
        plt.rc('legend', fontsize=label_fontsize)
        plt.rc('figure', titlesize=title_fontsize)
        
        # Create the SHAP plot
        shap.summary_plot(
            shap_values=shap_values,
            features=features,
            feature_names=feature_names,
            plot_type=plot_type,
            show=False,
            plot_size=figsize,
            max_display=max_display_features,
            alpha=0.7,
            color=plt.get_cmap('viridis')
        )
        
        # Add professional styling
        ax = plt.gca()
        ax.set_xlabel("SHAP Value Impact", fontsize=title_fontsize, weight='bold')
        ax.set_ylabel("Features", fontsize=title_fontsize, weight='bold')
        plt.title(title, fontsize=title_fontsize, pad=20, weight='bold')
        
        # Manually fix tick fonts
        for tick in ax.get_xticklabels():
            tick.set_fontsize(label_fontsize)
            tick.set_rotation(45)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(label_fontsize)

        # Handle colorbar for dot plots
        if plot_type == "dot":
            cb = plt.gcf().axes[-1]
            # cb.set_ylabel("Feature Value", size=label_fontsize)
            cb.set_ylabel("", size=1)
            cb.tick_params(labelsize=label_fontsize - 2)
        
        # Save and clean up
        plt.savefig(
            full_save_path,
            bbox_inches='tight',
            facecolor='white', 
            format="svg"
        )
        plt.close()
        rcdefaults()  # Reset rc parameters to default
    
    # START
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_to_explain)
    
    # Handle different model types
    if task == 'classification':
        # Determine if multiclass
        try:
            is_multiclass = len(model.classes_) > 2
            class_names = model.classes_
        except AttributeError:
            is_multiclass = isinstance(shap_values, list) and len(shap_values) > 1
            class_names = list(range(len(shap_values))) if is_multiclass else [0, 1]
        
        if is_multiclass:
            for class_idx, (class_shap, class_name) in enumerate(zip(shap_values, class_names)):
                _create_shap_plot(
                    shap_values=class_shap,
                    features=features_to_explain,
                    feature_names=feature_names,
                    full_save_path=os.path.join(save_dir, f"SHAP_{target_id}_Class{class_name}.svg"),
                    plot_type=plot_type,
                    title=f"{model_name} - {target_id} (Class {class_name})"
                )
        else:
            # Handle binary classification (single array case)
            plot_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            _create_shap_plot(
                shap_values=plot_vals,
                features=features_to_explain,
                feature_names=feature_names,
                full_save_path=os.path.join(save_dir, f"SHAP_{target_id}.svg"),
                plot_type=plot_type,
                title=f"{model_name} - {target_id}"
            )
    
    else:  # Regression
        _create_shap_plot(
            shap_values=shap_values,
            features=features_to_explain,
            feature_names=feature_names,
            full_save_path=os.path.join(save_dir, f"SHAP_{target_id}.svg"),
            plot_type=plot_type,
            title=f"{model_name} - {target_id}"
        )
        
# TRAIN TEST PIPELINE
def train_test_pipeline(model, model_name: str, dataset_id: str, task: Literal["classification", "regression"],
             train_features: np.ndarray, train_target: np.ndarray,
             test_features: np.ndarray, test_target: np.ndarray,
             feature_names: list[str], target_id: str, scaler_object: Union[StandardScaler, MinMaxScaler, MaxAbsScaler],
             save_dir: str,
             debug: bool=False, save_model: bool=False):
    ''' 
    1. Train model.
    2. Evaluate model.
    3. SHAP values.
    
    Returns: Tuple(Trained model, Test-set Predictions)
    '''
    print(f"\tModel: {model_name} for Target: {target_id}...")
    trained_model = _train_model(model=model, train_features=train_features, train_target=train_target)
    if debug:
        print(f"Trained model object: {type(trained_model)}")
    local_save_directory = _local_directories(model_name=model_name, dataset_id=dataset_id, save_dir=save_dir)
    
    if save_model:
        _save_model(trained_model=trained_model, model_name=model_name, 
                    target_name=target_id, feature_names=feature_names, 
                    save_directory=local_save_directory, scaler_object=scaler_object)
        
    if task == "classification":
        y_pred = evaluate_model_classification(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_id=target_id)
        plot_roc_curve(true_labels=test_target,
                       probabilities_or_model=trained_model, model_name=model_name, 
                       target_name=target_id, save_directory=local_save_directory, 
                       input_features=test_features)
    elif task == "regression":
        y_pred = evaluate_model_regression(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_id=target_id)
    else:
        raise ValueError(f"Unrecognized task '{task}' for model training,")
    if debug:
        print(f"Predicted vector: {type(y_pred)} with shape: {y_pred.shape}")
    
    get_shap_values(model=trained_model, model_name=model_name, save_dir=local_save_directory,
                    features_to_explain=train_features, feature_names=feature_names, target_id=target_id, task=task)
    print("\t...done.")
    return trained_model, y_pred

###### 5. Execution ######
def run_pipeline(datasets_dir: str, save_dir: str, target_columns: list[str], task: Literal["classification", "regression"]="regression",
         resample_strategy: Literal[r"ADASYN", r'SMOTE', r'RANDOM', r'UNDERSAMPLE', None]=None, scaler: Literal["standard", "minmax", "maxabs"]="minmax", save_model: bool=False,
         test_size: float=0.2, debug:bool=False, L1_regularization: float=0.5, L2_regularization: float=0.5, learning_rate: float=0.005, random_state: int=101):
    #Check paths
    _check_paths(datasets_dir, save_dir)
    #Yield imputed dataset
    for dataframe, dataframe_name in yield_imputed_dataframe(datasets_dir):
        #Yield features dataframe and target dataframe
        for df_features, df_target, feature_names, target_name in dataset_yielder(df=dataframe, target_cols=target_columns):
            #Dataset pipeline
            X_train, y_train, X_test, y_test, scaler_object = dataset_pipeline(df_features=df_features, df_target=df_target, task=task,
                                                                resample_strategy=resample_strategy, scaler=scaler,
                                                                test_size=test_size, debug=debug, random_state=random_state)
            #Get models
            models_dict = get_models(task=task, is_balanced=False if resample_strategy is None else True, 
                                     L1_regularization=L1_regularization, L2_regularization=L2_regularization, learning_rate=learning_rate)
            #Train models
            for model_name, model in models_dict.items():
                train_test_pipeline(model=model, model_name=model_name, dataset_id=dataframe_name, task=task,
                                    train_features=X_train, train_target=y_train,
                                    test_features=X_test, test_target=y_test,
                                    feature_names=feature_names,target_id=target_name, scaler_object=scaler_object,
                                    debug=debug, save_dir=save_dir, save_model=save_model)
    print("\nTraining and evaluation complete.")
    
    
def _check_paths(datasets_dir: str, save_dir:str):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)   
    if not os.path.isdir(datasets_dir):
        raise IOError(f"Datasets directory '{datasets_dir}' not found.\nCheck path or run MICE script first.")
