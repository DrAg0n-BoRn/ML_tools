import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from typing import Literal, Union
import joblib

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
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
        print(f"Working on file: {filename}")
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
        - "XGB" - XGBClassifier
        - "LGBM" - LGBMClassifier
        - "HistGB" - HistGradientBoostingClassifier
    Regression Models:
        - "XGB" - XGBRegressor
        - "LGBM" - LGBMRegressor
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
        models["XGB"] = xgb.XGBClassifier(**xgb_params)
    else:
        xgb_params.update({'eval_metric': 'rmse'})
        models["XGB"] = xgb.XGBRegressor(**xgb_params)

    # LGBM Model
    if task == "classification":
        lgbm_params.update({
            'class_weight': None if is_balanced else 'balanced',
            'boosting_type': 'goss' if is_balanced else 'dart',
        })
        models["LGBM"] = lgb.LGBMClassifier(**lgbm_params)
    else:
        lgbm_params['boosting_type'] = 'dart'
        models["LGBM"] = lgb.LGBMRegressor(**lgbm_params)

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
def _standardize_data(train_features, test_features):
    global DATASCALER
    DATASCALER = StandardScaler()
    train_scaled = DATASCALER.fit_transform(train_features)
    test_scaled = DATASCALER.transform(test_features)
    return train_scaled, test_scaled

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
                     resample_strategy: Union[Literal[r"ADASYN", r'SMOTE', r'RANDOM', r'UNDERSAMPLE'], None], 
                     test_size: float=0.2, debug: bool=False, random_state: int=101):
    ''' 
    1. Make Train/Test splits
    2. Standardize Train and Test Features
    3. Oversample imbalanced classes (classification)
    
    Return a processed Tuple: (X_train, y_train, X_test, y_test)
    
    `(nD-array, 1D-array, nD-array, Series)`
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
    X_train_scaled, X_test_scaled = _standardize_data(train_features=X_train, test_features=X_test)
    
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
    
    return X_train_oversampled, y_train_oversampled, X_test_scaled, y_test

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
def _save_model(trained_model, model_name: str, target_name:str, feature_names: list[str], save_directory: str):
    full_path = os.path.join(save_directory, f"{model_name}_{target_name}.joblib")
    joblib.dump({'model': trained_model, 'scaler':DATASCALER, 'feature_names': feature_names, 'target_name':target_name}, full_path)


# function to evaluate the model and save metrics (Classification)
def evaluate_model_classification(model, model_name: str, 
                                   save_dir: str,
                                   x_test_scaled: np.ndarray, single_y_test: np.ndarray, 
                                   target_id: str):
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(single_y_test, y_pred)
    report: str = classification_report(single_y_test, y_pred, target_names=["Negative", "Positive"], output_dict=False) # type: ignore
    
    report_path = os.path.join(save_dir, f"Classification_Report_{target_id}.txt")
    # Write a new file
    with open(report_path, "w") as f:
        f.write(f"{model_name} - {target_id}\t\tAccuracy: {accuracy:.2f}\n")
        f.write(f"Classification Report:\n")
        f.write(report)
        
    #Generate confusion matrix
    disp_ = ConfusionMatrixDisplay.from_predictions(
        y_true=single_y_test, y_pred=y_pred,
        display_labels=["Negative", "Positive"],
        cmap=plt.cm.Blues,
        normalize="true"
    )
    plt.title(f"{model_name} - Confusion Matrix for {target_id}")
    plt.savefig(os.path.join(save_dir, f"Confusion_Matrix_{target_id}.png"))
    plt.close()

    return y_pred

# function to evaluate the model and save metrics (Regression)
def evaluate_model_regression(model, model_name: str, 
                               save_dir: str,
                               x_test_scaled: np.ndarray, single_y_test: np.ndarray, 
                               target_id: str):
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
        
    # Plot config
    figure_size = (12,8)
    alpha_transparency = 0.5
    title_fontsize = 14
    normal_fontsize = 12
    dpi_value = 300

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
    plt.savefig(os.path.join(save_dir, f"Residual_Plot_{target_id}.png"), bbox_inches='tight', dpi=dpi_value)
    plt.close()
    
    # Create true vs predicted values plot
    plt.figure(figsize=figure_size)
    plt.scatter(single_y_test, y_pred, alpha=alpha_transparency)
    plt.plot([single_y_test.min(), single_y_test.max()], 
             [single_y_test.min(), single_y_test.max()], 
             'k--', lw=2)
    plt.xlabel('True Values', fontsize=normal_fontsize)
    plt.ylabel('Predictions', fontsize=normal_fontsize)
    plt.title(f"{model_name} - True vs Predicted Values ({target_id})", fontsize=title_fontsize)
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"Regression_Plot_{target_id}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi_value)
    plt.close()

    return y_pred

# Get SHAP values
def get_shap_values(model, model_name: str, 
                   save_dir: str,
                   features_to_explain: np.ndarray, 
                   feature_names: list[str], 
                   target_id: str,
                   task: Literal["classification", "regression"]):
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
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 18))
        
        # Create the SHAP plot
        shap.summary_plot(
            shap_values=shap_values,
            features=features,
            feature_names=feature_names,
            plot_type=plot_type,
            show=False,
            plot_size=(12, 8),
            max_display=20,
            alpha=0.7,
            color=plt.get_cmap('viridis')
        )
        
        # Add professional styling
        ax = plt.gca()
        ax.set_xlabel("SHAP Value Impact", fontsize=12, weight='bold')
        ax.set_ylabel("Features", fontsize=12, weight='bold')
        plt.title(title, fontsize=14, pad=20, weight='bold')
        
        # Add explanatory text
        plt.text(0.5, -0.15, 
                "Negative SHAP ← Feature Impact → Positive SHAP",
                ha='center', va='center', 
                transform=ax.transAxes,
                fontsize=10,
                color='#666666')
        
        # Handle colorbar for dot plots
        if plot_type == "dot":
            cb = plt.gcf().axes[-1]
            cb.set_ylabel("Feature Value", size=10)
            cb.tick_params(labelsize=8)
        
        # Save and clean up
        plt.savefig(
            full_save_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.5,
            facecolor='white'
        )
        plt.close()
    
    # Start
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
                    full_save_path=os.path.join(save_dir, f"SHAP_{target_id}_Class{class_name}.png"),
                    plot_type="dot",
                    title=f"{model_name} - {target_id} (Class {class_name})"
                )
        else:
            # Handle binary classification (single array case)
            plot_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            _create_shap_plot(
                shap_values=plot_vals,
                features=features_to_explain,
                feature_names=feature_names,
                full_save_path=os.path.join(save_dir, f"SHAP_{target_id}.png"),
                plot_type="dot",
                title=f"{model_name} - {target_id}"
            )
    
    else:  # Regression
        _create_shap_plot(
            shap_values=shap_values,
            features=features_to_explain,
            feature_names=feature_names,
            full_save_path=os.path.join(save_dir, f"SHAP_{target_id}.png"),
            plot_type="bar",
            title=f"{model_name} - {target_id}"
        )
        
# TRAIN TEST PIPELINE
def _train_test_pipeline(model, model_name: str, dataset_id: str, task: Literal["classification", "regression"],
             train_features: np.ndarray, train_target: np.ndarray,
             test_features: np.ndarray, test_target: np.ndarray,
             feature_names: list[str], target_id: str, save_dir: str,
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
                    save_directory=local_save_directory)
        
    if task == "classification":
        y_pred = evaluate_model_classification(model=trained_model, model_name=model_name, save_dir=local_save_directory, 
                             x_test_scaled=test_features, single_y_test=test_target, target_id=target_id)
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
         resample_strategy: Literal[r"ADASYN", r'SMOTE', r'RANDOM', r'UNDERSAMPLE', None]=None, save_model: bool=False,
         test_size: float=0.2, debug:bool=False, L1_regularization: float=0.5, L2_regularization: float=0.5, learning_rate: float=0.005, random_state: int=101):
    #Check paths
    _check_paths(datasets_dir, save_dir)
    #Yield imputed dataset
    for dataframe, dataframe_name in yield_imputed_dataframe(datasets_dir):
        #Yield features dataframe and target dataframe
        for df_features, df_target, feature_names, target_name in dataset_yielder(df=dataframe, target_cols=target_columns):
            #Dataset pipeline
            X_train, y_train, X_test, y_test = dataset_pipeline(df_features=df_features, df_target=df_target, task=task,
                                                                resample_strategy=resample_strategy, test_size=test_size, debug=debug, random_state=random_state)
            #Get models
            models_dict = get_models(task=task, is_balanced=False if resample_strategy is None else True, 
                                     L1_regularization=L1_regularization, L2_regularization=L2_regularization, learning_rate=learning_rate)
            #Train models
            for model_name, model in models_dict.items():
                _train_test_pipeline(model=model, model_name=model_name, dataset_id=dataframe_name, task=task,
                                    train_features=X_train, train_target=y_train,
                                    test_features=X_test, test_target=y_test,
                                    feature_names=feature_names,target_id=target_name,
                                    debug=debug, save_dir=save_dir, save_model=save_model)
    print("\nTraining and evaluation complete.")
    
    
def _check_paths(datasets_dir: str, save_dir:str):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)   
    if not os.path.isdir(datasets_dir):
        raise IOError(f"Datasets directory '{datasets_dir}' not found.\nCheck path or run MICE script first.")
