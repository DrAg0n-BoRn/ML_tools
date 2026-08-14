import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import rcdefaults
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import learning_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             ConfusionMatrixDisplay, 
                             mean_absolute_error, 
                             mean_squared_error, 
                             r2_score, 
                             roc_curve, 
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
from pathlib import Path
from typing import Union, Optional, Literal

from ..path_manager import sanitize_filename, make_fullpath
from .._core import get_logger
from ..keys._keys import SHAPKeys


_LOGGER = get_logger("Ensemble Evaluation")


__all__ = [
    "evaluate_model_classification",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "evaluate_model_regression",
    "get_shap_values",
    "plot_learning_curves",
]


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
    cmap: Colormap = plt.cm.Blues, # type: ignore
    heatmap_cmap: str = "viridis"
) -> np.ndarray:
    """
    Evaluates a classification model, saves the classification report (text and heatmap) and the confusion matrix plot.

    Parameters:
        model: Trained classifier with .predict() method
        model_name: Identifier for the model
        save_dir: Directory where results are saved
        x_test_scaled: Feature matrix for test set
        single_y_test: True targets
        target_name: Target name
        figsize: Size of the confusion matrix figure (width, height)
        fontsize: Font size used for title, axis labels and ticks
        heatmap_cmap: Colormap for the classification report heatmap.
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
    sanitized_target_name = sanitize_filename(target_name)

    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(single_y_test, y_pred)
    
    # Generate report as dictionary for the heatmap
    report_dict = classification_report(
        single_y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=True
    )

    # text report to save
    report_text = classification_report(
        single_y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=False
    )

    # Save text report
    
    report_path = save_path / f"Classification_Report_{sanitized_target_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"{model_name} - {target_name}\t\tAccuracy: {accuracy:.2f}\n")
        f.write("Classification Report:\n")
        f.write(report_text) # type: ignore

    # 3. Create and save the classification report heatmap
    try:
        report_df = pd.DataFrame(report_dict).iloc[:-1, :].T
        plt.figure(figsize=figsize)
        sns.heatmap(report_df, annot=True, cmap=heatmap_cmap, fmt='.2f', 
                    annot_kws={"size": base_fontsize - 4}, vmin=0.0, vmax=1.0)
        plt.title(f"{model_name} - {target_name}", fontsize=base_fontsize)
        plt.xticks(fontsize=base_fontsize - 2)
        plt.yticks(fontsize=base_fontsize - 2)
        
        heatmap_path = save_path / f"Classification_Report_{sanitized_target_name}.svg"
        plt.savefig(heatmap_path, format="svg", bbox_inches="tight")
        plt.close()
    except Exception:
        _LOGGER.exception(f"Could not generate classification report heatmap for {target_name}:")

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
    disp.im_.set_clim(vmin=0.0, vmax=1.0)

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
    fig.savefig(fig_path, format="svg", bbox_inches="tight") # type: ignore
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
        if probabilities_or_model.ndim == 2: # type: ignore
            y_score = probabilities_or_model[:, 1] # type: ignore
        else:
            y_score = probabilities_or_model
            
    elif hasattr(probabilities_or_model, "predict_proba"):
        if input_features is None:
            _LOGGER.error("input_features must be provided when using a classifier.")
            raise ValueError()

        try:
            classes = probabilities_or_model.classes_ # type: ignore
            positive_class_index = list(classes).index(1)
        except (AttributeError, ValueError):
            positive_class_index = 1

        y_score = probabilities_or_model.predict_proba(input_features)[:, positive_class_index] # type: ignore

    else:
        _LOGGER.error("Unsupported type for 'probabilities_or_model'. Must be a NumPy array or a model with support for '.predict_proba()'.")
        raise TypeError()

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
    fig.savefig(full_save_path, bbox_inches="tight", format="svg") # type: ignore

    return fig


# Precision-Recall curve (Classification)
def plot_precision_recall_curve(
    true_labels: np.ndarray,
    probabilities_or_model: Union[np.ndarray, xgb.XGBClassifier, lgb.LGBMClassifier, object],
    model_name: str,
    target_name: str,
    save_directory: Union[str, Path],
    color: str = "teal",
    figure_size: tuple = (10, 10),
    linewidth: int = 2,
    base_fontsize: int = 24,
    input_features: Optional[np.ndarray] = None,
) -> plt.Figure: # type: ignore
    """
    Plots the Precision-Recall curve and computes Average Precision (AP) for binary classification.

    Parameters:
        true_labels: np.ndarray of shape (n_samples,), ground truth binary labels (0 or 1).
        probabilities_or_model: either predicted probabilities (ndarray), or a trained model with attribute `.predict_proba()`.
        model_name: Identifier for the model.
        target_name: Name of the target variable.
        save_directory: Path to the directory where the figure will be saved.
        color: str, color of the PR curve.
        figure_size: Tuple for figure size (width, height).
        linewidth: int, width of the plotted PR line.
        base_fontsize: int, base font size for titles and labels.
        input_features: np.ndarray, required if a model object is passed instead of probabilities.

    Returns:
        fig: matplotlib Figure object
    """
    # Determine predicted probabilities for the positive class
    if isinstance(probabilities_or_model, np.ndarray):
        if probabilities_or_model.ndim == 2:
            y_score = probabilities_or_model[:, 1]
        else:
            y_score = probabilities_or_model
            
    elif hasattr(probabilities_or_model, "predict_proba"):
        if input_features is None:
            _LOGGER.error("input_features must be provided when using a classifier.")
            raise ValueError()
        try:
            classes = probabilities_or_model.classes_ # type: ignore
            positive_class_index = list(classes).index(1)
        except (AttributeError, ValueError):
            positive_class_index = 1
        y_score = probabilities_or_model.predict_proba(input_features)[:, positive_class_index] # type: ignore
    else:
        _LOGGER.error("Unsupported type for 'probabilities_or_model'. Must be a NumPy array or a model with support for '.predict_proba()'.")
        raise TypeError()

    # Calculate PR curve and AP score
    precision, recall, _ = precision_recall_curve(true_labels, y_score)
    ap_score = average_precision_score(true_labels, y_score)

    # Plot
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(recall, precision, color=color, lw=linewidth, label=f"AP = {ap_score:.2f}")

    ax.set_title(f"{model_name} - {target_name}", fontsize=base_fontsize)
    ax.set_xlabel("Recall", fontsize=base_fontsize)
    ax.set_ylabel("Precision", fontsize=base_fontsize)
    ax.tick_params(axis='both', labelsize=base_fontsize)
    ax.legend(loc="lower left", fontsize=base_fontsize)
    ax.grid(True)
    fig.tight_layout()

    # Save figure
    save_path = make_fullpath(save_directory, make=True)
    sanitized_target_name = sanitize_filename(target_name)
    full_save_path = save_path / f"PR_Curve_{sanitized_target_name}.svg"
    fig.savefig(full_save_path, bbox_inches="tight", format="svg") # type: ignore
    plt.close(fig)

    return fig


# Calibration curve (classification)
def plot_calibration_curve(
    model,
    model_name: str,
    save_dir: Union[str, Path],
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    figure_size: tuple = (10, 10),
    base_fontsize: int = 24,
    n_bins: int = 15,
    line_color: str = 'darkorange'
) -> plt.Figure: # type: ignore
    """
    Plots the calibration curve (reliability diagram) for a classifier.

    Parameters:
        model: Trained classifier with .predict_proba() method.
        model_name: Identifier for the model.
        save_dir: Directory where the plot will be saved.
        x_test: Feature matrix for the test set.
        y_test: True labels for the test set.
        target_name: Name of the target variable.
        figure_size: Tuple for figure size (width, height).
        base_fontsize: Base font size for titles and labels.
        n_bins: Number of bins to discretize predictions into.

    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figure_size)
    
    # --- Step 1: Get probabilities from the estimator ---
    # We do this manually so we can pass them to from_predictions
    try:
        y_prob = model.predict_proba(x_test)
        # Use probabilities for the positive class (assuming binary)
        y_score = y_prob[:, 1]
    except Exception as e:
        _LOGGER.error(f"Could not get probabilities from model: {e}")
        plt.close(fig)
        return fig # Return empty figure

    # --- Step 2: Get binned data *without* plotting ---
    with plt.ioff(): 
        fig_temp, ax_temp = plt.subplots()
        cal_display_temp = CalibrationDisplay.from_predictions(
            y_test, 
            y_score, 
            n_bins=n_bins, 
            ax=ax_temp,
            name="temp"
        )
        line_x, line_y = cal_display_temp.line_.get_data() # type: ignore
        plt.close(fig_temp)

    # --- Step 3: Build the plot from scratch on ax ---

    # 3a. Plot the ideal diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # 3b. Use regplot for the regression line and its CI
    sns.regplot(
        x=line_x, 
        y=line_y,
        ax=ax,
        scatter=False,  # No scatter dots
        label=f"Calibration Curve ({n_bins} bins)",
        line_kws={
            'color': line_color,
            'linestyle': '--', 
            'linewidth': 2
        }
    )

    # --- Step 4: Apply original formatting ---
    ax.set_title(f"{model_name} - Reliability Curve for {target_name}", fontsize=base_fontsize)
    ax.tick_params(axis='both', labelsize=base_fontsize - 2)
    ax.set_xlabel("Mean Predicted Probability", fontsize=base_fontsize)
    ax.set_ylabel("Fraction of Positives", fontsize=base_fontsize)
    
    # Set limits
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    
    ax.legend(fontsize=base_fontsize - 4, loc='lower right')
    fig.tight_layout()

    # --- Step 5: Save figure (using original logic) ---
    save_path = make_fullpath(save_dir, make=True)
    sanitized_target_name = sanitize_filename(target_name)
    full_save_path = save_path / f"Calibration_Plot_{sanitized_target_name}.svg"
    fig.savefig(full_save_path, bbox_inches="tight", format="svg") # type: ignore
    plt.close(fig)

    return fig


# function to evaluate the model and save metrics (Regression)
def evaluate_model_regression(model, model_name: str, 
                               save_dir: Union[str,Path],
                               x_test_scaled: np.ndarray, single_y_test: np.ndarray, 
                               target_name: str,
                               figure_size: tuple = (12, 8),
                               alpha_transparency: float = 0.5,
                               base_fontsize: int = 24,
                               hist_bins: int = 30):
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
    residual_path = save_path / f"Residuals_Plot_{sanitized_target_name}.svg"
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
    plot_path = save_path / f"True_Vs_Predict_Plot_{sanitized_target_name}.svg"
    plt.savefig(plot_path, bbox_inches='tight', format="svg")
    plt.close()
    
    # Generate and save histogram of residuals
    plt.figure(figsize=figure_size)
    sns.histplot(residuals, bins=hist_bins, kde=True)
    plt.xlabel("Residual Value", fontsize=base_fontsize)
    plt.ylabel("Frequency", fontsize=base_fontsize)
    plt.title(f"{model_name} - Distribution of Residuals for {target_name}", fontsize=base_fontsize)
    plt.grid(True)
    plt.tight_layout()
    hist_path = save_path / f"Residuals_Distribution_{sanitized_target_name}.svg"
    plt.savefig(hist_path, bbox_inches='tight', format="svg")
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
    global_save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
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
                    
                # Save the summary data for the current class
                summary_save_path = global_save_path / f"SHAP_{sanitized_target_name}_{class_name}.csv"
                _save_summary_csv(
                    shap_values_for_summary=class_shap,
                    feature_names=feature_names,
                    save_path=summary_save_path
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
                
            # Save the summary data for the positive class
            shap_summary_filename = SHAPKeys.SAVENAME + ".csv"
            summary_save_path = global_save_path / shap_summary_filename
            _save_summary_csv(
                shap_values_for_summary=values,
                feature_names=feature_names,
                save_path=summary_save_path
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
        
        # Save the summary data to a CSV file
        shap_summary_filename = SHAPKeys.SAVENAME + ".csv"
        summary_save_path = global_save_path / shap_summary_filename
        _save_summary_csv(
            shap_values_for_summary=shap_values,
            feature_names=feature_names,
            save_path=summary_save_path
        )
        
    def _save_summary_csv(shap_values_for_summary: np.ndarray, feature_names: list[str], save_path: Path):
        """Calculates and saves the SHAP summary data to a CSV file."""
        mean_abs_shap = np.abs(shap_values_for_summary).mean(axis=0)
        
        # Create default feature names if none are provided
        current_feature_names = feature_names
        if current_feature_names is None:
            current_feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]
        
        summary_df = pd.DataFrame({
            SHAPKeys.FEATURE_COLUMN: feature_names,
            SHAPKeys.SHAP_VALUE_COLUMN: mean_abs_shap
        }).sort_values(SHAPKeys.SHAP_VALUE_COLUMN, ascending=False)
        
        summary_df.to_csv(save_path, index=False)
        # print(f"📝 SHAP summary data saved as '{save_path.name}'")
    
            
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


# Learning curves for regression and classification
def plot_learning_curves(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"],
    model_name: str,
    target_name: str,
    save_directory: Union[str, Path],
    cv: int = 5,
    n_jobs: int = -1,
    figure_size: tuple = (12, 8),
    base_fontsize: int = 24
):
    """
    Generates and saves a plot of the learning curves for a given estimator
    to diagnose bias vs. variance.
    
    Computationally expensive, requires a fresh, unfitted instance of the model.
    """
    save_path = make_fullpath(save_directory, make=True)
    sanitized_target_name = sanitize_filename(target_name)
    
    # Select scoring metric based on task
    scoring = "accuracy" if task == "classification" else "r2"

    train_sizes_abs, train_scores, val_scores, *_ = learning_curve(
        estimator, X, y, 
        cv=cv, 
        n_jobs=n_jobs, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=figure_size)
    ax.grid(True)

    # Plot the mean scores
    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes_abs, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Plot the standard deviation bands
    ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color="g")

    ax.set_title(f"{model_name} - Learning Curve for {target_name}", fontsize=base_fontsize)
    ax.set_xlabel("Training examples", fontsize=base_fontsize)
    ax.set_ylabel(f"Score ({scoring})", fontsize=base_fontsize)
    ax.legend(loc="best", fontsize=base_fontsize - 4)
    ax.tick_params(axis='both', labelsize=base_fontsize - 4)
    fig.tight_layout()

    # Save figure
    full_save_path = save_path / f"Learning_Curve_{sanitized_target_name}.svg"
    fig.savefig(full_save_path, bbox_inches="tight", format="svg")
    plt.close(fig)
    
