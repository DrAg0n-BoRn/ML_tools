import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    classification_report, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    roc_auc_score, 
    mean_squared_error,
    mean_absolute_error,
    r2_score, 
    median_absolute_error,
    precision_recall_curve,
    average_precision_score
)
import torch
import shap
from pathlib import Path
from .path_manager import make_fullpath
from ._logger import _LOGGER
from typing import Union, Optional, List
from ._script_info import _script_info
from .keys import SHAPKeys


__all__ = [
    "plot_losses", 
    "classification_metrics", 
    "regression_metrics",
    "shap_summary_plot",
    "plot_attention_importance"
]


def plot_losses(history: dict, save_dir: Union[str, Path]):
    """
    Plots training & validation loss curves from a history object.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss'.
        save_dir (str | Path): Directory to save the plot image.
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss and not val_loss:
        print("Warning: Loss history is empty or incomplete. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # Plot training loss only if data for it exists
    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, 'o-', label='Training Loss')
    
    # Plot validation loss only if data for it exists
    if val_loss:
        epochs = range(1, len(val_loss) + 1)
        ax.plot(epochs, val_loss, 'o-', label='Validation Loss')
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    save_path = save_dir_path / "loss_plot.svg"
    plt.savefig(save_path)
    _LOGGER.info(f"📉 Loss plot saved as '{save_path.name}'")

    plt.close(fig)


def classification_metrics(save_dir: Union[str, Path], y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, 
                           cmap: str = "Blues"):
    """
    Saves classification metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for ROC curve.
        cmap (str): Colormap for the confusion matrix.
        save_dir (str | Path): Directory to save plots.
    """
    print("--- Classification Report ---")
    # Generate report as both text and dictionary
    report_text: str = classification_report(y_true, y_pred) # type: ignore
    report_dict: dict = classification_report(y_true, y_pred, output_dict=True) # type: ignore
    print(report_text)
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # Save text report
    report_path = save_dir_path / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    _LOGGER.info(f"📝 Classification report saved as '{report_path.name}'")

    # --- Save Classification Report Heatmap ---
    try:
        plt.figure(figsize=(8, 6), dpi=100)
        sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap='viridis', fmt='.2f')
        plt.title("Classification Report")
        plt.tight_layout()
        heatmap_path = save_dir_path / "classification_report_heatmap.svg"
        plt.savefig(heatmap_path)
        _LOGGER.info(f"📊 Report heatmap saved as '{heatmap_path.name}'")
        plt.close()
    except Exception as e:
        _LOGGER.error(f"Could not generate classification report heatmap: {e}")

    # Save Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=100)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    cm_path = save_dir_path / "confusion_matrix.svg"
    plt.savefig(cm_path)
    _LOGGER.info(f"❇️ Confusion matrix saved as '{cm_path.name}'")
    plt.close(fig_cm)

    # Plotting logic for ROC and PR Curves
    if y_prob is not None and y_prob.ndim > 1 and y_prob.shape[1] >= 2:
        # Use probabilities of the positive class
        y_score = y_prob[:, 1]
        
        # --- Save ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6), dpi=100)
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True)
        roc_path = save_dir_path / "roc_curve.svg"
        plt.savefig(roc_path)
        _LOGGER.info(f"📈 ROC curve saved as '{roc_path.name}'")
        plt.close(fig_roc)

        # --- Save Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6), dpi=100)
        ax_pr.plot(recall, precision, label=f'AP = {ap_score:.2f}')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.legend(loc='lower left')
        ax_pr.grid(True)
        pr_path = save_dir_path / "pr_curve.svg"
        plt.savefig(pr_path)
        _LOGGER.info(f"📈 PR curve saved as '{pr_path.name}'")
        plt.close(fig_pr)
        
        # --- Save Calibration Plot ---
        if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
            y_score = y_prob[:, 1] # Use probabilities of the positive class
            
            fig_cal, ax_cal = plt.subplots(figsize=(8, 8), dpi=100)
            CalibrationDisplay.from_predictions(y_true, y_score, n_bins=15, ax=ax_cal)
            
            ax_cal.set_title('Reliability Curve')
            ax_cal.set_xlabel('Mean Predicted Probability')
            ax_cal.set_ylabel('Fraction of Positives')
            ax_cal.grid(True)
            plt.tight_layout()
            
            cal_path = save_dir_path / "calibration_plot.svg"
            plt.savefig(cal_path)
            _LOGGER.info(f"📈 Calibration plot saved as '{cal_path.name}'")
            plt.close(fig_cal)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Union[str, Path]):
    """
    Saves regression metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        save_dir (str | Path): Directory to save plots and report.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    report_lines = [
        "--- Regression Report ---",
        f"  Root Mean Squared Error (RMSE): {rmse:.4f}",
        f"  Mean Absolute Error (MAE):      {mae:.4f}",
        f"  Median Absolute Error (MedAE):  {medae:.4f}",
        f"  Coefficient of Determination (R²): {r2:.4f}"
    ]
    report_string = "\n".join(report_lines)
    # print(report_string)

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # Save text report
    report_path = save_dir_path / "regression_report.txt"
    report_path.write_text(report_string)
    _LOGGER.info(f"📝 Regression report saved as '{report_path.name}'")

    # Save residual plot
    residuals = y_true - y_pred
    fig_res, ax_res = plt.subplots(figsize=(8, 6), dpi=100)
    ax_res.scatter(y_pred, residuals, alpha=0.6)
    ax_res.axhline(0, color='red', linestyle='--')
    ax_res.set_xlabel("Predicted Values")
    ax_res.set_ylabel("Residuals")
    ax_res.set_title("Residual Plot")
    ax_res.grid(True)
    plt.tight_layout()
    res_path = save_dir_path / "residual_plot.svg"
    plt.savefig(res_path)
    _LOGGER.info(f"📈 Residual plot saved as '{res_path.name}'")
    plt.close(fig_res)

    # Save true vs predicted plot
    fig_tvp, ax_tvp = plt.subplots(figsize=(8, 6), dpi=100)
    ax_tvp.scatter(y_true, y_pred, alpha=0.6)
    ax_tvp.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax_tvp.set_xlabel('True Values')
    ax_tvp.set_ylabel('Predictions')
    ax_tvp.set_title('True vs. Predicted Values')
    ax_tvp.grid(True)
    plt.tight_layout()
    tvp_path = save_dir_path / "true_vs_predicted_plot.svg"
    plt.savefig(tvp_path)
    _LOGGER.info(f"📉 True vs. Predicted plot saved as '{tvp_path.name}'")
    plt.close(fig_tvp)
    
    # Save Histogram of Residuals
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6), dpi=100)
    sns.histplot(residuals, kde=True, ax=ax_hist)
    ax_hist.set_xlabel("Residual Value")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Distribution of Residuals")
    ax_hist.grid(True)
    plt.tight_layout()
    hist_path = save_dir_path / "residuals_histogram.svg"
    plt.savefig(hist_path)
    _LOGGER.info(f"📊 Residuals histogram saved as '{hist_path.name}'")
    plt.close(fig_hist)


def shap_summary_plot(model, 
                      background_data: Union[torch.Tensor,np.ndarray], 
                      instances_to_explain: Union[torch.Tensor,np.ndarray], 
                      feature_names: Optional[list[str]], 
                      save_dir: Union[str, Path]):
    """
    Calculates SHAP values and saves summary plots and data.

    Args:
        model (nn.Module): The trained PyTorch model.
        background_data (torch.Tensor): A sample of data for the explainer background.
        instances_to_explain (torch.Tensor): The specific data instances to explain.
        feature_names (list of str | None): Names of the features for plot labeling.
        save_dir (str | Path): Directory to save SHAP artifacts.
    """
    # everything to numpy
    if isinstance(background_data, np.ndarray):
        background_data_np = background_data
    else:
        background_data_np = background_data.numpy()
        
    if isinstance(instances_to_explain, np.ndarray):
        instances_to_explain_np = instances_to_explain
    else:
        instances_to_explain_np = instances_to_explain.numpy()
    
    # --- Data Validation Step ---
    if np.isnan(background_data_np).any() or np.isnan(instances_to_explain_np).any():
        _LOGGER.error("Input data for SHAP contains NaN values. Aborting explanation.")
        return
    
    print("\n--- SHAP Value Explanation ---")
    
    model.eval()
    model.cpu()
    
    # 1. Summarize the background data.
    # Summarize the background data using k-means. 10-50 clusters is a good starting point.
    background_summary = shap.kmeans(background_data_np, 30) 
    
    # 2. Define a prediction function wrapper that SHAP can use. It must take a numpy array and return a numpy array.
    def prediction_wrapper(x_np: np.ndarray) -> np.ndarray:
        # Convert numpy data to torch tensor
        x_torch = torch.from_numpy(x_np).float()
        with torch.no_grad():
            # Get model output
            output = model(x_torch)
        # Return as numpy array
        return output.cpu().numpy().flatten()

    # 3. Create the KernelExplainer
    explainer = shap.KernelExplainer(prediction_wrapper, background_summary)
    
    print("Calculating SHAP values with KernelExplainer...")
    shap_values = explainer.shap_values(instances_to_explain_np, l1_reg="aic")
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    plt.ioff()
    
    # Save Bar Plot
    bar_path = save_dir_path / "shap_bar_plot.svg"
    shap.summary_plot(shap_values, instances_to_explain_np, feature_names=feature_names, plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value Impact", labelpad=10)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(bar_path)
    _LOGGER.info(f"📊 SHAP bar plot saved as '{bar_path.name}'")
    plt.close()

    # Save Dot Plot
    dot_path = save_dir_path / "shap_dot_plot.svg"
    shap.summary_plot(shap_values, instances_to_explain_np, feature_names=feature_names, plot_type="dot", show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value Impact", labelpad=10)
    cb = plt.gcf().axes[-1]
    cb.set_ylabel("", size=1)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(dot_path)
    _LOGGER.info(f"📊 SHAP dot plot saved as '{dot_path.name}'")
    plt.close()

    # Save Summary Data to CSV
    shap_summary_filename = SHAPKeys.SAVENAME + ".csv"
    summary_path = save_dir_path / shap_summary_filename
    # Ensure the array is 1D before creating the DataFrame
    mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]
        
    summary_df = pd.DataFrame({
        SHAPKeys.FEATURE_COLUMN: feature_names,
        SHAPKeys.SHAP_VALUE_COLUMN: mean_abs_shap
    }).sort_values(SHAPKeys.SHAP_VALUE_COLUMN, ascending=False)
    
    summary_df.to_csv(summary_path, index=False)
    
    _LOGGER.info(f"📝 SHAP summary data saved as '{summary_path.name}'")
    plt.ion()


def plot_attention_importance(weights: List[torch.Tensor], feature_names: Optional[List[str]], save_dir: Union[str, Path], top_n: int = 10):
    """
    Aggregates attention weights and plots global feature importance.

    The plot shows the mean attention for each feature as a bar, with the
    standard deviation represented by error bars.

    Args:
        weights (List[torch.Tensor]): A list of attention weight tensors from each batch.
        feature_names (List[str] | None): Names of the features for plot labeling.
        save_dir (str | Path): Directory to save the plot and summary CSV.
        top_n (int): The number of top features to display in the plot.
    """
    if not weights:
        _LOGGER.error("Attention weights list is empty. Skipping importance plot.")
        return

    # --- Step 1: Aggregate data ---
    # Concatenate the list of tensors into a single large tensor
    full_weights_tensor = torch.cat(weights, dim=0)
    
    # Calculate mean and std dev across the batch dimension (dim=0)
    mean_weights = full_weights_tensor.mean(dim=0)
    std_weights = full_weights_tensor.std(dim=0)

    # --- Step 2: Create and save summary DataFrame ---
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(mean_weights))]
    
    summary_df = pd.DataFrame({
        'feature': feature_names,
        'mean_attention': mean_weights.numpy(),
        'std_attention': std_weights.numpy()
    }).sort_values('mean_attention', ascending=False)

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    summary_path = save_dir_path / "attention_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _LOGGER.info(f"📝 Attention summary data saved as '{summary_path.name}'")

    # --- Step 3: Create and save the plot for top N features ---
    plot_df = summary_df.head(top_n).sort_values('mean_attention', ascending=True)
    
    plt.figure(figsize=(10, 8), dpi=100)

    # Create horizontal bar plot with error bars
    plt.barh(
        y=plot_df['feature'],
        width=plot_df['mean_attention'],
        xerr=plot_df['std_attention'],
        align='center',
        alpha=0.7,
        ecolor='grey',
        capsize=3,
        color='cornflowerblue'
    )
    
    plt.title('Top Features by Attention')
    plt.xlabel('Average Attention Weight')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = save_dir_path / "attention_importance.svg"
    plt.savefig(plot_path)
    _LOGGER.info(f"📊 Attention importance plot saved as '{plot_path.name}'")
    plt.close()


def info():
    _script_info(__all__)
