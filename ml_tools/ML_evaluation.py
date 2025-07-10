import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    roc_auc_score, 
    mean_squared_error,
    mean_absolute_error,
    r2_score, 
    median_absolute_error
)
import torch
import shap
from pathlib import Path
from .utilities import make_fullpath
from .logger import _LOGGER
from typing import Union, Optional


__all__ = [
    "plot_losses", 
    "classification_metrics", 
    "regression_metrics",
    "shap_summary_plot"
]


def plot_losses(history: dict, save_dir: Optional[Union[str, Path]] = None):
    """
    Plots training & validation loss curves from a history object.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss'.
        save_dir (str | Path | None): Directory to save the plot image.
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
    
    if save_dir:
        save_dir_path = make_fullpath(save_dir, make=True)
        save_path = save_dir_path / "loss_plot.svg"
        plt.savefig(save_path)
        _LOGGER.info(f"📉 Loss plot saved as '{save_path.name}'")
    else:
        plt.show()
    plt.close(fig)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, 
                           cmap: str = "Blues", save_dir: Optional[Union[str, Path]] = None):
    """
    Displays and optionally saves classification metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for ROC curve.
        cmap (str): Colormap for the confusion matrix.
        save_dir (str | Path | None): Directory to save plots. If None, plots are shown not saved.
    """
    print("--- Classification Report ---")
    report: str = classification_report(y_true, y_pred) # type: ignore
    print(report)
    
    if save_dir:
        save_dir_path = make_fullpath(save_dir, make=True)
        # Save text report
        report_path = save_dir_path / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        _LOGGER.info(f"📝 Classification report saved as '{report_path.name}'")

        # Save Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=100)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        cm_path = save_dir_path / "confusion_matrix.svg"
        plt.savefig(cm_path)
        _LOGGER.info(f"❇️ Confusion matrix saved as '{cm_path.name}'")
        plt.close(fig_cm)

        # Save ROC Curve
        if y_prob is not None and y_prob.ndim > 1 and y_prob.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
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
    else:
        # Show plots if not saving
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap)
        plt.show()
        if y_prob is not None and y_prob.ndim > 1 and y_prob.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('ROC Curve')
            plt.show()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Optional[Union[str, Path]] = None):
    """
    Displays regression metrics and optionally saves plots and report.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        save_dir (str | None): Directory to save plots and report.
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
    print(report_string)

    if save_dir:
        save_dir_path = make_fullpath(save_dir, make=True)
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


def shap_summary_plot(model, background_data: torch.Tensor, instances_to_explain: torch.Tensor, 
                      feature_names: Optional[list[str]]=None, save_dir: Optional[Union[str, Path]] = None):
    """
    Calculates SHAP values and saves summary plots and data.

    Args:
        model (nn.Module): The trained PyTorch model.
        background_data (torch.Tensor): A sample of data for the explainer background.
        instances_to_explain (torch.Tensor): The specific data instances to explain.
        feature_names (list of str | None): Names of the features for plot labeling.
        save_dir (str | Path | None): Directory to save SHAP artifacts. If None, dot plot is shown.
    """
    print("\n--- SHAP Value Explanation ---")
    print("Calculating SHAP values... ")
    
    model.eval()
    model.cpu()
    
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(instances_to_explain)

    shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    if isinstance(shap_values, list):
        _LOGGER.info("Using SHAP values for the positive class (class 1) for plots.")

    if save_dir:
        save_dir_path = make_fullpath(save_dir, make=True)
        # Save Bar Plot
        bar_path = save_dir_path / "shap_bar_plot.svg"
        shap.summary_plot(shap_values_for_plot, instances_to_explain, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(bar_path)
        _LOGGER.info(f"📊 SHAP bar plot saved as '{bar_path.name}'")
        plt.close()

        # Save Dot Plot
        dot_path = save_dir_path / "shap_dot_plot.svg"
        shap.summary_plot(shap_values_for_plot, instances_to_explain, feature_names=feature_names, plot_type="dot", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(dot_path)
        _LOGGER.info(f"📊 SHAP dot plot saved as '{dot_path.name}'")
        plt.close()

        # Save Summary Data to CSV
        summary_path = save_dir_path / "shap_summary.csv"
        mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]
        summary_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap_value': mean_abs_shap
        }).sort_values('mean_abs_shap_value', ascending=False)
        summary_df.to_csv(summary_path, index=False)
        _LOGGER.info(f"📝 SHAP summary data saved as '{summary_path.name}'")
    else:
        _LOGGER.info("No save directory provided. Displaying SHAP dot plot.")
        shap.summary_plot(shap_values_for_plot, instances_to_explain, feature_names=feature_names, plot_type="dot")
