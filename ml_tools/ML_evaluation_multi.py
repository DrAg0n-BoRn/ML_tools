import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    hamming_loss,
    jaccard_score
)
from pathlib import Path
from typing import Union, List

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info

__all__ = [
    "multi_target_regression_metrics",
    "multi_label_classification_metrics",
    "multi_target_shap_summary_plot",
]


def multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_dir: Union[str, Path]
):
    """
    Calculates and saves regression metrics for each target individually.

    For each target, this function saves a residual plot and a true vs. predicted plot.
    It also saves a single CSV file containing the key metrics (RMSE, MAE, R², MedAE)
    for all targets.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_targets).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_targets).
        target_names (List[str]): A list of names for the target variables.
        save_dir (str | Path): Directory to save plots and the report.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        _LOGGER.error("y_true and y_pred must be 2D arrays for multi-target regression.")
        raise ValueError()
    if y_true.shape != y_pred.shape:
        _LOGGER.error("Shapes of y_true and y_pred must match.")
        raise ValueError()
    if y_true.shape[1] != len(target_names):
        _LOGGER.error("Number of target names must match the number of columns in y_true.")
        raise ValueError()

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    metrics_summary = []

    _LOGGER.info("--- Multi-Target Regression Evaluation ---")

    for i, name in enumerate(target_names):
        print(f"  -> Evaluating target: '{name}'")
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        sanitized_name = sanitize_filename(name)

        # --- Calculate Metrics ---
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        medae = median_absolute_error(true_i, pred_i)
        metrics_summary.append({
            'target': name,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'median_abs_error': medae
        })

        # --- Save Residual Plot ---
        residuals = true_i - pred_i
        fig_res, ax_res = plt.subplots(figsize=(8, 6), dpi=100)
        ax_res.scatter(pred_i, residuals, alpha=0.6, edgecolors='k', s=50)
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals (True - Predicted)")
        ax_res.set_title(f"Residual Plot for '{name}'")
        ax_res.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        res_path = save_dir_path / f"residual_plot_{sanitized_name}.svg"
        plt.savefig(res_path)
        plt.close(fig_res)

        # --- Save True vs. Predicted Plot ---
        fig_tvp, ax_tvp = plt.subplots(figsize=(8, 6), dpi=100)
        ax_tvp.scatter(true_i, pred_i, alpha=0.6, edgecolors='k', s=50)
        ax_tvp.plot([true_i.min(), true_i.max()], [true_i.min(), true_i.max()], 'k--', lw=2)
        ax_tvp.set_xlabel('True Values')
        ax_tvp.set_ylabel('Predicted Values')
        ax_tvp.set_title(f'True vs. Predicted Values for "{name}"')
        ax_tvp.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        tvp_path = save_dir_path / f"true_vs_predicted_plot_{sanitized_name}.svg"
        plt.savefig(tvp_path)
        plt.close(fig_tvp)

    # --- Save Summary Report ---
    summary_df = pd.DataFrame(metrics_summary)
    report_path = save_dir_path / "regression_report_multi.csv"
    summary_df.to_csv(report_path, index=False)
    _LOGGER.info(f"Full regression report saved to '{report_path.name}'")


def multi_label_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_names: List[str],
    save_dir: Union[str, Path],
    threshold: float = 0.5
):
    """
    Calculates and saves classification metrics for each label individually.

    This function first computes overall multi-label metrics (Hamming Loss, Jaccard Score)
    and then iterates through each label to generate and save individual reports,
    confusion matrices, ROC curves, and Precision-Recall curves.

    Args:
        y_true (np.ndarray): Ground truth binary labels, shape (n_samples, n_labels).
        y_prob (np.ndarray): Predicted probabilities, shape (n_samples, n_labels).
        target_names (List[str]): A list of names for the labels.
        save_dir (str | Path): Directory to save plots and reports.
        threshold (float): The probability threshold to convert probabilities into
                           binary predictions for metrics like the confusion matrix.
    """
    if y_true.ndim != 2 or y_prob.ndim != 2:
        _LOGGER.error("y_true and y_prob must be 2D arrays for multi-label classification.")
        raise ValueError()
    if y_true.shape != y_prob.shape:
        _LOGGER.error("Shapes of y_true and y_prob must match.")
        raise ValueError()
    if y_true.shape[1] != len(target_names):
        _LOGGER.error("Number of target names must match the number of columns in y_true.")
        raise ValueError()

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # Generate binary predictions from probabilities
    y_pred = (y_prob >= threshold).astype(int)

    _LOGGER.info("--- Multi-Label Classification Evaluation ---")

    # --- Calculate and Save Overall Metrics ---
    h_loss = hamming_loss(y_true, y_pred)
    j_score_micro = jaccard_score(y_true, y_pred, average='micro')
    j_score_macro = jaccard_score(y_true, y_pred, average='macro')

    overall_report = (
        f"Overall Multi-Label Metrics (Threshold = {threshold}):\n"
        f"--------------------------------------------------\n"
        f"Hamming Loss: {h_loss:.4f}\n"
        f"Jaccard Score (micro): {j_score_micro:.4f}\n"
        f"Jaccard Score (macro): {j_score_macro:.4f}\n"
        f"--------------------------------------------------\n"
    )
    print(overall_report)
    overall_report_path = save_dir_path / "classification_report_overall.txt"
    overall_report_path.write_text(overall_report)

    # --- Per-Label Metrics and Plots ---
    for i, name in enumerate(target_names):
        print(f"  -> Evaluating label: '{name}'")
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        prob_i = y_prob[:, i]
        sanitized_name = sanitize_filename(name)

        # --- Save Classification Report for the label ---
        report_text = classification_report(true_i, pred_i)
        report_path = save_dir_path / f"classification_report_{sanitized_name}.txt"
        report_path.write_text(report_text) # type: ignore

        # --- Save Confusion Matrix ---
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=100)
        ConfusionMatrixDisplay.from_predictions(true_i, pred_i, cmap="Blues", ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix for '{name}'")
        cm_path = save_dir_path / f"confusion_matrix_{sanitized_name}.svg"
        plt.savefig(cm_path)
        plt.close(fig_cm)

        # --- Save ROC Curve ---
        fpr, tpr, _ = roc_curve(true_i, prob_i)
        auc = roc_auc_score(true_i, prob_i)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6), dpi=100)
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title(f'ROC Curve for "{name}"')
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right'); ax_roc.grid(True, linestyle='--', alpha=0.6)
        roc_path = save_dir_path / f"roc_curve_{sanitized_name}.svg"
        plt.savefig(roc_path)
        plt.close(fig_roc)

        # --- Save Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(true_i, prob_i)
        ap_score = average_precision_score(true_i, prob_i)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6), dpi=100)
        ax_pr.plot(recall, precision, label=f'AP = {ap_score:.2f}')
        ax_pr.set_title(f'Precision-Recall Curve for "{name}"')
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.legend(loc='lower left'); ax_pr.grid(True, linestyle='--', alpha=0.6)
        pr_path = save_dir_path / f"pr_curve_{sanitized_name}.svg"
        plt.savefig(pr_path)
        plt.close(fig_pr)

    _LOGGER.info(f"All individual label reports and plots saved to '{save_dir_path.name}'")


def multi_target_shap_summary_plot(
    model: torch.nn.Module,
    background_data: Union[torch.Tensor, np.ndarray],
    instances_to_explain: Union[torch.Tensor, np.ndarray],
    feature_names: List[str],
    target_names: List[str],
    save_dir: Union[str, Path]
):
    """
    Calculates SHAP values for a multi-target model and saves summary plots for each target.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        background_data (torch.Tensor | np.ndarray): A sample of data for the explainer background.
        instances_to_explain (torch.Tensor | np.ndarray): The specific data instances to explain.
        feature_names (List[str]): Names of the features for plot labeling.
        target_names (List[str]): Names of the output targets.
        save_dir (str | Path): Directory to save SHAP artifacts.
    """
    # Convert all data to numpy
    background_data_np = background_data.numpy() if isinstance(background_data, torch.Tensor) else background_data
    instances_to_explain_np = instances_to_explain.numpy() if isinstance(instances_to_explain, torch.Tensor) else instances_to_explain

    if np.isnan(background_data_np).any() or np.isnan(instances_to_explain_np).any():
        _LOGGER.error("Input data for SHAP contains NaN values. Aborting explanation.")
        return

    _LOGGER.info("--- Multi-Target SHAP Value Explanation ---")
    model.eval()
    model.cpu()

    # 1. Summarize the background data.
    background_summary = shap.kmeans(background_data_np, 30)

    # 2. Define a prediction function wrapper for the multi-target model.
    def prediction_wrapper(x_np: np.ndarray) -> np.ndarray:
        x_torch = torch.from_numpy(x_np).float()
        with torch.no_grad():
            output = model(x_torch)
        return output.cpu().numpy()

    # 3. Create the KernelExplainer.
    explainer = shap.KernelExplainer(prediction_wrapper, background_summary)

    print("Calculating SHAP values with KernelExplainer...")
    # For multi-output models, shap_values is a list of arrays.
    shap_values_list = explainer.shap_values(instances_to_explain_np, l1_reg="aic")

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    plt.ioff()

    # 4. Iterate through each target's SHAP values and generate plots.
    for i, target_name in enumerate(target_names):
        print(f"  -> Generating SHAP plots for target: '{target_name}'")
        shap_values_for_target = shap_values_list[i]
        sanitized_target_name = sanitize_filename(target_name)

        # Save Bar Plot for the target
        shap.summary_plot(shap_values_for_target, instances_to_explain_np, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for '{target_name}'")
        plt.tight_layout()
        bar_path = save_dir_path / f"shap_bar_plot_{sanitized_target_name}.svg"
        plt.savefig(bar_path)
        plt.close()

        # Save Dot Plot for the target
        shap.summary_plot(shap_values_for_target, instances_to_explain_np, feature_names=feature_names, plot_type="dot", show=False)
        plt.title(f"SHAP Feature Importance for '{target_name}'")
        plt.tight_layout()
        dot_path = save_dir_path / f"shap_dot_plot_{sanitized_target_name}.svg"
        plt.savefig(dot_path)
        plt.close()

    plt.ion()
    _LOGGER.info(f"All SHAP plots saved to '{save_dir_path.name}'")


def info():
    _script_info(__all__)
