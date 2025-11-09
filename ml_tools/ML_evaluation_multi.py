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
from typing import Union, List, Literal, Optional
import warnings

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from ._keys import SHAPKeys
from .ML_configuration import (MultiTargetRegressionMetricsFormat,
                               _BaseRegressionFormat,
                               MultiLabelBinaryClassificationMetricsFormat,
                               _BaseMultiLabelFormat)


__all__ = [
    "multi_target_regression_metrics",
    "multi_label_classification_metrics",
    "multi_target_shap_summary_plot",
]

DPI_value = 250


def multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_dir: Union[str, Path],
    config: Optional[MultiTargetRegressionMetricsFormat] = None
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
        config (object): Formatting configuration object.
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
    
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseRegressionFormat()
    else:
        format_config = config

    # --- Set Matplotlib font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})

    _LOGGER.debug("--- Multi-Target Regression Evaluation ---")

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
        fig_res, ax_res = plt.subplots(figsize=(8, 6), dpi=DPI_value)
        ax_res.scatter(pred_i, residuals, 
                       alpha=format_config.scatter_alpha, 
                       edgecolors='k', 
                       s=50,
                       color=format_config.scatter_color) # Use config color
        ax_res.axhline(0, color=format_config.residual_line_color, linestyle='--') # Use config color
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals (True - Predicted)")
        ax_res.set_title(f"Residual Plot for '{name}'")
        ax_res.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        res_path = save_dir_path / f"residual_plot_{sanitized_name}.svg"
        plt.savefig(res_path)
        plt.close(fig_res)

        # --- Save True vs. Predicted Plot ---
        fig_tvp, ax_tvp = plt.subplots(figsize=(8, 6), dpi=DPI_value)
        ax_tvp.scatter(true_i, pred_i, 
                       alpha=format_config.scatter_alpha, 
                       edgecolors='k', 
                       s=50,
                       color=format_config.scatter_color) # Use config color
        ax_tvp.plot([true_i.min(), true_i.max()], [true_i.min(), true_i.max()], 
                    linestyle='--', 
                    lw=2,
                    color=format_config.ideal_line_color) # Use config color
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
    
    # --- Restore RC params ---
    plt.rcParams.update(original_rc_params)


def multi_label_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_names: List[str],
    save_dir: Union[str, Path],
    config: Optional[MultiLabelBinaryClassificationMetricsFormat] = None
):
    """
    Calculates and saves classification metrics for each label individually.

    This function first computes overall multi-label metrics (Hamming Loss, Jaccard Score)
    and then iterates through each label to generate and save individual reports,
    confusion matrices, ROC curves, and Precision-Recall curves.

    Args:
        y_true (np.ndarray): Ground truth binary labels, shape (n_samples, n_labels).
        y_pred (np.ndarray): Predicted binary labels, shape (n_samples, n_labels).
        y_prob (np.ndarray): Predicted probabilities, shape (n_samples, n_labels).
        target_names (List[str]): A list of names for the labels.
        save_dir (str | Path): Directory to save plots and reports.
        config (object): Formatting configuration object.
    """
    if y_true.ndim != 2 or y_prob.ndim != 2 or y_pred.ndim != 2:
        _LOGGER.error("y_true, y_pred, and y_prob must be 2D arrays for multi-label classification.")
        raise ValueError()
    if y_true.shape != y_prob.shape or y_true.shape != y_pred.shape:
        _LOGGER.error("Shapes of y_true, y_pred, and y_prob must match.")
        raise ValueError()
    if y_true.shape[1] != len(target_names):
        _LOGGER.error("Number of target names must match the number of columns in y_true.")
        raise ValueError()

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseMultiLabelFormat()
    else:
        format_config = config
    
    # y_pred is now passed in directly, no threshold needed.
    
    # --- Save current RC params and update font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})

    # _LOGGER.info("--- Multi-Label Classification Evaluation ---")

    # --- Calculate and Save Overall Metrics (using y_pred) ---
    h_loss = hamming_loss(y_true, y_pred)
    j_score_micro = jaccard_score(y_true, y_pred, average='micro')
    j_score_macro = jaccard_score(y_true, y_pred, average='macro')

    overall_report = (
        f"Overall Multi-Label Metrics:\n" # No threshold to report here
        f"--------------------------------------------------\n"
        f"Hamming Loss: {h_loss:.4f}\n"
        f"Jaccard Score (micro): {j_score_micro:.4f}\n"
        f"Jaccard Score (macro): {j_score_macro:.4f}\n"
        f"--------------------------------------------------\n"
    )
    # print(overall_report)
    overall_report_path = save_dir_path / "classification_report_overall.txt"
    overall_report_path.write_text(overall_report)

    # --- Per-Label Metrics and Plots ---
    for i, name in enumerate(target_names):
        print(f"  -> Evaluating label: '{name}'")
        true_i = y_true[:, i]
        pred_i = y_pred[:, i] # Use passed-in y_pred
        prob_i = y_prob[:, i] # Use passed-in y_prob
        sanitized_name = sanitize_filename(name)

        # --- Save Classification Report for the label (uses y_pred) ---
        report_text = classification_report(true_i, pred_i)
        report_path = save_dir_path / f"classification_report_{sanitized_name}.txt"
        report_path.write_text(report_text) # type: ignore

        # --- Save Confusion Matrix (uses y_pred) ---
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=DPI_value)
        disp_ = ConfusionMatrixDisplay.from_predictions(true_i, 
                                                pred_i, 
                                                cmap=format_config.cmap, # Use config cmap
                                                ax=ax_cm, 
                                                normalize='true',
                                                labels=[0, 1],
                                                display_labels=["Negative", "Positive"])
        
        disp_.im_.set_clim(vmin=0.0, vmax=1.0)
        
        # Turn off gridlines
        ax_cm.grid(False)
        
        # Manually update font size of cell texts
        for text in ax_cm.texts:
            text.set_fontsize(format_config.font_size) # Use config font_size

        fig_cm.tight_layout()
        
        ax_cm.set_title(f"Confusion Matrix for '{name}'")
        cm_path = save_dir_path / f"confusion_matrix_{sanitized_name}.svg"
        plt.savefig(cm_path)
        plt.close(fig_cm)

        # --- Save ROC Curve (uses y_prob) ---
        fpr, tpr, thresholds = roc_curve(true_i, prob_i)
        
        try:
            # Calculate Youden's J statistic (tpr - fpr)
            J = tpr - fpr
            # Find the index of the best threshold
            best_index = np.argmax(J)
            optimal_threshold = thresholds[best_index]
            best_tpr = tpr[best_index]
            best_fpr = fpr[best_index]
            
            # Define the filename
            threshold_filename = f"best_threshold_{sanitized_name}.txt"
            threshold_path = save_dir_path / threshold_filename
            
            # The class name is the target_name for this label
            class_name = name 
            
            # Create content for the file
            file_content = (
                f"Optimal Classification Threshold (Youden's J Statistic)\n"
                f"Class/Label: {class_name}\n"
                f"--------------------------------------------------\n"
                f"Threshold: {optimal_threshold:.6f}\n"
                f"True Positive Rate (TPR): {best_tpr:.6f}\n"
                f"False Positive Rate (FPR): {best_fpr:.6f}\n"
            )
            
            threshold_path.write_text(file_content, encoding="utf-8")
            _LOGGER.info(f"💾 Optimal threshold for '{name}' saved to '{threshold_path.name}'")

        except Exception as e:
            _LOGGER.warning(f"Could not calculate or save optimal threshold for '{name}': {e}")
        
        auc = roc_auc_score(true_i, prob_i)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6), dpi=DPI_value)
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color=format_config.ROC_PR_line) # Use config color
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title(f'ROC Curve for "{name}"')
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right'); ax_roc.grid(True, linestyle='--', alpha=0.6)
        roc_path = save_dir_path / f"roc_curve_{sanitized_name}.svg"
        plt.savefig(roc_path)
        plt.close(fig_roc)

        # --- Save Precision-Recall Curve (uses y_prob) ---
        precision, recall, _ = precision_recall_curve(true_i, prob_i)
        ap_score = average_precision_score(true_i, prob_i)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6), dpi=DPI_value)
        ax_pr.plot(recall, precision, label=f'AP = {ap_score:.2f}', color=format_config.ROC_PR_line) # Use config color
        ax_pr.set_title(f'Precision-Recall Curve for "{name}"')
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.legend(loc='lower left'); ax_pr.grid(True, linestyle='--', alpha=0.6)
        pr_path = save_dir_path / f"pr_curve_{sanitized_name}.svg"
        plt.savefig(pr_path)
        plt.close(fig_pr)
        
    # restore RC params
    plt.rcParams.update(original_rc_params)

    _LOGGER.info(f"All individual label reports and plots saved to '{save_dir_path.name}'")


def multi_target_shap_summary_plot(
    model: torch.nn.Module,
    background_data: Union[torch.Tensor, np.ndarray],
    instances_to_explain: Union[torch.Tensor, np.ndarray],
    feature_names: List[str],
    target_names: List[str],
    save_dir: Union[str, Path],
    device: torch.device = torch.device('cpu'),
    explainer_type: Literal['deep', 'kernel'] = 'kernel'
):
    """
    Calculates SHAP values for a multi-target model and saves summary plots and data for each target.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        background_data (torch.Tensor | np.ndarray): A sample of data for the explainer background.
        instances_to_explain (torch.Tensor | np.ndarray): The specific data instances to explain.
        feature_names (List[str]): Names of the features for plot labeling.
        target_names (List[str]): Names of the output targets.
        save_dir (str | Path): Directory to save SHAP artifacts.
        device (torch.device): The torch device for SHAP calculations.
        explainer_type (Literal['deep', 'kernel']): The explainer to use.
            - 'deep': Uses shap.DeepExplainer. Fast and efficient.
            - 'kernel': Uses shap.KernelExplainer. Model-agnostic but slow and memory-intensive.
    """
    _LOGGER.info(f"--- Multi-Target SHAP Value Explanation (Using: {explainer_type.upper()}Explainer) ---")
    model.eval()
    # model.cpu()

    shap_values_list = None
    instances_to_explain_np = None

    if explainer_type == 'deep':
        # --- 1. Use DeepExplainer ---
        
        # Ensure data is torch.Tensor
        if isinstance(background_data, np.ndarray):
            background_data = torch.from_numpy(background_data).float()
        if isinstance(instances_to_explain, np.ndarray):
            instances_to_explain = torch.from_numpy(instances_to_explain).float()
            
        if torch.isnan(background_data).any() or torch.isnan(instances_to_explain).any():
            _LOGGER.error("Input data for SHAP contains NaN values. Aborting explanation.")
            return

        background_data = background_data.to(device)
        instances_to_explain = instances_to_explain.to(device)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            explainer = shap.DeepExplainer(model, background_data)
            
        # print("Calculating SHAP values with DeepExplainer...")
        # DeepExplainer returns a list of arrays for multi-output models
        shap_values_list = explainer.shap_values(instances_to_explain)
        instances_to_explain_np = instances_to_explain.cpu().numpy()

    elif explainer_type == 'kernel':
        # --- 2. Use KernelExplainer  ---
        _LOGGER.warning(
            "KernelExplainer is memory-intensive and slow. Consider reducing the number of instances to explain if the process terminates unexpectedly."
        )
        
        # Convert all data to numpy
        background_data_np = background_data.numpy() if isinstance(background_data, torch.Tensor) else background_data
        instances_to_explain_np = instances_to_explain.numpy() if isinstance(instances_to_explain, torch.Tensor) else instances_to_explain

        if np.isnan(background_data_np).any() or np.isnan(instances_to_explain_np).any():
            _LOGGER.error("Input data for SHAP contains NaN values. Aborting explanation.")
            return

        background_summary = shap.kmeans(background_data_np, 30)

        def prediction_wrapper(x_np: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x_np).float().to(device)
            with torch.no_grad():
                output = model(x_torch)
            return output.cpu().numpy() # Return full multi-output array

        explainer = shap.KernelExplainer(prediction_wrapper, background_summary)
        # print("Calculating SHAP values with KernelExplainer...")
        # KernelExplainer also returns a list of arrays for multi-output models
        shap_values_list = explainer.shap_values(instances_to_explain_np, l1_reg="aic")
        # instances_to_explain_np is already set
        
    else:
        _LOGGER.error(f"Invalid explainer_type: '{explainer_type}'. Must be 'deep' or 'kernel'.")
        raise ValueError("Invalid explainer_type")

    # --- 3. Plotting and Saving (Common Logic) ---
    
    if shap_values_list is None or instances_to_explain_np is None:
        _LOGGER.error("SHAP value calculation failed. Aborting plotting.")
        return
        
    # Ensure number of SHAP value arrays matches number of target names
    if len(shap_values_list) != len(target_names):
        _LOGGER.error(
            f"SHAP explanation mismatch: Model produced {len(shap_values_list)} "
            f"outputs, but {len(target_names)} target_names were provided."
        )
        return

    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    plt.ioff()

    # Iterate through each target's SHAP values and generate plots.
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
        if plt.gcf().axes and len(plt.gcf().axes) > 1:
            cb = plt.gcf().axes[-1]
            cb.set_ylabel("", size=1)
        plt.tight_layout()
        dot_path = save_dir_path / f"shap_dot_plot_{sanitized_target_name}.svg"
        plt.savefig(dot_path)
        plt.close()
        
        # --- Save Summary Data to CSV for this target ---
        shap_summary_filename = f"{SHAPKeys.SAVENAME}_{sanitized_target_name}.csv"
        summary_path = save_dir_path / shap_summary_filename
        
        # For a specific target, shap_values_for_target is just a 2D array
        mean_abs_shap = np.abs(shap_values_for_target).mean(axis=0).flatten()
        
        summary_df = pd.DataFrame({
            SHAPKeys.FEATURE_COLUMN: feature_names,
            SHAPKeys.SHAP_VALUE_COLUMN: mean_abs_shap
        }).sort_values(SHAPKeys.SHAP_VALUE_COLUMN, ascending=False)
        
        summary_df.to_csv(summary_path, index=False)
        
    plt.ion()
    _LOGGER.info(f"All SHAP plots saved to '{save_dir_path.name}'")


def info():
    _script_info(__all__)
