import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    hamming_loss,
    jaccard_score,
    f1_score
)
from pathlib import Path
from typing import Union, Optional
import warnings

from ..ML_configuration._config_metrics import (_BaseMultiLabelFormat,
                                         _BaseClassificationFormat,
                                        FormatBinaryClassificationMetrics,
                                        FormatMultiClassClassificationMetrics,
                                        FormatBinaryImageClassificationMetrics,
                                        FormatMultiClassImageClassificationMetrics,
                                        FormatMultiLabelBinaryClassificationMetrics)

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from .._helpers import wrap_text
from ..keys._config import _EvaluationConfig

from ._radar_plots import (
    mpl_to_plotly_rgba,
    calculate_smart_font_size,
    calculate_smart_margin_left_right,
    save_radar_chart
)


_LOGGER = get_logger("Classification Metrics")


__all__ = [
    "classification_metrics",
    "multi_label_classification_metrics",
]


DPI_value = _EvaluationConfig.DPI
CLASSIFICATION_PLOT_SIZE = _EvaluationConfig.CLASSIFICATION_PLOT_SIZE


# =====================================================================
# PRIVATE HELPER FUNCTIONS
# =====================================================================

def _save_classification_report_heatmap(report_dict: dict, save_path: Path, format_config, cm_font_size: int, cm_tick_size: int, title: str = "Classification Report Heatmap"):
    """Generates and saves a seaborn heatmap for the classification report."""
    try:
        report_df = pd.DataFrame(report_dict)
        report_df = report_df.drop(columns=['accuracy'], errors='ignore')
        if 'support' in report_df.index:
            report_df = report_df.drop(index='support')

        plot_df = report_df.T
        fig_height = max(5.0, len(plot_df.index) * 0.5 + 4.0)
        fig_width = _EvaluationConfig.HEATMAP_WIDTH 

        fig_heat, ax_heat = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI_value)

        sns.heatmap(plot_df, annot=True, cmap=format_config.cmap, fmt='.2f', vmin=0.0, vmax=1.0, cbar_kws={'shrink': 0.9}, ax=ax_heat)
        
        ax_heat.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
        
        for text in ax_heat.texts:
            text.set_fontsize(cm_tick_size)
            
        cbar = ax_heat.collections[0].colorbar
        
        if cbar is None:
            _LOGGER.warning("Colorbar not found in the heatmap. Skipping colorbar font size adjustment.")
        else:        
            cbar.ax.tick_params(labelsize=cm_tick_size - 4) 

        ax_heat.tick_params(axis='x', labelsize=cm_tick_size, pad=_EvaluationConfig.LABEL_PADDING)
        ax_heat.tick_params(axis='y', labelsize=cm_tick_size, pad=_EvaluationConfig.LABEL_PADDING, rotation=0) 

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig_heat)
        _LOGGER.info(f"📊 Report heatmap saved as '{save_path.name}'")
    except Exception as e:
        _LOGGER.error(f"Could not generate classification report heatmap: {e}")

def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[list], display_labels: Optional[list], 
                           save_path: Path, format_config, cm_font_size: int, cm_tick_size: int, title: str = "Confusion Matrix"):
    """Generates and saves a dynamically sized confusion matrix."""
    n_classes = len(labels) if labels is not None else len(np.unique(y_true))
    fig_w = max(9, n_classes * 0.8 + 3)
    fig_h = max(8, n_classes * 0.8 + 2)
    
    fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI_value)
    disp_ = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=format_config.cmap, ax=ax_cm, 
                                            normalize='true', labels=labels, display_labels=display_labels, colorbar=False)
    
    disp_.im_.set_clim(vmin=0.0, vmax=1.0)
    ax_cm.grid(False)
    
    final_font_size = cm_font_size + 2 if n_classes <= 2 else cm_font_size - n_classes
    for text in ax_cm.texts:
        text.set_fontsize(final_font_size)
    
    ax_cm.tick_params(axis='x', labelsize=cm_tick_size)
    ax_cm.tick_params(axis='y', labelsize=cm_tick_size)
    
    if n_classes > 3:
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    ax_cm.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size + 2)
    ax_cm.set_xlabel(ax_cm.get_xlabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
    ax_cm.set_ylabel(ax_cm.get_ylabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
    
    cbar = fig_cm.colorbar(disp_.im_, ax=ax_cm, shrink=0.8)
    cbar.ax.tick_params(labelsize=cm_tick_size) 
    
    fig_cm.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig_cm)
    _LOGGER.info(f"❇️ Confusion matrix saved as '{save_path.name}'")

def _save_roc_and_threshold(y_true_binary: np.ndarray, y_score: np.ndarray, class_name: str, 
                            save_dir_path: Path, save_suffix: str, format_config, auc: float, title: str):
    """Generates and saves the ROC curve and the optimal Youden's J threshold."""
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    
    try:
        J = tpr - fpr
        best_index = np.argmax(J)
        optimal_threshold = thresholds[best_index]
        
        threshold_path = save_dir_path / f"best_threshold{save_suffix}.txt"
        file_content = (
            f"Optimal Classification Threshold (Youden's J Statistic)\n"
            f"Class: {class_name}\n"
            f"--------------------------------------------------\n"
            f"Threshold: {optimal_threshold:.6f}\n"
            f"True Positive Rate (TPR): {tpr[best_index]:.6f}\n"
            f"False Positive Rate (FPR): {fpr[best_index]:.6f}\n"
        )
        threshold_path.write_text(file_content, encoding="utf-8")
        _LOGGER.info(f"📝 Optimal threshold saved as '{threshold_path.name}'")
    except Exception as e:
        _LOGGER.warning(f"Could not calculate or save optimal threshold: {e}")
        
    fig_roc, ax_roc = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
    ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color=format_config.ROC_PR_line)
    ax_roc.plot([0, 1], [0, 1], 'k--')

    ax_roc.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    ax_roc.set_xlabel('False Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax_roc.set_ylabel('True Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    
    ax_roc.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax_roc.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax_roc.legend(loc='lower right', fontsize=format_config.legend_size)
    
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_dir_path / f"roc_curve{save_suffix}.svg", bbox_inches='tight')
    plt.close(fig_roc)

def _save_pr_curve(y_true_binary: np.ndarray, y_score: np.ndarray, save_path: Path, format_config, ap_score: float, title: str):
    """Generates and saves the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
    
    fig_pr, ax_pr = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
    ax_pr.plot(recall, precision, label=f'Avg Precision = {ap_score:.2f}', color=format_config.ROC_PR_line)

    ax_pr.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    ax_pr.set_xlabel('Recall', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax_pr.set_ylabel('Precision', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    
    ax_pr.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax_pr.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax_pr.legend(loc='lower left', fontsize=format_config.legend_size)
    
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig_pr)

def _save_calibration_plot(y_true_binary: np.ndarray, y_score: np.ndarray, save_path: Path, format_config, title: str):
    """Generates and saves the model Calibration (reliability) plot."""
    fig_cal, ax_cal = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
    
    user_chosen_bins = format_config.calibration_bins
    if not isinstance(user_chosen_bins, int) or user_chosen_bins <= 0:
        n_samples = y_true_binary.shape[0]
        if n_samples < 200: dynamic_bins = 5
        elif n_samples < 1000: dynamic_bins = 10
        else: dynamic_bins = 15
    else:
        dynamic_bins = user_chosen_bins
    
    prob_true, prob_pred = calibration_curve(y_true_binary, y_score, n_bins=dynamic_bins)
    prob_true = np.concatenate(([0.0], prob_true, [1.0]))
    prob_pred = np.concatenate(([0.0], prob_pred, [1.0]))

    ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax_cal.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Model calibration", color=format_config.ROC_PR_line)
    
    ax_cal.set_title(title, pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
    ax_cal.set_xlabel('Mean Predicted Probability', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    ax_cal.set_ylabel('Fraction of Positives', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
    
    ax_cal.set_ylim(0.0, 1.0) 
    ax_cal.set_xlim(0.0, 1.0)
    
    ax_cal.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax_cal.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax_cal.legend(loc='lower right', fontsize=format_config.legend_size)
    
    ax_cal.spines['top'].set_visible(False)
    ax_cal.spines['right'].set_visible(False)
    ax_cal.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig_cal)


# =====================================================================
# MAIN FUNCTIONS
# =====================================================================

def classification_metrics(save_dir: Union[str, Path], 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None, 
                           class_map: Optional[dict[str,int]] = None,
                           config: Optional[Union[FormatBinaryClassificationMetrics,
                                                FormatMultiClassClassificationMetrics,
                                                FormatBinaryImageClassificationMetrics,
                                                FormatMultiClassImageClassificationMetrics]] = None):
    """
    Saves classification metrics and plots for Binary and Multi-Class scenarios.
    
    Args:
        save_dir (str | Path): Directory to save plots.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities for ROC curve.
        class_map (dict): Optional mapping of class names to integer labels.
        config (object): Formatting configuration object.
    """
    format_config = config if config is not None else _BaseClassificationFormat()
    
    cm_font_size = format_config.cm_font_size
    cm_tick_size = cm_font_size - 4
    
    map_labels, map_display_labels, plot_display_labels = None, None, None
    if class_map:
        try:
            sorted_items = sorted(class_map.items(), key=lambda item: item[1])
            map_labels = [item[1] for item in sorted_items]
            map_display_labels = [item[0] for item in sorted_items]
            plot_display_labels = [wrap_text(mapped_name) for mapped_name in map_display_labels]
        except Exception as e:
            _LOGGER.warning(f"Could not parse 'class_map': {e}")

    report_text: str = classification_report(y_true, y_pred, labels=map_labels, target_names=map_display_labels, zero_division=0) # type: ignore
    report_dict: dict = classification_report(y_true, y_pred, output_dict=True, labels=map_labels, target_names=plot_display_labels, zero_division=0) # type: ignore
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    report_path = save_dir_path / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    _LOGGER.info(f"📝 Classification report saved as '{report_path.name}'")

    _save_classification_report_heatmap(report_dict, save_dir_path / "classification_report_heatmap.svg", format_config, cm_font_size, cm_tick_size)
    _save_confusion_matrix(y_true, y_pred, map_labels, plot_display_labels, save_dir_path / "confusion_matrix.svg", format_config, cm_font_size, cm_tick_size)

    if y_prob is not None and y_prob.ndim == 2:
        num_classes = y_prob.shape[1]
        
        if num_classes == 2:
            class_indices_to_plot = [1]
            plot_titles, save_suffixes = [""], [""]
            _LOGGER.debug("Generating binary classification plots (ROC, PR, Calibration).")
        elif num_classes > 2:
            _LOGGER.debug(f"Generating One-vs-Rest plots for {num_classes} classes.")
            class_indices_to_plot = list(range(num_classes))
            
            if map_display_labels and len(map_display_labels) == num_classes:
                safe_names = [sanitize_filename(name) for name in map_display_labels]
                plot_titles = [f"'{name}'" for name in map_display_labels]
                save_suffixes = [f"_{safe_names[i]}" for i in class_indices_to_plot]
            else:
                plot_titles = [f"'Class {i}'" for i in class_indices_to_plot]
                save_suffixes = [f"_class_{i}" for i in class_indices_to_plot]
        else:
            _LOGGER.warning(f"Probability array has invalid shape {y_prob.shape}. Skipping ROC/PR/Calibration plots.")
            return

        for i, class_index in enumerate(class_indices_to_plot):
            plot_title = plot_titles[i]
            save_suffix = save_suffixes[i]
            y_score = y_prob[:, class_index]
            y_true_binary = (y_true == class_index).astype(int)
            
            class_name = map_display_labels[class_index] + (" (vs. Rest)" if num_classes > 2 else "") if (map_display_labels and class_index < len(map_display_labels)) else (plot_title.strip() or "Binary Positive Class")
            
            auc: float = roc_auc_score(y_true_binary, y_score) # type: ignore
            ap_score: float = average_precision_score(y_true_binary, y_score) # type: ignore
            
            roc_master_title = "Receiver Operating Characteristic" + (f"\n{plot_title}" if plot_title.strip() else "")
            pr_master_title = "Precision-Recall Curve" + (f"\n{plot_title}" if plot_title.strip() else "")
            cal_master_title = "Calibration Curve" + (f"\n{plot_title}" if plot_title.strip() else "")
            
            _save_roc_and_threshold(y_true_binary, y_score, class_name, save_dir_path, save_suffix, format_config, auc, roc_master_title)
            _save_pr_curve(y_true_binary, y_score, save_dir_path / f"pr_curve{save_suffix}.svg", format_config, ap_score, pr_master_title)
            _save_calibration_plot(y_true_binary, y_score, save_dir_path / f"calibration_plot{save_suffix}.svg", format_config, cal_master_title)
        
        _LOGGER.info(f"📈 Saved {len(class_indices_to_plot)} sets of ROC, Precision-Recall, and Calibration plots.")


def multi_label_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_names: list[str],
    save_dir: Union[str, Path],
    config: Optional[FormatMultiLabelBinaryClassificationMetrics] = None
):
    """
    Calculates and saves classification metrics for each label individually in a multilabel binary setting.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels (2D array).
        y_pred (np.ndarray): Predicted binary labels (2D array).
        y_prob (np.ndarray): Predicted probabilities for each label (2D array).
        target_names (list[str]): List of label names corresponding to columns in y_true/y_pred/y_prob.
        save_dir (Union[str, Path]): Directory to save reports and plots.
        config (Optional[FormatMultiLabelBinaryClassificationMetrics]): Optional formatting configuration object.
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
    format_config = config if config is not None else _BaseMultiLabelFormat()
    
    cm_font_size = format_config.cm_font_size
    cm_tick_size = cm_font_size - 4
    
    auc_scores, ap_scores, f1_scores = [], [], []

    # Overall Metrics
    h_loss = hamming_loss(y_true, y_pred)
    j_score_micro = jaccard_score(y_true, y_pred, average='micro')
    j_score_macro = jaccard_score(y_true, y_pred, average='macro')

    overall_report = (
        f"Overall Multi-Label Metrics:\n"
        f"--------------------------------------------------\n"
        f"Hamming Loss: {h_loss:.4f}\n"
        f"Jaccard Score (micro): {j_score_micro:.4f}\n"
        f"Jaccard Score (macro): {j_score_macro:.4f}\n"
        f"--------------------------------------------------\n"
    )
    (save_dir_path / "classification_report.txt").write_text(overall_report)

    # Full Heatmap
    full_report_dict: dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0) # type: ignore
    _save_classification_report_heatmap(full_report_dict, save_dir_path / "classification_report_heatmap.svg", format_config, cm_font_size, cm_tick_size)

    # Per-Label Logic
    for i, name in enumerate(target_names):
        name = name.strip()
        true_i, pred_i, prob_i = y_true[:, i], y_pred[:, i], y_prob[:, i]
        sanitized_name = sanitize_filename(name)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_val: float = roc_auc_score(true_i, prob_i) if len(np.unique(true_i)) > 1 else 0.0 # type: ignore
            f1_val: float = f1_score(true_i, pred_i, zero_division=0) # type: ignore
            ap_val: float = average_precision_score(true_i, prob_i) if len(np.unique(true_i)) > 1 else 0.0 # type: ignore
        
        auc_scores.append(0.0 if np.isnan(auc_val) else round(auc_val, 4))
        f1_scores.append(0.0 if np.isnan(f1_val) else round(f1_val, 4))
        ap_scores.append(0.0 if np.isnan(ap_val) else round(ap_val, 4))

        report_text = classification_report(true_i, pred_i, zero_division=0)
        (save_dir_path / f"classification_report_{sanitized_name}.txt").write_text(report_text) # type: ignore

        _save_confusion_matrix(true_i, pred_i, [0, 1], ["Negative", "Positive"], save_dir_path / f"confusion_matrix_{sanitized_name}.svg", 
                               format_config, format_config.font_size, format_config.ytick_size, title=f"Confusion Matrix\n'{name}'")
        
        _save_roc_and_threshold(true_i, prob_i, name, save_dir_path, f"_{sanitized_name}", format_config, auc_val, title=f"Receiver Operating Characteristic\n'{name}'")
        _save_pr_curve(true_i, prob_i, save_dir_path / f"pr_curve_{sanitized_name}.svg", format_config, ap_val, title=f"Precision-Recall Curve\n'{name}'")
        _save_calibration_plot(true_i, prob_i, save_dir_path / f"calibration_plot_{sanitized_name}.svg", format_config, title=f"Calibration Curve\n'{name}'")
        
    # Radar Charts
    if len(target_names) > 2:
        radar_dir = save_dir_path / "radar_charts"
        radar_dir.mkdir(exist_ok=True)
        
        line_hex = mcolors.to_hex(format_config.ROC_PR_line)
        fill_rgba = mpl_to_plotly_rgba(format_config.ROC_PR_line, 0.15) 
        
        smart_font_size = calculate_smart_font_size(len(target_names), format_config.font_size)
        dynamic_margin = calculate_smart_margin_left_right(max([len(str(name)) for name in target_names]))
        
        save_radar_chart(auc_scores, target_names, line_hex, fill_rgba, "ROC AUC across Labels", radar_dir / "auc_radar", dynamic_margin, smart_font_size)
        save_radar_chart(ap_scores, target_names, line_hex, fill_rgba, "Average Precision across Labels", radar_dir / "ap_radar", dynamic_margin, smart_font_size)
        save_radar_chart(f1_scores, target_names, line_hex, fill_rgba, "F1-Score across Labels", radar_dir / "f1_radar", dynamic_margin, smart_font_size)
                        
        _LOGGER.info(f"🌀 Radar charts saved to '{radar_dir.name}'")

    _LOGGER.info(f"All individual label reports and plots saved to '{save_dir_path.name}'")
