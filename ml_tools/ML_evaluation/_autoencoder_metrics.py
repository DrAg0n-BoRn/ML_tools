import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
from pathlib import Path
from typing import Union, Optional
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

from ..ML_configuration._metrics import FormatAutoencoderMetrics

from ..keys._keys import _EvaluationConfig
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

from ._helpers import check_and_abbreviate_name


_LOGGER = get_logger("AutoencoderMetrics")


__all__ = [
    "autoencoder_metrics"
]

DPI_value = _EvaluationConfig.DPI
REGRESSION_PLOT_SIZE = _EvaluationConfig.REGRESSION_PLOT_SIZE
CLASSIFICATION_PLOT_SIZE = _EvaluationConfig.CLASSIFICATION_PLOT_SIZE


def autoencoder_metrics(
    y_true_num: Optional[np.ndarray],
    y_pred_num: Optional[np.ndarray],
    num_target_names: Optional[list[str]],
    cat_true_list: Optional[list[np.ndarray]],
    cat_pred_list: Optional[list[np.ndarray]],
    cat_prob_list: Optional[list[np.ndarray]],
    cat_target_names: Optional[list[str]],
    cat_class_maps: Optional[list[Optional[dict[str, int]]]],
    save_dir: Union[str, Path],
    config: Optional[FormatAutoencoderMetrics] = None
):
    """
    Evaluates Autoencoder reconstruction performance for mixed tabular data.
    Generates global reconstruction metrics and per-feature summary tables.
    """
    if config is None:
        format_config = FormatAutoencoderMetrics()
    else:
        format_config = config
        
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")

    overall_report_lines = ["--- Autoencoder Global Reconstruction Report ---"]

    # 1. Evaluate Individual Numerical Features
    num_report_lines = _evaluate_numerical_features(
        y_true_num, y_pred_num, num_target_names, save_dir_path, format_config
    )
    overall_report_lines.extend(num_report_lines)

    # 2. Evaluate Individual Categorical Features
    cat_report_lines = _evaluate_categorical_features(
        cat_true_list, cat_pred_list, cat_prob_list, cat_target_names, cat_class_maps, save_dir_path, format_config
    )
    overall_report_lines.extend(cat_report_lines)

    # 3. Global Overview Plots
    _plot_global_feature_performance(
        y_true_num, y_pred_num, num_target_names,
        cat_true_list, cat_pred_list, cat_target_names,
        save_dir_path, format_config
    )
    
    _plot_sample_error_scatter(
        y_true_num, y_pred_num, cat_true_list, cat_pred_list, save_dir_path, format_config
    )
    
    _plot_error_correlation_heatmap(
        y_true_num, y_pred_num, num_target_names, save_dir_path, format_config
    )
    
    _plot_global_radar_chart(
        y_true_num, y_pred_num, num_target_names,
        cat_true_list, cat_pred_list, cat_target_names,
        save_dir_path, format_config
    )

    # 4. Save Overall Report
    report_string = "\n".join(overall_report_lines)
    report_path = save_dir_path / "global_autoencoder_report.txt"
    report_path.write_text(report_string, encoding="utf-8")
    _LOGGER.info(f"🌎 Global reconstruction report saved to '{report_path.name}'")


# =====================================================================
# PRIVATE HELPER FUNCTIONS
# =====================================================================

def _evaluate_numerical_features(
    y_true_num: Optional[np.ndarray],
    y_pred_num: Optional[np.ndarray],
    num_target_names: Optional[list[str]],
    save_dir_path: Path,
    format_config: FormatAutoencoderMetrics
) -> list[str]:
    """Evaluates numerical marginal distributions and returns report lines."""
    report_lines = []
    
    if not (y_true_num is not None and len(y_true_num) > 0 and 
            y_pred_num is not None and len(y_pred_num) > 0 and 
            num_target_names is not None and len(num_target_names) > 0):
        return report_lines

    sample_mse = np.mean(np.square(y_true_num - y_pred_num), axis=1)
    global_num_mse = np.mean(sample_mse)
    
    report_lines.append(f"\n[Numerical Features: {len(num_target_names)}]")
    report_lines.append(f"Global Mean Squared Error (MSE): {global_num_mse:.4f}")

    # Plot Distribution of Sample-wise Reconstruction Errors
    fig_err, ax_err = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
    ax_err.hist(sample_mse, bins=format_config.hist_bins, color=format_config.hist_color, alpha=0.7, edgecolor='black')
    ax_err.set_title("Numerical Reconstruction Error", fontsize=format_config.font_size + 2)
    ax_err.set_xlabel("Mean Squared Error per Sample", fontsize=format_config.font_size)
    ax_err.set_ylabel("Frequency", fontsize=format_config.font_size)
    
    ax_err.tick_params(axis='x', labelsize=format_config.xtick_size)
    ax_err.tick_params(axis='y', labelsize=format_config.ytick_size)
    ax_err.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    hist_path = save_dir_path / "global_numerical_error_distribution.svg"
    plt.savefig(hist_path)
    plt.close(fig_err)

    # Calculate Per-Feature Numerical Summary 
    metrics_summary = []
    for i, name in enumerate(num_target_names):
        true_i = y_true_num[:, i]
        pred_i = y_pred_num[:, i]
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        metrics_summary.append({
            'Feature': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2-score': r2,
        })
        
    summary_df = pd.DataFrame(metrics_summary)
    csv_path = save_dir_path / "numerical_reconstruction_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    _LOGGER.info(f"🔢 Numerical summary saved to '{csv_path.name}'")

    return report_lines


def _evaluate_categorical_features(
    cat_true_list: Optional[list[np.ndarray]],
    cat_pred_list: Optional[list[np.ndarray]],
    cat_prob_list: Optional[list[np.ndarray]],
    cat_target_names: Optional[list[str]],
    cat_class_maps: Optional[list[Optional[dict[str, int]]]],
    save_dir_path: Path,
    format_config: FormatAutoencoderMetrics
) -> list[str]:
    """Evaluates categorical marginal distributions and returns report lines."""
    report_lines = []
    
    if not (cat_true_list is not None and len(cat_true_list) > 0 and 
            cat_pred_list is not None and len(cat_pred_list) > 0 and 
            cat_target_names is not None and len(cat_target_names) > 0):
        return report_lines
        
    report_lines.append(f"\n[Categorical Features: {len(cat_target_names)}]")
    
    global_accuracies = []
    cat_metrics_summary = []
    
    for i, feat_name in enumerate(cat_target_names):
        y_true_c = cat_true_list[i]
        y_pred_c = cat_pred_list[i]
        
        feat_name_abbrev = check_and_abbreviate_name(feat_name)
        
        acc = accuracy_score(y_true_c, y_pred_c)
        f1_macro = f1_score(y_true_c, y_pred_c, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true_c, y_pred_c, average='weighted', zero_division=0)
        
        global_accuracies.append(acc)
        
        cat_metrics_summary.append({
            'Feature': feat_name,
            'Accuracy': acc,
            'F1-score (Macro)': f1_macro,
            'F1-score (Weighted)': f1_weighted
        })
        
        # 1. Confusion Matrix
        plot_labels = None
        plot_display_labels = None
        if cat_class_maps is not None:
            class_map = cat_class_maps[i]
            if class_map is not None:
                sorted_map = sorted(class_map.items(), key=lambda item: item[1])
                plot_display_labels = [item[0] for item in sorted_map]
                plot_labels = [item[1] for item in sorted_map]

        n_classes = len(plot_labels) if plot_labels is not None else len(np.unique(y_true_c))
        fig_w = max(9, n_classes * 0.8 + 3)
        fig_h = max(8, n_classes * 0.8 + 2)

        fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI_value)
        
        disp_ = ConfusionMatrixDisplay.from_predictions(
            y_true_c, y_pred_c, cmap=format_config.cmap, ax=ax_cm, 
            normalize='true', labels=plot_labels, display_labels=plot_display_labels, colorbar=False
        )
        
        disp_.im_.set_clim(vmin=0.0, vmax=1.0)
        ax_cm.grid(False)
        
        cm_font_size = format_config.cm_font_size
        cm_tick_size = cm_font_size - 4
        
        final_font_size = cm_font_size + 2
        if n_classes > 2: 
             final_font_size = cm_font_size - n_classes
        
        for text in ax_cm.texts:
            text.set_fontsize(final_font_size)
        
        ax_cm.tick_params(axis='x', labelsize=cm_tick_size)
        ax_cm.tick_params(axis='y', labelsize=cm_tick_size)
        
        if n_classes > 3:
            plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

        ax_cm.set_title(f"Reconstruction: {feat_name_abbrev}", pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size + 2)
        ax_cm.set_xlabel(ax_cm.get_xlabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
        ax_cm.set_ylabel(ax_cm.get_ylabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
        
        cbar = fig_cm.colorbar(disp_.im_, ax=ax_cm, shrink=0.8)
        cbar.ax.tick_params(labelsize=cm_tick_size)
        
        plt.tight_layout()
        cm_path = save_dir_path / f"categorical_cm_{sanitize_filename(feat_name)}.svg"
        plt.savefig(cm_path)
        plt.close(fig_cm)

        # 2. Confidence Distribution Plot
        if cat_prob_list is not None:
            probs_c = cat_prob_list[i]
            max_probs = np.clip(np.max(probs_c, axis=1), 0.0, 1.0)
            
            fig_prob, ax_prob = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
            correct_mask = (y_true_c == y_pred_c)
            
            correct_probs = max_probs[correct_mask]
            incorrect_probs = max_probs[~correct_mask]
            
            if isinstance(format_config.confidence_bins, int):
                bins = np.linspace(0.0, 1.0, format_config.confidence_bins + 1)
            else:
                bins = format_config.confidence_bins
            
            if len(correct_probs) > 0:
                ax_prob.hist(correct_probs, bins=bins, alpha=0.6, color='tab:green', label='Correct Reconstructions', density=True)
            if len(incorrect_probs) > 0:
                ax_prob.hist(incorrect_probs, bins=bins, alpha=0.6, color='tab:red', label='Incorrect Reconstructions', density=True)
            
            ax_prob.set_title(f"Reconstruction Confidence: {feat_name_abbrev}", fontsize=format_config.font_size)
            ax_prob.set_xlabel("Max Predicted Probability", fontsize=format_config.xtick_size)
            ax_prob.set_ylabel("Density", fontsize=format_config.ytick_size)
            
            ax_prob.set_xlim(-0.1, 1.1)
            ax_prob.set_xticks(np.arange(0.0, 1.1, 0.1))
            
            ax_prob.legend(fontsize=format_config.font_size - 4)
            ax_prob.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            prob_path = save_dir_path / f"categorical_confidence_{sanitize_filename(feat_name)}.svg"
            plt.savefig(prob_path)
            plt.close(fig_prob)
    
    _LOGGER.info(f"📊 Saved Confusion Matrices and Distribution Plots for categorical features to '{save_dir_path.name}'")    
    
    report_lines.append(f"Macro Average Categorical Accuracy: {np.mean(global_accuracies):.4f}")
    
    cat_summary_df = pd.DataFrame(cat_metrics_summary)
    cat_csv_path = save_dir_path / "categorical_reconstruction_summary.csv"
    cat_summary_df.to_csv(cat_csv_path, index=False)
    _LOGGER.info(f"🔢 Categorical summary saved to '{cat_csv_path.name}'")

    return report_lines


def _plot_global_feature_performance(y_true_num: Optional[np.ndarray], 
                                     y_pred_num: Optional[np.ndarray], 
                                     num_target_names: Optional[list[str]],
                                     cat_true_list: Optional[list[np.ndarray]], 
                                     cat_pred_list: Optional[list[np.ndarray]], 
                                     cat_target_names: Optional[list[str]],
                                     save_dir_path: Path, 
                                     format_config: FormatAutoencoderMetrics) -> None:
    """
    [PRIVATE] Helper function to plot sorted global feature performance.
    Uses MAE for numerical features (lower is better) and F1-Macro for categorical (higher is better).
    """
    try:
        has_num = y_true_num is not None and y_pred_num is not None and num_target_names is not None and len(num_target_names) > 0
        has_cat = cat_true_list is not None and cat_pred_list is not None and cat_target_names is not None and len(cat_target_names) > 0
        
        if not has_num and not has_cat:
            return

        n_plots = sum([has_num, has_cat])
        
        # Scale figure height dynamically based on the maximum number of features to ensure readability
        max_features = max(len(num_target_names) if has_num else 0, len(cat_target_names) if has_cat else 0)
        fig_height = max(6.0, max_features * 0.4)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(max(8, 7 * n_plots), fig_height), dpi=DPI_value)
        
        # Make axes iterable if there's only one plot
        if n_plots == 1:
            axes = [axes]
            
        ax_idx = 0
        
        # Numerical Features (MAE)
        if has_num:
            mae_scores = []
            abbr_names = [check_and_abbreviate_name(name) for name in num_target_names]
            for i in range(len(num_target_names)):
                mae = mean_absolute_error(y_true_num[:, i], y_pred_num[:, i])
                mae_scores.append(mae)
            
            # Sort by MAE (ascending, so best/lowest error is at the top)
            sorted_indices = np.argsort(mae_scores)
            sorted_maes = [mae_scores[i] for i in sorted_indices]
            sorted_names = [abbr_names[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_names))
            axes[ax_idx].barh(y_pos, sorted_maes, color='tab:blue', alpha=0.7)
            axes[ax_idx].set_yticks(y_pos)
            axes[ax_idx].set_yticklabels(sorted_names, fontsize=format_config.ytick_size)
            axes[ax_idx].invert_yaxis()  # Best (lowest MAE) at the top
            
            axes[ax_idx].set_xlabel('Mean Absolute Error', fontsize=format_config.font_size)
            axes[ax_idx].set_title('Numerical Reconstruction Performance', fontsize=format_config.font_size + 2)
            axes[ax_idx].grid(axis='x', linestyle='--', alpha=0.6)
            axes[ax_idx].tick_params(axis='x', labelsize=format_config.xtick_size)
            ax_idx += 1
            
        # Categorical Features (F1-Macro)
        if has_cat:
            f1_scores = []
            abbr_cat_names = [check_and_abbreviate_name(name) for name in cat_target_names]
            for i in range(len(cat_target_names)):
                f1 = f1_score(cat_true_list[i], cat_pred_list[i], average='macro', zero_division=0)
                f1_scores.append(f1)
            
            # Sort by F1 (descending, so best/highest score is at the top)
            sorted_indices = np.argsort(f1_scores)[::-1]
            sorted_f1s = [f1_scores[i] for i in sorted_indices]
            sorted_cat_names = [abbr_cat_names[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_cat_names))
            axes[ax_idx].barh(y_pos, sorted_f1s, color='tab:orange', alpha=0.7)
            axes[ax_idx].set_yticks(y_pos)
            axes[ax_idx].set_yticklabels(sorted_cat_names, fontsize=format_config.ytick_size)
            axes[ax_idx].invert_yaxis()  # Best (highest F1) at the top
            
            axes[ax_idx].set_xlabel('F1-Macro Score', fontsize=format_config.font_size)
            axes[ax_idx].set_title('Categorical Reconstruction Performance', fontsize=format_config.font_size + 2)
            axes[ax_idx].set_xlim(0, 1.05)
            axes[ax_idx].grid(axis='x', linestyle='--', alpha=0.6)
            axes[ax_idx].tick_params(axis='x', labelsize=format_config.xtick_size)

        plt.tight_layout()
        plot_path = save_dir_path / "global_feature_performance.svg"
        plt.savefig(plot_path)
        plt.close(fig)
        
        _LOGGER.info(f"📊 Global feature performance bar chart saved to '{plot_path.name}'")
    except Exception as e:
        _LOGGER.error(f"Failed to generate global feature performance plot: {e}")


def _plot_sample_error_scatter(y_true_num: Optional[np.ndarray], 
                               y_pred_num: Optional[np.ndarray], 
                               cat_true_list: Optional[list[np.ndarray]], 
                               cat_pred_list: Optional[list[np.ndarray]], 
                               save_dir_path: Path, 
                               format_config: FormatAutoencoderMetrics) -> None:
    """
    [PRIVATE] Helper function to plot sample-wise error scatter (Anomaly Detection).
    X-axis: Numerical Sample MAE. Y-axis: Categorical Misclassifications.
    """
    try:
        has_num = y_true_num is not None and y_pred_num is not None and len(y_true_num) > 0
        has_cat = cat_true_list is not None and cat_pred_list is not None and len(cat_true_list) > 0
        
        if not has_num and not has_cat:
            return

        fig, ax = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
        
        x_data = None
        y_data = None
        x_label = ""
        y_label = ""
        
        # Calculate numerical sample-wise MAE
        if has_num:
            # Absolute error per feature, then mean across features for each sample
            x_data = np.mean(np.abs(y_true_num - y_pred_num), axis=1) 
            x_label = "Sample Mean Absolute Error (Numerical)"
            
        # Calculate categorical sample-wise misclassifications
        if has_cat:
            # Stack arrays to shape (n_samples, n_features)
            cat_true_stacked = np.column_stack(cat_true_list)
            cat_pred_stacked = np.column_stack(cat_pred_list)
            # Count misclassifications per sample
            y_data = np.sum(cat_true_stacked != cat_pred_stacked, axis=1)
            y_label = "Categorical Misclassifications (Count)"

        if has_num and has_cat:
            # Both: 2D Scatter
            ax.scatter(x_data, y_data, alpha=0.6, color='tab:purple', edgecolors='none')
            ax.set_xlabel(x_label, fontsize=format_config.font_size)
            ax.set_ylabel(y_label, fontsize=format_config.font_size)
            ax.set_title("Global Sample-wise Error", fontsize=format_config.font_size + 2)
            # Ensure y-axis only shows integers for misclassifications
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
        elif has_num:
            # Only num: 1D Scatter with jitter for visibility
            y_zeros = np.zeros_like(x_data)
            jitter = np.random.normal(0, 0.05, size=len(x_data))
            ax.scatter(x_data, y_zeros + jitter, alpha=0.6, color='tab:blue', edgecolors='none')
            ax.set_xlabel(x_label, fontsize=format_config.font_size)
            ax.set_yticks([])
            ax.set_ylabel("Density", fontsize=format_config.font_size)
            ax.set_title("Global Sample-wise Error (Numerical)", fontsize=format_config.font_size + 2)
            
        elif has_cat:
            # Only cat: 1D Scatter with jitter for visibility
            x_zeros = np.zeros_like(y_data)
            jitter = np.random.normal(0, 0.05, size=len(y_data))
            ax.scatter(x_zeros + jitter, y_data, alpha=0.6, color='tab:orange', edgecolors='none')
            ax.set_ylabel(y_label, fontsize=format_config.font_size)
            ax.set_xticks([])
            ax.set_xlabel("Density", fontsize=format_config.font_size)
            ax.set_title("Global Sample-wise Error (Categorical)", fontsize=format_config.font_size + 2)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelsize=format_config.xtick_size)
        ax.tick_params(axis='y', labelsize=format_config.ytick_size)
        
        plt.tight_layout()
        plot_path = save_dir_path / "global_sample_error_scatter.svg"
        plt.savefig(plot_path)
        plt.close(fig)
        
        _LOGGER.info(f"📊 Sample-wise error scatter plot saved to '{plot_path.name}'")
        
    except Exception as e:
        _LOGGER.error(f"Failed to generate sample-wise error scatter plot: {e}")


def _plot_error_correlation_heatmap(y_true_num: Optional[np.ndarray], 
                                    y_pred_num: Optional[np.ndarray], 
                                    num_target_names: Optional[list[str]], 
                                    save_dir_path: Path, 
                                    format_config: FormatAutoencoderMetrics) -> None:
    """
    [PRIVATE] Helper function to plot the correlation of absolute reconstruction errors 
    across numerical features. Helps identify if errors in certain features are linked.
    """
    if y_true_num is None or y_pred_num is None or num_target_names is None or len(num_target_names) < 2:
        return
        
    try:
        # Calculate absolute reconstruction errors
        abs_errors = np.abs(y_true_num - y_pred_num)
        
        # Abbreviate names for the plot to prevent overlapping text
        abbr_names = [check_and_abbreviate_name(name) for name in num_target_names]
        
        # Create DataFrame and calculate correlation matrix
        error_df = pd.DataFrame(abs_errors, columns=abbr_names)
        error_corr = error_df.corr().fillna(0) # Fill NaNs in case of zero variance in errors
        
        # Dynamically scale figure size based on number of features
        num_feats = len(abbr_names)
        fig_size_xy = max(8, num_feats * 0.8)
        fig, ax = plt.subplots(figsize=(fig_size_xy, fig_size_xy), dpi=DPI_value)
        
        # Only show text annotations if there are 15 or fewer features to avoid clutter
        show_annotations = num_feats <= 15
        
        # Adjust font sizes
        title_fs = max(14, format_config.font_size - 8)
        annot_fs = max(10, format_config.font_size - max(4, num_feats // 2))
        
        sns.heatmap(error_corr, 
                    annot=show_annotations, 
                    fmt=".2f", 
                    cmap=format_config.cmap, 
                    annot_kws={"size": annot_fs},
                    ax=ax,
                    vmin=-1.0, vmax=1.0)
                    
        ax.set_title("Reconstruction Error Correlation\n(Numerical Features)", 
                     fontsize=title_fs, pad=15)
                     
        ax.tick_params(axis='x', labelsize=format_config.xtick_size - 2)
        ax.tick_params(axis='y', labelsize=format_config.ytick_size - 2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Increase colorbar tick size
        if ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=format_config.ytick_size - 4)
        
        plt.tight_layout()
        plot_path = save_dir_path / "global_error_correlation_heatmap.svg"
        plt.savefig(plot_path)
        plt.close(fig)
        
        _LOGGER.info(f"📊 Error correlation heatmap saved to '{plot_path.name}'")
        
    except Exception as e:
        _LOGGER.error(f"Failed to generate error correlation heatmap: {e}")


def _plot_global_radar_chart(y_true_num: Optional[np.ndarray], 
                             y_pred_num: Optional[np.ndarray], 
                             num_target_names: Optional[list[str]],
                             cat_true_list: Optional[list[np.ndarray]], 
                             cat_pred_list: Optional[list[np.ndarray]], 
                             cat_target_names: Optional[list[str]],
                             save_dir_path: Path, 
                             format_config: FormatAutoencoderMetrics) -> None:
    """
    [PRIVATE] Helper function to plot a global reconstruction radar chart.
    Plots Numerical MAE (closer to center is better) and Categorical F1 (closer to edge is better).
    """
    # Radar charts require at least 3 features to form a proper polygon
    has_num = y_true_num is not None and y_pred_num is not None and num_target_names is not None and len(num_target_names) > 2
    has_cat = cat_true_list is not None and cat_pred_list is not None and cat_target_names is not None and len(cat_target_names) > 2
    
    if not has_num and not has_cat:
        return

    try:
        n_plots = sum([has_num, has_cat])
        fig = plt.figure(figsize=(max(7, 7 * n_plots), 7), dpi=DPI_value)
        
        plot_idx = 1
        
        # Numerical Radar (MAE)
        if has_num:
            ax = fig.add_subplot(1, n_plots, plot_idx, polar=True)
            abbr_names = [check_and_abbreviate_name(name) for name in num_target_names]
            
            mae_scores = []
            for i in range(len(num_target_names)):
                mae = mean_absolute_error(y_true_num[:, i], y_pred_num[:, i])
                mae_scores.append(mae)
                
            # Close the loop to connect the last point back to the first
            mae_scores = mae_scores + [mae_scores[0]]
            angles = [n / float(len(abbr_names)) * 2 * math.pi for n in range(len(abbr_names))]
            angles += angles[:1]
            
            ax.plot(angles, mae_scores, linewidth=2, linestyle='solid', color='tab:blue')
            ax.fill(angles, mae_scores, 'tab:blue', alpha=0.2)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(abbr_names, fontsize=format_config.xtick_size)
            
            ax.set_title("Numerical MAE", fontsize=format_config.font_size + 2, pad=20)
            plot_idx += 1
            
        # Categorical Radar (F1-Macro)
        if has_cat:
            ax = fig.add_subplot(1, n_plots, plot_idx, polar=True)
            abbr_cat_names = [check_and_abbreviate_name(name) for name in cat_target_names]
            
            f1_scores = []
            for i in range(len(cat_target_names)):
                f1 = f1_score(cat_true_list[i], cat_pred_list[i], average='macro', zero_division=0)
                f1_scores.append(f1)
                
            # Close the loop
            f1_scores = f1_scores + [f1_scores[0]]
            angles = [n / float(len(abbr_cat_names)) * 2 * math.pi for n in range(len(abbr_cat_names))]
            angles += angles[:1]
            
            ax.plot(angles, f1_scores, linewidth=2, linestyle='solid', color='tab:orange')
            ax.fill(angles, f1_scores, 'tab:orange', alpha=0.2)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(abbr_cat_names, fontsize=format_config.xtick_size)
            ax.set_ylim(0, 1.0)
            
            ax.set_title("Categorical F1-Macro", fontsize=format_config.font_size + 2, pad=20)

        plt.tight_layout()
        plot_path = save_dir_path / "global_radar_chart.svg"
        plt.savefig(plot_path)
        plt.close(fig)
        
        _LOGGER.info(f"📊 Global radar chart saved to '{plot_path.name}'")
        
    except Exception as e:
        _LOGGER.error(f"Failed to generate radar chart: {e}")

