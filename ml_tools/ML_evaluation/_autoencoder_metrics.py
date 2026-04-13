import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    Args:
        y_true_num: 2D array of true numerical values (samples x num_features).
        y_pred_num: 2D array of predicted numerical values (samples x num_features).
        num_target_names: List of names for numerical features.
        cat_true_list: List of 1D arrays of true categorical labels for each categorical feature.
        cat_pred_list: List of 1D arrays of predicted categorical labels for each categorical feature.
        cat_prob_list: List of 2D arrays of predicted probabilities for each categorical feature (samples x num_classes).
        cat_target_names: List of names for categorical features.
        cat_class_maps: List of optional dictionaries mapping class names to indices for each categorical feature.
        save_dir: Directory path to save the evaluation results.
        config: Optional configuration object for formatting the evaluation outputs.
    """
    if config is None:
        format_config = FormatAutoencoderMetrics()
    else:
        format_config = config
        
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # _LOGGER.info(f"Starting Autoencoder evaluation. Saving to '{save_dir_path.name}'")

    overall_report_lines = ["--- Autoencoder Global Reconstruction Report ---"]

    # ==========================================
    # 1. Global Numerical Metrics & Distribution
    # ==========================================
    if (y_true_num is not None and len(y_true_num) > 0 and 
        y_pred_num is not None and len(y_pred_num) > 0 and 
        num_target_names is not None and len(num_target_names) > 0):
        
        # Calculate sample-wise MSE (useful for anomaly detection later)
        sample_mse = np.mean(np.square(y_true_num - y_pred_num), axis=1)
        global_num_mse = np.mean(sample_mse)
        
        overall_report_lines.append(f"\n[Numerical Features: {len(num_target_names)}]")
        overall_report_lines.append(f"Global Mean Squared Error (MSE): {global_num_mse:.4f}")

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

    # ==========================================
    # 2. Global Categorical Metrics
    # ==========================================
    if (cat_true_list is not None and len(cat_true_list) > 0 and 
        cat_pred_list is not None and len(cat_pred_list) > 0 and 
        cat_target_names is not None and len(cat_target_names) > 0):
        
        overall_report_lines.append(f"\n[Categorical Features: {len(cat_target_names)}]")
        
        global_accuracies = []
        cat_metrics_summary = []
        
        for i, feat_name in enumerate(cat_target_names):
            y_true_c = cat_true_list[i]
            y_pred_c = cat_pred_list[i]
            
            # abbreviate feature name if too long for plotting
            feat_name_abbrev = check_and_abbreviate_name(feat_name)
            
            # Calculate metrics
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
            
            # -- Visualizations --
            
            # 1. Confusion Matrix
            plot_labels = None
            plot_display_labels = None
            if cat_class_maps is not None:
                class_map = cat_class_maps[i]
                if class_map is not None:
                    # Sort the dictionary by values to ensure correct index mapping
                    sorted_map = sorted(class_map.items(), key=lambda item: item[1])
                    plot_display_labels = [item[0] for item in sorted_map]
                    plot_labels = [item[1] for item in sorted_map]

            # Dynamic Size Calculation
            n_classes = len(plot_labels) if plot_labels is not None else len(np.unique(y_true_c))
            fig_w = max(9, n_classes * 0.8 + 3)
            fig_h = max(8, n_classes * 0.8 + 2)

            fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI_value)
            
            disp_ = ConfusionMatrixDisplay.from_predictions(
                y_true_c, 
                y_pred_c, 
                cmap=format_config.cmap, 
                ax=ax_cm, 
                normalize='true',
                labels=plot_labels,
                display_labels=plot_display_labels,
                colorbar=False
            )
            
            disp_.im_.set_clim(vmin=0.0, vmax=1.0)
            ax_cm.grid(False)
            
            # Font configurations
            cm_font_size = format_config.cm_font_size
            cm_tick_size = cm_font_size - 4
            
            # Font clash check for large matrices
            final_font_size = cm_font_size + 2
            if n_classes > 2: 
                 final_font_size = cm_font_size - n_classes
            
            for text in ax_cm.texts:
                text.set_fontsize(final_font_size)
            
            # Update Ticks
            ax_cm.tick_params(axis='x', labelsize=cm_tick_size)
            ax_cm.tick_params(axis='y', labelsize=cm_tick_size)
            
            if n_classes > 3:
                plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

            # Set titles and labels with padding
            ax_cm.set_title(f"Reconstruction: {feat_name_abbrev}", pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size + 2)
            ax_cm.set_xlabel(ax_cm.get_xlabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
            ax_cm.set_ylabel(ax_cm.get_ylabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
            
            # Adjust colorbar
            cbar = fig_cm.colorbar(disp_.im_, ax=ax_cm, shrink=0.8)
            cbar.ax.tick_params(labelsize=cm_tick_size)
            
            plt.tight_layout()
            
            cm_path = save_dir_path / f"categorical_cm_{sanitize_filename(feat_name)}.svg"
            plt.savefig(cm_path)
            plt.close(fig_cm)

            # 2. Confidence Distribution Plot
            if cat_prob_list is not None:
                probs_c = cat_prob_list[i]
                max_probs = np.max(probs_c, axis=1)
                
                # Clip probabilities to ensure they are within [0, 1] for plotting, in case of any numerical issues
                max_probs = np.clip(max_probs, 0.0, 1.0)
                
                fig_prob, ax_prob = plt.subplots(figsize=REGRESSION_PLOT_SIZE, dpi=DPI_value)
                correct_mask = (y_true_c == y_pred_c)
                
                correct_probs = max_probs[correct_mask]
                incorrect_probs = max_probs[~correct_mask]
                
                # Explicitly define bin edges to prevent zero-width bins when all probabilities are 1.0
                if isinstance(format_config.confidence_bins, int):
                    bins = np.linspace(0.0, 1.0, format_config.confidence_bins + 1)
                else:
                    bins = format_config.confidence_bins
                
                if len(correct_probs) > 0:
                    ax_prob.hist(correct_probs, bins=bins, alpha=0.6, color='tab:green', label='Correct Reconstructions', density=True) # type: ignore
                if len(incorrect_probs) > 0:
                    ax_prob.hist(incorrect_probs, bins=bins, alpha=0.6, color='tab:red', label='Incorrect Reconstructions', density=True) # type: ignore
                
                ax_prob.set_title(f"Reconstruction Confidence: {feat_name_abbrev}", fontsize=format_config.font_size)
                ax_prob.set_xlabel("Max Predicted Probability", fontsize=format_config.xtick_size)
                ax_prob.set_ylabel("Density", fontsize=format_config.ytick_size)
                
                # Force the X-axis to display the full probability range
                ax_prob.set_xlim(-0.1, 1.1)
                ax_prob.set_xticks(np.arange(0.0, 1.1, 0.1))
                
                ax_prob.legend(fontsize=format_config.font_size - 4)
                ax_prob.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                prob_path = save_dir_path / f"categorical_confidence_{sanitize_filename(feat_name)}.svg"
                plt.savefig(prob_path)
                plt.close(fig_prob)
        
        _LOGGER.info(f"📊 Saved Confusion Matrices and Distribution Plots for categorical features to '{save_dir_path.name}'")    
        
        overall_report_lines.append(f"Macro Average Categorical Accuracy: {np.mean(global_accuracies):.4f}")
        
        # Save Per-Feature Categorical Summary
        cat_summary_df = pd.DataFrame(cat_metrics_summary)
        cat_csv_path = save_dir_path / "categorical_reconstruction_summary.csv"
        cat_summary_df.to_csv(cat_csv_path, index=False)
        _LOGGER.info(f"🔢 Categorical summary saved to '{cat_csv_path.name}'")

    # ==========================================
    # 3. Save Overall Report
    # ==========================================
    report_string = "\n".join(overall_report_lines)
    report_path = save_dir_path / "global_autoencoder_report.txt"
    report_path.write_text(report_string, encoding="utf-8")
    _LOGGER.info(f"🌎 Global reconstruction report saved to '{report_path.name}'")
