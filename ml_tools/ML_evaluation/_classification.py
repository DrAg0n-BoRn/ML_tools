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
    precision_recall_curve,
    average_precision_score,
    hamming_loss,
    jaccard_score
)
from pathlib import Path
from typing import Union, Optional

from ..ML_configuration._metrics import (_BaseMultiLabelFormat,
                                         _BaseClassificationFormat,
                                        FormatBinaryClassificationMetrics,
                                        FormatMultiClassClassificationMetrics,
                                        FormatBinaryImageClassificationMetrics,
                                        FormatMultiClassImageClassificationMetrics,
                                        FormatMultiLabelBinaryClassificationMetrics)

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import _EvaluationConfig


_LOGGER = get_logger("Classification Metrics")


__all__ = [
    "classification_metrics",
    "multi_label_classification_metrics",
]


DPI_value = _EvaluationConfig.DPI
CLASSIFICATION_PLOT_SIZE = _EvaluationConfig.CLASSIFICATION_PLOT_SIZE


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
    Saves classification metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities for ROC curve.
        config (object): Formatting configuration object.
        save_dir (str | Path): Directory to save plots.
    """
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseClassificationFormat()
    else:
        format_config = config
    
    # --- Set Font Sizes ---
    xtick_size = format_config.xtick_size
    ytick_size = format_config.ytick_size
    legend_size = format_config.legend_size
    
    # config font size for heatmap
    cm_font_size = format_config.cm_font_size
    cm_tick_size = cm_font_size - 4
    
    # --- Parse class_map ---
    map_labels = None
    map_display_labels = None
    if class_map:
        # Sort the map by its values (the indices) to ensure correct order
        try:
            sorted_items = sorted(class_map.items(), key=lambda item: item[1])
            map_labels = [item[1] for item in sorted_items]
            map_display_labels = [item[0] for item in sorted_items]
        except Exception as e:
            _LOGGER.warning(f"Could not parse 'class_map': {e}")
            map_labels = None
            map_display_labels = None
    
    # Generate report as both text and dictionary
    report_text: str = classification_report(y_true, y_pred, labels=map_labels, target_names=map_display_labels) # type: ignore
    report_dict: dict = classification_report(y_true, y_pred, output_dict=True, labels=map_labels, target_names=map_display_labels) # type: ignore
    # print(report_text)
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # Save text report
    report_path = save_dir_path / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    _LOGGER.info(f"📝 Classification report saved as '{report_path.name}'")

    # --- Save Classification Report Heatmap ---
    try:
        # Create DataFrame from report
        report_df = pd.DataFrame(report_dict)

        # 1. Robust Cleanup: Drop by name, not position
        # Remove 'accuracy' column if it exists (handles the scalar value issue)
        report_df = report_df.drop(columns=['accuracy'], errors='ignore')

        # Remove 'support' row explicitly (safer than iloc[:-1])
        if 'support' in report_df.index:
            report_df = report_df.drop(index='support')

        # 2. Transpose: Rows = Classes, Cols = Metrics
        plot_df = report_df.T

        # 3. Dynamic Height Calculation
        # (Base height of 4 + 0.5 inches per class row)
        fig_height = max(5.0, len(plot_df.index) * 0.5 + 4.0)
        fig_width = 8.0 # Set a fixed width

        # --- Use calculated dimensions, not the config constant ---
        fig_heat, ax_heat = plt.subplots(figsize=(fig_width, fig_height), dpi=_EvaluationConfig.DPI)

        # sns.set_theme(font_scale=1.4)
        sns.heatmap(plot_df, 
                    annot=True, 
                    cmap=format_config.cmap, 
                    fmt='.2f',
                    vmin=0.0,
                    vmax=1.0,
                    cbar_kws={'shrink': 0.9}) # Shrink colorbar slightly to fit better

        # sns.set_theme(font_scale=1.0)

        ax_heat.set_title("Classification Report Heatmap", pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
        
        # manually increase the font size of the elements
        for text in ax_heat.texts:
            text.set_fontsize(cm_tick_size)
            
        # manually increase the size of the colorbar ticks
        cbar = ax_heat.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cm_tick_size - 4) # type: ignore

        # Update Ticks
        ax_heat.tick_params(axis='x', labelsize=cm_tick_size, pad=_EvaluationConfig.LABEL_PADDING)
        ax_heat.tick_params(axis='y', labelsize=cm_tick_size, pad=_EvaluationConfig.LABEL_PADDING, rotation=0) # Ensure Y labels are horizontal

        plt.tight_layout()
        
        heatmap_path = save_dir_path / "classification_report_heatmap.svg"
        plt.savefig(heatmap_path)
        _LOGGER.info(f"📊 Report heatmap saved as '{heatmap_path.name}'")
        plt.close(fig_heat)
        
    except Exception as e:
        _LOGGER.error(f"Could not generate classification report heatmap: {e}")
    
    # --- labels for Confusion Matrix ---
    plot_labels = map_labels
    plot_display_labels = map_display_labels
    
    # 1. DYNAMIC SIZE CALCULATION
    # Calculate figure size based on number of classes. 
    n_classes = len(plot_labels) if plot_labels is not None else len(np.unique(y_true))
    # Ensure a minimum size so very small matrices aren't tiny
    fig_w = max(9, n_classes * 0.8 + 3)
    fig_h = max(8, n_classes * 0.8 + 2)
    
    # Use the calculated size instead of CLASSIFICATION_PLOT_SIZE
    fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI_value)
    disp_ = ConfusionMatrixDisplay.from_predictions(y_true, 
                                            y_pred, 
                                            cmap=format_config.cmap, 
                                            ax=ax_cm, 
                                            normalize='true',
                                            labels=plot_labels,
                                            display_labels=plot_display_labels,
                                            colorbar=False)
    
    disp_.im_.set_clim(vmin=0.0, vmax=1.0)
    
    # Turn off gridlines
    ax_cm.grid(False)
    
    # 2. CHECK FOR FONT CLASH
    # If matrix is huge, force text smaller. If small, allow user config.
    final_font_size = cm_font_size + 2
    if n_classes > 2: 
         final_font_size = cm_font_size - n_classes  # Decrease font size for larger matrices
    
    for text in ax_cm.texts:
        text.set_fontsize(final_font_size)
    
    # Update Ticks for Confusion Matrix
    ax_cm.tick_params(axis='x', labelsize=cm_tick_size)
    ax_cm.tick_params(axis='y', labelsize=cm_tick_size)
    
    #if more than 3 classes, rotate x ticks
    if n_classes > 3:
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    # Set titles and labels with padding
    ax_cm.set_title("Confusion Matrix", pad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size + 2)
    ax_cm.set_xlabel(ax_cm.get_xlabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
    ax_cm.set_ylabel(ax_cm.get_ylabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=cm_font_size)
    
    # --- ADJUST COLORBAR FONT & SIZE---
    # Manually add the colorbar with the 'shrink' parameter
    cbar = fig_cm.colorbar(disp_.im_, ax=ax_cm, shrink=0.8)

    # Update the tick size on the new cbar object
    cbar.ax.tick_params(labelsize=cm_tick_size) 

    # (Optional) add a label to the bar itself (e.g. "Probability")
    # cbar.set_label('Probability', fontsize=12)
    
    fig_cm.tight_layout()
    
    cm_path = save_dir_path / "confusion_matrix.svg"
    plt.savefig(cm_path)
    _LOGGER.info(f"❇️ Confusion matrix saved as '{cm_path.name}'")
    plt.close(fig_cm)


    # Plotting logic for ROC, PR, and Calibration Curves
    if y_prob is not None and y_prob.ndim == 2:
        num_classes = y_prob.shape[1]
        
        # --- Determine which classes to loop over ---
        class_indices_to_plot = []
        plot_titles = []
        save_suffixes = []

        if num_classes == 2:
            # Binary case: Only plot for the positive class (index 1)
            class_indices_to_plot = [1]
            plot_titles = [""] # No extra title
            save_suffixes = [""] # No extra suffix
            _LOGGER.debug("Generating binary classification plots (ROC, PR, Calibration).")
        
        elif num_classes > 2:
            _LOGGER.debug(f"Generating One-vs-Rest plots for {num_classes} classes.")
            # Multiclass case: Plot for every class (One-vs-Rest)
            class_indices_to_plot = list(range(num_classes))
            
            # --- Use class_map names if available ---
            use_generic_names = True
            if map_display_labels and len(map_display_labels) == num_classes:
                try:
                    # Ensure labels are safe for filenames
                    safe_names = [sanitize_filename(name) for name in map_display_labels]
                    plot_titles = [f" ({name} vs. Rest)" for name in map_display_labels]
                    save_suffixes = [f"_{safe_names[i]}" for i in class_indices_to_plot]
                    use_generic_names = False
                except Exception as e:
                    _LOGGER.warning(f"Failed to use 'class_map' for plot titles: {e}. Reverting to generic names.")
                    use_generic_names = True
            
            if use_generic_names:
                plot_titles = [f" (Class {i} vs. Rest)" for i in class_indices_to_plot]
                save_suffixes = [f"_class_{i}" for i in class_indices_to_plot]
        
        else:
            # Should not happen, but good to check
            _LOGGER.warning(f"Probability array has invalid shape {y_prob.shape}. Skipping ROC/PR/Calibration plots.")

        # --- Loop and generate plots ---
        for i, class_index in enumerate(class_indices_to_plot):
            plot_title = plot_titles[i]
            save_suffix = save_suffixes[i]

            # Get scores for the current class
            y_score = y_prob[:, class_index]
            
            # Binarize y_true for the current class
            y_true_binary = (y_true == class_index).astype(int)
            
            # --- Save ROC Curve ---
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
            
            try:
                # Calculate Youden's J statistic (tpr - fpr)
                J = tpr - fpr
                # Find the index of the best threshold
                best_index = np.argmax(J)
                optimal_threshold = thresholds[best_index]
                
                # Define the filename
                threshold_filename = f"best_threshold{save_suffix}.txt"
                threshold_path = save_dir_path / threshold_filename
                
                # Get the class name for the report
                class_name = ""
                # Check if we have display labels and the current index is valid
                if map_display_labels and class_index < len(map_display_labels):
                    class_name = map_display_labels[class_index]
                    if num_classes > 2:
                        # Add 'vs. Rest' for multiclass one-vs-rest plots
                        class_name += " (vs. Rest)"
                else:
                    # Fallback to the generic title or default binary name
                    class_name = plot_title.strip() or "Binary Positive Class"
                
                # Create content for the file
                file_content = (
                    f"Optimal Classification Threshold (Youden's J Statistic)\n"
                    f"Class: {class_name}\n"
                    f"--------------------------------------------------\n"
                    f"Threshold: {optimal_threshold:.6f}\n"
                    f"True Positive Rate (TPR): {tpr[best_index]:.6f}\n"
                    f"False Positive Rate (FPR): {fpr[best_index]:.6f}\n"
                )
                
                threshold_path.write_text(file_content, encoding="utf-8")
                _LOGGER.info(f"💾 Optimal threshold saved as '{threshold_path.name}'")

            except Exception as e:
                _LOGGER.warning(f"Could not calculate or save optimal threshold: {e}")
            
            # Calculate AUC. 
            auc = roc_auc_score(y_true_binary, y_score) 
            
            fig_roc, ax_roc = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color=format_config.ROC_PR_line)
            ax_roc.plot([0, 1], [0, 1], 'k--')
            # use "ROC" if extra title, else use "Receiver Operating Characteristic" title
            if plot_title.strip():
                ax_roc.set_title(f'ROC{plot_title}', pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            else:
                ax_roc.set_title(f'Receiver Operating Characteristic', pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            ax_roc.set_xlabel('False Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            ax_roc.set_ylabel('True Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            
            # Apply Ticks and Legend sizing
            ax_roc.tick_params(axis='x', labelsize=xtick_size)
            ax_roc.tick_params(axis='y', labelsize=ytick_size)
            ax_roc.legend(loc='lower right', fontsize=legend_size)
            
            ax_roc.grid(True)
            roc_path = save_dir_path / f"roc_curve{save_suffix}.svg"
            
            plt.tight_layout()
            
            plt.savefig(roc_path)
            plt.close(fig_roc)

            # --- Save Precision-Recall Curve ---
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            ap_score = average_precision_score(y_true_binary, y_score)
            fig_pr, ax_pr = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
            ax_pr.plot(recall, precision, label=f'Avg Precision = {ap_score:.2f}', color=format_config.ROC_PR_line)
            # Use "PR Curve" if extra title, else use "Precision-Recall Curve" title
            if plot_title.strip():
                ax_pr.set_title(f'PR Curve{plot_title}', pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            else:
                ax_pr.set_title(f'Precision-Recall Curve', pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            ax_pr.set_xlabel('Recall', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            ax_pr.set_ylabel('Precision', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            
            # Apply Ticks and Legend sizing
            ax_pr.tick_params(axis='x', labelsize=xtick_size)
            ax_pr.tick_params(axis='y', labelsize=ytick_size)
            ax_pr.legend(loc='lower left', fontsize=legend_size)
            
            ax_pr.grid(True)
            pr_path = save_dir_path / f"pr_curve{save_suffix}.svg"
            
            plt.tight_layout()
            
            plt.savefig(pr_path)
            plt.close(fig_pr)
            
            # --- Save Calibration Plot ---
            fig_cal, ax_cal = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)

            # --- Step 1: Get binned data *without* plotting ---
            with plt.ioff(): # Suppress showing the temporary plot
                fig_temp, ax_temp = plt.subplots()
                cal_display_temp = CalibrationDisplay.from_predictions(
                    y_true_binary, # Use binarized labels
                    y_score, 
                    n_bins=format_config.calibration_bins, 
                    ax=ax_temp,
                    name="temp" # Add a name to suppress potential warnings
                )
                # Get the x, y coordinates of the binned data
                line_x, line_y = cal_display_temp.line_.get_data() # type: ignore
                plt.close(fig_temp) # Close the temporary plot

            # --- Step 2: Build the plot from scratch ---
            ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            
            sns.regplot(
                x=line_x, 
                y=line_y,
                ax=ax_cal,
                scatter=False, 
                label=f"Model calibration",
                line_kws={
                    'color': format_config.ROC_PR_line, 
                    'linestyle': '--', 
                    'linewidth': 2,
                    }
            )
            
            ax_cal.set_title(f'Reliability Curve{plot_title}', pad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size + 2)
            ax_cal.set_xlabel('Mean Predicted Probability', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            ax_cal.set_ylabel('Fraction of Positives', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=format_config.font_size)
            
            # --- Step 3: Set final limits *after* plotting ---
            ax_cal.set_ylim(0.0, 1.0) 
            ax_cal.set_xlim(0.0, 1.0)
            
            # Apply Ticks and Legend sizing
            ax_cal.tick_params(axis='x', labelsize=xtick_size)
            ax_cal.tick_params(axis='y', labelsize=ytick_size)
            ax_cal.legend(loc='lower right', fontsize=legend_size)
            
            ax_cal.grid(True)
            plt.tight_layout()
            
            cal_path = save_dir_path / f"calibration_plot{save_suffix}.svg"
            plt.savefig(cal_path)
            plt.close(fig_cal)
            
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

    # ticks and legend font sizes
    xtick_size = format_config.xtick_size
    ytick_size = format_config.ytick_size
    legend_size = format_config.legend_size
    base_font_size = format_config.font_size

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
    overall_report_path = save_dir_path / "classification_report.txt"
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
        fig_cm, ax_cm = plt.subplots(figsize=_EvaluationConfig.CM_SIZE, dpi=_EvaluationConfig.DPI)
        disp_ = ConfusionMatrixDisplay.from_predictions(true_i, 
                                                pred_i, 
                                                cmap=format_config.cmap, # Use config cmap
                                                ax=ax_cm, 
                                                normalize='true',
                                                labels=[0, 1],
                                                display_labels=["Negative", "Positive"],
                                                colorbar=False)
        
        disp_.im_.set_clim(vmin=0.0, vmax=1.0)
        
        # Turn off gridlines
        ax_cm.grid(False)
        
        # Manually update font size of cell texts
        for text in ax_cm.texts:
            text.set_fontsize(base_font_size + 2) # Use config font_size
            
        # Apply ticks 
        ax_cm.tick_params(axis='x', labelsize=xtick_size)
        ax_cm.tick_params(axis='y', labelsize=ytick_size)
        
        # Set titles and labels with padding
        ax_cm.set_title(f"Confusion Matrix for '{name}'", pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        ax_cm.set_xlabel(ax_cm.get_xlabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_cm.set_ylabel(ax_cm.get_ylabel(), labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        
        # --- ADJUST COLORBAR FONT & SIZE---
        # Manually add the colorbar with the 'shrink' parameter
        cbar = fig_cm.colorbar(disp_.im_, ax=ax_cm, shrink=0.8)

        # Update the tick size on the new cbar object
        cbar.ax.tick_params(labelsize=ytick_size)  # type: ignore
        
        plt.tight_layout()
        
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
        fig_roc, ax_roc = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color=format_config.ROC_PR_line) # Use config color
        ax_roc.plot([0, 1], [0, 1], 'k--')
        
        ax_roc.set_title(f'ROC Curve for "{name}"', pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        ax_roc.set_xlabel('False Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_roc.set_ylabel('True Positive Rate', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        
        # Apply ticks and legend font size
        ax_roc.tick_params(axis='x', labelsize=xtick_size)
        ax_roc.tick_params(axis='y', labelsize=ytick_size)
        ax_roc.legend(loc='lower right', fontsize=legend_size)
        
        ax_roc.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        roc_path = save_dir_path / f"roc_curve_{sanitized_name}.svg"
        plt.savefig(roc_path)
        plt.close(fig_roc)

        # --- Save Precision-Recall Curve (uses y_prob) ---
        precision, recall, _ = precision_recall_curve(true_i, prob_i)
        ap_score = average_precision_score(true_i, prob_i)
        fig_pr, ax_pr = plt.subplots(figsize=CLASSIFICATION_PLOT_SIZE, dpi=DPI_value)
        ax_pr.plot(recall, precision, label=f'AP = {ap_score:.2f}', color=format_config.ROC_PR_line) # Use config color
        ax_pr.set_title(f'Precision-Recall Curve for "{name}"', pad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size + 2)
        ax_pr.set_xlabel('Recall', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        ax_pr.set_ylabel('Precision', labelpad=_EvaluationConfig.LABEL_PADDING, fontsize=base_font_size)
        
        # Apply ticks and legend font size
        ax_pr.tick_params(axis='x', labelsize=xtick_size)
        ax_pr.tick_params(axis='y', labelsize=ytick_size)
        ax_pr.legend(loc='lower left', fontsize=legend_size)
        
        ax_pr.grid(True, linestyle='--', alpha=0.6)
        
        fig_pr.tight_layout()
        
        pr_path = save_dir_path / f"pr_curve_{sanitized_name}.svg"
        plt.savefig(pr_path)
        plt.close(fig_pr)

    _LOGGER.info(f"All individual label reports and plots saved to '{save_dir_path.name}'")

