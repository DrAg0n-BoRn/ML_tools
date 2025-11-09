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
from typing import Union, Optional, List, Literal
import warnings

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from ._keys import SHAPKeys, PyTorchLogKeys
from .ML_configuration import (RegressionMetricsFormat,
                               BinaryClassificationMetricsFormat,
                               MultiClassClassificationMetricsFormat,
                               BinaryImageClassificationMetricsFormat,
                               MultiClassImageClassificationMetricsFormat,
                               _BaseClassificationFormat,
                               _BaseRegressionFormat)


__all__ = [
    "plot_losses", 
    "classification_metrics", 
    "regression_metrics",
    "shap_summary_plot",
    "plot_attention_importance"
]

DPI_value = 250


def plot_losses(history: dict, save_dir: Union[str, Path]):
    """
    Plots training & validation loss curves from a history object.
    Also plots the learning rate if available in the history.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss'.
        save_dir (str | Path): Directory to save the plot image.
    """
    train_loss = history.get(PyTorchLogKeys.TRAIN_LOSS, [])
    val_loss = history.get(PyTorchLogKeys.VAL_LOSS, [])
    lr_history = history.get(PyTorchLogKeys.LEARNING_RATE, [])
    
    if not train_loss and not val_loss:
        _LOGGER.warning("Loss history is empty or incomplete. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI_value)
    
    # --- Plot Losses (Left Y-axis) ---
    line_handles = [] # To store line objects for the legend
    
    # Plot training loss only if data for it exists
    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        line1, = ax.plot(epochs, train_loss, 'o-', label='Training Loss', color='tab:blue')
        line_handles.append(line1)
    
    # Plot validation loss only if data for it exists
    if val_loss:
        epochs = range(1, len(val_loss) + 1)
        line2, = ax.plot(epochs, val_loss, 'o-', label='Validation Loss', color='tab:orange')
        line_handles.append(line2)
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, linestyle='--')
    
    # --- Plot Learning Rate (Right Y-axis) ---
    if lr_history:
        ax2 = ax.twinx() # Create a second y-axis
        epochs = range(1, len(lr_history) + 1)
        line3, = ax2.plot(epochs, lr_history, 'g--', label='Learning Rate')
        line_handles.append(line3)
        
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        # Use scientific notation if the LR is very small
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Combine legends from both axes
    ax.legend(handles=line_handles, loc='best')
    
    # ax.grid(True)
    plt.tight_layout()    
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    save_path = save_dir_path / "loss_plot.svg"
    plt.savefig(save_path)
    _LOGGER.info(f"📉 Loss plot saved as '{save_path.name}'")

    plt.close(fig)


def classification_metrics(save_dir: Union[str, Path], 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None, 
                           class_map: Optional[dict[str,int]] = None,
                           config: Optional[Union[BinaryClassificationMetricsFormat,
                                                MultiClassClassificationMetricsFormat,
                                                BinaryImageClassificationMetricsFormat,
                                                MultiClassImageClassificationMetricsFormat]] = None):
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
    
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})
    
    # print("--- Classification Report ---")
    
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
        
        # 1. Drop the 'accuracy' column (single float)
        if 'accuracy' in report_df.columns:
            report_df = report_df.drop(columns=['accuracy'])
        
        # 2. Select all metric rows *except* the last one ('support')
        # 3. Transpose the DataFrame
        plot_df = report_df.iloc[:-1, :].T
        
        fig_height = max(5.0, len(plot_df.index) * 0.5 + 2.0)
        plt.figure(figsize=(7, fig_height), dpi=DPI_value)

        sns.set_theme(font_scale=1.2) # Scale seaborn font
        sns.heatmap(plot_df, 
                    annot=True, 
                    cmap=format_config.cmap, 
                    fmt='.2f',
                    vmin=0.0,
                    vmax=1.0)
        sns.set_theme(font_scale=1.0) # Reset seaborn scale
        plt.title("Classification Report Heatmap")
        plt.tight_layout()
        heatmap_path = save_dir_path / "classification_report_heatmap.svg"
        plt.savefig(heatmap_path)
        _LOGGER.info(f"📊 Report heatmap saved as '{heatmap_path.name}'")
        plt.close()
    except Exception as e:
        _LOGGER.error(f"Could not generate classification report heatmap: {e}")
        
    # --- labels for Confusion Matrix ---
    plot_labels = map_labels
    plot_display_labels = map_display_labels
    
    # Save Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=DPI_value)
    disp_ = ConfusionMatrixDisplay.from_predictions(y_true, 
                                            y_pred, 
                                            cmap=format_config.cmap, 
                                            ax=ax_cm, 
                                            normalize='true',
                                            labels=plot_labels,
                                            display_labels=plot_display_labels)
    
    disp_.im_.set_clim(vmin=0.0, vmax=1.0)
    
    # Turn off gridlines
    ax_cm.grid(False)
    
    # Manually update font size of cell texts
    for text in ax_cm.texts:
        text.set_fontsize(format_config.font_size)

    fig_cm.tight_layout()
    
    ax_cm.set_title("Confusion Matrix")
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
            
            fig_roc, ax_roc = plt.subplots(figsize=(6, 6), dpi=DPI_value)
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color=format_config.ROC_PR_line)
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_title(f'Receiver Operating Characteristic{plot_title}')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right')
            ax_roc.grid(True)
            roc_path = save_dir_path / f"roc_curve{save_suffix}.svg"
            plt.savefig(roc_path)
            plt.close(fig_roc)

            # --- Save Precision-Recall Curve ---
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            ap_score = average_precision_score(y_true_binary, y_score)
            fig_pr, ax_pr = plt.subplots(figsize=(6, 6), dpi=DPI_value)
            ax_pr.plot(recall, precision, label=f'Avg Precision = {ap_score:.2f}', color=format_config.ROC_PR_line)
            ax_pr.set_title(f'Precision-Recall Curve{plot_title}')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.legend(loc='lower left')
            ax_pr.grid(True)
            pr_path = save_dir_path / f"pr_curve{save_suffix}.svg"
            plt.savefig(pr_path)
            plt.close(fig_pr)
            
            # --- Save Calibration Plot ---
            fig_cal, ax_cal = plt.subplots(figsize=(8, 8), dpi=DPI_value)

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
                label=f"Calibration Curve ({format_config.calibration_bins} bins)",
                line_kws={
                    'color': format_config.ROC_PR_line, 
                    'linestyle': '--', 
                    'linewidth': 2,
                    }
            )
            
            ax_cal.set_title(f'Reliability Curve{plot_title}')
            ax_cal.set_xlabel('Mean Predicted Probability')
            ax_cal.set_ylabel('Fraction of Positives')
            
            # --- Step 3: Set final limits *after* plotting ---
            ax_cal.set_ylim(0.0, 1.0) 
            ax_cal.set_xlim(0.0, 1.0)
            
            ax_cal.legend(loc='lower right')
            ax_cal.grid(True)
            plt.tight_layout()
            
            cal_path = save_dir_path / f"calibration_plot{save_suffix}.svg"
            plt.savefig(cal_path)
            plt.close(fig_cal)
            
        _LOGGER.info(f"📈 Saved {len(class_indices_to_plot)} sets of ROC, Precision-Recall, and Calibration plots.")
            
    # restore RC params
    plt.rcParams.update(original_rc_params)


def regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_dir: Union[str, Path],
    config: Optional[RegressionMetricsFormat] = None
):
    """
    Saves regression metrics and plots.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        save_dir (str | Path): Directory to save plots and report.
        config (RegressionMetricsFormat, optional): Formatting configuration object.
    """
    
    # --- Parse Config or use defaults ---
    if config is None:
        # Create a default config if one wasn't provided
        format_config = _BaseRegressionFormat()
    else:
        format_config = config
        
    # --- Set Matplotlib font size ---
    original_rc_params = plt.rcParams.copy()
    plt.rcParams.update({'font.size': format_config.font_size})
    
    # --- Calculate Metrics ---
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

    # --- Save residual plot ---
    residuals = y_true - y_pred
    fig_res, ax_res = plt.subplots(figsize=(8, 6), dpi=DPI_value)
    ax_res.scatter(y_pred, residuals, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_res.axhline(0, color=format_config.residual_line_color, linestyle='--')
    ax_res.set_xlabel("Predicted Values")
    ax_res.set_ylabel("Residuals")
    ax_res.set_title("Residual Plot")
    ax_res.grid(True)
    plt.tight_layout()
    res_path = save_dir_path / "residual_plot.svg"
    plt.savefig(res_path)
    _LOGGER.info(f"📈 Residual plot saved as '{res_path.name}'")
    plt.close(fig_res)

    # --- Save true vs predicted plot ---
    fig_tvp, ax_tvp = plt.subplots(figsize=(8, 6), dpi=DPI_value)
    ax_tvp.scatter(y_true, y_pred, 
                   alpha=format_config.scatter_alpha, 
                   color=format_config.scatter_color)
    ax_tvp.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                linestyle='--', 
                lw=2,
                color=format_config.ideal_line_color)
    ax_tvp.set_xlabel('True Values')
    ax_tvp.set_ylabel('Predictions')
    ax_tvp.set_title('True vs. Predicted Values')
    ax_tvp.grid(True)
    plt.tight_layout()
    tvp_path = save_dir_path / "true_vs_predicted_plot.svg"
    plt.savefig(tvp_path)
    _LOGGER.info(f"📉 True vs. Predicted plot saved as '{tvp_path.name}'")
    plt.close(fig_tvp)
    
    # --- Save Histogram of Residuals ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6), dpi=DPI_value)
    sns.histplot(residuals, kde=True, ax=ax_hist, 
                 bins=format_config.hist_bins, 
                 color=format_config.scatter_color)
    ax_hist.set_xlabel("Residual Value")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Distribution of Residuals")
    ax_hist.grid(True)
    plt.tight_layout()
    hist_path = save_dir_path / "residuals_histogram.svg"
    plt.savefig(hist_path)
    _LOGGER.info(f"📊 Residuals histogram saved as '{hist_path.name}'")
    plt.close(fig_hist)
    
    # --- Restore RC params ---
    plt.rcParams.update(original_rc_params)
    

def shap_summary_plot(model, 
                      background_data: Union[torch.Tensor,np.ndarray], 
                      instances_to_explain: Union[torch.Tensor,np.ndarray], 
                      feature_names: Optional[list[str]], 
                      save_dir: Union[str, Path],
                      device: torch.device = torch.device('cpu'),
                      explainer_type: Literal['deep', 'kernel'] = 'kernel'):
    """
    Calculates SHAP values and saves summary plots and data.

    Args:
        model (nn.Module): The trained PyTorch model.
        background_data (torch.Tensor): A sample of data for the explainer background.
        instances_to_explain (torch.Tensor): The specific data instances to explain.
        feature_names (list of str | None): Names of the features for plot labeling.
        save_dir (str | Path): Directory to save SHAP artifacts.
        device (torch.device): The torch device for SHAP calculations.
        explainer_type (Literal['deep', 'kernel']): The explainer to use.
            - 'deep': Uses shap.DeepExplainer. Fast and efficient for
              PyTorch models.
            - 'kernel': Uses shap.KernelExplainer. Model-agnostic but EXTREMELY
              slow and memory-intensive.
    """
    
    _LOGGER.info(f"📊 Running SHAP Value Explanation Using {explainer_type.upper()} Explainer")
    
    model.eval()
    # model.cpu() # Run explanations on CPU
    
    shap_values = None
    instances_to_explain_np = None

    if explainer_type == 'deep':
        # --- 1. Use DeepExplainer  ---
        
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
        shap_values = explainer.shap_values(instances_to_explain)
        instances_to_explain_np = instances_to_explain.cpu().numpy()

    elif explainer_type == 'kernel':
        # --- 2. Use KernelExplainer ---
        _LOGGER.warning(
            "KernelExplainer is memory-intensive and slow. Consider reducing the number of instances to explain if the process terminates unexpectedly."
        )

        # Ensure data is np.ndarray
        if isinstance(background_data, torch.Tensor):
            background_data_np = background_data.cpu().numpy()
        else:
            background_data_np = background_data
            
        if isinstance(instances_to_explain, torch.Tensor):
            instances_to_explain_np = instances_to_explain.cpu().numpy()
        else:
            instances_to_explain_np = instances_to_explain
        
        if np.isnan(background_data_np).any() or np.isnan(instances_to_explain_np).any():
            _LOGGER.error("Input data for SHAP contains NaN values. Aborting explanation.")
            return
        
        # Summarize background data
        background_summary = shap.kmeans(background_data_np, 30) 
        
        def prediction_wrapper(x_np: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x_np).float().to(device)
            with torch.no_grad():
                output = model(x_torch)
            # Return as numpy array
            return output.cpu().numpy()

        explainer = shap.KernelExplainer(prediction_wrapper, background_summary)
        # print("Calculating SHAP values with KernelExplainer...")
        shap_values = explainer.shap_values(instances_to_explain_np, l1_reg="aic")
        # instances_to_explain_np is already set
    
    else:
        _LOGGER.error(f"Invalid explainer_type: '{explainer_type}'. Must be 'deep' or 'kernel'.")
        raise ValueError()
    
    if not isinstance(shap_values, list) and shap_values.ndim == 3 and shap_values.shape[2] == 1: # type: ignore
        # _LOGGER.info("Squeezing SHAP values from (N, F, 1) to (N, F) for regression plot.")
        shap_values = shap_values.squeeze(-1) # type: ignore

    # --- 3. Plotting and Saving ---
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    plt.ioff()
    
    # Convert instances to a DataFrame. robust way to ensure SHAP correctly maps values to feature names.
    if feature_names is None:
        # Create generic names if none were provided
        num_features = instances_to_explain_np.shape[1]
        feature_names = [f'feature_{i}' for i in range(num_features)]
        
    instances_df = pd.DataFrame(instances_to_explain_np, columns=feature_names)
    
    # Save Bar Plot
    bar_path = save_dir_path / "shap_bar_plot.svg"
    shap.summary_plot(shap_values, instances_df, plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value Impact", labelpad=10)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(bar_path)
    _LOGGER.info(f"📊 SHAP bar plot saved as '{bar_path.name}'")
    plt.close()

    # Save Dot Plot
    dot_path = save_dir_path / "shap_dot_plot.svg"
    shap.summary_plot(shap_values, instances_df, plot_type="dot", show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value Impact", labelpad=10)
    if plt.gcf().axes and len(plt.gcf().axes) > 1:
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
    
    # Handle multi-class (list of arrays) vs. regression (single array)
    if isinstance(shap_values, list):
        mean_abs_shap = np.abs(np.stack(shap_values)).mean(axis=0).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    mean_abs_shap = mean_abs_shap.flatten()
        
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
    
    plt.figure(figsize=(10, 8), dpi=DPI_value)

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
