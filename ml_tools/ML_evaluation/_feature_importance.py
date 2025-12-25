import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap
from pathlib import Path
from typing import Union, Optional, Literal
import warnings

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import SHAPKeys, _EvaluationConfig


_LOGGER = get_logger("Feature Importance")


__all__ = [
    "shap_summary_plot",
    "plot_attention_importance",
    "multi_target_shap_summary_plot",
]


DPI_value = _EvaluationConfig.DPI


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


def plot_attention_importance(weights: list[torch.Tensor], feature_names: Optional[list[str]], save_dir: Union[str, Path], top_n: int = 10):
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


def multi_target_shap_summary_plot(
    model: torch.nn.Module,
    background_data: Union[torch.Tensor, np.ndarray],
    instances_to_explain: Union[torch.Tensor, np.ndarray],
    feature_names: list[str],
    target_names: list[str],
    save_dir: Union[str, Path],
    device: torch.device = torch.device('cpu'),
    explainer_type: Literal['deep', 'kernel'] = 'kernel'
):
    """
    DEPRECATED
    
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
    _LOGGER.warning("This function is deprecated and may be removed in future versions. Use Captum module instead.")
    
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

