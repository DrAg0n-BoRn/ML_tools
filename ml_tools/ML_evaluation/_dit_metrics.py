import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional
from scipy.stats import wasserstein_distance, ks_2samp

from ..ML_configuration import FormatTabularDiffusionMetrics

from ..keys._keys import _EvaluationConfig
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

from ._helpers import check_and_abbreviate_name


_LOGGER = get_logger("DiTMetrics")


__all__ = [
    "dit_generation_metrics"
]

DPI_value = _EvaluationConfig.DPI
DISTRIBUTION_PLOT_SIZE = _EvaluationConfig.REGRESSION_PLOT_SIZE


def dit_generation_metrics(
    real_num: Optional[np.ndarray],
    gen_num: Optional[np.ndarray],
    num_target_names: Optional[list[str]],
    real_cat_list: Optional[list[np.ndarray]],
    gen_cat_list: Optional[list[np.ndarray]],
    cat_target_names: Optional[list[str]],
    cat_class_maps: Optional[list[Optional[dict[str, int]]]],
    save_dir: Union[str, Path],
    config: Optional[FormatTabularDiffusionMetrics] = None
):
    """
    Evaluates Diffusion generation performance by comparing real vs. generated distributions for tabular data. 
    
    Args:
        real_num: 2D array of real numerical features (shape: [n_samples, n_num_features])
        gen_num: 2D array of generated numerical features (shape: [n_samples, n_num_features])
        num_target_names: List of names for numerical features (length should match num features in real_num/gen_num)
        real_cat_list: List of 1D arrays for real categorical features (each array shape: [n_samples])
        gen_cat_list: List of 1D arrays for generated categorical features (each array shape: [n_samples])
        cat_target_names: List of names for categorical features (length should match length of real_cat_list/gen_cat_list)
        cat_class_maps: Optional list of dicts mapping categorical class labels to integers for each categorical feature (length should match cat_target_names)
        save_dir: Directory path to save evaluation reports and plots
        config: Optional configuration object for formatting plots and reports (if None, defaults will be used)
    """
    if config is None:
        format_config = FormatTabularDiffusionMetrics()
    else:
        format_config = config
        
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    # _LOGGER.info(f"Starting DiT generation evaluation. Saving to '{save_dir_path.name}'")

    overall_report_lines = ["--- Diffusion Model Generation Report ---"]
    
    # if no feature names are provided, make generic ones for reporting purposes
    if num_target_names is None and real_num is not None:
        num_target_names = [f"Feature_{i}" for i in range(real_num.shape[1])]
        
    if cat_target_names is None and real_cat_list is not None:
        cat_target_names = [f"CatFeature_{i}" for i in range(len(real_cat_list))]

    # ==========================================
    # 1. Continuous Features (Marginal Distributions)
    # ==========================================
    if real_num is not None and gen_num is not None and num_target_names is not None:
        # _LOGGER.info("Processing continuous feature distributions...")
        overall_report_lines.append(f"\n[Continuous Features: {len(num_target_names)}]")
        
        metrics_summary = []
        
        for i, name in enumerate(num_target_names):
            real_i = real_num[:, i]
            gen_i = gen_num[:, i]
            
            # abbreviate names for plots
            abbreviated_name = check_and_abbreviate_name(name)
            
            # Calculate Statistical Distances
            w_dist = wasserstein_distance(real_i, gen_i)
            ks_stat, ks_pval = ks_2samp(real_i, gen_i)
            
            metrics_summary.append({
                'Feature': name,
                'Wasserstein Distance': w_dist,
                'KS Statistic': ks_stat,
                'KS p-value': ks_pval
            })
            
            # Plot KDE Overlays
            fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
            sns.kdeplot(real_i, fill=True, color=format_config.real_color, alpha=format_config.alpha, label='Real Data', ax=ax)
            sns.kdeplot(gen_i, fill=True, color=format_config.gen_color, alpha=format_config.alpha, label='Generated Data', ax=ax)
            
            ax.set_title(f"Distribution Comparison: {abbreviated_name}", fontsize=format_config.font_size + 2)
            ax.set_xlabel("Value", fontsize=format_config.font_size)
            ax.set_ylabel("Density", fontsize=format_config.font_size)
            
            ax.tick_params(axis='x', labelsize=format_config.xtick_size)
            ax.tick_params(axis='y', labelsize=format_config.ytick_size)
            ax.legend(fontsize=format_config.legend_size)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plot_path = save_dir_path / f"kde_{sanitize_filename(name)}.svg"
            plt.savefig(plot_path)
            plt.close(fig)
            
        summary_df = pd.DataFrame(metrics_summary)
        csv_path = save_dir_path / "continuous_generation_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        _LOGGER.info(f"🔢 Continuous distribution summary saved to '{csv_path.name}'")
        overall_report_lines.append(f"Average Wasserstein Distance: {summary_df['Wasserstein Distance'].mean():.4f}")

    # ==========================================
    # 2. Categorical Features (Proportions)
    # ==========================================
    if real_cat_list is not None and gen_cat_list is not None and cat_target_names is not None:
        # _LOGGER.info("Processing categorical feature distributions...")
        overall_report_lines.append(f"\n[Categorical Features: {len(cat_target_names)}]")
        
        cat_metrics_summary = []
        
        for i, feat_name in enumerate(cat_target_names):
            real_c = real_cat_list[i]
            gen_c = gen_cat_list[i]
            
            # abbreviate names for plots
            abbreviated_cat_name = check_and_abbreviate_name(feat_name)
            
            # Count frequencies
            real_counts = pd.Series(real_c).value_counts(normalize=True)
            gen_counts = pd.Series(gen_c).value_counts(normalize=True)
            
            # Align indices to compare
            all_classes = sorted(list(set(real_counts.index) | set(gen_counts.index)))
            real_props = np.array([real_counts.get(cls, 0.0) for cls in all_classes])
            gen_props = np.array([gen_counts.get(cls, 0.0) for cls in all_classes])
            
            # Total Variation Distance (TVD) for discrete distributions
            tvd = 0.5 * np.sum(np.abs(real_props - gen_props))
            
            cat_metrics_summary.append({
                'Feature': feat_name,
                'Total Variation Distance': tvd
            })
            
            # Resolve labels if class map exists
            plot_labels = all_classes
            if cat_class_maps is not None and i < len(cat_class_maps) and cat_class_maps[i] is not None:
                inv_map = {v: k for k, v in cat_class_maps[i].items()} # type: ignore
                plot_labels = [inv_map.get(cls, str(cls)) for cls in all_classes]

            # Bar plot
            x = np.arange(len(all_classes))
            width = 0.35

            fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
            ax.bar(x - width/2, real_props, width, label='Real Data', color=format_config.real_color, alpha=format_config.alpha)
            ax.bar(x + width/2, gen_props, width, label='Generated Data', color=format_config.gen_color, alpha=format_config.alpha)

            ax.set_title(f"Proportion Comparison: {abbreviated_cat_name}", fontsize=format_config.font_size + 2)
            ax.set_xlabel("Categories", fontsize=format_config.font_size)
            ax.set_ylabel("Proportion", fontsize=format_config.font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_labels, rotation=45 if len(plot_labels) > 3 else 0, ha='right')
            ax.legend(fontsize=format_config.legend_size)
            ax.grid(True, linestyle='--', alpha=0.6, axis='y')
            
            plt.tight_layout()
            plot_path = save_dir_path / f"bar_{sanitize_filename(feat_name)}.svg"
            plt.savefig(plot_path)
            plt.close(fig)
            
        cat_summary_df = pd.DataFrame(cat_metrics_summary)
        cat_csv_path = save_dir_path / "categorical_generation_summary.csv"
        cat_summary_df.to_csv(cat_csv_path, index=False)
        _LOGGER.info(f"🔢 Categorical distribution summary saved to '{cat_csv_path.name}'")
        overall_report_lines.append(f"Average Total Variation Distance: {cat_summary_df['Total Variation Distance'].mean():.4f}")
    
    # ==========================================
    # 3. Multivariate Relationships (Numerical Correlation)
    # ==========================================
    if real_num is not None and gen_num is not None and num_target_names is not None and real_num.shape[1] > 1:
        # _LOGGER.info("Processing numerical feature correlations...")
        overall_report_lines.append(f"\n[Multivariate Relationships: Numerical Features]")
        
        # Calculate Pearson correlation matrices
        real_df = pd.DataFrame(real_num, columns=num_target_names)
        gen_df = pd.DataFrame(gen_num, columns=num_target_names)
        
        # Fill NaNs with 0 in case some features generated as pure constants (variance=0)
        real_corr = real_df.corr().fillna(0)
        gen_corr = gen_df.corr().fillna(0)
        
        # Calculate difference matrix
        corr_diff = real_corr - gen_corr
        corr_diff_abs = corr_diff.abs()
        
        # Calculate Correlation Matrix Error 
        # Using the upper triangle to avoid duplicating off-diagonal elements and excluding the diagonal (always 1)
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        
        if mask.sum() > 0:
            corr_mae = corr_diff_abs.where(mask).mean().mean()
            corr_mse = (corr_diff ** 2).where(mask).mean().mean()
        else:
            corr_mae = 0.0
            corr_mse = 0.0
            
        overall_report_lines.append(f"Correlation Matrix MAE: {corr_mae:.4f}")
        overall_report_lines.append(f"Correlation Matrix MSE: {corr_mse:.4f}")
        
        # Plot Correlation Difference Heatmap
        # Scale figure size dynamically if there are many features
        fig_size_xy = max(8, len(num_target_names) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size_xy, fig_size_xy), dpi=DPI_value)
        
        # Only display annotations if there are fewer than 15 features to avoid clutter
        show_annotations = len(num_target_names) <= 15
        
        sns.heatmap(corr_diff_abs, 
                    annot=show_annotations, 
                    fmt=".2f", 
                    cmap="Reds", 
                    cbar_kws={'label': 'Absolute Correlation Difference'},
                    ax=ax,
                    vmin=0, vmax=1.0)
                    
        ax.set_title("Absolute Difference in Feature Correlations (Real vs Gen)", fontsize=format_config.font_size + 2)
        plt.tight_layout()
        plot_path = save_dir_path / "correlation_difference_heatmap.svg"
        plt.savefig(plot_path)
        plt.close(fig)
        
        _LOGGER.info(f"🔗 Correlation matrix metrics calculated and heatmap saved to '{plot_path.name}'")
    
    # ==========================================
    # 4. Save Overall Report
    # ==========================================
    report_string = "\n".join(overall_report_lines)
    report_path = save_dir_path / "global_generation_report.txt"
    report_path.write_text(report_string, encoding="utf-8")
    _LOGGER.info(f"🌎 Global generation report saved to '{report_path.name}'")
