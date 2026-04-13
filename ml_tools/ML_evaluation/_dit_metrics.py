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
    if (real_num is not None and len(real_num) > 0 and 
        gen_num is not None and len(gen_num) > 0 and 
        num_target_names is not None and len(num_target_names) > 0):
        
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
            
            # Calculate Relative Wasserstein Distance (handle zero variance)
            real_std = np.std(real_i)
            rel_w_dist = w_dist / real_std if real_std > 0 else np.nan
            
            metrics_summary.append({
                'Feature': name,
                'Wasserstein Distance': w_dist,
                'Relative Wasserstein Distance': rel_w_dist,
                'KS Statistic': ks_stat,
                'KS p-value': ks_pval
            })
            
            # Plot KDE Overlays
            fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
            
            # Handle zero variance for Real Data
            if np.isclose(np.std(real_i), 0, atol=1e-5):
                ax.axvline(x=real_i[0], color=format_config.real_color, linestyle='--', linewidth=2.5, label='Real Data (Constant)')
            else:
                sns.kdeplot(real_i, fill=True, color=format_config.real_color, alpha=format_config.alpha, label='Real Data', ax=ax)
            
            # Handle zero variance for Generated Data
            if np.isclose(np.std(gen_i), 0, atol=1e-5):
                ax.axvline(x=gen_i[0], color=format_config.gen_color, linestyle='--', linewidth=2.5, label='Generated Data (Constant)')
            else:
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
                
        if not summary_df.empty:
            # Calculate averages, skipping NaNs automatically with pandas .mean()
            avg_w_dist = summary_df['Wasserstein Distance'].mean()
            avg_rel_w_dist = summary_df['Relative Wasserstein Distance'].mean()
            avg_ks_stat = summary_df['KS Statistic'].mean()
            
            overall_report_lines.append(f"Average Wasserstein Distance: {avg_w_dist:.4f}")
            overall_report_lines.append(f"Average Relative Wasserstein Distance: {avg_rel_w_dist:.4f}")
            overall_report_lines.append(f"Average KS Statistic: {avg_ks_stat:.4f}")
        else:
            overall_report_lines.append("Average Continuous Metrics: N/A")

    # ==========================================
    # 2. Categorical Features (Proportions)
    # ==========================================
    if (real_cat_list is not None and len(real_cat_list) > 0 and 
        gen_cat_list is not None and len(gen_cat_list) > 0 and 
        cat_target_names is not None and len(cat_target_names) > 0):
        
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
        
        if not cat_summary_df.empty and 'Total Variation Distance' in cat_summary_df.columns:
            overall_report_lines.append(f"Average Total Variation Distance: {cat_summary_df['Total Variation Distance'].mean():.4f}")
        else:
            overall_report_lines.append("Average Total Variation Distance: N/A")
    
    # ==========================================
    # 3. Multivariate Relationships (Numerical Correlation)
    # ==========================================
    if (real_num is not None and len(real_num) > 0 and 
        gen_num is not None and len(gen_num) > 0 and 
        num_target_names is not None and len(num_target_names) > 1):
        
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
                    cmap=format_config.cmap, 
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


# Metrics explanation:
"""
We generally want to answer two main questions: 
    1. Do the individual features look realistic? (Marginal Distributions)
    2. Do the features interact with each other realistically? (Multivariate Relationships).

1. Marginal Distributions: Categorical Data

    1.1 Total Variation Distance (TVD):
    
    Lower is better
    
    - A measure of the difference between two discrete probability distributions (categorical features).
    - When looking at a bar chart of the "Real" categories overlapping with the "Generated" categories. 
    TVD is exactly half of the sum of the absolute differences between the heights of those bars.
    - TVD ranges from 0 to 1, where 0 means the distributions are identical and 1 means they are completely different.

2. Marginal Distributions: Continuous Data
    
    2.1 Wasserstein Distance (Earth Mover's Distance):
    
    Lower is better
    
    - A metric for continuous features that measures the minimum "cost" 
    required to transform the generated distribution into the real distribution.
    - Imagine the real distribution as a specific landscape of dirt piles, and the generated distribution as a different landscape. 
    The Wasserstein distance calculates the minimum amount of "dirt" you have to move, 
    multiplied by the distance you have to move it, to make the generated landscape look exactly like the real one.
    - Ranges from 0 to infinity, where 0 means the distributions are identical.
    
    2.2 Kolmogorov-Smirnov (KS) Statistic:
    
    Lower is better
    
    - A metric that compares the Cumulative Distribution Functions (CDFs) of two continuous arrays.
    - Look at the CDF curves (which start at 0 and climb to 100%). 
    The KS Statistic is simply the maximum vertical gap between the real data's curve and the generated data's curve at any single point.
    - Ranges from 0 to 1, where 0 means the distributions are identical. A higher KS statistic indicates a greater difference between the two distributions.
    
    2.3 KS p-value:
    
    Higher is better (ideally > 0.05)
    
    - The statistical confidence level accompanying the KS Statistic. 
    It tests the "null hypothesis" that both your real and generated samples were drawn from the exact same underlying distribution.
    - A high p-value (commonly above 0.05) means you fail to reject the null hypothesis, 
    suggesting that the real and generated data could plausibly come from the same distribution (what we want).
    - A low p-value means you reject the null hypothesis, indicating a statistically significant difference between the distributions.
    - NOTE: Large datasets tend to yield very low p-values. In practice, it's often more informative to look at the KS Statistic itself rather than relying solely on the p-value for large samples.

3. Multivariate Relationships: Continuous Data

    3.1 Correlation Matrix Mean Absolute Error (MAE):
    
    Lower is better
    
    - The average absolute difference between the correlation matrix of the real data and the correlation matrix of the generated data.
    - If Feature A and Feature B have a correlation of 0.8 in the real data, but a correlation of 0.6 in the generated data, the absolute error is 0.2.
    - A lower MAE indicates that the generated data better captures the relationships between features as observed in the real data.

    3.2 Correlation Matrix Mean Squared Error (MSE):
    
    Lower is better
    
    - Similar to MAE but squares the differences before averaging, which penalizes larger errors more heavily.
    - A lower MSE indicates that the generated data better captures the relationships between features, 
    with a stronger emphasis on avoiding large discrepancies in correlations compared to the real data.
    
"""
