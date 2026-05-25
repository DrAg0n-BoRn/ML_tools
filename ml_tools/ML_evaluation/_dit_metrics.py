import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency
# PCA and ROC analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from ..ML_configuration import FormatTabularDiffusionMetrics

from ..keys._keys import _EvaluationConfig
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger

# from ._helpers import check_and_abbreviate_name
from ._helpers import wrap_text


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
    continuous_lines = _evaluate_continuous_features(
        real_num, gen_num, num_target_names, save_dir_path, format_config
    )
    overall_report_lines.extend(continuous_lines)

    # ==========================================
    # 2. Categorical Features (Proportions)
    # ==========================================
    categorical_lines = _evaluate_categorical_features(
        real_cat_list, gen_cat_list, cat_target_names, cat_class_maps, save_dir_path, format_config
    )
    overall_report_lines.extend(categorical_lines)
    
    # ==========================================
    # 3. Multivariate Relationships (Numerical Correlation)
    # ==========================================
    numerical_corr_lines = _evaluate_numerical_correlations(
        real_num, gen_num, num_target_names, save_dir_path, format_config
    )
    overall_report_lines.extend(numerical_corr_lines)
    
    # ==========================================
    # 4. Global Multivariate Projections (PCA)
    # ==========================================
    if real_num is not None and gen_num is not None and num_target_names is not None and len(num_target_names) > 1:
        overall_report_lines.append(f"\n[Global Projection: PCA]")
        _plot_pca_projection(real_num, gen_num, save_dir_path, format_config)
        overall_report_lines.append("- PCA 2D projection plot generated.")

    # ==========================================
    # 5. Global Categorical Associations (Cramer's V)
    # ==========================================
    if real_cat_list is not None and gen_cat_list is not None and cat_target_names is not None and len(cat_target_names) > 1:
        overall_report_lines.append(f"\n[Global Categorical Associations: Cramer's V]")
        _plot_cramers_v_heatmap(real_cat_list, gen_cat_list, cat_target_names, save_dir_path, format_config)
        overall_report_lines.append("- Cramer's V difference heatmap generated.")

    # ==========================================
    # 6. Global Realism Evaluation (Discriminator ROC)
    # ==========================================
    has_num = real_num is not None and gen_num is not None and len(real_num) > 0
    has_cat = real_cat_list is not None and gen_cat_list is not None and len(real_cat_list) > 0
    if has_num or has_cat:
        overall_report_lines.append(f"\n[Global Realism: Discriminator ROC]")
        _plot_discriminator_roc(real_num, gen_num, real_cat_list, gen_cat_list, save_dir_path, format_config)
        overall_report_lines.append("- Discriminator ROC curve generated (target ideal AUC is ~0.50).")
    
    # ==========================================
    # 7. Save Overall Report
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

# =====================================================================
# PRIVATE HELPER FUNCTIONS
# =====================================================================

def _evaluate_continuous_features(
    real_num: Optional[np.ndarray],
    gen_num: Optional[np.ndarray],
    num_target_names: Optional[list[str]],
    save_dir_path: Path,
    format_config: FormatTabularDiffusionMetrics
) -> list[str]:
    """Evaluates continuous marginal distributions and returns report lines."""
    report_lines = []
    
    if not (real_num is not None and len(real_num) > 0 and 
            gen_num is not None and len(gen_num) > 0 and 
            num_target_names is not None and len(num_target_names) > 0):
        return report_lines

    report_lines.append(f"\n[Continuous Features: {len(num_target_names)}]")
    metrics_summary = []
    
    local_save_dir = save_dir_path / "numerical_distributions"
    local_save_dir.mkdir(exist_ok=True)
    
    for i, name in enumerate(num_target_names):
        real_i = real_num[:, i]
        gen_i = gen_num[:, i]
        
        # abbreviated_name = check_and_abbreviate_name(name)
        
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
        
        if np.isclose(np.std(real_i), 0, atol=1e-5):
            ax.axvline(x=real_i[0], color=format_config.real_color, linestyle='--', linewidth=2.5, label='Real Data (Constant)')
        else:
            sns.kdeplot(real_i, fill=True, color=format_config.real_color, alpha=format_config.alpha, label='Real Data', ax=ax)
        
        if np.isclose(np.std(gen_i), 0, atol=1e-5):
            ax.axvline(x=gen_i[0], color=format_config.gen_color, linestyle='--', linewidth=2.5, label='Generated Data (Constant)')
        else:
            sns.kdeplot(gen_i, fill=True, color=format_config.gen_color, alpha=format_config.alpha, label='Generated Data', ax=ax)    
        
        # ax.set_title(name, fontsize=format_config.font_size + 2, pad=_EvaluationConfig.LABEL_PADDING) # Remove title, use x-label instead for cleaner look
        ax.set_xlabel(name, fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        ax.set_ylabel("Density", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        
        ax.tick_params(axis='x', labelsize=format_config.xtick_size)
        # Remove the confusing density numbers but keep the axis line
        ax.set_yticks([])
        # ax.tick_params(axis='y', labelsize=format_config.ytick_size)
        ax.legend(fontsize=format_config.legend_size)
        ax.grid(True, linestyle='--', alpha=0.6, axis='x') # use X axis only
        
        # Turn off the top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = local_save_dir / f"{sanitize_filename(name)}_kde.svg"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
    summary_df = pd.DataFrame(metrics_summary)
    csv_path = save_dir_path / "continuous_generation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    _LOGGER.info(f"📊 Continuous distribution summary saved to '{csv_path.name}'")
            
    if not summary_df.empty:
        avg_w_dist = summary_df['Wasserstein Distance'].mean()
        avg_rel_w_dist = summary_df['Relative Wasserstein Distance'].mean()
        avg_ks_stat = summary_df['KS Statistic'].mean()
        
        report_lines.append(f"Average Wasserstein Distance: {avg_w_dist:.4f}")
        report_lines.append(f"Average Relative Wasserstein Distance: {avg_rel_w_dist:.4f}")
        report_lines.append(f"Average KS Statistic: {avg_ks_stat:.4f}")
    else:
        report_lines.append("Average Continuous Metrics: N/A")
        
    return report_lines


def _evaluate_categorical_features(
    real_cat_list: Optional[list[np.ndarray]],
    gen_cat_list: Optional[list[np.ndarray]],
    cat_target_names: Optional[list[str]],
    cat_class_maps: Optional[list[Optional[dict[str, int]]]],
    save_dir_path: Path,
    format_config: FormatTabularDiffusionMetrics
) -> list[str]:
    """Evaluates categorical marginal distributions and returns report lines."""
    report_lines = []
    
    if not (real_cat_list is not None and len(real_cat_list) > 0 and 
            gen_cat_list is not None and len(gen_cat_list) > 0 and 
            cat_target_names is not None and len(cat_target_names) > 0):
        return report_lines
        
    report_lines.append(f"\n[Categorical Features: {len(cat_target_names)}]")
    cat_metrics_summary = []
    
    local_save_dir = save_dir_path / "categorical_proportions"
    local_save_dir.mkdir(exist_ok=True)
    
    for i, feat_name in enumerate(cat_target_names):
        real_c = real_cat_list[i]
        gen_c = gen_cat_list[i]
        
        real_counts = pd.Series(real_c).value_counts(normalize=True)
        gen_counts = pd.Series(gen_c).value_counts(normalize=True)
        
        all_classes = sorted(list(set(real_counts.index) | set(gen_counts.index)))
        real_props = np.array([real_counts.get(cls, 0.0) for cls in all_classes])
        gen_props = np.array([gen_counts.get(cls, 0.0) for cls in all_classes])
        
        tvd = 0.5 * np.sum(np.abs(real_props - gen_props))
        
        cat_metrics_summary.append({
            'Feature': feat_name,
            'Total Variation Distance': tvd
        })
        
        plot_labels = all_classes
        if cat_class_maps is not None and i < len(cat_class_maps) and cat_class_maps[i] is not None:
            inv_map = {v: k for k, v in cat_class_maps[i].items()} # type: ignore
            plot_labels = [inv_map.get(cls, str(cls)) for cls in all_classes]
            
        # Wrap long category class names
        plot_labels = [wrap_text(str(label)) for label in plot_labels]

        x = np.arange(len(all_classes))
        width = 0.35

        fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
        ax.bar(x - width/2, real_props, width, label='Real Data', color=format_config.real_color, alpha=format_config.alpha)
        ax.bar(x + width/2, gen_props, width, label='Generated Data', color=format_config.gen_color, alpha=format_config.alpha)

        # ax.set_title(feat_name, fontsize=format_config.font_size + 2, pad=_EvaluationConfig.LABEL_PADDING)
        ax.set_xlabel(feat_name, fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING) # Use x-label for feature name instead of title for cleaner and consistent look
        ax.set_ylabel("Proportion", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        ax.set_xticks(x)
        
        # smart font size adjustment based on number of categories to prevent overcrowding
        font_shrink_constant = 20
        ax.set_xticklabels(plot_labels, 
                   rotation=45 if len(plot_labels) > 3 else 0, 
                   ha='right' if len(plot_labels) > 3 else 'center',
                   fontsize=max(format_config.xtick_size // 2, int(format_config.xtick_size * (font_shrink_constant / (font_shrink_constant + len(plot_labels))))))
        
        ax.tick_params(axis='y', labelsize=format_config.ytick_size)
        ax.legend(fontsize=format_config.legend_size)
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Turn off the top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = local_save_dir / f"{sanitize_filename(feat_name)}_bar.svg"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
    cat_summary_df = pd.DataFrame(cat_metrics_summary)
    cat_csv_path = save_dir_path / "categorical_generation_summary.csv"
    cat_summary_df.to_csv(cat_csv_path, index=False)
    _LOGGER.info(f"📊 Categorical distribution summary saved to '{cat_csv_path.name}'")
    
    if not cat_summary_df.empty and 'Total Variation Distance' in cat_summary_df.columns:
        report_lines.append(f"Average Total Variation Distance: {cat_summary_df['Total Variation Distance'].mean():.4f}")
    else:
        report_lines.append("Average Total Variation Distance: N/A")
        
    return report_lines


def _evaluate_numerical_correlations(
    real_num: Optional[np.ndarray],
    gen_num: Optional[np.ndarray],
    num_target_names: Optional[list[str]],
    save_dir_path: Path,
    format_config: FormatTabularDiffusionMetrics
) -> list[str]:
    """Evaluates numerical multivariate relationships (correlations) and returns report lines."""
    report_lines = []
    
    if not (real_num is not None and len(real_num) > 0 and 
            gen_num is not None and len(gen_num) > 0 and 
            num_target_names is not None and len(num_target_names) > 1):
        return report_lines
        
    report_lines.append(f"\n[Multivariate Relationships: Numerical Features]")
    
    wrapped_num_names = [wrap_text(name) for name in num_target_names]
    
    real_df = pd.DataFrame(real_num, columns=wrapped_num_names)
    gen_df = pd.DataFrame(gen_num, columns=wrapped_num_names)
    
    real_corr = real_df.corr().fillna(0)
    gen_corr = gen_df.corr().fillna(0)
    
    corr_diff = real_corr - gen_corr
    corr_diff_abs = corr_diff.abs()
    
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    
    if mask.sum() > 0:
        corr_mae = corr_diff_abs.where(mask).mean().mean()
        corr_mse = (corr_diff ** 2).where(mask).mean().mean()
    else:
        corr_mae = 0.0
        corr_mse = 0.0
        
    report_lines.append(f"Correlation Matrix MAE: {corr_mae:.4f}")
    report_lines.append(f"Correlation Matrix MSE: {corr_mse:.4f}")
    
    num_feats = len(num_target_names)
    fig_size_xy = max(8, num_feats * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size_xy, fig_size_xy), dpi=DPI_value)
    
    show_annotations = num_feats <= 15
    title_fs = max(14, format_config.font_size - 8)
    annot_fs = max(10, format_config.font_size - max(4, num_feats // 2))
    
    sns.heatmap(corr_diff_abs, 
                mask=mask,
                annot=show_annotations, 
                fmt=".2f", 
                cmap=format_config.cmap, 
                cbar_kws={}, 
                annot_kws={"size": annot_fs},
                ax=ax,
                vmin=0, vmax=1.0)
                
    ax.set_title("Absolute Difference in Numerical Associations\n(Real vs Generated)", fontsize=title_fs, pad=_EvaluationConfig.LABEL_PADDING)
                 
    ax.tick_params(axis='x', labelsize=format_config.xtick_size - 2)
    ax.tick_params(axis='y', labelsize=format_config.ytick_size - 2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=format_config.ytick_size - 4)
    
    plt.tight_layout()
    plot_path = save_dir_path / "correlation_difference_heatmap.svg"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    
    _LOGGER.info(f"🔗 Correlation matrix metrics calculated and heatmap saved to '{plot_path.name}'")
    
    return report_lines


def _plot_pca_projection(real_num: np.ndarray, 
                         gen_num: np.ndarray, 
                         save_dir_path: Path, 
                         format_config: FormatTabularDiffusionMetrics) -> None:
    """
    [PRIVATE] Helper function to plot a 2D PCA projection of numerical features.
    """
    if real_num is None or gen_num is None or real_num.shape[1] < 2:
        return
        
    try:
        # Standardize based on real data distribution
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_num)
        gen_scaled = scaler.transform(gen_num)
        
        # Fit PCA on real data to define the manifold
        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_scaled)
        gen_pca = pca.transform(gen_scaled)
        
        fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
        
        # Add Density contours (overlapping areas with high transparency)
        sns.kdeplot(x=real_pca[:, 0], y=real_pca[:, 1], 
                    fill=True, alpha=0.2, color=format_config.real_color, 
                    ax=ax, zorder=1)
        sns.kdeplot(x=gen_pca[:, 0], y=gen_pca[:, 1], 
                    fill=True, alpha=0.2, color=format_config.gen_color, 
                    ax=ax, zorder=1)
        
        # Plot Real Data Scatter (on top of density)
        ax.scatter(real_pca[:, 0], real_pca[:, 1], 
                   c=format_config.real_color, 
                   alpha=format_config.alpha, 
                   label='Real Data', 
                   s=10, zorder=2, edgecolors='none')
        
        # Plot Generated Data Scatter
        ax.scatter(gen_pca[:, 0], gen_pca[:, 1], 
                   c=format_config.gen_color, 
                   alpha=format_config.alpha, 
                   label='Generated Data', 
                   s=10, zorder=2, edgecolors='none')
                   
        explained_variance = pca.explained_variance_ratio_
        ax.set_title("2D PCA Projection (Numerical Features)", fontsize=format_config.font_size + 2, pad=_EvaluationConfig.LABEL_PADDING)
        ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.1%})", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.1%})", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        
        ax.tick_params(axis='x', labelsize=format_config.xtick_size)
        ax.tick_params(axis='y', labelsize=format_config.ytick_size)
        ax.legend(fontsize=format_config.legend_size - 4) # Slightly smaller legend font
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Turn off the top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = save_dir_path / "pca_projection_numerical.svg"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
        _LOGGER.info(f"📊 PCA projection plot saved to '{plot_path.name}'")
    except Exception as e:
        _LOGGER.error(f"Failed to generate PCA plot: {e}")

        
def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Cramer's V statistic for categorical-categorical association."""
    crosstab = pd.crosstab(x, y)
    if crosstab.empty or crosstab.size == 0:
        return 0.0
    
    chi2, _, _, _ = chi2_contingency(crosstab, correction=False)
    n = crosstab.sum().sum()
    r, k = crosstab.shape
    min_dim = min(k - 1, r - 1)
    
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _plot_cramers_v_heatmap(real_cat_list: list[np.ndarray], 
                            gen_cat_list: list[np.ndarray], 
                            cat_target_names: list[str], 
                            save_dir_path: Path, 
                            format_config: FormatTabularDiffusionMetrics) -> None:
    """
    [PRIVATE] Helper function to plot Cramer's V categorical association difference heatmap.
    """
    num_cat = len(cat_target_names) if cat_target_names else 0
    if not real_cat_list or not gen_cat_list or num_cat < 2:
        return
        
    try:
        # Initialize matrices
        real_corr = np.zeros((num_cat, num_cat))
        gen_corr = np.zeros((num_cat, num_cat))
        
        # Calculate pairwise Cramer's V
        for i in range(num_cat):
            for j in range(num_cat):
                if i == j:
                    real_corr[i, j] = 1.0
                    gen_corr[i, j] = 1.0
                elif i < j:
                    # Calculate for Real Data
                    cv_real = _cramers_v(real_cat_list[i], real_cat_list[j])
                    real_corr[i, j] = cv_real
                    real_corr[j, i] = cv_real
                    
                    # Calculate for Generated Data
                    cv_gen = _cramers_v(gen_cat_list[i], gen_cat_list[j])
                    gen_corr[i, j] = cv_gen
                    gen_corr[j, i] = cv_gen

        # Wrap names for the plot
        wrapped_cat_names = [wrap_text(name) for name in cat_target_names]

        real_df = pd.DataFrame(real_corr, index=wrapped_cat_names, columns=wrapped_cat_names)
        gen_df = pd.DataFrame(gen_corr, index=wrapped_cat_names, columns=wrapped_cat_names)
        
        corr_diff_abs = (real_df - gen_df).abs()
        
        # Dynamically scale figure size based on number of categorical features
        fig_size_xy = max(8, num_cat * 0.8)
        fig, ax = plt.subplots(figsize=(fig_size_xy, fig_size_xy), dpi=DPI_value)
        
        show_annotations = num_cat <= 15
        
        # Adjust font sizes for heatmaps to prevent clipping and balance elements
        title_fs = max(14, format_config.font_size - 8)
        # Dynamically scale annotation size: large for few categories, smaller for many
        annot_fs = max(10, format_config.font_size - max(4, num_cat // 2))
        
        # create a lower triangular mask to only show one triangle of the heatmap since it's symmetric
        cat_mask = np.triu(np.ones_like(corr_diff_abs, dtype=bool), k=1)
        
        sns.heatmap(corr_diff_abs,
                    mask=cat_mask,
                    annot=show_annotations, 
                    fmt=".2f", 
                    cmap=format_config.cmap, 
                    cbar_kws={}, # Removed label
                    annot_kws={"size": annot_fs},
                    ax=ax,
                    vmin=0, vmax=1.0)
                    
        ax.set_title("Absolute Difference in Categorical Associations\n(Real vs Generated)", fontsize=title_fs, pad=_EvaluationConfig.LABEL_PADDING)
        
        # Update Ticks and explicitly rotate them 45 degrees
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
        
        plot_path = save_dir_path / "cramers_v_difference_heatmap.svg"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
        _LOGGER.info(f"📊 Cramer's V heatmap saved to '{plot_path.name}'")
    except Exception as e:
        _LOGGER.error(f"Failed to generate Cramer's V heatmap: {e}")


def _plot_discriminator_roc(real_num: Optional[np.ndarray], 
                            gen_num: Optional[np.ndarray], 
                            real_cat_list: Optional[list[np.ndarray]], 
                            gen_cat_list: Optional[list[np.ndarray]], 
                            save_dir_path: Path, 
                            format_config: FormatTabularDiffusionMetrics) -> None:
    """
    [PRIVATE] Helper function to plot Discriminator ROC curve to evaluate global realism.
    Trains a lightweight classifier to distinguish real (class 0) from generated (class 1) data.
    An AUC close to 0.5 indicates highly realistic generated data.
    """
    try:
        X_real_parts = []
        X_gen_parts = []
        
        # 1. Prepare Numerical Features
        if real_num is not None and gen_num is not None and len(real_num) > 0:
            X_real_parts.append(real_num)
            X_gen_parts.append(gen_num)
            
        # 2. Prepare Categorical Features (Factorize to handle strings safely)
        if real_cat_list is not None and gen_cat_list is not None and len(real_cat_list) > 0:
            real_cat_encoded = []
            gen_cat_encoded = []
            for r_cat, g_cat in zip(real_cat_list, gen_cat_list):
                combined = np.concatenate([r_cat, g_cat])
                codes, _ = pd.factorize(combined)
                real_cat_encoded.append(codes[:len(r_cat)])
                gen_cat_encoded.append(codes[len(r_cat):])
            
            X_real_parts.append(np.column_stack(real_cat_encoded))
            X_gen_parts.append(np.column_stack(gen_cat_encoded))
            
        if not X_real_parts or not X_gen_parts:
            return
            
        X_real = np.hstack(X_real_parts)
        X_gen = np.hstack(X_gen_parts)
        
        # 3. Create labels (Real = 0, Generated = 1)
        y_real = np.zeros(len(X_real))
        y_gen = np.ones(len(X_gen))
        
        X = np.vstack([X_real, X_gen])
        y = np.concatenate([y_real, y_gen])
        
        # Safety fallback for NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # 4. Train/Test Split & Train Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # 5. Predict and calculate Metrics
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        # 6. Plotting
        fig, ax = plt.subplots(figsize=DISTRIBUTION_PLOT_SIZE, dpi=DPI_value)
        
        ax.plot(fpr, tpr, color=format_config.gen_color, linewidth=2, 
                label=f'Discriminator AUC = {auc_score:.2f}')
        
        # Ideal line for generative models is the random guess diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal AUC = 0.5')
        
        ax.set_title("Discriminator ROC Curve (Real vs Generated)", fontsize=format_config.font_size + 2, pad=_EvaluationConfig.LABEL_PADDING)
        ax.set_xlabel("False Positive Rate", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        ax.set_ylabel("True Positive Rate", fontsize=format_config.font_size, labelpad=_EvaluationConfig.LABEL_PADDING)
        
        ax.tick_params(axis='x', labelsize=format_config.xtick_size)
        ax.tick_params(axis='y', labelsize=format_config.ytick_size)
        ax.legend(loc='lower right', fontsize=format_config.legend_size)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Turn off the top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = save_dir_path / "discriminator_roc_curve.svg"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
        _LOGGER.info(f"📊 Discriminator ROC curve saved to '{plot_path.name}' (AUC: {auc_score:.2f})")
        
    except Exception as e:
        _LOGGER.error(f"Failed to generate Discriminator ROC curve: {e}")

