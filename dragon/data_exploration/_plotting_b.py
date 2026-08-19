import pandas as pd
import matplotlib
# Use non-interactive backend for parallel plotting to avoid GUI issues
matplotlib.use("Agg") 
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from pathlib import Path
from joblib import Parallel, delayed

from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from .._helpers import wrap_text, get_valid_seaborn_color


_LOGGER = get_logger("Data Exploration: Visualization")


__all__ = [
    "plot_pairgrid_continuous_vs_target"
]


def _worker_plot_pairgrid(
    target: str,
    df_continuous: pd.DataFrame,
    df_targets: pd.DataFrame,
    valid_features: list[str],
    save_path: Path,
    font_scaling: float,
    validated_color: str,
    hexbin_lower: bool,
    log_diagonal: bool
) -> dict:
    """
    Isolated top-level worker function to plot a PairGrid for a single target.
    """
    try:
        # 1. Prevent duplicate column crashes
        features_to_check = [f for f in valid_features if f != target]
        if not features_to_check:
            return {"target": target, "status": "skipped", "reason": "no_independent_features"}

        # 2. Memory-efficient correlation
        corrs = df_continuous[features_to_check].corrwith(
            df_targets[target], 
            drop=True, 
            numeric_only=True
        ).abs()
        
        top_features = corrs.nlargest(4).index.tolist()
        
        if not top_features:
            return {"target": target, "status": "skipped", "reason": "no_correlated_features"}

        # 3. Safely subset only the required columns and drop NaNs
        plot_df = pd.concat([
            df_continuous[top_features], 
            df_targets[target]
        ], axis=1).dropna()

        # 4. Guardrail against plotting errors on tiny datasets
        if len(plot_df) < 5:
            return {"target": target, "status": "skipped", "reason": f"insufficient_data (n={len(plot_df)})"}

        # Format column names
        plot_df = plot_df.rename(columns=lambda x: wrap_text(x))

        # Generate colormap safely inside the worker process
        custom_cmap = sns.light_palette(validated_color, as_cmap=True)

        # 5. Plotting Context
        with sns.plotting_context("notebook", font_scale=font_scaling):
            facet_height = max(4.0, 2.5 * font_scaling)
            
            g = sns.PairGrid(plot_df, height=facet_height)
            
            # Upper: Scatterplot (Rasterized to keep SVG file sizes tiny)
            g.map_upper(sns.scatterplot, alpha=0.6, color=validated_color, edgecolor="none", rasterized=True)
            
            # Lower: Hexbin OR KDE
            if hexbin_lower:
                # Helper function to intercept and delete Seaborn's secret 'color' argument
                def hexbin_wrapper(x, y, **kwargs):
                    kwargs.pop("color", None)
                    plt.hexbin(x, y, **kwargs)

                g.map_lower(
                    hexbin_wrapper, 
                    gridsize=30, 
                    cmap=custom_cmap, 
                    mincnt=1, 
                    bins='log',
                    edgecolors='none',
                    rasterized=True
                )
            else:
                # rasterized=True ensures complex contour shapes are flattened into pixels
                g.map_lower(
                    sns.kdeplot, 
                    fill=True, 
                    cmap=custom_cmap, 
                    alpha=0.6, 
                    warn_singular=False, 
                    rasterized=True
                )
            
            # Diagonal: Histogram
            if log_diagonal:
                # Use Matplotlib's native hist to prevent the log(0) infinite stretch
                g.map_diag(plt.hist, log=True, color=validated_color, edgecolor='none', bins=30)
            else:
                # Default behavior
                g.map_diag(sns.histplot, kde=True, color=validated_color)

            # Aesthetic Formatting
            base_font = 12 * font_scaling
            g.figure.suptitle(f"Pairwise Relationships - {target}", y=1.02, fontsize=base_font + 2)
            
            for ax in g.axes.flatten():
                if ax is not None:
                    ax.xaxis.label.set_size(base_font)
                    ax.yaxis.label.set_size(base_font)
                    ax.tick_params(axis='both', labelsize=base_font - 2)
                    ax.grid(True, linestyle='--', alpha=0.3)
            
            # Save Plot
            safe_target = sanitize_filename(target)
            plot_filename = f"PairGrid_{safe_target}.svg"
            full_plot_path = save_path / plot_filename
            
            g.savefig(full_plot_path, bbox_inches='tight')
            
            # Force memory cleanup
            plt.close('all')
            
        return {"target": target, "status": "success"}

    except Exception as e:
        plt.close('all')
        return {"target": target, "status": "error", "reason": str(e)}


def plot_pairgrid_continuous_vs_target(
    df_continuous: pd.DataFrame,
    df_targets: pd.DataFrame,
    save_dir: Union[str, Path],
    verbose: int = 2,
    font_scaling: float = 1.5,
    color: str = "tab:orange",
    n_jobs: int = -1,
    hexbin_lower: bool = False,
    log_diagonal: bool = False
):
    """
    Plots a PairGrid of the top 4 most correlated continuous features against each target.
    
    Each plot is saved as an SVG file for high-quality vector graphics. 
    If multiple targets are provided, plots will be saved in a subdirectory named 'PairPlots' within the specified save_dir. 
    
    Plots a upper triangle scatterplot, a lower triangle KDE or hexbin plot, and a diagonal histogram for each target.
    The function is optimized for large datasets by using parallel processing (1 core per target) and rasterization to reduce SVG file sizes.

    Args:
        df_continuous (pd.DataFrame): DataFrame containing continuous feature columns.
        df_targets (pd.DataFrame): DataFrame containing numeric target columns.
        save_dir (str | Path): Base directory for saving plots.
        verbose (int): Verbosity level for logging.
        font_scaling (float): Multiplier for all text elements in the plots.
        color (str): A valid Seaborn color string for the plots.
        n_jobs (int): The number of parallel workers (1 core per target). -1 uses all available cores.
        hexbin_lower (bool): If True, uses a lightning-fast hexbin plot for the lower triangle. Recommended for large datasets. If False, uses a standard KDE plot.
        log_diagonal (bool): If True, applies a log scale to the y-axis of the diagonal histograms. Recommended for skewed distributions. If False, uses the default histogram with KDE overlay.
    
    <br>
        
    ## [Matplotlib Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
    """
    # validate jobs parameter
    if n_jobs < -1 or n_jobs == 0:
        _LOGGER.error("Invalid n_jobs parameter. Must be -1 or a positive integer.")
        return
    
    valid_targets = [col for col in df_targets.columns if is_numeric_dtype(df_targets[col])]
    if not valid_targets:
        _LOGGER.error("No valid numeric target columns provided in df_targets.")
        return

    valid_features = [col for col in df_continuous.columns if is_numeric_dtype(df_continuous[col])]
    if not valid_features:
        _LOGGER.error("No valid numeric feature columns provided in df_continuous.")
        return

    base_save_path = make_fullpath(save_dir, make=True, enforce="directory")
    if len(valid_targets) > 1:
        save_path = base_save_path / "PairPlots"
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = base_save_path

    validated_color = get_valid_seaborn_color(color)
    
    # get the real number of jobs to use
    if n_jobs == -1 or n_jobs > len(valid_targets):
        real_jobs = len(valid_targets)
    else:
        real_jobs = n_jobs

    if verbose >= 2 and real_jobs != 1:
        _LOGGER.info(f"Starting parallel generation of PairGrids using {real_jobs} cores.")

    # Execute in Parallel
    results = Parallel(n_jobs=real_jobs, backend="loky")(
        delayed(_worker_plot_pairgrid)(
            target=target,
            df_continuous=df_continuous,
            df_targets=df_targets,
            valid_features=valid_features,
            save_path=save_path,
            font_scaling=font_scaling,
            validated_color=validated_color,
            hexbin_lower=hexbin_lower,
            log_diagonal=log_diagonal
        )
        for target in valid_targets
    )

    # Parse Results & Log
    total_plots = 0
    for res in results:
        if res is None:
            _LOGGER.error("Received None result from a worker. This indicates an unexpected failure.")
            continue
        
        target = res["target"]
        if res["status"] == "success":
            total_plots += 1
            if verbose >= 3:
                _LOGGER.info(f"Successfully plotted PairGrid for '{target}'.")
        elif res["status"] == "skipped":
            if verbose >= 1:
                _LOGGER.warning(f"Skipping '{target}': {res['reason']}.")
        elif res["status"] == "error":
            _LOGGER.error(f"Failed to save PairGrid for '{target}'. Error: {res['reason']}")

    if verbose >= 2:
        _LOGGER.info(f"Successfully saved {total_plots} PairGrid plot(s) to '{save_path.name}'.")
