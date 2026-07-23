import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from pathlib import Path

from ..optimization_tools import (
    create_optimization_bounds, 
    load_continuous_bounds_template,
    plot_optimal_feature_distributions_from_dataframe
)
from ..ML_inference import DragonInferenceHandler
from ..schema import FeatureSchema
from ..ML_configuration import DragonOptimizerConfig
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import MLTaskKeys, ParetoOptimizationKeys

from ._single_manual import FitnessEvaluator, create_pytorch_problem, run_optimization


_LOGGER = get_logger("Dragon Optimizer")


__all__ = [
    "DragonOptimizer",
]


class DragonOptimizer:
    """
    A wrapper class for setting up and running EvoTorch optimization tasks for regression models.

    This class combines the functionality of `FitnessEvaluator`, `create_pytorch_problem`, and
    `run_optimization` into a single, streamlined workflow. 
    
    SNES and CEM algorithms do not accept bounds, the given bounds will be used as an initial starting point.

    Example:
        >>> # 1. Define configuration
        >>> config = DragonOptimizerConfig(
        ...     target_name="my_target",
        ...     task="max",
        ...     continuous_bounds_map="path/to/bounds",
        ...     save_directory="/path/to/results",
        ...     algorithm="Genetic"
        ... )
        >>>
        >>> # 2. Initialize the optimizer
        >>> optimizer = DragonOptimizer(
        ...     inference_handler=my_handler,
        ...     schema=schema,
        ...     config=config
        ... )
        >>> # 3. Run the optimization
        >>> best_result = optimizer.run()
    """
    def __init__(self,
                 inference_handler: DragonInferenceHandler,
                 schema: FeatureSchema,
                 config: DragonOptimizerConfig):
        """
        Initializes the optimizer by creating the EvoTorch problem and searcher.

        Args:
            inference_handler (DragonInferenceHandler): 
                An initialized inference handler containing the model.
            schema (FeatureSchema): 
                The definitive schema object.
            config (DragonOptimizerConfig):
                Configuration object containing optimization parameters.
        """
        # --- Store schema ---
        self.schema = schema
        # --- Store inference handler ---
        self.inference_handler = inference_handler
        
        # --- Store config ---
        self.config = config
        
        # Ensure only Regression tasks are used
        allowed_tasks = [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]
        if self.inference_handler.task not in allowed_tasks:
            _LOGGER.error(f"DragonOptimizer only supports {allowed_tasks}. Got '{self.inference_handler.task}'.")
            raise ValueError()
        
        # --- store target name ---
        self.target_name = config.target_name
        
        # --- flag to control single vs multi-target ---
        self.is_multi_target = False
        
        # --- 1. Create bounds from schema ---
        # Handle bounds loading if it's a path
        raw_bounds_map = config.continuous_bounds_map
        if isinstance(raw_bounds_map, (str, Path)):
            continuous_bounds = load_continuous_bounds_template(raw_bounds_map)
        else:
            continuous_bounds = raw_bounds_map

        # Robust way to get bounds
        bounds = create_optimization_bounds(
            schema=schema,
            continuous_bounds_map=continuous_bounds,
            start_at_zero=config.discretize_start_at_zero
        )
        
        # Resolve target index if multi-target
        target_index = None
        
        if self.inference_handler.target_ids is None:
            # This should be caught by ML_inference logic
            _LOGGER.error("The provided inference handler does not have 'target_ids' defined.")
            raise ValueError()

        if self.target_name not in self.inference_handler.target_ids:
            _LOGGER.error(f"Target name '{self.target_name}' not found in the inference handler's 'target_ids': {self.inference_handler.target_ids}")
            raise ValueError()

        if len(self.inference_handler.target_ids) == 1:
            # Single target regression
            target_index = None
            _LOGGER.info(f"Optimization locked to single-target model '{self.target_name}'.")
        else:
            # Multi-target regression (optimizing one specific column)
            target_index = self.inference_handler.target_ids.index(self.target_name)
            self.is_multi_target = True
            _LOGGER.info(f"Optimization locked to target '{self.target_name}' (Index {target_index}) in a multi-target model.")
        
        # --- 2. Make a fitness function ---
        self.evaluator = FitnessEvaluator(
            inference_handler=inference_handler,
            # Get categorical info from the schema
            categorical_index_map=schema.categorical_index_map,
            discretize_start_at_zero=config.discretize_start_at_zero,
            target_index=target_index
        )
        
        # --- 3. Create the problem and searcher factory ---
        self.problem, self.searcher_factory = create_pytorch_problem(
            evaluator=self.evaluator,
            bounds=bounds,
            task=config.task, # type: ignore
            algorithm=config.algorithm, # type: ignore
            population_size=config.population_size,
            **config.searcher_kwargs
        )

    def run(self,
            plots_and_log: bool = True) -> Optional[dict]:
        """
        Runs the evolutionary optimization process using the pre-configured settings.

        The `feature_names` are automatically pulled from the `FeatureSchema`
        provided during initialization.

        Args:
            plots_and_log (bool): If True, generates convergence and distribution plots.

        Returns:
            Optional[dict]: A dictionary with the best result if repetitions is 1, otherwise None.
        """
        # Pass inference handler and target names for multi-target only
        if self.is_multi_target:
            target_names_to_pass = self.inference_handler.target_ids
            inference_handler_to_pass = self.inference_handler
        else:
            target_names_to_pass = None
            inference_handler_to_pass = None
        
        # Call the existing run function, passing info from the schema
        result_dict, log_df, csv_path = run_optimization(
            problem=self.problem,
            searcher_factory=self.searcher_factory,
            num_generations=self.config.generations,
            target_name=self.target_name,
            save_dir=self.config.save_directory,
            save_format=self.config.save_format, # type: ignore
            # Get the definitive feature names (as a list) from the schema
            feature_names=list(self.schema.feature_names),
            # Get categorical info from the schema
            categorical_map=self.schema.categorical_index_map,
            categorical_mappings=self.schema.categorical_mappings,
            repetitions=self.config.repetitions,
            verbose=plots_and_log, # log and plots requested
            discretize_start_at_zero=self.config.discretize_start_at_zero,
            all_target_names=target_names_to_pass,
            inference_handler=inference_handler_to_pass
        )

        if plots_and_log:
            self._generate_plots(log_df, csv_path)
            
        return result_dict

    def _generate_plots(self, log_df: Optional[pd.DataFrame], csv_path: Path):
        """Orchestrates the generation of visualizations for the single-target optimizer."""
        save_dir = make_fullpath(self.config.save_directory, make=True, enforce="directory")
        plot_dir = make_fullpath(save_dir / ParetoOptimizationKeys.OPTIMIZATION_PLOTS_DIR, make=True, enforce="directory")
        
        # _LOGGER.info("Generating optimization visualization plots...")

        # 1. Convergence History Plot
        if log_df is not None and not log_df.empty:
            self._plot_optimization_history(log_df, plot_dir)

        # Ensure the CSV exists before attempting distribution plots (especially relevant for repetitions > 1)
        if csv_path.exists():
            try:
                df_results = pd.read_csv(csv_path)
                
                # PairGrid and Distribution plots are most useful when we have a surface of solutions
                if len(df_results) > 1:
                    # 2. Input Feature Distributions (Histograms/KDEs)
                    plot_optimal_feature_distributions_from_dataframe(
                        dataframe=df_results,
                        save_dir=plot_dir,
                        verbose=False,
                        target_columns=[self.target_name] # Exclude target from being plotted as a feature
                    )
                    
                    # 3. PairGrid: Top Continuous Features vs Target
                    self._plot_feature_vs_target(df_results, plot_dir)
                else:
                    _LOGGER.debug("Skipping distribution/correlation plots: Requires repetitions > 1.")

            except Exception as e:
                _LOGGER.error(f"Failed to load or plot result distributions from CSV: {e}")
                
    def _plot_optimization_history(self, log_df: pd.DataFrame, save_dir: Path):
        """Generates convergence plots (Best/Mean/Worst) over generations."""
        fig, ax = plt.subplots(figsize=self.config.plot_size, dpi=ParetoOptimizationKeys.DPI)
        
        # EvoTorch PandasLogger standard columns: 'iter', 'best_eval', 'mean_eval', 'worst_eval'
        x_col = 'iter' if 'iter' in log_df.columns else log_df.index
            
        if 'best_eval' in log_df.columns:
            ax.plot(log_df[x_col], log_df['best_eval'], label='Best Fitness', color='#55a868', linewidth=2)
            
        if 'mean_eval' in log_df.columns:
            ax.plot(log_df[x_col], log_df['mean_eval'], label='Mean Fitness', color='#4c72b0', linewidth=2)
            
        if 'worst_eval' in log_df.columns:
            ax.plot(log_df[x_col], log_df['worst_eval'], label='Worst Fitness', color='#c44e52', linestyle='--', alpha=0.7)
            
        # Scientific Formatting
        base_font = self.config.plot_font_size
        ax.set_title(f"Convergence History: {self.target_name}", 
                     fontsize=base_font + 2, pad=ParetoOptimizationKeys.FONT_PAD)
        ax.set_xlabel("Generation", fontsize=base_font, labelpad=ParetoOptimizationKeys.FONT_PAD)
        ax.set_ylabel("Fitness", fontsize=base_font, labelpad=ParetoOptimizationKeys.FONT_PAD)
        
        ax.tick_params(axis='both', labelsize=base_font - 2)
        ax.legend(loc='best', fontsize=base_font - 2)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Remove top and right spines for a cleaner aesthetic
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        save_path = save_dir / f"Convergence_{sanitize_filename(self.target_name)}.svg"
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        _LOGGER.info(f"📈 Convergence history plot saved to: '{save_path}'")

    def _plot_feature_vs_target(self, df: pd.DataFrame, save_dir: Path):
        """Plots a PairGrid of the top continuous features vs the single target."""
        continuous_features = self.schema.continuous_feature_names
        if not continuous_features:
            _LOGGER.info("No continuous features found in schema; skipping PairGrid plot.")
            return
            
        plot_cols = [c for c in continuous_features if c in df.columns]
        if not plot_cols:
            _LOGGER.info("No continuous features from schema are present in the results DataFrame; skipping PairGrid plot.")
            return
            
        target = self.target_name
        if target not in df.columns:
            _LOGGER.warning(f"Target '{target}' not found in results DataFrame; skipping PairGrid plot.")
            return
            
        if len(plot_cols) > 4:
            corrs = df[plot_cols + [target]].corr(numeric_only=True)[target].abs().drop(target)
            plot_cols = corrs.nlargest(4).index.tolist()
            
        vars_to_plot = plot_cols + [target]
        
        # PairGrid handles sizes differently (using height per facet), scale it relative to the config
        facet_height = self.config.plot_size[1] / 2.3
        
        g = sns.PairGrid(df[vars_to_plot], height=facet_height)
        g.map_upper(sns.scatterplot, alpha=0.6, color="#4c72b0", edgecolor="w")
        g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.6)
        g.map_diag(sns.histplot, kde=True, color="#4c72b0")
        
        # Scientific Formatting
        base_font = self.config.plot_font_size
        g.figure.suptitle(f"Feature vs Target Trade-offs ({target})", y=1.02, fontsize=base_font + 2)
        
        for ax in g.axes.flatten():
            ax.xaxis.label.set_size(base_font)
            ax.yaxis.label.set_size(base_font)
            ax.tick_params(axis='both', labelsize=base_font - 2)
            ax.grid(True, linestyle='--', alpha=0.3)
            
        save_path = save_dir / f"PairGrid_{sanitize_filename(target)}.svg"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        _LOGGER.info(f"📊 PairGrid plot saved to: '{save_path}'")
