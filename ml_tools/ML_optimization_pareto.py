import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Literal, Union, Tuple, List, Optional, Dict
from tqdm import tqdm

from evotorch.algorithms import GeneticAlgorithm
from evotorch import Problem
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from evotorch.operators import functional as func_ops

from .ML_inference import DragonInferenceHandler
from .optimization_tools import create_optimization_bounds, plot_optimal_feature_distributions
from .math_utilities import discretize_categorical_values
from .utilities import save_dataframe_filename
from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from ._keys import PyTorchInferenceKeys, MLTaskKeys
from ._schema import FeatureSchema


__all__ = [
    "DragonParetoOptimizer"
    ]


class DragonParetoOptimizer:
    """
    A specialized optimizer for Multi-Target Regression tasks using Pareto Fronts (NSGA-II).

    This class identifies the set of optimal trade-off solutions (the Pareto Front)
    where improving one target would worsen another.

    Features:
    - Supports mixed optimization directions (e.g., Maximize Profit, Minimize Risk).
    - Handles categorical constraints via feature schema.
    - Automatically generates Pareto plots (2D/3D Scatter and Parallel Coordinates).
    - Uses EvoTorch's GeneticAlgorithm which behaves like NSGA-II for multi-objective problems.
    """

    def __init__(self,
                 inference_handler: DragonInferenceHandler,
                 schema: FeatureSchema,
                 target_objectives: Dict[str, Literal["min", "max"]],
                 continuous_bounds_map: Dict[str, Tuple[float, float]],
                 population_size: int = 200,
                 discretize_start_at_zero: bool = True):
        """
        Initialize the Pareto Optimizer.

        Args:
            inference_handler (DragonInferenceHandler): Validated model handler.
            schema (FeatureSchema): Feature schema for bounds and types.
            target_objectives (Dict[str, "min"|"max"]): Dictionary mapping target names to optimization direction.
                Example: {"price": "max", "error": "min"}
            continuous_bounds_map (Dict): Bounds for continuous features {name: (min, max)}.
            population_size (int): Size of the genetic population.
            discretize_start_at_zero (bool): Categorical encoding start index.
        """
        self.inference_handler = inference_handler
        self.schema = schema
        self.target_objectives = target_objectives
        self.discretize_start_at_zero = discretize_start_at_zero

        # --- 1. Validation ---
        if self.inference_handler.task != MLTaskKeys.MULTITARGET_REGRESSION:
            _LOGGER.error(f"DragonParetoOptimizer requires '{MLTaskKeys.MULTITARGET_REGRESSION}'. Got '{self.inference_handler.task}'.")
            raise ValueError()

        if not self.inference_handler.target_ids:
            _LOGGER.error("Inference Handler has no 'target_ids' defined.")
            raise ValueError()

        # Map user targets to model output indices
        self.target_indices = []
        self.objective_senses = []
        self.ordered_target_names = []

        for name, direction in target_objectives.items():
            if name not in self.inference_handler.target_ids:
                raise ValueError(f"Target '{name}' not found in model targets: {self.inference_handler.target_ids}")
            
            idx = self.inference_handler.target_ids.index(name)
            self.target_indices.append(idx)
            self.objective_senses.append(direction)
            self.ordered_target_names.append(name)

        _LOGGER.info(f"Pareto Optimization setup for: {self.ordered_target_names}")

        # --- 2. Bounds Setup ---
        # Uses the external tool which reads the schema to set correct bounds for both continuous and categorical
        bounds = create_optimization_bounds(
            schema=schema,
            continuous_bounds_map=continuous_bounds_map,
            start_at_zero=discretize_start_at_zero
        )
        self.lower_bounds = list(bounds[0])
        self.upper_bounds = list(bounds[1])

        # --- 3. Evaluator Setup ---
        self.evaluator = ParetoFitnessEvaluator(
            inference_handler=inference_handler,
            target_indices=self.target_indices,
            categorical_index_map=schema.categorical_index_map,
            discretize_start_at_zero=discretize_start_at_zero
        )

        # --- 4. EvoTorch Problem & Algorithm ---
        self.problem = Problem(
            objective_sense=self.objective_senses,
            objective_func=self.evaluator,
            solution_length=len(self.lower_bounds),
            bounds=(self.lower_bounds, self.upper_bounds),
            device=inference_handler.device,
            vectorized=True
        )

        # GeneticAlgorithm. It automatically applies NSGA-II logic (Pareto sorting) when problem is multi-objective.
        self.algorithm = GeneticAlgorithm(
            self.problem,
            popsize=population_size,
            operators=[
                SimulatedBinaryCrossOver(self.problem, tournament_size=3, eta=20.0, cross_over_rate=1.0),
                GaussianMutation(self.problem, stdev=0.1)
            ],
            re_evaluate=False # model is deterministic
        )

    def run(self, 
            generations: int, 
            save_dir: Union[str, Path],
            log_interval: int = 10,
            verbose: bool = True) -> pd.DataFrame:
        """
        Execute the optimization with progress tracking and periodic logging.

        Args:
            generations (int): Number of generations to evolve.
            save_dir (str|Path): Directory to save results and plots.
            log_interval (int): How often (in generations) to log population statistics.
            verbose (bool): If True, enables logging and progress bar.

        Returns:
            pd.DataFrame: A DataFrame containing the non-dominated solutions (Pareto Front).
        """
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        
        if verbose:
            _LOGGER.info(f"🧬 Starting NSGA-II (GeneticAlgorithm) for {generations} generations...")

        # --- Optimization Loop with Progress Bar ---
        # We use a manual loop instead of self.algorithm.run() to inject logging/tqdm
        with tqdm(total=generations, desc="Evolving Pareto Front", disable=not verbose, unit="gen") as pbar:
            for gen in range(1, generations + 1):
                self.algorithm.step()
                
                # Periodic Logging of Population Stats
                if verbose and (gen % log_interval == 0 or gen == generations):
                    stats_msg = [f"Gen {gen}:"]
                    
                    # Get current population values (shape: [pop_size, n_targets])
                    # These are the raw predictions from the model
                    current_evals = self.algorithm.population.evals
                    
                    for i, target_name in enumerate(self.ordered_target_names):
                        # Extract column for this target
                        # Note: EvoTorch evals are on the same device as the problem
                        vals = current_evals[:, i]
                        
                        v_mean = float(vals.mean())
                        v_min = float(vals.min())
                        v_max = float(vals.max())
                        
                        stats_msg.append(f"{target_name}: {v_mean:.3f} (Range: {v_min:.3f}-{v_max:.3f})")
                    
                    _LOGGER.info(" | ".join(stats_msg))
                
                pbar.update(1)

        # --- Extract Pareto Front ---
        # Manually identify the Pareto front from the final population using domination counts
        final_pop = self.algorithm.population
        
        # Calculate domination counts (0 means non-dominated / Pareto optimal)
        domination_counts = func_ops.domination_counts(final_pop.evals, objective_sense=self.objective_senses)
        is_pareto = (domination_counts == 0)
        
        pareto_pop = final_pop[is_pareto]
        
        if len(pareto_pop) == 0:
            _LOGGER.warning("No strictly non-dominated solutions found (rare). Using best available front.")
            pareto_pop = final_pop
        
        # Inputs (Features)
        features_tensor = pareto_pop.values.cpu().numpy()
        
        # Outputs (Targets) - We re-evaluate to get exact predictions aligned with our names.
        with torch.no_grad():
             # We use the internal evaluator logic to get the exact target values corresponding to indices
            targets_tensor = self.evaluator(pareto_pop.values).cpu().numpy()

        # --- Post-Process Features (Discretization) ---
        # Ensure categorical columns are perfect integers
        if self.schema.categorical_index_map:
            features_final = discretize_categorical_values(
                features_tensor, 
                self.schema.categorical_index_map, 
                self.discretize_start_at_zero
            )
        else:
            features_final = features_tensor

        # --- Create DataFrame ---
        # 1. Features
        df_dict = {}
        for i, name in enumerate(self.schema.feature_names):
            df_dict[name] = features_final[:, i]
        
        # 2. Targets
        for i, name in enumerate(self.ordered_target_names):
            df_dict[name] = targets_tensor[:, i]

        pareto_df = pd.DataFrame(df_dict)

        # --- Reverse Mapping (Label Restoration) ---
        # Convert integer categorical values back to human-readable strings using the Schema
        if self.schema.categorical_mappings:
            _LOGGER.debug("Restoring categorical string labels...")

            for name, mapping in self.schema.categorical_mappings.items():
                if name in pareto_df.columns:
                    inv_map = {v: k for k, v in mapping.items()}
                    pareto_df[name] = pareto_df[name].apply(
                        lambda x: inv_map.get(int(x), x) if not pd.isna(x) else x
                    )
        
        # --- Save ---
        filename = "Pareto_Front_Solutions"
        save_dataframe_filename(pareto_df, save_path, filename)
        
        if verbose:
            _LOGGER.info(f"✅ Optimization complete. Found {len(pareto_df)} non-dominated solutions.")
            _LOGGER.info(f"💾 Solutions saved to '{save_path}/{filename}.csv'")

        # --- Plotting ---
        self._generate_plots(pareto_df, save_path)

        return pareto_df
    
    def _generate_plots(self, df: pd.DataFrame, save_dir: Path):
        """Orchestrates the generation of visualizations."""
        plot_dir = make_fullpath(save_dir / "ParetoPlots", make=True)
        
        n_objectives = len(self.ordered_target_names)
        
        # 1. Parallel Coordinates (Good for ANY number of targets)
        self._plot_parallel_coordinates(df, plot_dir)

        # 2. Pairplot (Good for inspecting trade-offs)
        self._plot_pairgrid(df, plot_dir)
        
        # 3. Specific 2D/3D Scatter plots
        if n_objectives == 2:
            self._plot_pareto_2d(df, plot_dir)
        elif n_objectives == 3:
            self._plot_pareto_3d(df, plot_dir)
            
        # 4. Input Feature Distributions
        # This utilizes the existing tool to plot histograms/KDEs of the INPUTS that resulted in these Pareto optimal solutions.
        _LOGGER.debug("Generating input feature distribution plots...")
        plot_optimal_feature_distributions(
            results_dir=save_dir, 
            verbose=False,
            target_columns=self.ordered_target_names # Exclude targets from being plotted as features
        )

    def _plot_parallel_coordinates(self, df: pd.DataFrame, save_dir: Path):
        """Creates a normalized parallel coordinates plot of the targets."""
        targets_df = df[self.ordered_target_names].copy()
        
        # Normalize for visualization (Min-Max scaling)
        norm_df = (targets_df - targets_df.min()) / (targets_df.max() - targets_df.min())
        norm_df['Solution_Index'] = range(len(norm_df))

        plt.figure(figsize=(12, 6))
        pd.plotting.parallel_coordinates(norm_df, 'Solution_Index', color='#4c72b0', alpha=0.3)
        plt.title("Parallel Coordinates of Pareto Front (Normalized)", fontsize=14)
        plt.legend().remove() # Remove legend as it's just indices
        plt.ylabel("Normalized Value")
        plt.tight_layout()
        plt.savefig(save_dir / "Pareto_Parallel_Coords.svg")
        plt.close()

    def _plot_pairgrid(self, df: pd.DataFrame, save_dir: Path):
        """Matrix of scatter plots for targets."""
        g = sns.PairGrid(df[self.ordered_target_names]) # type: ignore
        g.map_upper(sns.scatterplot, color="#4c72b0", alpha=0.6)
        g.map_lower(sns.scatterplot, color="#4c72b0", alpha=0.6)
        g.map_diag(sns.histplot, color="#4c72b0")
        g.savefig(save_dir / "Pareto_PairGrid.svg")
        plt.close()

    def _plot_pareto_2d(self, df: pd.DataFrame, save_dir: Path):
        """Standard 2D scatter plot."""
        x_name, y_name = self.ordered_target_names[0], self.ordered_target_names[1]
        
        plt.figure(figsize=(10, 8))
        
        # Use a color gradient based on the Y-axis to make "better" values visually distinct
        sns.scatterplot(
            data=df, 
            x=x_name, 
            y=y_name, 
            hue=y_name,      # Color by Y value
            palette="viridis", 
            s=100, 
            alpha=0.8, 
            edgecolor='k',
            legend=False
        )
        
        plt.title(f"Pareto Front: {x_name} vs {y_name}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add simple annotation for the 'corners' (extremes)
        # Find min/max for annotations
        for idx in [df[x_name].idxmin(), df[x_name].idxmax()]:
            row = df.loc[idx]
            plt.annotate(
                f"({row[x_name]:.2f}, {row[y_name]:.2f})",
                (row[x_name], row[y_name]),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
                fontsize=9,
                fontweight='bold'
            )

        plt.savefig(save_dir / f"Pareto_2D_{sanitize_filename(x_name)}_vs_{sanitize_filename(y_name)}.svg")
        plt.close()
        
    def _plot_pareto_3d(self, df: pd.DataFrame, save_dir: Path):
        """
        3D scatter plot with depth coloring and multiple viewing angles.
        """
        x_name, y_name, z_name = self.ordered_target_names[:3]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Color by Z-value to provide depth cues
        sc = ax.scatter(
            df[x_name], 
            df[y_name], 
            df[z_name], 
            c=df[z_name], 
            cmap='viridis', 
            s=80, 
            alpha=0.8, 
            edgecolor='k',
            depthshade=True # Matplotlib shading based on distance
        )
        
        ax.set_xlabel(x_name, labelpad=10)
        ax.set_ylabel(y_name, labelpad=10)
        ax.set_zlabel(z_name, labelpad=10)
        ax.set_title(f"3D Pareto Front: {x_name} vs {y_name} vs {z_name}", fontsize=12)
        
        # Add colorbar to quantify the depth
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label(z_name)

        # 2. Save Multiple Angles (Since the output is static)
        # View 1: Default
        plt.savefig(save_dir / "Pareto_3D_View_Default.svg", bbox_inches='tight')
        
        # View 2: Top-down-ish (XY plane emphasis)
        ax.view_init(elev=60, azim=45)
        plt.savefig(save_dir / "Pareto_3D_View_TopDown.svg", bbox_inches='tight')
        
        # View 3: Side profile (XZ/YZ emphasis)
        ax.view_init(elev=20, azim=135)
        plt.savefig(save_dir / "Pareto_3D_View_Side.svg", bbox_inches='tight')
        
        plt.close()


class ParetoFitnessEvaluator:
    """
    Evaluates fitness for Multi-Objective optimization.
    Returns a tensor of shape (batch_size, n_selected_targets).
    """
    def __init__(self,
                 inference_handler: DragonInferenceHandler,
                 target_indices: List[int],
                 categorical_index_map: Optional[Dict[int, int]] = None,
                 discretize_start_at_zero: bool = True):
        
        self.inference_handler = inference_handler
        self.target_indices = target_indices
        self.categorical_index_map = categorical_index_map
        self.discretize_start_at_zero = discretize_start_at_zero
        self.device = inference_handler.device

    def __call__(self, solution_tensor: torch.Tensor) -> torch.Tensor:
        # Clone to allow modification
        processed_tensor = solution_tensor.clone()
        
        # 1. Apply Discretization (Soft rounding for gradient compatibility if needed, 
        # but NSGA2 is derivative-free, so hard clamping is fine)
        if self.categorical_index_map:
            for col_idx, cardinality in self.categorical_index_map.items():
                rounded = torch.floor(processed_tensor[:, col_idx] + 0.5)
                min_b = 0 if self.discretize_start_at_zero else 1
                max_b = cardinality - 1 if self.discretize_start_at_zero else cardinality
                processed_tensor[:, col_idx] = torch.clamp(rounded, min_b, max_b)

        # 2. Inference
        # Returns dict with key PREDICTIONS -> tensor shape (batch, total_targets)
        preds = self.inference_handler.predict_batch(processed_tensor)[PyTorchInferenceKeys.PREDICTIONS]
        
        # 3. Filter for selected targets
        # Return shape (batch, n_selected_targets)
        return preds[:, self.target_indices]


def info():
    _script_info(__all__)
