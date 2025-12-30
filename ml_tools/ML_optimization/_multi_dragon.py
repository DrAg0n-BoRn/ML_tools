import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from typing import Literal, Union, Optional
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

from evotorch.algorithms import GeneticAlgorithm
from evotorch import Problem
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from evotorch.operators import functional as func_ops

from ..SQL import DragonSQL
from ..ML_inference import DragonInferenceHandler
from ..ML_inference._chain_inference import DragonChainInference
from ..ML_configuration import DragonParetoConfig
from ..optimization_tools import create_optimization_bounds, plot_optimal_feature_distributions_from_dataframe, load_continuous_bounds_template
from ..utilities import save_dataframe_filename
from ..IO_tools import save_json
from ..schema import FeatureSchema

from ..math_utilities import discretize_categorical_values
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys, ParetoOptimizationKeys


_LOGGER = get_logger("DragonParetoOptimizer")


__all__ = [
    "DragonParetoOptimizer"
]


class DragonParetoOptimizer:
    """
    A specialized optimizer for Multi-Target Regression tasks using Pareto Fronts (NSGA-II).

    This class identifies the set of optimal trade-off solutions (the Pareto Front)
    where improving one target would worsen another.

    Features:
    - Supports DragonInferenceHandler (Multi-Target Regression) AND DragonChainInference.
    - Supports mixed optimization directions (e.g., Maximize Profit, Minimize Risk).
    - Handles categorical constraints via feature schema.
    - Automatically generates Pareto plots (2D/3D Scatter and Parallel Coordinates).
    - Uses EvoTorch's GeneticAlgorithm which behaves like NSGA-II for multi-objective problems.
    """

    def __init__(self,
                 inference_handler: Union[DragonInferenceHandler, DragonChainInference],
                 schema: FeatureSchema,
                 config: DragonParetoConfig):
        """
        Initialize the Pareto Optimizer.

        Args:
            inference_handler (DragonInferenceHandler | DragonChainInference): Validated model handler.
            schema (FeatureSchema): Feature schema for bounds and types.
            config (DragonParetoConfig): Configuration for the Pareto optimizer.
        """
        self.inference_handler = inference_handler
        self.schema = schema
        self.config = config
        
        self.target_objectives = config.target_objectives
        self.discretize_start_at_zero = config.discretize_start_at_zero
        
        # Initialize state for results
        self.pareto_front: Optional[pd.DataFrame] = None
        self._metrics_dir: Optional[Path] = None
        
        # Detect and validate handler
        self.is_chain = isinstance(self.inference_handler, DragonChainInference)
        
        # used for debug
        self._debug: bool = False

        # --- 1. Validation ---
        if not self.is_chain:
            # Standard Handler Validation
            if self.inference_handler.task != MLTaskKeys.MULTITARGET_REGRESSION: # type: ignore
                 _LOGGER.error(f"DragonParetoOptimizer with a standard handler requires '{MLTaskKeys.MULTITARGET_REGRESSION}'. Got '{self.inference_handler.task}'.") # type: ignore
                 raise ValueError()

        if not self.inference_handler.target_ids:
            _LOGGER.error("Inference Handler has no 'target_ids' defined.")
            raise ValueError()

        # Map user targets to model output indices
        self.target_indices = []
        self.objective_senses = []
        self.ordered_target_names = []
        
        available_targets = self.inference_handler.target_ids

        for name, direction in self.target_objectives.items():
            if name not in available_targets:
                _LOGGER.error(f"Target '{name}' not found in model targets: {available_targets}")
                raise ValueError()
            
            if direction not in ["min" , "max"]:
                _LOGGER.error(f"Invalid optimization direction '{direction}' for target '{name}'. Use 'min' or 'max'.")
                raise ValueError()
            
            # For standard handlers, we need indices to slice the output tensor.
            # For chain handlers, we just rely on name matching, but we track index for consistency.
            idx = available_targets.index(name)
            self.target_indices.append(idx)
            self.objective_senses.append(direction)
            self.ordered_target_names.append(name)

        _LOGGER.info(f"Pareto Optimization setup for: {self.ordered_target_names}")

        # --- 2. Bounds Setup ---
        # check type
        raw_bounds_map = config.continuous_bounds_map
        if isinstance(raw_bounds_map, (str, Path)):
            continuous_bounds = load_continuous_bounds_template(raw_bounds_map)
        elif isinstance(raw_bounds_map, dict):
            continuous_bounds = raw_bounds_map
        else:
            _LOGGER.error(f"Invalid type for 'continuous_bounds_map' in config. Expected dict or Path. Got {type(raw_bounds_map)}.")
            raise ValueError()
        
        # Uses the external tool which reads the schema to set correct bounds for both continuous and categorical
        bounds = create_optimization_bounds(
            schema=schema,
            continuous_bounds_map=continuous_bounds,
            start_at_zero=self.discretize_start_at_zero
        )
        self.lower_bounds = list(bounds[0])
        self.upper_bounds = list(bounds[1])

        # --- 3. Evaluator Setup ---
        self.evaluator = _ParetoFitnessEvaluator(
            inference_handler=inference_handler,
            target_indices=self.target_indices, # Used by Standard Handler
            target_names=self.ordered_target_names, # Used by Chain Handler
            categorical_index_map=schema.categorical_index_map,
            discretize_start_at_zero=self.discretize_start_at_zero,
            is_chain=self.is_chain
        )

        # --- 4. EvoTorch Problem & Algorithm ---
        device = inference_handler.handlers[0].device if self.is_chain else inference_handler.device # type: ignore
        
        self.problem = Problem(
            objective_sense=self.objective_senses,
            objective_func=self.evaluator,
            solution_length=len(self.lower_bounds),
            bounds=(self.lower_bounds, self.upper_bounds),
            device=device,
            vectorized=True
        )

        # GeneticAlgorithm. It automatically applies NSGA-II logic (Pareto sorting) when problem is multi-objective.
        self.algorithm = GeneticAlgorithm(
            self.problem,
            popsize=config.population_size,
            operators=[
                SimulatedBinaryCrossOver(self.problem, tournament_size=3, eta=20.0, cross_over_rate=1.0),
                GaussianMutation(self.problem, stdev=0.1)
            ],
            re_evaluate=False # model is deterministic
        )

    def run(self) -> pd.DataFrame:
        """
        Execute the optimization with progress tracking and periodic logging.

        Returns:
            pd.DataFrame: A DataFrame containing the non-dominated solutions (Pareto Front).
        """
        generations = self.config.generations
        save_dir = self.config.save_directory
        log_interval = self.config.log_interval
        
        save_path = make_fullpath(save_dir, make=True, enforce="directory")
        log_file = save_path / "optimization_log.txt"
        
        self._metrics_dir = save_path
        
        _LOGGER.info(f"🧬 Starting NSGA-II (GeneticAlgorithm) for {generations} generations...")
        
        # Initialize log file
        with open(log_file, "w") as f:
            f.write(f"Pareto Optimization Log - {generations} Generations\n")
            f.write("=" * 60 + "\n")
            
        # History tracking for visualization
        history_records = []

        # --- Optimization Loop with Progress Bar ---
        with tqdm(total=generations, desc="Evolving Pareto Front", unit="gen") as pbar:
            for gen in range(1, generations + 1):
                self.algorithm.step()
                
                # Capture stats for history (every generation for smooth plots)
                current_evals = self.algorithm.population.evals.clone() # type: ignore
                
                gen_stats = {}
                for i, target_name in enumerate(self.ordered_target_names):
                    vals = current_evals[:, i]
                    v_mean = float(vals.mean())
                    v_min = float(vals.min())
                    v_max = float(vals.max())
                    
                    # Store for plotting
                    history_records.append({
                        "Generation": gen,
                        "Target": target_name,
                        "Mean": v_mean,
                        "Min": v_min,
                        "Max": v_max
                    })
                    
                    gen_stats[target_name] = (v_mean, v_min, v_max)
                
                # Periodic Logging of Population Stats to FILE
                if gen % log_interval == 0 or gen == generations:
                    stats_msg = [f"Gen {gen}:"]
                    for t_name, (v_mean, v_min, v_max) in gen_stats.items():
                        stats_msg.append(f"{t_name}: {v_mean:.3f} (Range: {v_min:.3f}-{v_max:.3f})")
                    
                    log_line = " | ".join(stats_msg)
                    
                    # Write to file
                    with open(log_file, "a") as f:
                        f.write(log_line + "\n")
                
                pbar.update(1)
        
        # --- Post-Optimization Visualization ---
        if history_records:
            _LOGGER.debug("Generating optimization history plots...")
            history_df = pd.DataFrame(history_records)
            self._plot_optimization_history(history_df, save_path)

        # --- Extract Pareto Front ---
        # Manually identify the Pareto front from the final population using domination counts
        final_pop = self.algorithm.population
        
        # Clone the evals tensor. EvoTorch 'evals' is often a ReadOnlyTensor or BatchedTensorImpl
        evals_tensor = final_pop.evals.clone()
        
        # Calculate domination counts (0 means non-dominated / Pareto optimal)
        domination_counts = func_ops.domination_counts(evals_tensor, objective_sense=self.objective_senses)
        is_pareto = (domination_counts == 0)
        
        pareto_pop = final_pop[is_pareto]
        
        if len(pareto_pop) == 0:
            _LOGGER.warning("No strictly non-dominated solutions found (rare). Using best available front.")
            pareto_pop = final_pop
        
        # Inputs (Features)
        features_tensor = pareto_pop.values.cpu().numpy() # type: ignore
        
        # Outputs (Targets) - We re-evaluate to get exact predictions aligned with our names.
        with torch.no_grad():
             # We use the internal evaluator logic to get the exact target values corresponding to indices
            targets_tensor = self.evaluator(pareto_pop.values).cpu().numpy() # type: ignore

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
        
        # --- Store DataFrame ---
        self.pareto_front = pareto_df
        
        _LOGGER.info(f"Optimization complete. Found {len(pareto_df)} non-dominated solutions.")

        # --- Plotting ---
        self._generate_plots(pareto_df, save_path)

        return pareto_df
    
    def save_solutions(self, 
                       csv_if_exists: Literal['fail', 'replace', 'append'] = 'replace',
                       save_to_sql: bool = False,
                       sql_table_name: Optional[str] = None,
                       sql_if_exists: Literal['fail', 'replace', 'append'] = 'replace') -> None:
        """
        Saves the current Pareto front to a CSV file. Optionally saves to a SQL database.

        Args:
            csv_if_exists (str): Behavior if CSV file exists ('fail', 'replace', 'append').
            save_to_sql (bool): If True, also writes the results to a SQLite database in the save_dir.
            sql_table_name (str, optional): Specific table name for SQL. If None, uses the solutions filename.
            sql_if_exists (str): Behavior if SQL table exists ('fail', 'replace', 'append').
        """
        if self.pareto_front is None:
            _LOGGER.error("Cannot save solutions: No Pareto front found. Run the optimizer first.")
            raise ValueError()
        
        # handle directory
        save_path = self._metrics_dir
        if save_path is None:
            _LOGGER.error("No save directory found. Cannot save solutions.")
            raise ValueError()
        
        # unpack values from config
        filename = self.config.solutions_filename
        columns_to_round = self.config.columns_to_round
        float_precision = self.config.float_precision

        # Create a copy to avoid modifying the internal state
        df_to_save = self.pareto_front.copy()

        # Apply rounding to specific columns
        if columns_to_round:
            # Validate columns exist
            missing_cols = [c for c in columns_to_round if c not in df_to_save.columns]
            if missing_cols:
                _LOGGER.error(f"Save failed: The following columns to round were not found in the results: {missing_cols}")
                raise ValueError()

            # columns should be continuous columns only, validate against schema
            invalid_cols = [c for c in columns_to_round if c not in self.schema.continuous_feature_names]
            if invalid_cols:
                _LOGGER.error(f"Save failed: The following columns to round are not continuous features: {invalid_cols}")
                raise ValueError()

            for col in columns_to_round:
                # Round to nearest integer (handle floating point drift) and cast to int
                # This ensures 4.999 becomes 5, and saves as "5" not "5.0"
                df_to_save[col] = df_to_save[col].round().astype(int)
                _LOGGER.debug(f"Column '{col}' rounded to nearest integer.")
                
        # Validate float precision using a default value if invalid
        if float_precision < 0:
            float_precision = 4
            _LOGGER.warning("Invalid float_precision provided. Using default of 4.")
        
        # Select columns that are still floats (excludes those converted to int above)
        float_cols = df_to_save.select_dtypes(include=['float', 'float32', 'float64']).columns
        if len(float_cols) > 0:
            df_to_save[float_cols] = df_to_save[float_cols].round(float_precision)

        # Save CSV
        # sanitize filename and add extension if missing
        sanitized_filename = sanitize_filename(filename)
        csv_filename = sanitized_filename if sanitized_filename.lower().endswith(".csv") else f"{sanitized_filename}.csv"
        full_csv_path = save_path / csv_filename
        
        # Logic to handle Append/Fail/Replace for CSV
        if csv_if_exists == 'append' and full_csv_path.exists():
            try:
                # Append mode: write without header, index=False to match standard data exports
                df_to_save.to_csv(full_csv_path, mode='a', header=False, index=False)
                _LOGGER.info(f"💾 Pareto solutions APPENDED to CSV: '{save_path.name}/{csv_filename}'. Added {len(df_to_save)} rows.")
            except Exception as e:
                _LOGGER.error(f"Failed to append CSV: {e}")
                raise e
        elif csv_if_exists == 'fail' and full_csv_path.exists():
            _LOGGER.error(f"File '{full_csv_path}' already exists and csv_if_exists='fail'.")
            raise FileExistsError()
        else:
            # Default 'replace' or new file creation using the existing utility
            save_dataframe_filename(df=df_to_save, save_dir=save_path, filename=csv_filename, verbose=1)
            _LOGGER.info(f"💾 Pareto solutions saved to CSV: '{save_path.name}/{csv_filename}'. Shape: {df_to_save.shape}")
        
        # Save optimization bounds as JSON for reference (debug mode)
        if self._debug:
            try:
                # Create a human-readable map of feature_name -> [low, high]
                bounds_data = {}
                for i, name in enumerate(self.schema.feature_names):
                    low = self.lower_bounds[i]
                    high = self.upper_bounds[i]

                    # Check if this feature is categorical
                    # Categorical bounds are internally floats like (-0.5, 3.5) for [0, 3] (cardinality 4)
                    # We revert this logic for readability: int(low + 0.5) to int(high - 0.5)
                    if self.schema.categorical_index_map and i in self.schema.categorical_index_map:
                        readable_low = int(low + 0.5)
                        readable_high = int(high - 0.5)
                        bounds_data[name] = [readable_low, readable_high]
                    else:
                        # Continuous features are kept as floats
                        bounds_data[name] = [low, high]
                
                save_json(
                    data=bounds_data, 
                    directory=save_path, 
                    filename="all_debug_optimization_bounds.json", 
                    verbose=False
                )
                _LOGGER.info(f"💾 Optimization bounds saved to: '{save_path.name}/all_debug_optimization_bounds.json'")
                
            except Exception as e:
                _LOGGER.warning(f"Failed to save debug optimization bounds to JSON: {e}")
        
        # --- 2. Save SQL (Optional) ---
        if save_to_sql:
            db_path = save_path / ParetoOptimizationKeys.SQL_DATABASE_FILENAME
            target_table = sql_table_name if sql_table_name else sanitized_filename.rstrip(".csv")

            try:
                with DragonSQL(db_path) as db:
                    db.insert_from_dataframe(
                        table_name=target_table,
                        df=df_to_save,
                        if_exists=sql_if_exists
                    )
                _LOGGER.info(f"💾 Pareto solutions saved to SQL Table '{target_table}' in '{db_path.name}'")
            except Exception as e:
                _LOGGER.error(f"Failed to save solutions to SQL: {e}")
                # do not raise here to ensure the CSV save (which happened first) is not considered 'failed'

    def _generate_plots(self, df: pd.DataFrame, save_dir: Path):
        """Orchestrates the generation of visualizations."""
        plot_dir = make_fullpath(save_dir / ParetoOptimizationKeys.PARETO_PLOTS_DIR, make=True)
        
        n_objectives = len(self.ordered_target_names)
        
        # 1. Parallel Coordinates (Good for ANY number of targets)
        self._plot_parallel_coordinates(df, plot_dir)

        # 2. Pairplot (Good for inspecting trade-offs)
        self._plot_pairgrid(df, plot_dir)
        
        # 3. Specific 2D Scatter plots
        if n_objectives == 2:
            self._plot_pareto_2d(df, plot_dir)
            
        # 4. Input Feature Distributions
        # This utilizes the existing tool to plot histograms/KDEs of the INPUTS that resulted in these Pareto optimal solutions.
        _LOGGER.debug("Generating input feature distribution plots...")
        plot_optimal_feature_distributions_from_dataframe(
            dataframe=df,
            save_dir=save_dir,
            verbose=False,
            target_columns=self.ordered_target_names # Exclude targets from being plotted as features
        )

    def _plot_parallel_coordinates(self, df: pd.DataFrame, save_dir: Path):
        """Creates a normalized parallel coordinates plot of the targets."""
        targets_df = df[self.ordered_target_names].copy()
        
        # Normalize (Min-Max scaling)
        norm_df = (targets_df - targets_df.min()) / (targets_df.max() - targets_df.min())
        
        # Use the last column as the color scale (Gradient)
        color_col = self.ordered_target_names[-1]
        color_values = norm_df[color_col].values
        
        # --- 1. Static Matplotlib Plot (with Gradient) ---
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # use LineCollection for efficient gradient plotting
        # Create segments: shape (n_lines, n_points, 2)
        # x coordinates are 0, 1, 2... corresponding to columns
        x_coords = range(len(self.ordered_target_names))
        segments = []
        for i in range(len(norm_df)):
            y_coords = norm_df.iloc[i].values
            segments.append(list(zip(x_coords, y_coords)))
            
        lc = LineCollection(segments, cmap='viridis', alpha=0.4)
        lc.set_array(color_values) # Map colors to the last objective
        lc.set_linewidth(1.5)
        
        ax.add_collection(lc) # type: ignore
        ax.set_xlim(-0.1, len(self.ordered_target_names) - 0.9)
        ax.set_ylim(-0.05, 1.05)
        
        # Customizing axes
        ax.set_xticks(x_coords)
        ax.set_xticklabels(self.ordered_target_names, rotation=15)
        ax.set_ylabel("Normalized Value (0=Worst, 1=Best)")
        ax.set_title(f"Parallel Coordinates (Colored by {color_col})", fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add Colorbar
        cbar = plt.colorbar(lc, ax=ax, pad=0.02)
        cbar.set_label(f"Normalized {color_col}")
        
        plt.tight_layout()
        plt.savefig(save_dir / "Pareto_Parallel_Coords.svg")
        plt.close()

        # --- 2. Interactive Plotly Plot ---
        _LOGGER.debug("Generating interactive Parallel Coordinates with Plotly...")
        
        # Plotly expects the raw values (it handles normalization internally for the visual if needed, 
        # but usually it's better to show real units in the interactive tooltip).
        # We construct dimensions manually to ensure correct labeling.
        dims = []
        for col in self.ordered_target_names:
            dims.append(dict(
                range=[df[col].min(), df[col].max()],
                label=col,
                values=df[col]
            ))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col)
            ),
            dimensions=dims
        ))
        
        fig.update_layout(
            title="Interactive Parallel Coordinates (Drag axes to filter)",
            height=600
        )
        
        html_path = save_dir / "Pareto_Parallel_Coords_Interactive.html"
        fig.write_html(str(html_path))

    def _plot_pairgrid(self, df: pd.DataFrame, save_dir: Path):
        """
        Matrix of scatter plots for targets.
        Enhanced to color points by the last objective and improve readability.
        """
        cols = self.ordered_target_names
        
        # Define the variable to use for coloring (Gradient)
        # use the last column (usually the Z-axis in 3D plots)
        hue_col = cols[-1] if len(cols) > 0 else None
        
        # --- 1. Initialize PairGrid ---
        # height=4 increases the physical size of each subplot (default is 2.5)
        # df[cols] is a dataframe, add safe cast
        g = sns.PairGrid(df[cols], diag_sharey=False, height=4, aspect=1.1) # type: ignore
        
        # --- 2. Custom Scatter Function ---
        # This allows us to pass a continuous 'hue' (gradient) based on the last column without breaking PairGrid (which usually expects categorical hues).
        def scatter_with_continuous_hue(x, y, **kwargs):
            # Map hue to the 'hue_col' values derived from the original dataframe
            # Note: x.index aligns with df.index
            ax = plt.gca()
            points = ax.scatter(x, y, c=df.loc[x.index, hue_col], 
                                cmap='viridis', s=70, alpha=0.8, edgecolors='w', linewidth=0.5)
            return points

        # --- 3. Map Plots ---
        # Lower & Upper: Scatter plots colored by the 3rd objective
        g.map_upper(scatter_with_continuous_hue)
        g.map_lower(scatter_with_continuous_hue)
        
        # Diagonal: Histogram with KDE overlay
        g.map_diag(sns.histplot, kde=True, color='#4c72b0', alpha=0.6, linewidth=0)

        # --- 4. Aesthetics & Labels ---
        # Add a title
        g.figure.suptitle(f"Pareto Front Trade-offs (Hue: {hue_col})", y=1.02, fontsize=16)

        # Iterate over all axes to manually adjust fonts and rotation
        for ax in g.axes.flatten():
            # Manually increase axis label size (Column names)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            
            # Manually increase tick label size (Values)
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Rotate X-axis labels to prevent overlap
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')
            
            # Ensure grids are on for readability
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.savefig(save_dir / "Pareto_PairGrid.svg", bbox_inches='tight')
        plt.close()

    def _plot_pareto_2d(self, df: pd.DataFrame, save_dir: Path):
        """Standard 2D scatter plot."""
        x_name, y_name = self.ordered_target_names[0], self.ordered_target_names[1]
        
        plt.figure(figsize=self.config.plot_size, dpi=ParetoOptimizationKeys.DPI)
        
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
        
        plt.title(f"Pareto Front: {x_name} vs {y_name}", fontsize=self.config.plot_font_size + 2, pad=ParetoOptimizationKeys.FONT_PAD)
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
        
    def plot_pareto_3d(self, 
                       x_target: Union[int, str],
                       y_target: Union[int, str],
                       z_target: Union[int, str],
                       hue_target: Optional[Union[int, str]] = None):
        """
        Generate 3D visualizations for specific targets.
        
        Args:
            x_target (int|str): Index or name of the target for the X axis.
            y_target (int|str): Index or name of the target for the Y axis.
            z_target (int|str): Index or name of the target for the Z axis.
            hue_target (int|str, optional): Index or name of the target for coloring. Defaults to z_target if None.
        """
        if self._metrics_dir is None:
            _LOGGER.error("No save directory specified and no previous optimization directory found.")
            raise ValueError()
        save_path_root = self._metrics_dir
        
        save_path = make_fullpath(save_path_root / ParetoOptimizationKeys.PARETO_PLOTS_DIR, make=True, enforce="directory")
        
        df = self.pareto_front
        if df is None:
            _LOGGER.error("Pareto front data is not available. Please run the optimization first.")
            raise ValueError()
        
        # Helper to resolve index/name to string column name
        def resolve_name(t: Union[int, str]) -> str:
            if isinstance(t, int):
                if 0 <= t < len(self.ordered_target_names):
                    return self.ordered_target_names[t]
                _LOGGER.error(f"Target index {t} is out of bounds. Valid range is 0 to {len(self.ordered_target_names)-1}.")
                raise IndexError()
            if t not in df.columns:
                _LOGGER.error(f"Target '{t}' not found in DataFrame columns.")
                raise ValueError()
            return t

        # Resolve columns
        x_name = resolve_name(x_target)
        y_name = resolve_name(y_target)
        z_name = resolve_name(z_target)
        hue_name = resolve_name(hue_target) if hue_target is not None else z_name

        _LOGGER.info(f"Generating 3D Pareto Plot: {x_name} vs {y_name} vs {z_name} (Hue: {hue_name})")

        # --- 1. Static Matplotlib Plot ---
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        sc = ax.scatter(
            df[x_name], 
            df[y_name], 
            df[z_name],  # type: ignore
            c=df[hue_name], 
            cmap='viridis', 
            s=80, 
            alpha=0.8, 
            edgecolor='k',
            depthshade=True
        )
        
        ax.set_xlabel(x_name, labelpad=15)
        ax.set_ylabel(y_name, labelpad=15)
        ax.set_zlabel(z_name, labelpad=15)
        ax.set_title(f"3D Pareto Front", fontsize=14, pad=15)
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label(hue_name, labelpad=15)
        
        plt.tight_layout()

        # create a subdirectory to keep plots organized and avoid overwriting
        fname_suffix = f"{sanitize_filename(x_name)}_{sanitize_filename(y_name)}_{sanitize_filename(z_name)}"
        sub_dir_path = save_path / fname_suffix
        sub_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save Standard View
        plt.savefig(sub_dir_path / f"Pareto_3D.svg", bbox_inches='tight')
        plt.close()
        
        # --- 2. Interactive Plotly Plot ---
        fig_html = px.scatter_3d(
            df, 
            x=x_name, 
            y=y_name, 
            z=z_name,
            color=hue_name,
            title=f"Interactive 3D Pareto Front",
            labels={x_name: x_name, y_name: y_name, z_name: z_name, hue_name: hue_name},
            opacity=0.8
        )
        
        fig_html.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
        
        html_path = sub_dir_path / f"Pareto_3D_Interactive.html"
        fig_html.write_html(str(html_path))

    def _plot_optimization_history(self, history_df: pd.DataFrame, save_dir: Path):
        """
        Generates convergence plots (Mean/Min/Max) for each objective over generations.
        
        Args:
            history_df: DataFrame with cols [Generation, Target, Mean, Min, Max]
            save_dir: Base directory to save plots
        """
        # Create subdirectory for history plots
        plot_dir = make_fullpath(save_dir / ParetoOptimizationKeys.HISTORY_PLOTS_DIR, make=True, enforce="directory")
        
        unique_targets = history_df["Target"].unique()
        
        for target in unique_targets:
            subset = history_df[history_df["Target"] == target]
            
            # Determine direction (just for annotation/context if needed, but plotting stats is neutral)
            direction = self.target_objectives.get(target, "unknown")
            
            plt.figure(figsize=self.config.plot_size, dpi=ParetoOptimizationKeys.DPI)
            
            # Plot Mean
            plt.plot(subset["Generation"], subset["Mean"], label="Population Mean", color="#4c72b0", linewidth=2)
            
            # Plot Min/Max Range
            plt.fill_between(
                subset["Generation"], 
                subset["Min"], 
                subset["Max"], 
                color="#4c72b0", 
                alpha=0.15, 
                label="Min-Max Range"
            )
            
            # Plot extremes as dashed lines
            plt.plot(subset["Generation"], subset["Min"], linestyle="--", color="#55a868", alpha=0.6, linewidth=1, label="Min")
            plt.plot(subset["Generation"], subset["Max"], linestyle="--", color="#c44e52", alpha=0.6, linewidth=1, label="Max")

            plt.title(f"Convergence History: {target} ({direction.upper()})", fontsize=self.config.plot_font_size + 2, pad=ParetoOptimizationKeys.FONT_PAD)
            plt.xlabel("Generation", labelpad=ParetoOptimizationKeys.FONT_PAD, fontsize=self.config.plot_font_size)
            plt.ylabel("Target Value", labelpad=ParetoOptimizationKeys.FONT_PAD, fontsize=self.config.plot_font_size)
            plt.legend(loc='best', fontsize=self.config.plot_font_size)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(fontsize=self.config.plot_font_size - 4)
            plt.yticks(fontsize=self.config.plot_font_size - 4)
            
            plt.tight_layout()
            
            fname = f"Convergence_{sanitize_filename(target)}.svg"
            plt.savefig(plot_dir / fname, bbox_inches='tight')
            plt.close()

class _ParetoFitnessEvaluator:
    """
    Evaluates fitness for Multi-Objective optimization.
    Returns a tensor of shape (batch_size, n_selected_targets).
    
    Handles both Standard DragonInferenceHandler and DragonChainInference.
    """
    def __init__(self,
                 inference_handler: Union[DragonInferenceHandler, DragonChainInference],
                 target_indices: list[int],
                 target_names: list[str],
                 categorical_index_map: Optional[dict[int, int]] = None,
                 discretize_start_at_zero: bool = True,
                 is_chain: bool = False):
        
        self.inference_handler = inference_handler
        self.target_indices = target_indices
        self.target_names = target_names
        self.categorical_index_map = categorical_index_map
        self.discretize_start_at_zero = discretize_start_at_zero
        # Determine device from handler (Chain or Standard)
        if is_chain:
            # Chain stores a list of handlers, grab device from the first
            self.device = inference_handler.handlers[0].device # type: ignore
        else:
            self.device = inference_handler.device # type: ignore
            
        self.is_chain = is_chain

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

        # 2. Inference & Selection
        if self.is_chain:
            # --- Chain Logic ---
            # Output is Dict[target_name, Tensor(N, 1 or N)]
            raw_output = self.inference_handler.predict_batch(processed_tensor)
            
            # Extract specific targets in order and stack them
            selected_tensors = []
            for name in self.target_names:
                if name not in raw_output:
                    _LOGGER.error(f"Target '{name}' not found in chain inference output.")
                    raise KeyError()
                
                t = raw_output[name]
                # Ensure it is (Batch, 1) before stacking
                if t.ndim == 1:
                    t = t.unsqueeze(1)
                selected_tensors.append(t)
            
            # Stack along dim 1 -> (Batch, N_Selected_Targets)
            return torch.cat(selected_tensors, dim=1)
            
        else:
            # --- Standard Logic ---
            # Output is Dict['predictions', Tensor(N, Total_Targets)]
            preds = self.inference_handler.predict_batch(processed_tensor)[PyTorchInferenceKeys.PREDICTIONS]
            
            # Slice specific indices -> (Batch, N_Selected_Targets)
            return preds[:, self.target_indices]

