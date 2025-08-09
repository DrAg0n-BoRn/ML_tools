import pandas as pd
import torch
import numpy as np
import evotorch
from evotorch.algorithms import NSGA2
from evotorch.logging import PandasLogger
from typing import Literal, Union, Tuple, List, Optional, Any, Callable
from pathlib import Path
from tqdm.auto import trange
from functools import partial
from contextlib import nullcontext
import matplotlib.pyplot as plt
import seaborn as sns

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .ML_inference import PyTorchInferenceHandlerMulti # Using the multi-target handler
from .keys import PyTorchInferenceKeys
from .utilities import threshold_binary_values, save_dataframe
from .SQL import DatabaseManager # Added for SQL saving

__all__ = [
    "create_multi_objective_problem",
    "run_multi_objective_optimization",
    "plot_pareto_front"
]


def create_multi_objective_problem(
    inference_handler: PyTorchInferenceHandlerMulti,
    bounds: Tuple[List[float], List[float]],
    binary_features: int,
    objective_senses: Tuple[Literal["min", "max"], ...],
    algorithm: Literal["NSGA2"] = "NSGA2",
    population_size: int = 200,
    **searcher_kwargs
) -> Tuple[evotorch.Problem, Callable[[], Any]]:
    """
    Creates and configures an EvoTorch Problem and a Searcher for multi-objective optimization.

    This function sets up a problem where the goal is to optimize multiple conflicting
    objectives simultaneously, using an algorithm like NSGA2 to find the Pareto front.

    Args:
        inference_handler (PyTorchInferenceHandlerMulti): An initialized handler for the multi-target model.
        bounds (tuple[list[float], list[float]]): Lower and upper bounds for the solution features.
        binary_features (int): Number of binary features at the end of the feature vector.
        objective_senses (Tuple[Literal["min", "max"], ...]): A tuple specifying the optimization
            goal for each target (e.g., ("max", "min", "max")). The length of this tuple
            must match the number of outputs from the model.
        algorithm (str): The multi-objective search algorithm to use. Currently supports "NSGA2".
        population_size (int): The number of solutions in each generation.
        **searcher_kwargs: Additional keyword arguments for the search algorithm's constructor.

    Returns:
        A tuple containing the configured multi-objective Problem and the Searcher factory.
    """
    lower_bounds, upper_bounds = list(bounds[0]), list(bounds[1])

    if binary_features > 0:
        lower_bounds.extend([0.45] * binary_features)
        upper_bounds.extend([0.55] * binary_features)

    solution_length = len(lower_bounds)
    device = inference_handler.device

    def fitness_func(solution_tensor: torch.Tensor) -> torch.Tensor:
        """
        The fitness function for a multi-objective problem.
        It returns the entire output tensor from the model. EvoTorch handles the rest.
        """
        # The handler returns a tensor of shape [batch_size, num_targets]
        predictions = inference_handler.predict_batch(solution_tensor)[PyTorchInferenceKeys.PREDICTIONS]
        return predictions

    if algorithm == "NSGA2":
        problem = evotorch.Problem(
            objective_sense=objective_senses,
            objective_func=fitness_func,
            solution_length=solution_length,
            bounds=(lower_bounds, upper_bounds),
            device=device,
            vectorized=True,
            num_actors='max' # Use available CPU cores
        )
        SearcherClass = NSGA2
        if 'popsize' not in searcher_kwargs:
            searcher_kwargs['popsize'] = population_size
    else:
        raise ValueError(f"Unknown multi-objective algorithm '{algorithm}'.")

    searcher_factory = partial(SearcherClass, problem, **searcher_kwargs)
    return problem, searcher_factory


def run_multi_objective_optimization(
    problem: evotorch.Problem,
    searcher_factory: Callable[[], Any],
    num_generations: int,
    run_name: str,
    binary_features: int,
    save_dir: Union[str, Path],
    feature_names: List[str],
    target_names: List[str],
    save_format: Literal['csv', 'sqlite', 'both'] = 'csv',
    verbose: bool = True
):
    """
    Runs the multi-objective evolutionary optimization process to find the Pareto front.

    This function executes a multi-objective algorithm (like NSGA2) and saves the
    entire set of non-dominated solutions (the Pareto front) to the specified format(s).
    It also generates and saves a plot of the Pareto front.

    Args:
        problem (evotorch.Problem): The configured multi-objective problem.
        searcher_factory (Callable): A factory function to generate a fresh searcher instance.
        num_generations (int): The number of generations to run the algorithm.
        run_name (str): A name for this optimization run, used for filenames/table names.
        binary_features (int): Number of binary features in the solution vector.
        save_dir (str | Path): The directory where the result files will be saved.
        feature_names (List[str]): Names of the solution features for labeling columns.
        target_names (List[str]): Names of the target objectives for labeling columns.
        save_format (str): The format to save results in ('csv', 'sqlite', or 'both').
        verbose (bool): If True, attaches a logger and saves the evolution history.
    """
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    sanitized_run_name = sanitize_filename(run_name)
    
    if len(target_names) != problem.num_objectives:
        raise ValueError("The number of `target_names` must match the number of objectives in the problem.")

    searcher = searcher_factory()
    _LOGGER.info(f"🤖 Starting multi-objective optimization with {searcher.__class__.__name__} for {num_generations} generations...")

    logger = PandasLogger(searcher) if verbose else None
    searcher.run(num_generations)

    pareto_front = searcher.status["pareto_front"]
    _LOGGER.info(f"✅ Optimization complete. Found {len(pareto_front)} non-dominated solutions.")

    solutions_np = pareto_front.values.cpu().numpy()
    objectives_np = pareto_front.evals.cpu().numpy()

    if binary_features > 0:
        solutions_np = threshold_binary_values(input_array=solutions_np, binary_values=binary_features)

    results_df = pd.DataFrame(solutions_np, columns=feature_names)
    objective_cols = []
    for i, name in enumerate(target_names):
        col_name = f"predicted_{name}"
        results_df[col_name] = objectives_np[:, i]
        objective_cols.append(col_name)

    # --- Saving Logic ---
    if save_format in ['csv', 'both']:
        csv_path = save_path / f"pareto_front_{sanitized_run_name}.csv"
        results_df.to_csv(csv_path, index=False)
        _LOGGER.info(f"📄 Pareto front data saved to '{csv_path.name}'")

    if save_format in ['sqlite', 'both']:
        db_path = save_path / "Optimization_Multi.db"
        with DatabaseManager(db_path) as db:
            db.insert_from_dataframe(
                table_name=sanitized_run_name,
                df=results_df,
                if_exists='replace'
            )
        _LOGGER.info(f"🗃️ Pareto front data saved to table '{sanitized_run_name}' in '{db_path.name}'")

    # --- Plotting Logic ---
    plot_pareto_front(
        results_df,
        objective_cols=objective_cols,
        save_path=save_path / f"pareto_plot_{sanitized_run_name}.svg"
    )

    if logger:
        log_df = logger.to_dataframe()
        save_dataframe(df=log_df, save_dir=save_path / "EvolutionLogs", filename=f"log_{sanitized_run_name}")


def plot_pareto_front(results_df: pd.DataFrame, objective_cols: List[str], save_path: Path):
    """
    Generates and saves a plot of the Pareto front.

    - For 2 objectives, it creates a 2D scatter plot.
    - For 3 objectives, it creates a 3D scatter plot.
    - For >3 objectives, it creates a scatter plot matrix (pairs plot).

    Args:
        results_df (pd.DataFrame): DataFrame containing the optimization results.
        objective_cols (List[str]): The names of the columns that hold the objective values.
        save_path (Path): The full path (including filename) to save the SVG plot.
    """
    num_objectives = len(objective_cols)
    _LOGGER.info(f"🎨 Generating Pareto front plot for {num_objectives} objectives...")

    plt.style.use('seaborn-v0_8-whitegrid')

    if num_objectives == 2:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.scatter(results_df[objective_cols[0]], results_df[objective_cols[1]], alpha=0.7, edgecolors='k')
        ax.set_xlabel(objective_cols[0])
        ax.set_ylabel(objective_cols[1])
        ax.set_title("Pareto Front (2D)")
    
    elif num_objectives == 3:
        fig = plt.figure(figsize=(9, 7), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(results_df[objective_cols[0]], results_df[objective_cols[1]], results_df[objective_cols[2]], alpha=0.7, depthshade=True)
        ax.set_xlabel(objective_cols[0])
        ax.set_ylabel(objective_cols[1])
        ax.set_zlabel(objective_cols[2])
        ax.set_title("Pareto Front (3D)")

    else: # > 3 objectives
        _LOGGER.info("  -> More than 3 objectives found, generating a scatter plot matrix.")
        g = sns.pairplot(results_df[objective_cols], diag_kind="kde", plot_kws={'alpha': 0.6})
        g.fig.suptitle("Pareto Front (Pairs Plot)", y=1.02)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        _LOGGER.info(f"📊 Pareto plot saved to '{save_path.name}'")
        return

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    _LOGGER.info(f"📊 Pareto plot saved to '{save_path.name}'")

