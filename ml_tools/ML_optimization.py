import torch
import numpy    #handling torch to numpy
import evotorch
from evotorch.algorithms import CMAES, SteadyStateGA
from evotorch.logging import StdOutLogger
from typing import Literal, Union, Tuple, List, Optional
from pathlib import Path
from tqdm.auto import trange
from contextlib import nullcontext

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .ML_inference import PyTorchInferenceHandler
from .keys import PyTorchInferenceKeys
from .SQL import DatabaseManager
from .optimization_tools import _save_result
from .utilities import threshold_binary_values


__all__ = [
    "create_pytorch_problem",
    "run_optimization"
]


def create_pytorch_problem(
    handler: PyTorchInferenceHandler,
    bounds: Tuple[List[float], List[float]],
    binary_features: int,
    task: Literal["minimize", "maximize"],
    algorithm: Literal["CMAES", "GA"] = "CMAES",
    verbose: bool = False,
    **searcher_kwargs
) -> Tuple[evotorch.Problem, evotorch.Searcher]: # type: ignore
    """
    Creates and configures an EvoTorch Problem and Searcher for a PyTorch model.

    Args:
        handler (PyTorchInferenceHandler): An initialized inference handler
            containing the model and weights.
        bounds (tuple[list[float], list[float]]): A tuple containing the lower
            and upper bounds for the solution features.
        binary_features (int): Number of binary features located at the END of the feature vector. Will be automatically added to the bounds.
        task (str): The optimization goal, either "minimize" or "maximize".
        algorithm (str): The search algorithm to use, "CMAES" or "GA" (SteadyStateGA).
        verbose (bool): Add an Evotorch logger for real-time console updates.
        **searcher_kwargs: Additional keyword arguments to pass to the
            selected search algorithm's constructor (e.g., stdev_init=0.5 for CMAES).

    Returns:
        Tuple:
        A tuple containing the configured evotorch.Problem and evotorch.Searcher.
    """
    lower_bounds, upper_bounds = bounds
    
    # add binary bounds
    if binary_features > 0:
        lower_bounds.extend([0.45] * binary_features)
        upper_bounds.extend([0.55] * binary_features)
    
    solution_length = len(lower_bounds)
    device = handler.device

    # Define the fitness function that EvoTorch will call.
    @evotorch.decorators.to_tensor # type: ignore
    @evotorch.decorators.on_aux_device(device)
    def fitness_func(solution_tensor: torch.Tensor) -> torch.Tensor:
        # Directly use the continuous-valued tensor from the optimizer for prediction
        predictions = handler.predict_batch(solution_tensor)[PyTorchInferenceKeys.PREDICTIONS]
        return predictions.flatten()

    # Create the Problem instance.
    problem = evotorch.Problem(
        objective_sense=task,
        objective_func=fitness_func,
        solution_length=solution_length,
        initial_bounds=(lower_bounds, upper_bounds),
        device=device,
    )

    # Create the selected searcher instance.
    if algorithm == "CMAES":
        searcher = CMAES(problem, **searcher_kwargs)
    elif algorithm == "GA":
        searcher = SteadyStateGA(problem, **searcher_kwargs)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'CMAES' or 'GA'.")

    # Add a logger for real-time console updates.
    # This gives the user immediate feedback on the optimization progress.
    if verbose:
        _ = StdOutLogger(searcher)

    return problem, searcher


def run_optimization(
    problem: evotorch.Problem,
    searcher: evotorch.Searcher, # type: ignore
    num_generations: int,
    target_name: str,
    binary_features: int,
    save_dir: Union[str, Path],
    save_format: Literal['csv', 'sqlite', 'both'],
    feature_names: Optional[List[str]],
    repetitions: int = 1
) -> Optional[dict]:
    """
    Runs the evolutionary optimization process, with support for multiple repetitions.

    This function serves as the main engine for the optimization task. It takes a
    configured Problem and a Searcher from EvoTorch and executes the optimization
    for a specified number of generations.

    It has two modes of operation:
    1.  **Single Run (repetitions=1):** Executes the optimization once, saves the
        single best result to a CSV file, and returns it as a dictionary.
    2.  **Iterative Analysis (repetitions > 1):** Executes the optimization
        multiple times. Results from each run are streamed incrementally to the
        specified file formats (CSV and/or SQLite database). In this mode,
        the function returns None.

    Args:
        problem (evotorch.Problem): The configured problem instance, which defines
            the objective function, solution space, and optimization sense.
        searcher (evotorch.Searcher): The configured searcher instance, which
            contains the evolutionary algorithm (e.g., CMAES, GA).
        num_generations (int): The total number of generations to run the
            search algorithm for in each repetition.
        target_name (str): Target name that will also be used for the CSV filename and SQL table.
        binary_features (int): Number of binary features located at the END of the feature vector.
        save_dir (str | Path): The directory where the result file(s) will be saved.
        save_format (Literal['csv', 'sqlite', 'both'], optional): The format for
            saving results during iterative analysis. Defaults to 'both'.
        feature_names (List[str], optional): Names of the solution features for
            labeling the output files. If None, generic names like 'feature_0',
            'feature_1', etc., will be created. Defaults to None.
        repetitions (int, optional): The number of independent times to run the
            entire optimization process. Defaults to 1.

    Returns:
        Optional[dict]: A dictionary containing the best feature values and the
        fitness score if `repetitions` is 1. Returns `None` if `repetitions`
        is greater than 1, as results are streamed to files instead.
    """
    # preprocess paths
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    sanitized_target_name = sanitize_filename(target_name)
    if not sanitized_target_name.endswith(".csv"):
        sanitized_target_name = sanitized_target_name + ".csv"
    
    csv_path = save_path / sanitized_target_name
    
    db_path = save_path / "Optimization.db"
    db_table_name = target_name
    
    # preprocess feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(problem.solution_length)] # type: ignore
    
    # --- SINGLE RUN LOGIC ---
    if repetitions <= 1:
        _LOGGER.info(f"🤖 Starting optimization with {searcher.__class__.__name__} for {num_generations} generations...")
        for _ in trange(num_generations, desc="Optimizing"):
            searcher.step()

        best_solution_tensor, best_fitness = searcher.best
        best_solution_np = best_solution_tensor.cpu().numpy()
        
        # threshold binary features
        if binary_features > 0:
            best_solution_thresholded = threshold_binary_values(input_array=best_solution_np, binary_values=binary_features)
        else:
            best_solution_thresholded = best_solution_np

        result_dict = {name: value for name, value in zip(feature_names, best_solution_thresholded)}
        result_dict[target_name] = best_fitness.item()
        
        _save_result(result_dict, 'csv', csv_path) # Single run defaults to CSV
        _LOGGER.info(f"✅ Optimization complete. Best solution saved to '{csv_path.name}'")
        return result_dict

    # --- MULTIPLE REPETITIONS LOGIC ---
    else:
        _LOGGER.info(f"🏁 Starting optimal solution space analysis with {repetitions} repetitions...")

        db_context = DatabaseManager(db_path) if save_format in ['sqlite', 'both'] else nullcontext()
        
        with db_context as db_manager:
            if db_manager:
                schema = {name: "REAL" for name in feature_names}
                schema[target_name] = "REAL"
                db_manager.create_table(db_table_name, schema)

            for i in trange(repetitions, desc="Repetitions"):
                _LOGGER.info(f"--- Starting Repetition {i+1}/{repetitions} ---")
                
                # CRITICAL: Re-initialize the searcher to ensure each run is independent
                searcher.reset()

                for _ in range(num_generations): # Inner loop does not need a progress bar
                    searcher.step()

                best_solution_tensor, best_fitness = searcher.best
                best_solution_np = best_solution_tensor.cpu().numpy()
                
                # threshold binary features
                if binary_features > 0:
                    best_solution_thresholded = threshold_binary_values(input_array=best_solution_np, binary_values=binary_features)
                else:
                    best_solution_thresholded = best_solution_np
                
                result_dict = {name: value for name, value in zip(feature_names, best_solution_thresholded)}
                result_dict[target_name] = best_fitness.item()
                
                # Save each result incrementally
                _save_result(result_dict, save_format, csv_path, db_manager, db_table_name)
        
        _LOGGER.info(f"✅ Optimal solution space complete. Results saved to '{save_path}'")
        return None


def info():
    _script_info(__all__)
