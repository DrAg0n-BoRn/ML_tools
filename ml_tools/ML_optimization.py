import pandas # logger
import torch
import numpy    #handling torch to numpy
import evotorch
from evotorch.algorithms import SNES, CEM, GeneticAlgorithm
from evotorch.logging import PandasLogger
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from typing import Literal, Union, Tuple, List, Optional, Any, Callable
from pathlib import Path
from tqdm.auto import trange
from contextlib import nullcontext
from functools import partial

from .path_manager import make_fullpath, sanitize_filename
from ._logger import _LOGGER
from ._script_info import _script_info
from .ML_inference import PyTorchInferenceHandler
from .keys import PyTorchInferenceKeys
from .SQL import DatabaseManager
from .optimization_tools import _save_result
from .utilities import threshold_binary_values, save_dataframe


__all__ = [
    "MLOptimizer",
    "create_pytorch_problem",
    "run_optimization"
]


class MLOptimizer:
    """
    A wrapper class for setting up and running EvoTorch optimization tasks.

    This class combines the functionality of `create_pytorch_problem` and
    `run_optimization` functions into a single, streamlined workflow. 
    
    SNES and CEM algorithms do not accept bounds, the given bounds will be used as an initial starting point.

    Example:
        >>> # 1. Initialize the optimizer with model and search parameters
        >>> optimizer = MLOptimizer(
        ...     inference_handler=my_handler,
        ...     bounds=(lower_bounds, upper_bounds),
        ...     number_binary_features=2,
        ...     task="max",
        ...     algorithm="Genetic"
        ... )
        >>> # 2. Run the optimization and save the results
        >>> best_result = optimizer.run(
        ...     num_generations=100,
        ...     target_name="my_target",
        ...     feature_names=my_feature_names,
        ...     save_dir="/path/to/results",
        ...     save_format="csv"
        ... )
    """
    def __init__(self,
                 inference_handler: PyTorchInferenceHandler,
                 bounds: Tuple[List[float], List[float]],
                 number_binary_features: int,
                 task: Literal["min", "max"],
                 algorithm: Literal["SNES", "CEM", "Genetic"] = "Genetic",
                 population_size: int = 200,
                 **searcher_kwargs):
        """
        Initializes the optimizer by creating the EvoTorch problem and searcher.

        Args:
            inference_handler (PyTorchInferenceHandler): An initialized inference handler containing the model and weights.
            bounds (tuple[list[float], list[float]]): A tuple containing the lower and upper bounds for the solution features.
            number_binary_features (int): Number of binary features located at the END of the feature vector.
            task (str): The optimization goal, either "min" or "max".
            algorithm (str): The search algorithm to use ("SNES", "CEM", "Genetic").
            population_size (int): Population size for CEM and GeneticAlgorithm.
            **searcher_kwargs: Additional keyword arguments for the selected search algorithm's constructor.
        """
        # Call the existing factory function to get the problem and searcher factory
        self.problem, self.searcher_factory = create_pytorch_problem(
            inference_handler=inference_handler,
            bounds=bounds,
            binary_features=number_binary_features,
            task=task,
            algorithm=algorithm,
            population_size=population_size,
            **searcher_kwargs
        )
        # Store binary_features count to pass it to the run function later
        self._binary_features = number_binary_features

    def run(self,
            num_generations: int,
            target_name: str,
            save_dir: Union[str, Path],
            feature_names: Optional[List[str]],
            save_format: Literal['csv', 'sqlite', 'both'],
            repetitions: int = 1,
            verbose: bool = True) -> Optional[dict]:
        """
        Runs the evolutionary optimization process using the pre-configured settings.

        Args:
            num_generations (int): The total number of generations for each repetition.
            target_name (str): Target name used for the CSV filename and/or SQL table.
            save_dir (str | Path): The directory where result files will be saved.
            feature_names (List[str] | None): Names of the solution features for labeling output. If None, generic names like 'feature_0', 'feature_1', ... , will be created.
            save_format (Literal['csv', 'sqlite', 'both']): The format for saving results.
            repetitions (int): The number of independent times to run the optimization.
            verbose (bool): If True, enables detailed logging.

        Returns:
            Optional[dict]: A dictionary with the best result if repetitions is 1, otherwise None.
        """
        # Call the existing run function with the stored problem, searcher, and binary feature count
        return run_optimization(
            problem=self.problem,
            searcher_factory=self.searcher_factory,
            num_generations=num_generations,
            target_name=target_name,
            binary_features=self._binary_features,
            save_dir=save_dir,
            save_format=save_format,
            feature_names=feature_names,
            repetitions=repetitions,
            verbose=verbose
        )


def create_pytorch_problem(
    inference_handler: PyTorchInferenceHandler,
    bounds: Tuple[List[float], List[float]],
    binary_features: int,
    task: Literal["min", "max"],
    algorithm: Literal["SNES", "CEM", "Genetic"] = "Genetic",
    population_size: int = 200,
    **searcher_kwargs
) -> Tuple[evotorch.Problem, Callable[[], Any]]:
    """
    Creates and configures an EvoTorch Problem and a Searcher factory class for a PyTorch model.
    
    SNES and CEM do not accept bounds, the given bounds will be used as an initial starting point.
    
    The Genetic Algorithm works directly with the bounds, and operators such as SimulatedBinaryCrossOver and GaussianMutation.
    
    Args:
        inference_handler (PyTorchInferenceHandler): An initialized inference handler containing the model and weights.
        bounds (tuple[list[float], list[float]]): A tuple containing the lower and upper bounds for the solution features.
        binary_features (int): Number of binary features located at the END of the feature vector. Will be automatically added to the bounds.
        task (str): The optimization goal, either "minimize" or "maximize".
        algorithm (str): The search algorithm to use.
        population_size (int): Used for CEM and GeneticAlgorithm.
        **searcher_kwargs: Additional keyword arguments to pass to the
            selected search algorithm's constructor (e.g., stdev_init=0.5 for CMAES).

    Returns:
        Tuple:
        A tuple containing the configured Problem and Searcher.
    """
    # Create copies to avoid modifying the original lists passed in the `bounds` tuple
    lower_bounds = list(bounds[0])
    upper_bounds = list(bounds[1])
    
    # add binary bounds
    if binary_features > 0:
        lower_bounds.extend([0.48] * binary_features)
        upper_bounds.extend([0.52] * binary_features)
    
    solution_length = len(lower_bounds)
    device = inference_handler.device

    # Define the fitness function that EvoTorch will call.
    def fitness_func(solution_tensor: torch.Tensor) -> torch.Tensor:
        # Directly use the continuous-valued tensor from the optimizer for prediction
        predictions = inference_handler.predict_batch(solution_tensor)[PyTorchInferenceKeys.PREDICTIONS]
        return predictions.flatten()
    
    
    # Create the Problem instance.
    if algorithm == "CEM" or algorithm == "SNES":
        problem = evotorch.Problem(
            objective_sense=task,
            objective_func=fitness_func,
            solution_length=solution_length,
            initial_bounds=(lower_bounds, upper_bounds),
            device=device,
            vectorized=True #Use batches
        )
        
        # If stdev_init is not provided, calculate it based on the bounds (used for SNES and CEM)
        if 'stdev_init' not in searcher_kwargs:
            # Calculate stdev for each parameter as 25% of its search range
            stdevs = [abs(up - low) * 0.25 for low, up in zip(lower_bounds, upper_bounds)]
            searcher_kwargs['stdev_init'] = torch.tensor(stdevs, dtype=torch.float32, requires_grad=False)
        
        if algorithm == "SNES":
            SearcherClass = SNES
        elif algorithm == "CEM":
            SearcherClass = CEM
            # Set a defaults for CEM if not provided
            if 'popsize' not in searcher_kwargs:
                searcher_kwargs['popsize'] = population_size
            if 'parenthood_ratio' not in searcher_kwargs:
                searcher_kwargs['parenthood_ratio'] = 0.2   #float 0.0 - 1.0
        
    elif algorithm == "Genetic":
        problem = evotorch.Problem(
            objective_sense=task,
            objective_func=fitness_func,
            solution_length=solution_length,
            bounds=(lower_bounds, upper_bounds),
            device=device,
            vectorized=True #Use batches
        )

        operators = [
            SimulatedBinaryCrossOver(problem,
                                    tournament_size=3,
                                    eta=0.6),
            GaussianMutation(problem,
                            stdev=0.4)
        ]
        
        searcher_kwargs["operators"] = operators
        if 'popsize' not in searcher_kwargs:
            searcher_kwargs['popsize'] = population_size
        
        SearcherClass = GeneticAlgorithm
        
    else:
        _LOGGER.error(f"Unknown algorithm '{algorithm}'.")
        raise ValueError()
    
    # Create a factory function with all arguments pre-filled
    searcher_factory = partial(SearcherClass, problem, **searcher_kwargs)

    return problem, searcher_factory


def run_optimization(
    problem: evotorch.Problem,
    searcher_factory: Callable[[],Any],
    num_generations: int,
    target_name: str,
    binary_features: int,
    save_dir: Union[str, Path],
    save_format: Literal['csv', 'sqlite', 'both'],
    feature_names: Optional[List[str]],
    repetitions: int = 1,
    verbose: bool = True
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
        searcher_factory (Callable): The searcher factory to generate fresh evolutionary algorithms.
        num_generations (int): The total number of generations to run the search algorithm for in each repetition.
        target_name (str): Target name that will also be used for the CSV filename and SQL table.
        binary_features (int): Number of binary features located at the END of the feature vector.
        save_dir (str | Path): The directory where the result file(s) will be saved.
        save_format (Literal['csv', 'sqlite', 'both'], optional): The format for
            saving results during iterative analysis.
        feature_names (List[str], optional): Names of the solution features for
            labeling the output files. If None, generic names like 'feature_0',
            'feature_1', etc., will be created.
        repetitions (int, optional): The number of independent times to run the
            entire optimization process.
        verbose (bool): Add an Evotorch Pandas logger saved as a csv. Only for the first repetition.

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
        searcher = searcher_factory()
        _LOGGER.info(f"🤖 Starting optimization with {searcher.__class__.__name__} Algorithm for {num_generations} generations...")
        # for _ in trange(num_generations, desc="Optimizing"):
        #     searcher.step()
        
        # Attach logger if requested
        if verbose:
            pandas_logger = PandasLogger(searcher)
            
        searcher.run(num_generations) # Use the built-in run method for simplicity
            
        # # DEBUG new searcher objects
        # for status_key in searcher.iter_status_keys():
        #     print("===", status_key, "===")
        #     print(searcher.status[status_key])
        #     print()
        
        # Get results from the .status dictionary 
        # SNES and CEM use the key 'center' to get mean values if needed    best_solution_tensor = searcher.status["center"]
        best_solution_container = searcher.status["pop_best"]
        best_solution_tensor = best_solution_container.values
        best_fitness = best_solution_container.evals

        best_solution_np = best_solution_tensor.cpu().numpy()
        
        # threshold binary features
        if binary_features > 0:
            best_solution_thresholded = threshold_binary_values(input_array=best_solution_np, binary_values=binary_features)
        else:
            best_solution_thresholded = best_solution_np

        result_dict = {name: value for name, value in zip(feature_names, best_solution_thresholded)}
        result_dict[target_name] = best_fitness.item()
        
        _save_result(result_dict, 'csv', csv_path) # Single run defaults to CSV
        
        # Process logger
        if verbose:
            _handle_pandas_log(pandas_logger, save_path=save_path, target_name=target_name)
        
        _LOGGER.info(f"Optimization complete. Best solution saved to '{csv_path.name}'")
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
            
            print("")
            # Repetitions loop
            pandas_logger = None
            for i in trange(repetitions, desc="Repetitions"):
                # CRITICAL: Create a fresh searcher for each run using the factory
                searcher = searcher_factory()
                
                # Attach logger if requested
                if verbose and i==0:
                    pandas_logger = PandasLogger(searcher)
                
                searcher.run(num_generations) # Use the built-in run method for simplicity
                
                # Get results from the .status dictionary 
                # SNES and CEM use the key 'center' to get mean values if needed    best_solution_tensor = searcher.status["center"]
                best_solution_container = searcher.status["pop_best"]
                best_solution_tensor = best_solution_container.values
                best_fitness = best_solution_container.evals
             
                best_solution_np = best_solution_tensor.cpu().numpy()
                
                # threshold binary features
                if binary_features > 0:
                    best_solution_thresholded = threshold_binary_values(input_array=best_solution_np, binary_values=binary_features)
                else:
                    best_solution_thresholded = best_solution_np
                
                # make results dictionary
                result_dict = {name: value for name, value in zip(feature_names, best_solution_thresholded)}
                result_dict[target_name] = best_fitness.item()
                
                # Save each result incrementally
                _save_result(result_dict, save_format, csv_path, db_manager, db_table_name)
            
        # Process logger
        if pandas_logger is not None:
            _handle_pandas_log(pandas_logger, save_path=save_path, target_name=target_name)      
        
        _LOGGER.info(f"Optimal solution space complete. Results saved to '{save_path}'")
        return None


def _handle_pandas_log(logger: PandasLogger, save_path: Path, target_name: str):
    log_dataframe = logger.to_dataframe()
    save_dataframe(df=log_dataframe, save_dir=save_path / "EvolutionLogs", filename=target_name)


def info():
    _script_info(__all__)
