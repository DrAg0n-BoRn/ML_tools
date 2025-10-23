import pandas # logger
import torch
import numpy    #handling torch to numpy
import evotorch
from evotorch.algorithms import SNES, CEM, GeneticAlgorithm
from evotorch.logging import PandasLogger
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from typing import Literal, Union, Tuple, List, Optional, Any, Callable, Dict
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
from .utilities import save_dataframe_filename
from .math_utilities import discretize_categorical_values


__all__ = [
    "MLOptimizer",
    "FitnessEvaluator",
    "create_pytorch_problem",
    "run_optimization"
]


class MLOptimizer:
    """
    A wrapper class for setting up and running EvoTorch optimization tasks.

    This class combines the functionality of `FitnessEvaluator`, `create_pytorch_problem`, and
    `run_optimization` into a single, streamlined workflow. 
    
    SNES and CEM algorithms do not accept bounds, the given bounds will be used as an initial starting point.

    Example:
        >>> # 1. Get categorical info from preprocessing steps
        >>> # e.g., from data_exploration.encode_categorical_features
        >>> cat_mappings = {'feature_C': {'A': 0, 'B': 1}, 'feature_D': {'X': 0, 'Y': 1}}
        >>> # e.g., from data_exploration.create_transformer_categorical_map
        >>> # Assumes feature_C is at index 2 (cardinality 2) and feature_D is at index 3 (cardinality 2)
        >>> cat_index_map = {2: 2, 3: 2}
        >>>
        >>> # 2. Initialize the optimizer
        >>> optimizer = MLOptimizer(
        ...     inference_handler=my_handler,
        ...     bounds=(lower_bounds, upper_bounds), # Bounds for ALL features
        ...     task="max",
        ...     algorithm="Genetic",
        ...     categorical_index_map=cat_index_map,
        ...     categorical_mappings=cat_mappings,
        ... )
        >>> # 3. Run the optimization
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
                 task: Literal["min", "max"],
                 algorithm: Literal["SNES", "CEM", "Genetic"] = "Genetic",
                 population_size: int = 200,
                 categorical_index_map: Optional[Dict[int, int]] = None,
                 categorical_mappings: Optional[Dict[str, Dict[str, int]]] = None,
                 discretize_start_at_zero: bool = True,
                 **searcher_kwargs):
        """
        Initializes the optimizer by creating the EvoTorch problem and searcher.

        Args:
            inference_handler (PyTorchInferenceHandler): An initialized inference handler containing the model and weights.
            bounds (tuple[list[float], list[float]]): A tuple containing the lower and upper bounds for ALL solution features. 
                Use the `optimization_tools.create_optimization_bounds()` helper to easily generate this and ensure unbiased categorical bounds.
            task (str): The optimization goal, either "min" or "max".
            algorithm (str): The search algorithm to use ("SNES", "CEM", "Genetic").
            population_size (int): Population size for CEM and GeneticAlgorithm.
            categorical_index_map (Dict[int, int] | None): Used to discretize values after optimization. Maps {column_index: cardinality}.
            categorical_mappings (Dict[str, Dict[str, int]] | None): Used to map discrete integer values back to strings (e.g., {0: 'Category_A'}) before saving.
            discretize_start_at_zero (bool): 
                True if the discrete encoding starts at 0 (e.g., [0, 1, 2]).
                False if it starts at 1 (e.g., [1, 2, 3]).
            **searcher_kwargs: Additional keyword arguments for the selected search algorithm's constructor.
        """
        # Make a fitness function
        self.evaluator = FitnessEvaluator(
            inference_handler=inference_handler,
            categorical_index_map=categorical_index_map,
            discretize_start_at_zero=discretize_start_at_zero
        )
        
        # Call the existing factory function to get the problem and searcher factory
        self.problem, self.searcher_factory = create_pytorch_problem(
            evaluator=self.evaluator,
            bounds=bounds,
            task=task,
            algorithm=algorithm,
            population_size=population_size,
            **searcher_kwargs
        )
        # Store categorical info to pass to the run function
        self.categorical_map = categorical_index_map
        self.categorical_mappings = categorical_mappings
        self.discretize_start_at_zero = discretize_start_at_zero

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
            feature_names (List[str] | None): Names of the solution features for labeling output.
                If None, generic names like 'feature_0', 'feature_1', ... , will be created.
            save_format (Literal['csv', 'sqlite', 'both']): The format for saving results.
            repetitions (int): The number of independent times to run the optimization.
            verbose (bool): If True, enables detailed logging.

        Returns:
            Optional[dict]: A dictionary with the best result if repetitions is 1, otherwise None.
        """
        # Call the existing run function with the stored problem, searcher, and categorical info
        return run_optimization(
            problem=self.problem,
            searcher_factory=self.searcher_factory,
            num_generations=num_generations,
            target_name=target_name,
            save_dir=save_dir,
            save_format=save_format,
            feature_names=feature_names,
            repetitions=repetitions,
            verbose=verbose,
            categorical_map=self.categorical_map,
            categorical_mappings=self.categorical_mappings,
            discretize_start_at_zero=self.discretize_start_at_zero
        )
        
        
class FitnessEvaluator:
    """
    A callable class that wraps the PyTorch model inference handler and performs
    on-the-fly discretization for the EvoTorch fitness function.

    This class is automatically instantiated by MLOptimizer and passed to
    create_pytorch_problem, encapsulating the evaluation logic.
    """
    def __init__(self,
                 inference_handler: PyTorchInferenceHandler,
                 categorical_index_map: Optional[Dict[int, int]] = None,
                 discretize_start_at_zero: bool = True):
        """
        Initializes the fitness evaluator.

        Args:
            inference_handler (PyTorchInferenceHandler): 
                An initialized inference handler containing the model.
            categorical_index_map (Dict[int, int] | None): 
                Maps {column_index: cardinality} for discretization.
            discretize_start_at_zero (bool): 
                True if discrete encoding starts at 0.
        """
        self.inference_handler = inference_handler
        self.categorical_index_map = categorical_index_map
        self.discretize_start_at_zero = discretize_start_at_zero
        
        # Expose the device
        self.device = self.inference_handler.device

    def __call__(self, solution_tensor: torch.Tensor) -> torch.Tensor:
        """
        This is the fitness function EvoTorch will call.

        It receives a batch of continuous solutions, discretizes the
        categorical ones, and returns the model's predictions.
        """
        # Clone to avoid modifying the optimizer's internal state (SNES, CEM, GA)
        processed_tensor = solution_tensor.clone()
        
        if self.categorical_index_map:
            for col_idx, cardinality in self.categorical_index_map.items():
                # 1. Round (using torch.floor(x + 0.5) for "round half up" behavior)
                rounded_col = torch.floor(processed_tensor[:, col_idx] + 0.5)
                
                # 2. Determine clamping bounds
                min_bound = 0 if self.discretize_start_at_zero else 1
                max_bound = cardinality - 1 if self.discretize_start_at_zero else cardinality
                
                # 3. Clamp the values and update the processed tensor
                processed_tensor[:, col_idx] = torch.clamp(rounded_col, min_bound, max_bound)

        # Use the *processed_tensor* for prediction
        predictions = self.inference_handler.predict_batch(processed_tensor)[PyTorchInferenceKeys.PREDICTIONS]
        return predictions.flatten()


def create_pytorch_problem(
    evaluator: FitnessEvaluator,
    bounds: Tuple[List[float], List[float]],
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
        evaluator (FitnessEvaluator): A callable class that wraps the model inference and handles on-the-fly discretization.
        bounds (tuple[list[float], list[float]]): A tuple containing the lower and upper bounds for the solution features.
            Use the `optimization_tools.create_optimization_bounds()` helper to easily generate this and ensure unbiased categorical bounds.
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
    
    solution_length = len(lower_bounds)
    device = evaluator.device

    # Create the Problem instance.
    if algorithm == "CEM" or algorithm == "SNES":
        problem = evotorch.Problem(
            objective_sense=task,
            objective_func=evaluator,
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
            objective_func=evaluator,
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
    save_dir: Union[str, Path],
    save_format: Literal['csv', 'sqlite', 'both'],
    feature_names: Optional[List[str]],
    repetitions: int = 1,
    verbose: bool = True,
    categorical_map: Optional[Dict[int, int]] = None,
    categorical_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    discretize_start_at_zero: bool = True
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
        save_dir (str | Path): The directory where the result file(s) will be saved.
        save_format (Literal['csv', 'sqlite', 'both'], optional): The format for
            saving results during iterative analysis.
        feature_names (List[str], optional): Names of the solution features for
            labeling the output files. If None, generic names like 'feature_0',
            'feature_1', etc., will be created.
        repetitions (int, optional): The number of independent times to run the
            entire optimization process.
        verbose (bool): Add an Evotorch Pandas logger saved as a csv. Only for the first repetition.
        categorical_index_map (Dict[int, int] | None): Used to discretize values after optimization. Maps {column_index: cardinality}.
        categorical_mappings (Dict[str, Dict[str, int]] | None): Used to map discrete integer values back to strings (e.g., {0: 'Category_A'}) before saving.
        discretize_start_at_zero (bool): 
            True if the discrete encoding starts at 0 (e.g., [0, 1, 2]).
            False if it starts at 1 (e.g., [1, 2, 3]).

    Returns:
        Optional[dict]: A dictionary containing the best feature values and the
        fitness score if `repetitions` is 1. Returns `None` if `repetitions`
        is greater than 1, as results are streamed to files instead.
    """
    # --- 1. Setup Paths and Feature Names ---
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    sanitized_target_name = sanitize_filename(target_name)
    if not sanitized_target_name.endswith(".csv"):
        sanitized_target_name = sanitized_target_name + ".csv"
    
    csv_path = save_path / sanitized_target_name
    db_path = save_path / "Optimization.db"
    db_table_name = target_name
    
    # Use problem's solution_length to create default names if none provided
    if feature_names is None:
        feat_len = problem.solution_length
        feature_names = [f"feature_{i}" for i in range(feat_len)] # type: ignore
    
    # --- 2. Run Optimization ---
    # --- SINGLE RUN LOGIC ---
    if repetitions <= 1:
        _LOGGER.info(f"🤖 Starting optimization for {num_generations} generations...")
        
        result_dict, pandas_logger = _run_single_optimization_rep(
            searcher_factory=searcher_factory,
            num_generations=num_generations,
            feature_names=feature_names,
            target_name=target_name,
            categorical_map=categorical_map,
            discretize_start_at_zero=discretize_start_at_zero,
            attach_logger=verbose
        )
        
        # Single run defaults to CSV, pass mappings for reverse mapping
        _save_result(
            result_dict=result_dict, 
            save_format='csv', 
            csv_path=csv_path,
            categorical_mappings=categorical_mappings
        )
        
        if pandas_logger:
            _handle_pandas_log(pandas_logger, save_path=save_path, target_name=target_name)
        
        _LOGGER.info(f"Optimization complete. Best solution saved to '{csv_path.name}'")
        return result_dict

    # --- MULTIPLE REPETITIONS LOGIC ---
    else:
        _LOGGER.info(f"🏁 Starting optimal solution space analysis with {repetitions} repetitions...")
        
        first_run_logger = None # To store the logger from the first rep
        db_context = DatabaseManager(db_path) if save_format in ['sqlite', 'both'] else nullcontext()
        
        with db_context as db_manager:
            # --- Setup Database Schema (if applicable) ---
            if db_manager:
                schema = {}
                categorical_cols = set(categorical_mappings.keys()) if categorical_mappings else set()
                
                for name in feature_names:
                    schema[name] = "TEXT" if name in categorical_cols else "REAL"
                schema[target_name] = "REAL"
                
                db_manager.create_table(db_table_name, schema)
            
            # --- Repetitions Loop ---
            print("")
            for i in trange(repetitions, desc="Repetitions"):
                
                # Only attach a logger for the first repetition if verbose
                attach_logger = verbose and (i == 0)
                
                result_dict, pandas_logger = _run_single_optimization_rep(
                    searcher_factory=searcher_factory,
                    num_generations=num_generations,
                    feature_names=feature_names,
                    target_name=target_name,
                    categorical_map=categorical_map,
                    discretize_start_at_zero=discretize_start_at_zero,
                    attach_logger=attach_logger
                )
                
                if pandas_logger:
                    first_run_logger = pandas_logger
                
                # Save each result incrementally
                _save_result(
                    result_dict=result_dict, 
                    save_format=save_format, 
                    csv_path=csv_path, 
                    db_manager=db_manager, 
                    db_table_name=db_table_name, 
                    categorical_mappings=categorical_mappings
                )
            
        if first_run_logger:
            _handle_pandas_log(first_run_logger, save_path=save_path, target_name=target_name)      
        
        _LOGGER.info(f"Optimal solution space complete. Results saved to '{save_path}'")
        return None


def _run_single_optimization_rep(
    searcher_factory: Callable[[],Any],
    num_generations: int,
    feature_names: List[str],
    target_name: str,
    categorical_map: Optional[Dict[int, int]],
    discretize_start_at_zero: bool,
    attach_logger: bool
) -> Tuple[dict, Optional[PandasLogger]]:
    """
    Internal helper to run one full optimization repetition.
    
    Handles searcher creation, logging, running, and result post-processing.
    """
    # CRITICAL: Create a fresh searcher for each run using the factory
    searcher = searcher_factory()
    
    # Attach logger if requested
    pandas_logger = PandasLogger(searcher) if attach_logger else None
    
    # Run the optimization
    searcher.run(num_generations)
    
    # Get the best result
    best_solution_container = searcher.status["pop_best"]
    best_solution_tensor = best_solution_container.values
    best_fitness = best_solution_container.evals
 
    best_solution_np = best_solution_tensor.cpu().numpy()
    
    # Discretize categorical/binary features
    if categorical_map:
        best_solution_thresholded = discretize_categorical_values(
            input_array=best_solution_np,
            categorical_info=categorical_map,
            start_at_zero=discretize_start_at_zero
        )
    else:
        best_solution_thresholded = best_solution_np
    
    # Format results into a dictionary
    result_dict = {name: value for name, value in zip(feature_names, best_solution_thresholded)}
    result_dict[target_name] = best_fitness.item()
    
    return result_dict, pandas_logger


def _handle_pandas_log(logger: PandasLogger, save_path: Path, target_name: str):
    log_dataframe = logger.to_dataframe()
    save_dataframe_filename(df=log_dataframe, save_dir=save_path / "EvolutionLogs", filename=target_name)


def info():
    _script_info(__all__)
