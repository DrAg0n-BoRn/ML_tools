from typing import Union, Optional, Any, Literal
from pathlib import Path

from .._core import get_logger
from ..path_manager import make_fullpath
from ..keys._keys import MLTaskKeys

from ._base_model_config import _BaseModelParams


_LOGGER = get_logger("ML Configuration")


__all__ = [    
    # --- Training Config ---
    "DragonTrainingConfig",
    "DragonParetoConfig",
    "DragonOptimizerConfig",
]


class DragonTrainingConfig(_BaseModelParams):
    """
    Configuration object for the training process.
    
    Can be unpacked as a dictionary for logging or accessed as an object.
    
    Accepts arbitrary keyword arguments which are set as instance attributes.
    """
    def __init__(self,
                 validation_size: float,
                 test_size: float,
                 initial_learning_rate: float,
                 batch_size: int,
                 task: str,
                 device: str,
                 finalized_filename: str,
                 random_state: int = 101,
                 **kwargs: Any) -> None:
        """  
        Args:
            validation_size (float): Proportion of data for validation set.
            test_size (float): Proportion of data for test set.
            initial_learning_rate (float): Starting learning rate.
            batch_size (int): Number of samples per training batch.
            task (str): Type of ML task (use TaskKeys).
            device (str): Device to run training on.
            finalized_filename (str): Filename for the Dragon ML Finalized-file.
            random_state (int): Seed for reproducibility.
            **kwargs: Additional training parameters as key-value pairs.
        """
        self.validation_size = validation_size
        self.test_size = test_size
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.device = device
        self.finalized_filename = finalized_filename
        self.random_state = random_state        
        
        # validate task
        if task not in MLTaskKeys.ALL_TASKS:
            _LOGGER.error(f"Invalid task '{task}'. Must be one of: {MLTaskKeys.ALL_TASKS}")
            raise ValueError()
        self.task = task
        
        # Process kwargs with validation
        for key, value in kwargs.items():
            # Python guarantees 'key' is a string for **kwargs
            
            # Allow None in value
            if value is None:
                setattr(self, key, value)
                continue
            
            if isinstance(value, dict):
                _LOGGER.error("Nested dictionaries are not supported, unpack them first.")
                raise TypeError()
            
            # Check if value is a number or a string or a JSON supported type, except dict
            if not isinstance(value, (str, int, float, bool, list, tuple)):
                _LOGGER.error(f"Invalid type for configuration '{key}': {type(value).__name__}")
                raise TypeError()
            
            setattr(self, key, value)


class DragonParetoConfig(_BaseModelParams):
    """
    Configuration object for the Pareto Optimization process.
    """
    def __init__(self,
                 save_directory: Union[str, Path],
                 target_objectives: dict[str, Literal["min", "max"]],
                 continuous_bounds_map: Union[dict[str, tuple[float, float]], dict[str, list[float]], str, Path],
                 columns_to_round: Optional[list[str]] = None,
                 population_size: int = 500,
                 generations: int = 1000,
                 solutions_filename: str = "NonDominatedSolutions",
                 float_precision: int = 4,
                 log_interval: int = 10,
                 plot_size: tuple[int, int] = (10, 7),
                 plot_font_size: int = 16,
                 discretize_start_at_zero: bool = True):
        """  
        Configure the Pareto Optimizer.

        Args:
            save_directory (str | Path): Directory to save artifacts.
            target_objectives (Dict[str, "min"|"max"]): Dictionary mapping target names to optimization direction.
                Example: {"price": "max", "error": "min"}
            continuous_bounds_map (Dict): Bounds for continuous features {name: (min, max)}. Or a path/str to a directory containing the "optimization_bounds.json" file.
            columns_to_round (List[str] | None): List of continuous column names that should be rounded to the nearest integer.
            population_size (int): Size of the genetic population.
            generations (int): Number of generations to run.
            solutions_filename (str): Filename for saving Pareto solutions.
            float_precision (int): Number of decimal places to round standard float columns.
            log_interval (int): Interval for logging progress.
            plot_size (Tuple[int, int]): Size of the 2D plots.
            plot_font_size (int): Font size for plot text.
            discretize_start_at_zero (bool): Categorical encoding start index. True=0, False=1.
        """
        # Validate string or Path
        valid_save_dir = make_fullpath(save_directory, make=True, enforce="directory")
        
        if isinstance(continuous_bounds_map, (str, Path)):
            continuous_bounds_map = make_fullpath(continuous_bounds_map, make=False, enforce="directory")
        
        self.save_directory = valid_save_dir
        self.target_objectives = target_objectives
        self.continuous_bounds_map = continuous_bounds_map
        self.columns_to_round = columns_to_round
        self.population_size = population_size
        self.generations = generations
        self.solutions_filename = solutions_filename
        self.float_precision = float_precision
        self.log_interval = log_interval
        self.plot_size = plot_size
        self.plot_font_size = plot_font_size
        self.discretize_start_at_zero = discretize_start_at_zero


class DragonOptimizerConfig(_BaseModelParams):
    """
    Configuration object for the Single-Objective DragonOptimizer.
    """
    def __init__(self,
                 target_name: str,
                 task: Literal["min", "max"],
                 continuous_bounds_map: Union[dict[str, tuple[float, float]], str, Path],
                 save_directory: Union[str, Path],
                 save_format: Literal['csv', 'sqlite', 'both'] = 'csv',
                 algorithm: Literal["SNES", "CEM", "Genetic"] = "Genetic",
                 population_size: int = 500,
                 generations: int = 1000,
                 repetitions: int = 1,
                 discretize_start_at_zero: bool = True,
                 **searcher_kwargs: Any):
        """
        Args:
            target_name (str): The name of the target variable to optimize.
            task (str): The optimization goal, either "min" or "max".
            continuous_bounds_map (Dict | str | Path): Dictionary {feature_name: (min, max)} or path to "optimization_bounds.json".
            save_directory (str | Path): Directory to save results.
            save_format (str): Format for saving results ('csv', 'sqlite', 'both').
            algorithm (str): Search algorithm ("SNES", "CEM", "Genetic").
            population_size (int): Population size for CEM and GeneticAlgorithm.
            generations (int): Number of generations per repetition.
            repetitions (int): Number of independent optimization runs.
            discretize_start_at_zero (bool): True if discrete encoding starts at 0.
            **searcher_kwargs: Additional arguments for the specific search algorithm 
                               (e.g., stdev_init for SNES).
        """
        # Validate paths
        self.save_directory = make_fullpath(save_directory, make=True, enforce="directory")
        
        if isinstance(continuous_bounds_map, (str, Path)):
            self.continuous_bounds_map = make_fullpath(continuous_bounds_map, make=False, enforce="directory")
        else:
            self.continuous_bounds_map = continuous_bounds_map

        # Core params
        self.target_name = target_name
        self.task = task
        self.save_format = save_format
        self.algorithm = algorithm
        self.population_size = population_size
        self.generations = generations
        self.repetitions = repetitions
        self.discretize_start_at_zero = discretize_start_at_zero
        
        # Store algorithm specific kwargs
        self.searcher_kwargs = searcher_kwargs

        # Basic Validation
        if self.task not in ["min", "max"]:
             _LOGGER.error(f"Invalid task '{self.task}'. Must be 'min' or 'max'.")
             raise ValueError()
             
        valid_algos = ["SNES", "CEM", "Genetic"]
        if self.algorithm not in valid_algos:
            _LOGGER.error(f"Invalid algorithm '{self.algorithm}'. Must be one of {valid_algos}.")
            raise ValueError()

