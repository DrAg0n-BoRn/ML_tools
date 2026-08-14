from typing import Union, Optional, Any, Literal
from pathlib import Path

from .._core import get_logger
from ..path_manager import make_fullpath

from ._base_model_config import _BaseModelParams


_LOGGER = get_logger("Optimization Configuration")


__all__ = [    
    "DragonParetoConfig",
    "DragonOptimizerConfig",
]


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
                 plot_size: tuple[int, int] = (10, 7),
                 plot_font_size: int = 16,
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
            plot_size (tuple[int, int]): Size of the plots.
            plot_font_size (int): Base Font size for the plots.
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
        self.plot_size = plot_size
        self.plot_font_size = plot_font_size
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

