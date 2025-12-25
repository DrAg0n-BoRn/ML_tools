from typing import Literal, Union, Optional
from pathlib import Path

from ..optimization_tools import create_optimization_bounds
from ..ML_inference import DragonInferenceHandler
from ..schema import FeatureSchema

from .._core import get_logger
from ..keys._keys import MLTaskKeys

from ._single_manual import FitnessEvaluator, create_pytorch_problem, run_optimization


_LOGGER = get_logger("DragonOptimizer")


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
        >>> # 1. Define bounds for continuous features
        >>> cont_bounds = {'feature_A': (0, 100), 'feature_B': (-10, 10)}
        >>>
        >>> # 2. Initialize the optimizer
        >>> optimizer = DragonOptimizer(
        ...     inference_handler=my_handler,
        ...     schema=schema,
        ...     target_name="my_target",
        ...     continuous_bounds_map=cont_bounds,
        ...     task="max",
        ...     algorithm="Genetic",
        ... )
        >>> # 3. Run the optimization
        >>> best_result = optimizer.run(
        ...     num_generations=100,
        ...     save_dir="/path/to/results",
        ...     save_format="csv"
        ... )
    """
    def __init__(self,
                 inference_handler: DragonInferenceHandler,
                 schema: FeatureSchema,
                 target_name: str,
                 continuous_bounds_map: dict[str, tuple[float, float]],
                 task: Literal["min", "max"],
                 algorithm: Literal["SNES", "CEM", "Genetic"] = "Genetic",
                 population_size: int = 200,
                 discretize_start_at_zero: bool = True,
                 **searcher_kwargs):
        """
        Initializes the optimizer by creating the EvoTorch problem and searcher.

        Args:
            inference_handler (DragonInferenceHandler): 
                An initialized inference handler containing the model.
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            target_name (str): 
                target name to optimize.
            continuous_bounds_map (Dict[str, Tuple[float, float]]):
                A dictionary mapping the *name* of each **continuous** feature
                to its (min_bound, max_bound) tuple.
            task (str): The optimization goal, either "min" or "max".
            
            algorithm (str): The search algorithm to use ("SNES", "CEM", "Genetic").
            population_size (int): Population size for CEM and GeneticAlgorithm.
            discretize_start_at_zero (bool): 
                True if the discrete encoding starts at 0 (e.g., [0, 1, 2]).
                False if it starts at 1 (e.g., [1, 2, 3]).
            **searcher_kwargs: Additional keyword arguments for the selected 
                               search algorithm's constructor.
        """
        # --- Store schema ---
        self.schema = schema
        # --- Store inference handler ---
        self.inference_handler = inference_handler
        
        # Ensure only Regression tasks are used
        allowed_tasks = [MLTaskKeys.REGRESSION, MLTaskKeys.MULTITARGET_REGRESSION]
        if self.inference_handler.task not in allowed_tasks:
            _LOGGER.error(f"DragonOptimizer only supports {allowed_tasks}. Got '{self.inference_handler.task}'.")
            raise ValueError(f"Invalid Task: {self.inference_handler.task}")
        
        # --- store target name ---
        self.target_name = target_name
        
        # --- flag to control single vs multi-target ---
        self.is_multi_target = False
        
        # --- 1. Create bounds from schema ---
        # This is the robust way to get bounds
        bounds = create_optimization_bounds(
            schema=schema,
            continuous_bounds_map=continuous_bounds_map,
            start_at_zero=discretize_start_at_zero
        )
        
        # Resolve target index if multi-target
        target_index = None
        
        if self.inference_handler.target_ids is None:
            # This should be caught by ML_inference logic
            _LOGGER.error("The provided inference handler does not have 'target_ids' defined.")
            raise ValueError()

        if target_name not in self.inference_handler.target_ids:
            _LOGGER.error(f"Target name '{target_name}' not found in the inference handler's 'target_ids': {self.inference_handler.target_ids}")
            raise ValueError()

        if len(self.inference_handler.target_ids) == 1:
            # Single target regression
            target_index = None
            _LOGGER.info(f"Optimization locked to single-target model '{target_name}'.")
        else:
            # Multi-target regression (optimizing one specific column)
            target_index = self.inference_handler.target_ids.index(target_name)
            self.is_multi_target = True
            _LOGGER.info(f"Optimization locked to target '{target_name}' (Index {target_index}) in a multi-target model.")
        
        # --- 2. Make a fitness function ---
        self.evaluator = FitnessEvaluator(
            inference_handler=inference_handler,
            # Get categorical info from the schema
            categorical_index_map=schema.categorical_index_map,
            discretize_start_at_zero=discretize_start_at_zero,
            target_index=target_index
        )
        
        # --- 3. Create the problem and searcher factory ---
        self.problem, self.searcher_factory = create_pytorch_problem(
            evaluator=self.evaluator,
            bounds=bounds,
            task=task,
            algorithm=algorithm,
            population_size=population_size,
            **searcher_kwargs
        )
        
        # --- 4. Store other info needed by run() ---
        self.discretize_start_at_zero = discretize_start_at_zero

    def run(self,
            num_generations: int,
            save_dir: Union[str, Path],
            save_format: Literal['csv', 'sqlite', 'both'],
            repetitions: int = 1,
            verbose: bool = True) -> Optional[dict]:
        """
        Runs the evolutionary optimization process using the pre-configured settings.

        The `feature_names` are automatically pulled from the `FeatureSchema`
        provided during initialization.

        Args:
            num_generations (int): The total number of generations for each repetition.
            save_dir (str | Path): The directory where result files will be saved.
            save_format (Literal['csv', 'sqlite', 'both']): The format for saving results.
            repetitions (int): The number of independent times to run the optimization.
            verbose (bool): If True, enables detailed logging.

        Returns:
            Optional[dict]: A dictionary with the best result if repetitions is 1, 
                            otherwise None.
        """
        # Pass inference handler and target names for multi-target only
        if self.is_multi_target:
            target_names_to_pass = self.inference_handler.target_ids
            inference_handler_to_pass = self.inference_handler
        else:
            target_names_to_pass = None
            inference_handler_to_pass = None
        
        # Call the existing run function, passing info from the schema
        return run_optimization(
            problem=self.problem,
            searcher_factory=self.searcher_factory,
            num_generations=num_generations,
            target_name=self.target_name,
            save_dir=save_dir,
            save_format=save_format,
            # Get the definitive feature names (as a list) from the schema
            feature_names=list(self.schema.feature_names),
            # Get categorical info from the schema
            categorical_map=self.schema.categorical_index_map,
            categorical_mappings=self.schema.categorical_mappings,
            repetitions=repetitions,
            verbose=verbose,
            discretize_start_at_zero=self.discretize_start_at_zero,
            all_target_names=target_names_to_pass,
            inference_handler=inference_handler_to_pass
        )
        
