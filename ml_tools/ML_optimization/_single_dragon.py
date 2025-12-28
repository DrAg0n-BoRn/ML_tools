from typing import Literal, Union, Optional
from pathlib import Path

from ..optimization_tools import create_optimization_bounds, load_continuous_bounds_template
from ..ML_inference import DragonInferenceHandler
from ..schema import FeatureSchema
from ..ML_configuration import DragonOptimizerConfig

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
            verbose: bool = True) -> Optional[dict]:
        """
        Runs the evolutionary optimization process using the pre-configured settings.

        The `feature_names` are automatically pulled from the `FeatureSchema`
        provided during initialization.

        Args:
            verbose (bool): If True, enables detailed logging.

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
        return run_optimization(
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
            verbose=verbose,
            discretize_start_at_zero=self.config.discretize_start_at_zero,
            all_target_names=target_names_to_pass,
            inference_handler=inference_handler_to_pass
        )
        
