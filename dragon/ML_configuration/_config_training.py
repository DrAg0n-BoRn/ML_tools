from typing import Union, Optional, Any, Literal
from pathlib import Path

from ..schema import FeatureSchema

from .._core import get_logger
from ..keys._keys import MLTaskKeys


_LOGGER = get_logger("Training Configuration")


__all__ = [    
    "DragonTrainingConfig",
]


class DragonTrainingConfig:
    """
    Configuration object for the training process.
    
    Accepts arbitrary keyword arguments which are set as instance attributes.
    
    Fully compatible with custom training loggers via `to_log()`.
    """
    def __init__(self, *, #enforce keyword args
                 finalized_filename: str,
                 initial_learning_rate: float,
                 batch_size: int,
                 device: str,
                 task: Literal["regression",
                               "multitarget regression",
                               "binary classification",
                               "multiclass classification",
                               "multilabel binary classification",
                               "binary image classification",
                               "multiclass image classification",
                               "binary segmentation",
                               "multiclass segmentation",
                               "object detection",
                               "autoregressive-sequence-to-sequence",
                               "autoregressive-sequence-to-value",
                               "exogenous-sequence-to-sequence",
                               "exogenous-sequence-to-value",
                               "diffusion",
                               "autoencoder"],
                 # Optional for data splitting
                 validation_size: Optional[float] = None,
                 test_size: Optional[float] = None,
                 targets: Optional[Union[list[str], str]] = None,
                 random_state: Optional[int] = None,
                 # optional for callbacks
                 weight_decay: Optional[float] = None,
                 early_stop_patience: Optional[int] = None,
                 scheduler_patience: Optional[int] = None,
                 scheduler_lr_factor: Optional[float] = None,
                 monitor_metric: Optional[Union[Literal["Validation Loss"], Literal["Training Loss"], str]] = None,
                 **kwargs: Any) -> None:
        """
        Args:
            finalized_filename (str): Name of the Finalized-file.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            batch_size (int): Batch size for training.
            device (str): Device to use for training (e.g., "cpu", "cuda:0").
            task (str): Task type for training. Must be one of the predefined tasks in MLTaskKeys.ALL_TASKS.
            validation_size (float | None): Optional fraction of data to use for validation (between 0 and 1).
            test_size (float | None): Optional fraction of data to use for testing (between 0 and 1).
            targets (List[str] | str | None): Optional list of target column names or a single target name.
            random_state (int | None): Optional random seed for reproducibility.
            weight_decay (float | None): Optional weight decay for optimizers.
            early_stop_patience (int | None): Optional patience for early stopping.
            scheduler_patience (int | None): Optional patience for learning rate scheduler.
            scheduler_lr_factor (float | None): Optional factor for reducing learning rate in scheduler.
            monitor_metric (str | None): Optional metric to monitor for callbacks (e.g., "Validation Loss").
            **kwargs: Additional training parameters as key-value pairs.
        """
        self.finalized_filename = finalized_filename
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.device = device
        
        # validate task
        if task not in MLTaskKeys.ALL_TASKS:
            _LOGGER.error(f"Invalid task '{task}'. Must be one of: {MLTaskKeys.ALL_TASKS}")
            raise ValueError()
        self.task = task
        
        # Optional parameters to be obtained through getter properties
        self._validation_size = validation_size
        self._test_size = test_size
        self._targets = targets
        self._random_state = random_state
        self._weight_decay = weight_decay
        self._early_stop_patience = early_stop_patience
        self._scheduler_patience = scheduler_patience
        self._scheduler_lr_factor = scheduler_lr_factor
        self._monitor_metric = monitor_metric
        
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
    
    ### Getter properties for optional parameters with validation ###
    @property
    def validation_size(self) -> float:
        if self._validation_size is None:
            _LOGGER.error("Validation size is not set.")
            raise ValueError()
        return self._validation_size
    
    @property
    def test_size(self) -> float:
        if self._test_size is None:
            _LOGGER.error("Test size is not set.")
            raise ValueError()
        return self._test_size
    
    @property
    def targets(self) -> Union[list[str], str]:
        if self._targets is None:
            _LOGGER.error("Targets are not set.")
            raise ValueError()
        return self._targets
    
    @property
    def random_state(self) -> int:
        if self._random_state is None:
            _LOGGER.error("Random state is not set.")
            raise ValueError()
        return self._random_state
    
    @property
    def weight_decay(self) -> float:
        if self._weight_decay is None:
            _LOGGER.error("Weight decay is not set.")
            raise ValueError()
        return self._weight_decay
    
    @property
    def early_stop_patience(self) -> int:
        if self._early_stop_patience is None:
            _LOGGER.error("Early stop patience is not set.")
            raise ValueError()
        return self._early_stop_patience
    
    @property
    def scheduler_patience(self) -> int:
        if self._scheduler_patience is None:
            _LOGGER.error("Scheduler patience is not set.")
            raise ValueError()
        return self._scheduler_patience
    
    @property
    def scheduler_lr_factor(self) -> float:
        if self._scheduler_lr_factor is None:
            _LOGGER.error("Scheduler learning rate factor is not set.")
            raise ValueError()
        return self._scheduler_lr_factor
    
    @property
    def monitor_metric(self) -> Union[Literal["Validation Loss"], Literal["Training Loss"], str]:
        if self._monitor_metric is None:
            _LOGGER.error("Monitor metric is not set.")
            raise ValueError()
        return self._monitor_metric
    
    ### Logging and representation methods ###
    def _get_public_state(self) -> dict[str, Any]:
        """Safely extracts all public attributes, kwargs, and valid properties."""
        state = {}
        # 1. Capture direct attributes and dynamic kwargs
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                state[k] = v
                
        # 2. Capture properties
        properties = [
            'validation_size', 
            'test_size', 
            'targets', 
            'random_state',
            'weight_decay', 
            'early_stop_patience', 
            'scheduler_patience',
            'scheduler_lr_factor', 
            'monitor_metric'
        ]
        
        for prop in properties:
            if getattr(self, f"_{prop}", None) is not None:
                state[prop] = getattr(self, prop)

        return state
    
    def to_log(self) -> dict[str, Any]:
        """        
        Returns a dictionary for JSON logging.
        
        Converts Path and FeatureSchema to strings if detected.
        """
        clean_dict = {}
        for k, v in self._get_public_state().items():
            if isinstance(v, FeatureSchema):
                clean_dict[k] = repr(v)
            elif isinstance(v, Path):
                clean_dict[k] = str(v)
            else:
                clean_dict[k] = v
        return clean_dict
    
    def __repr__(self) -> str:
        """Returns a formatted multi-line string representation of the public state."""
        class_name = self.__class__.__name__
        params = []
        for k, v in self._get_public_state().items():
            params.append(f"  {k}={repr(v)}")
            
        params_str = ",\n".join(params)
        return f"{class_name}(\n{params_str}\n)"
