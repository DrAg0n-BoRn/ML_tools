from typing import Any, Union, Literal
from collections.abc import Mapping


class DragonCheckpointConfig(Mapping):
    """
    Configuration class for managing checkpointing settings in the Dragon ML framework.
    
    Allows `**` unpacking and merging with other mappings using the `|` operator.
    """
    def __init__(self, *, # parameter names are keyword-only
                monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                save_three_best: bool = True, 
                mode: Literal['min', 'max'] = 'min', 
                verbose: int = 1):
        """
        Optional configuration for checkpointing during model training.
        
        Args:
            monitor (str): Metric to monitor. If "both", the sum of training loss and validation loss is used.
            save_three_best (bool): 
                - If True, keeps the top 3 best checkpoints found during training (based on metric).
                - If False, keeps the 3 most recent checkpoints (rolling window).
            mode (str): One of {'min', 'max'}. Condition to determine if the monitored metric is improving. 'min' means lower is better, 'max' means higher is better.
            verbose (int): Verbosity mode.
        """
        self.monitor = monitor
        self.save_three_best = save_three_best
        self.mode = mode
        self.verbose = verbose

    def __getitem__(self, key: str):
        # Safer than getattr, ensures we only access valid instance variables
        return self.__dict__[key]

    def __iter__(self):
        # Only iterate over the explicitly defined attributes
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
    
    def __or__(self, other) -> dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(self) | dict(other)
        return NotImplemented
    
    def __ror__(self, other) -> dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(other) | dict(self)
        return NotImplemented
    
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        # Safely loop over the instance dictionary
        params_str = ",\n".join(f"  {k}={repr(v)}" for k, v in self.__dict__.items())
        return f"{class_name}(\n{params_str}\n)"
