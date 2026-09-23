from typing import Literal, Union


class DragonDDPConfig:
    """
    Configuration class for managing DDP training settings in the Dragon ML Parallel framework.
    """
    def __init__(self, *,
                 dataloader_workers: int = -1,
                 
                 # Checkpoint config
                 checkpoint_monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 checkpoint_save_three_best: bool = True,
                 checkpoint_mode: Literal['min', 'max'] = 'min',
                 checkpoint_verbose: int = 1,
                 
                 # Early Stopping config
                 use_early_stopping: bool = True,
                 early_stopping_monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 early_stopping_min_delta: float = 0.0,
                 early_stopping_patience: int = 25,
                 early_stopping_mode: Literal['min', 'max'] = 'min',
                 early_stopping_verbose: int = 1,
                 
                 # Plateau Scheduler config
                 use_plateau_scheduler: bool = True,
                 scheduler_monitor: Union[Literal["Training Loss", "Validation Loss", "both"], str] = "Validation Loss",
                 scheduler_mode: Literal['min', 'max'] = 'min',
                 scheduler_factor: float = 0.6,
                 scheduler_patience: int = 4,
                 scheduler_threshold: float = 1e-4,
                 scheduler_threshold_mode: Literal['rel', 'abs'] = 'rel',
                 scheduler_cooldown: int = 0,
                 scheduler_min_lr: float = 0.0,
                 scheduler_eps: float = 1e-8,
                 scheduler_verbose: int = 1):
        """
        Unified configuration for Distributed Data Parallel (DDP) training.
        
        Args:
            dataloader_workers (int): Number of workers for DataLoader. -1 uses all available CPU cores.
            checkpoint_monitor (str): Metric to monitor for checkpointing.
            checkpoint_save_three_best (bool): Save the three best checkpoints based on the monitored metric.
            checkpoint_mode (str): Mode for checkpointing ('min' or 'max').
            checkpoint_verbose (int): Verbosity level for checkpointing.
            use_early_stopping (bool): Enable or disable early stopping.
            early_stopping_monitor (str): Metric to monitor for early stopping.
            early_stopping_min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            early_stopping_mode (str): Mode for early stopping ('min' or 'max').
            early_stopping_verbose (int): Verbosity level for early stopping.
            use_plateau_scheduler (bool): Enable or disable the ReduceLROnPlateau scheduler.
            scheduler_monitor (str): Metric to monitor for the learning rate scheduler.
            scheduler_mode (str): Mode for the learning rate scheduler ('min' or 'max').
            scheduler_factor (float): Factor by which the learning rate will be reduced.
            scheduler_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            scheduler_threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
            scheduler_threshold_mode (str): Mode for the threshold ('rel' or 'abs').
            scheduler_cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
            scheduler_min_lr (float): Lower bound on the learning rate.
            scheduler_eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.
            scheduler_verbose (int): Verbosity level for the learning rate scheduler.
        """
        self.dataloader_workers = dataloader_workers
        
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_save_three_best = checkpoint_save_three_best
        self.checkpoint_mode = checkpoint_mode
        self.checkpoint_verbose = checkpoint_verbose
        
        self.use_early_stopping = use_early_stopping
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_verbose = early_stopping_verbose
        
        self.use_plateau_scheduler = use_plateau_scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.scheduler_threshold_mode = scheduler_threshold_mode
        self.scheduler_cooldown = scheduler_cooldown
        self.scheduler_min_lr = scheduler_min_lr
        self.scheduler_eps = scheduler_eps
        self.scheduler_verbose = scheduler_verbose
    
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        params_str = ",\n".join(f"  {k}={repr(v)}" for k, v in self.__dict__.items())
        return f"{class_name}(\n{params_str}\n)"
