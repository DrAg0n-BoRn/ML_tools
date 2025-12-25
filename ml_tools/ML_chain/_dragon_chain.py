import pandas as pd
from typing import Optional, Literal

from ..ML_inference import DragonInferenceHandler

from .._core import get_logger

from ._chaining_tools import (
    augment_dataset_with_predictions,
    augment_dataset_with_predictions_multi,
    prepare_chaining_dataset,
)


_LOGGER = get_logger("DragonChainOrchestrator")


__all__ = [
    "DragonChainOrchestrator",
]

class DragonChainOrchestrator:
    """
    Manages the data flow for a sequential chain of ML models (Model 1 -> Model 2 -> ... -> Model N).
    
    This orchestrator maintains a master copy of the dataset that grows as models are applied.
    1. Use `get_training_data` to extract a clean, target-specific subset for training a model.
    2. Train your model externally.
    3. Use `update_with_inference` to run that model on the master dataset and append predictions 
       as features for subsequent steps.
    """
    def __init__(self, initial_dataset: pd.DataFrame, all_targets: list[str]):
        """
        Args:
            initial_dataset (pd.DataFrame): The starting dataframe with original features and all ground truth targets.
            all_targets (list[str]): A list of all ground truth target column names present in the dataset.
        """
        # Validation: Ensure targets exist
        missing = [t for t in all_targets if t not in initial_dataset.columns]
        if missing:
            _LOGGER.error(f"The following targets were not found in the initial dataset: {missing}")
            raise ValueError()

        self.current_dataset = initial_dataset.copy()
        self.all_targets = all_targets
        _LOGGER.info(f"Orchestrator initialized with {len(initial_dataset)} samples, {len(initial_dataset.columns) - len(all_targets)} features, and {len(all_targets)} targets.")

    def get_training_data(
        self, 
        target_subset: list[str], 
        dropna_how: Literal["any", "all"] = "all"
    ) -> pd.DataFrame:
        """
        Generates a clean dataframe tailored for training a specific step in the chain.
        
        This method does NOT modify the internal state. It returns a view with:
        - Current features (including previous model predictions).
        - Only the specified `target_subset`.
        - Rows cleaned based on `dropna_how`.
        
        Args:
            target_subset (list[str]): The targets for the current model.
            dropna_how (Literal["any", "all"]): "any" drops row if any target is missing; "all" drops if all are missing.

        Returns:
            pd.DataFrame: A prepared dataframe for training.
        """
        _LOGGER.info(f"Extracting training data for targets {target_subset}...")
        return prepare_chaining_dataset(
            dataset=self.current_dataset, 
            all_targets=self.all_targets, 
            target_subset=target_subset, 
            dropna_how=dropna_how,
            verbose=False
        )

    def update_with_inference(
        self, 
        handler: DragonInferenceHandler, 
        batch_size: int = 4096
    ) -> None:
        """
        Runs inference using the provided handler on the full internal dataset and appends the results as new features.
        
        This updates the internal state of the Orchestrator. Subsequent calls to `get_training_data` 
        will include these new prediction columns as features with a standardized prefix.

        Args:
            handler (DragonInferenceHandler): The trained model handler.
            batch_size (int): Batch size for inference.
        """
        _LOGGER.info(f"Orchestrator: Updating internal state with predictions from handler (Targets: {handler.target_ids})...")
        
        # We use the existing utility to handle the augmentation
        # This keeps the logic consistent (drop GT -> predict -> concat GT)
        self.current_dataset = augment_dataset_with_predictions(
            handler=handler,
            dataset=self.current_dataset,
            ground_truth_targets=self.all_targets,
            batch_size=batch_size
        )
        
        _LOGGER.debug(f"Orchestrator State updated. Current feature count (approx): {self.current_dataset.shape[1] - len(self.all_targets)}")
        
    def update_with_ensemble(
        self,
        handlers: list[DragonInferenceHandler],
        prefixes: Optional[list[str]] = None,
        batch_size: int = 4096
    ) -> None:
        """
        Runs multiple independent inference handlers (e.g. for Stacking) on the full internal dataset 
        and appends all results as new features.
        
        Args:
            handlers (list[DragonInferenceHandler]): List of trained model handlers.
            prefixes (list[str], optional): Prefixes for each model's columns.
            batch_size (int): Batch size for inference.
        """
        _LOGGER.info(f"Orchestrator: Updating internal state with ensemble of {len(handlers)} models...")
        
        self.current_dataset = augment_dataset_with_predictions_multi(
            handlers=handlers,
            dataset=self.current_dataset,
            ground_truth_targets=self.all_targets,
            model_prefixes=prefixes,
            batch_size=batch_size
        )
        
        new_feat_count = self.current_dataset.shape[1] - len(self.all_targets)
        _LOGGER.debug(f"Orchestrator: State updated. Total current features: {new_feat_count}")

    @property
    def latest_dataset(self) -> pd.DataFrame:
        """Returns a copy of the current master dataset including all accumulated predictions."""
        return self.current_dataset.copy()

