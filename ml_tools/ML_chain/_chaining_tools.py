import pandas as pd
import numpy as np
from math import ceil
from typing import Optional, Literal

from ..ML_inference import DragonInferenceHandler

from ..keys._keys import MLTaskKeys, PyTorchInferenceKeys, ChainKeys
from .._core import get_logger


_LOGGER = get_logger("ML Chain")


__all__ = [
    "augment_dataset_with_predictions",
    "augment_dataset_with_predictions_multi",
    "prepare_chaining_dataset",
]


def augment_dataset_with_predictions(
    handler: DragonInferenceHandler,
    dataset: pd.DataFrame,
    ground_truth_targets: list[str],
    batch_size: int = 4096
) -> pd.DataFrame:
    """
    Uses a DragonInferenceHandler to generate predictions for a dataset and appends them as new feature columns with a standardized prefix.
    
    This function splits the features from the ground truth targets, runs inference in batches to ensure
    memory efficiency, and returns a unified DataFrame containing:
    [Original Features] + [New Predictions] + [Original Targets].

    Args:
        handler (DragonInferenceHandler): The loaded inference handler. Must have `target_ids` set.
        dataset (pd.DataFrame): The input pandas DataFrame containing features and ground truth targets.
        ground_truth_targets (List[str]): A list of column names in `dataset` representing the actual targets.
            These are removed from the input features during inference and appended to the end of the result.
        batch_size (int, optional): The number of samples to process in a single inference step. 
            Prevents OOM errors on large datasets. Defaults to 4096.

    Returns:
        pd.DataFrame: A new DataFrame with the augmented features and re-ordered columns.

    Raises:
        ValueError: If `handler.target_ids` is None or if ground truth columns are missing.
    """
    # --- 1. Validation ---
    if handler.target_ids is None:
        _LOGGER.error("The provided Inference Handler does not have 'target_ids' set.")
        raise ValueError()

    missing_cols = [col for col in ground_truth_targets if col not in dataset.columns]
    if missing_cols:
        _LOGGER.error(f"The following ground truth target columns were not found in the dataset: {missing_cols}")
        raise ValueError()

    # --- 2. Preparation ---
    # Separate features (X) and ground truth targets (y)
    # We copy X to avoid SettingWithCopy warnings when we add columns later
    X = dataset.drop(columns=ground_truth_targets).copy()
    y = dataset[ground_truth_targets].copy()
    
    total_samples = len(X)
    num_batches = ceil(total_samples / batch_size)
    
    _LOGGER.info(f"Starting inference augmentation. Processing {total_samples} samples in {num_batches} batches.")

    # Container for collected predictions
    # collect numpy arrays here and vstack them at the end
    all_predictions: list[np.ndarray] = []

    # --- 3. Batched Inference ---
    # iterate using index slicing to keep memory footprint low
    for i in range(0, total_samples, batch_size):
        # Slice the current batch of features
        batch_df = X.iloc[i : i + batch_size]
        
        # Convert to numpy for the handler
        batch_features = batch_df.to_numpy()
        
        # Run inference (returns dict of numpy arrays)
        # predict_batch_numpy handles device transfer and formatting
        outputs = handler.predict_batch_numpy(batch_features)
        
        # Extract the specific tensor based on task
        if handler.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
            # For classification, we use the Labels (integers)
            batch_preds = outputs[PyTorchInferenceKeys.LABELS]
        else:
            # For regression (single or multi-target), we use Predictions (floats)
            batch_preds = outputs[PyTorchInferenceKeys.PREDICTIONS]

        # Ensure shape consistency: (Batch, Targets)
        # If the model output is 1D (e.g. single target regression or binary class), reshape to (N, 1)
        if batch_preds.ndim == 1:
            batch_preds = batch_preds.reshape(-1, 1)

        all_predictions.append(batch_preds)

    # --- 4. Assembly ---
    # Concatenate all batches into one large array
    full_prediction_array = np.vstack(all_predictions)
    
    # Generate new column names
    new_col_names = [f"{ChainKeys.CHAIN_PREDICTION_PREFIX}{tid}" for tid in handler.target_ids]
    
    # Verify dimensions match
    if full_prediction_array.shape[1] != len(new_col_names):
        _LOGGER.error(f"Model output shape {full_prediction_array.shape} does not match the number of target_ids {len(new_col_names)}.")
        raise ValueError()

    # Assign predictions to X as new columns
    # This modifies X in-place (on the copy created earlier)
    for idx, col_name in enumerate(new_col_names):
        X[col_name] = full_prediction_array[:, idx]

    # Join with original ground truth targets
    # Structure: [Original Features] + [Predictions] + [Ground Truth]
    unified_dataset = pd.concat([X, y], axis=1)

    _LOGGER.info(f"Dataset augmentation complete. New shape: {unified_dataset.shape}")
    
    return unified_dataset


def augment_dataset_with_predictions_multi(
    handlers: list[DragonInferenceHandler],
    dataset: pd.DataFrame,
    ground_truth_targets: list[str],
    model_prefixes: Optional[list[str]] = None,
    batch_size: int = 4096
) -> pd.DataFrame:
    """
    Runs multiple independent inference handlers on the same dataset and appends all predictions 
    as new feature columns (Stacking/Ensemble generation).

    This function is designed for the first layer of a stacking architecture, where multiple 
    models view the same features independently.

    Args:
        handlers (List[DragonInferenceHandler]): A list of loaded inference handlers. 
            All must have `target_ids` set.
        dataset (pd.DataFrame): The input pandas DataFrame containing features and ground truth targets.
        ground_truth_targets (List[str]): A list of column names in `dataset` representing the actual targets.
            These are removed from the input features during inference and appended to the very end of the result.
        model_prefixes (List[str], optional): A list of string prefixes, one for each handler. 
            These are prepended to the column names to distinguish predictions from different models 
            (e.g., ["rf_", "nn_"] -> "rf_pred_price", "nn_pred_price"). 
            If None, defaults to ["m0_", "m1_", "m2_", ...].
        batch_size (int, optional): The number of samples to process in a single inference step. 
            Defaults to 4096.

    Returns:
        pd.DataFrame: A new DataFrame with the original features, followed by all model predictions, 
        followed by the ground truth targets.

    Raises:
        ValueError: If `model_prefixes` length does not match `handlers` length, or if validation fails.
    """
    # --- 1. Validation ---
    if not handlers:
        _LOGGER.warning("No handlers provided to augment_dataset_with_predictions_multi. Returning original dataset.")
        return dataset.copy()

    for i, h in enumerate(handlers):
        if h.target_ids is None:
            _LOGGER.error(f"Handler at index {i} does not have 'target_ids' set.")
            raise ValueError()

    missing_cols = [col for col in ground_truth_targets if col not in dataset.columns]
    if missing_cols:
        _LOGGER.error(f"The following ground truth target columns were not found in the dataset: {missing_cols}")
        raise ValueError()
    
    # Handle Prefixes
    if model_prefixes is None:
        model_prefixes = [f"m{i}_" for i in range(len(handlers))]
    
    if len(model_prefixes) != len(handlers):
        _LOGGER.error(f"Length of model_prefixes ({len(model_prefixes)}) must match length of handlers ({len(handlers)}).")
        raise ValueError()

    # --- 2. Preparation ---
    # Separate features (X) and ground truth targets (y)
    X = dataset.drop(columns=ground_truth_targets).copy()
    y = dataset[ground_truth_targets].copy()
    
    total_samples = len(X)
    num_batches = ceil(total_samples / batch_size)
    
    _LOGGER.info(f"Starting multi-model augmentation. Processing {total_samples} samples with {len(handlers)} models.")

    # --- 3. Inference Loop (Model by Model) ---
    # We iterate models, then batches. This is safer for VRAM than loading all models and iterating batches once.
    
    for handler, prefix in zip(handlers, model_prefixes):
        
        # Container for this specific model's predictions
        model_predictions: list[np.ndarray] = []
        
        # Batched inference for this handler
        for i in range(0, total_samples, batch_size):
            batch_df = X.iloc[i : i + batch_size]
            batch_features = batch_df.to_numpy()
            
            # Run inference
            outputs = handler.predict_batch_numpy(batch_features)
            
            # Extract result based on task
            if handler.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
                batch_preds = outputs[PyTorchInferenceKeys.LABELS]
            else:
                batch_preds = outputs[PyTorchInferenceKeys.PREDICTIONS]

            # Reshape 1D -> 2D (N, 1)
            if batch_preds.ndim == 1:
                batch_preds = batch_preds.reshape(-1, 1)
                
            model_predictions.append(batch_preds)
        
        # Stack batches for this model
        full_model_output = np.vstack(model_predictions)
        
        # Generate Column Names: prefix + "pred_" + target_name
        # e.g. "m0_pred_price"
        # We explicitly add "pred_" to ensure clarity, unless the prefix already contains it.
        base_prefix = f"{prefix}pred_"
        col_names = [f"{base_prefix}{tid}" for tid in handler.target_ids] # type: ignore
        
        # Check dimensions
        if full_model_output.shape[1] != len(col_names):
             # This error usually happens if target_ids is length 1 but model outputs 2 cols (unlikely with this logic)
             _LOGGER.error(f"Model output shape {full_model_output.shape} mismatch with generated columns {col_names}.")
             raise ValueError()

        # Assign to X
        for idx, col_name in enumerate(col_names):
            X[col_name] = full_model_output[:, idx]

    # --- 4. Assembly ---
    # Join with original ground truth targets
    unified_dataset = pd.concat([X, y], axis=1)

    _LOGGER.info(f"Multi-model augmentation complete. New shape: {unified_dataset.shape}")
    
    return unified_dataset


def prepare_chaining_dataset(
    dataset: pd.DataFrame,
    all_targets: list[str],
    target_subset: list[str],
    dropna_how: Literal["any", "all"] = "all",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepares a dataset for a specific step in a model chain by isolating specific targets 
    and cleaning relevant rows.

    It performs the following operations:
    1. Validates input columns.
    2. Drops all target columns found in `all_targets` that are NOT in `target_subset`.
    3. Drops rows where the `target_subset` columns are null based on the `dropna_how` parameter.

    Args:
        dataset (pd.DataFrame): The ground truth DataFrame containing features and targets.
        all_targets (list[str]): A list of all potential target columns present in the dataset.
        target_subset (list[str]): The specific target columns to process for this iteration.
        dropna_how ("any" | "all"): Determines the condition for dropping rows based on null values in target columns.
            - "any" drops rows if any target column is null.
            - "all" drops rows only if all target columns are null.
    Returns:
        pd.DataFrame: A cleaned copy of the dataframe containing features and the `target_subset`.

    Raises:
        ValueError: If validation of columns fails.
    """
    # --- 1. Validation ---
    # Check if all_targets exist in dataset
    missing_all = [t for t in all_targets if t not in dataset.columns]
    if missing_all:
        _LOGGER.error(f"The following 'all_targets' were not found in the dataset: {missing_all}")
        raise ValueError()

    # Check if target_subset is actually a subset of all_targets
    # (convert to set for efficient subset checking, but keep lists for reporting)
    if not set(target_subset).issubset(set(all_targets)):
        invalid_subset = [t for t in target_subset if t not in all_targets]
        _LOGGER.error(f"The provided 'target_subset' contains columns not listed in 'all_targets': {invalid_subset}")
        raise ValueError()

    if verbose:
        _LOGGER.info(f"Preparing dataset for targets: {target_subset}")

    # --- 2. Preparation ---
    df = dataset.copy()
    initial_shape = df.shape

    # Identify targets to drop (targets in the global list but not in the specific request)
    targets_to_drop = [t for t in all_targets if t not in target_subset]
    
    if targets_to_drop:
        df.drop(columns=targets_to_drop, inplace=True)

    # --- 3. Cleaning ---
    # Apply dropna strategy
    # how='any' -> drop if any NaN in subset
    # how='all' -> drop if all NaNs in subset
    df.dropna(subset=target_subset, how=dropna_how, inplace=True)
    
    rows_dropped = initial_shape[0] - df.shape[0]
    _LOGGER.debug(f"Dropped {len(targets_to_drop)} irrelevant target columns.")
    if verbose:
        _LOGGER.info(f"Dropped {rows_dropped} rows due to missing target values. Final shape: {df.shape}")

    return df

