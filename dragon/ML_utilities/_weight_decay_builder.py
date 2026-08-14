from typing import Any
from torch import nn

from .._core import get_logger


_LOGGER = get_logger("Optimizer Utilities")


__all__ = [
    "build_optimizer_params",
]


def build_optimizer_params(model: nn.Module, weight_decay: float = 0.01, verbose: int = 2) -> list[dict[str, Any]]:
    """
    Groups model parameters to apply weight decay only to weights (matrices/embeddings),
    while excluding biases and normalization parameters (scales/shifts).

    Custom models should implement the `_get_non_decaying_parameters()` method to return a set of parameter names that should not have weight decay applied.

    Args:
        model (nn.Module): 
            The PyTorch model.
        weight_decay (float): 
            The L2 regularization coefficient for the weights. 
        verbose (int):
            Logging level.

    Returns:
        List (List[Dict[str, Any]]): A list of parameter groups formatted for PyTorch optimizers.
            - Group 0: 'params' = Weights (decay applied)
            - Group 1: 'params' = Biases/Norms (decay = 0.0)
    
    <br>
    
    ## Notes:
    
    - This function is designed to be used with PyTorch optimizers like AdamW or SGD.
    - Default Starting Point: 0.01 is the standard PyTorch default and a highly reliable baseline.
    - Transformers & Vision (DiT, ResNet, Tabular Transformers): Usually benefit from higher values, typically between 0.01 and 0.1. (Some large Vision Transformers even push up to 0.3).
    - MLPs & LSTMs: Generally prefer lower values, typically between 0.00001 and 0.00100, as they are more prone to underfitting with aggressive decay.
    - Dataset Size: Use higher weight decay for smaller datasets (to prevent memorization) and lower weight decay for massive datasets.
    - Tuning: If training loss drops but validation loss spikes (overfitting), increase this value. If training loss plateaus too early (underfitting), decrease it.
    """
    # if negative weight decay is provided, raise an error
    if weight_decay < 0.0:
        _LOGGER.error(f"Weight decay must be non-negative, but got {weight_decay}")
        raise ValueError()
    
    no_decay_strings = {"bias", "norm.weight", "norm.bias"}

    if hasattr(model, "_get_non_decaying_parameters"):
        custom_no_decay: set[str] = model._get_non_decaying_parameters() # type: ignore
        
        if verbose >= 3:
            _LOGGER.info(
                f"Model '{type(model).__name__}' provided custom non-decaying parameters: {custom_no_decay}"
            )
        if len(custom_no_decay) == 0 and verbose >= 3:
            _LOGGER.warning(
                f"Model '{type(model).__name__}' returned an empty set from `_get_non_decaying_parameters()`. "
                "This may indicate that no custom parameters were specified for exclusion from weight decay."
            )
        # Union merges the custom tabular params with standard PyTorch biases/norms
        no_decay_strings = no_decay_strings.union(custom_no_decay)
    else:
        if verbose >= 1:
            _LOGGER.warning(
                f"Model '{type(model).__name__}' is missing the custom method `_get_non_decaying_parameters()`. Using default PyTorch biases/norms for no-decay."
            )
    
    decay_params = []
    no_decay_params = []
    
    # 2. Iterate only over trainable parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check 1: Name match
        is_blacklisted_name = any(nd in name for nd in no_decay_strings)
        
        # Check 2: Dimensionality
        # Weights/Embeddings are 2D+, Biases/Norm Scales are 1D
        is_1d = param.ndim < 2

        if is_blacklisted_name or is_1d:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    if verbose >= 2:
        # Element calculations for accurate logging
        decay_elements = sum(p.numel() for p in decay_params)
        no_decay_elements = sum(p.numel() for p in no_decay_params)
        
        if weight_decay == 0.0:
            _LOGGER.info(f"Weight decay is 0.0. No parameters will have weight decay applied.")
        else:
            _LOGGER.info(
                f"Weight decay {weight_decay} applied:\n"
                f"    Decaying elements:     {decay_elements:,} (in {len(decay_params)} tensors)\n"
                f"    Non-decaying elements: {no_decay_elements:,} (in {len(no_decay_params)} tensors)"
            )
        
    return [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
        }
    ]
    
