from pathlib import Path
from typing import Union, Any, Iterable
from torch import nn

from ..serde import serialize_object_filename

from .._core import get_logger


_LOGGER = get_logger("Torch Utilities")


__all__ = [
    "build_optimizer_params",
    "set_parameter_requires_grad",
    "save_pretrained_transforms",
]


def build_optimizer_params(model: nn.Module, weight_decay: float = 0.01) -> list[dict[str, Any]]:
    """
    Groups model parameters to apply weight decay only to weights (matrices/embeddings),
    while excluding biases and normalization parameters (scales/shifts).

    This function uses a robust hybrid strategy:
    1. It excludes parameters matching standard names (e.g., "bias", "norm").
    2. It excludes any parameter with < 2 dimensions (vector parameters), which 
       automatically catches unnamed BatchNorm/LayerNorm weights in Sequential containers.

    Args:
        model (nn.Module): 
            The PyTorch model.
        weight_decay (float): 
            The L2 regularization coefficient for the weights. 
            (Default: 0.01)

    Returns:
        List[Dict[str, Any]]: A list of parameter groups formatted for PyTorch optimizers.
            - Group 0: 'params' = Weights (decay applied)
            - Group 1: 'params' = Biases/Norms (decay = 0.0)
    """
    # 1. Hard-coded strings for explicit safety
    no_decay_strings = {"bias", "LayerNorm", "BatchNorm", "GroupNorm", "norm.weight"}
    
    decay_params = []
    no_decay_params = []
    
    # 2. Iterate only over trainable parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check 1: Name match
        is_blacklisted_name = any(nd in name for nd in no_decay_strings)
        
        # Check 2: Dimensionality (Robust fallback)
        # Weights/Embeddings are 2D+, Biases/Norm Scales are 1D
        is_1d = param.ndim < 2

        if is_blacklisted_name or is_1d:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    _LOGGER.info(f"Weight decay configured:\n    Decaying parameters: {len(decay_params)}\n    Non-decaying parameters: {len(no_decay_params)}")

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


def set_parameter_requires_grad(
    model: nn.Module,
    unfreeze_last_n_params: int,
) -> int:
    """
    Freezes or unfreezes parameters in a model based on unfreeze_last_n_params.

    - N = 0: Freezes ALL parameters.
    - N > 0 and N < total: Freezes ALL parameters, then unfreezes the last N.
    - N >= total: Unfreezes ALL parameters.

    Note: 'N' refers to individual parameter tensors (e.g., `layer.weight`
    or `layer.bias`), not modules or layers. For example, to unfreeze
    the final nn.Linear layer, you would use N=2 (for its weight and bias).

    Args:
        model (nn.Module): The model to modify.
        unfreeze_last_n_params (int):
            The number of parameter tensors to unfreeze, starting from
            the end of the model.

    Returns:
        int: The total number of individual parameters (elements) that were set to `requires_grad=True`.
    """
    if unfreeze_last_n_params < 0:
        _LOGGER.error(f"unfreeze_last_n_params must be >= 0, but got {unfreeze_last_n_params}")
        raise ValueError()

    # --- Step 1: Get all parameter tensors ---
    all_params = list(model.parameters())
    total_param_tensors = len(all_params)

    # --- Case 1: N = 0 (Freeze ALL parameters) ---
    # early exit for the "freeze all" case.
    if unfreeze_last_n_params == 0:
        params_frozen = _set_params_grad(all_params, requires_grad=False)
        _LOGGER.warning(f"Froze all {total_param_tensors} parameter tensors ({params_frozen} total elements).")
        return 0  # 0 parameters unfrozen

    # --- Case 2: N >= total (Unfreeze ALL parameters) ---
    if unfreeze_last_n_params >= total_param_tensors:
        if unfreeze_last_n_params > total_param_tensors:
            _LOGGER.warning(f"Requested to unfreeze {unfreeze_last_n_params} params, but model only has {total_param_tensors}. Unfreezing all.")
        
        params_unfrozen = _set_params_grad(all_params, requires_grad=True)
        _LOGGER.info(f"Unfroze all {total_param_tensors} parameter tensors ({params_unfrozen} total elements) for training.")
        return params_unfrozen

    # --- Case 3: 0 < N < total (Standard: Freeze all, unfreeze last N) ---
    # Freeze ALL
    params_frozen = _set_params_grad(all_params, requires_grad=False)
    _LOGGER.info(f"Froze {params_frozen} parameters.")

    # Unfreeze the last N
    params_to_unfreeze = all_params[-unfreeze_last_n_params:]
    
    # these are all False, so the helper will set them to True
    params_unfrozen = _set_params_grad(params_to_unfreeze, requires_grad=True)

    _LOGGER.info(f"Unfroze the last {unfreeze_last_n_params} parameter tensors ({params_unfrozen} total elements) for training.")

    return params_unfrozen


def _set_params_grad(
    params: Iterable[nn.Parameter], 
    requires_grad: bool
) -> int:
    """
    A helper function to set the `requires_grad` attribute for an iterable
    of parameters and return the total number of elements changed.
    """
    params_changed = 0
    for param in params:
        if param.requires_grad != requires_grad:
            param.requires_grad = requires_grad
            params_changed += param.numel()
    return params_changed


def save_pretrained_transforms(model: nn.Module, output_dir: Union[str, Path]):
    """
    Checks a model for the 'self._pretrained_default_transforms' attribute, if found,
    serializes the returned transform object as a .joblib file.
    
    Used for wrapper vision models when initialized with pre-trained weights.

    This saves the callable transform object itself for
    later use, such as passing it directly to the 'transform_source'
    argument of the PyTorchVisionInferenceHandler.

    Args:
        model (nn.Module): The model instance to check.
        output_dir (str | Path): The directory where the transform file will be saved.
    """
    output_filename = "pretrained_model_transformations"

    # 1. Check for the "secret attribute"
    if not hasattr(model, '_pretrained_default_transforms'):
        _LOGGER.warning(f"Model of type {type(model).__name__} does not have the required attribute. No transformations saved.")
        return

    # 2. Get the transform object
    try:
        transform_obj = model._pretrained_default_transforms
    except Exception as e:
        _LOGGER.error(f"Error calling the required attribute on model: {e}")
        return

    # 3. Check if the object is actually there
    if transform_obj is None:
        _LOGGER.warning(f"Model {type(model).__name__} has the required attribute but returned None. No transforms saved.")
        return

    # 4. Serialize and save using serde
    try:
        serialize_object_filename(
            obj=transform_obj,
            save_dir=output_dir,
            filename=output_filename,
            verbose=True,
            raise_on_error=True
        )
        # _LOGGER.info(f"Successfully saved pretrained transforms to '{output_dir}'.")
    except Exception as e:
        _LOGGER.error(f"Failed to serialize transformations: {e}")
        raise
