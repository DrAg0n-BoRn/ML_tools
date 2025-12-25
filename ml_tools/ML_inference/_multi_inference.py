import torch
import numpy as np
from typing import Union, Literal, Any

from .._core import get_logger
from ..keys._keys import PyTorchInferenceKeys, MLTaskKeys

from ._dragon_inference import DragonInferenceHandler


_LOGGER = get_logger("Multi Inference")


__all__ = [
    "multi_inference_regression",
    "multi_inference_classification"
]


def multi_inference_regression(handlers: list[DragonInferenceHandler], 
                               feature_vector: Union[np.ndarray, torch.Tensor], 
                               output: Literal["numpy","torch"]="numpy") -> dict[str,Any]:
    """
    Performs regression inference using multiple models on a single feature vector.

    This function iterates through a list of DragonInferenceHandler objects,
    each configured for a different regression target. It runs a prediction for
    each handler using the same input feature vector and returns the results
    in a dictionary.
    
    The function adapts its behavior based on the input dimensions:
    - 1D input: Returns a dictionary mapping target ID to a single value.
    - 2D input: Returns a dictionary mapping target ID to a list of values.

    Args:
        handlers (list[DragonInferenceHandler]): A list of initialized inference
            handlers. Each handler must have a unique `target_id` and be configured with `task="regression"`.
        feature_vector (Union[np.ndarray, torch.Tensor]): An input sample (1D) or a batch of samples (2D) to be fed into each regression model.
        output (Literal["numpy", "torch"], optional): The desired format for the output predictions.
            - "numpy": Returns predictions as Python scalars or NumPy arrays.
            - "torch": Returns predictions as PyTorch tensors.

    Returns:
        (dict[str, Any]): A dictionary mapping each handler's `target_id` to its
        predicted regression values. 

    Raises:
        AttributeError: If any handler in the list is missing a `target_id`.
        ValueError: If any handler's `task` is not 'regression' or if the input `feature_vector` is not 1D or 2D.
    """
    # check batch dimension
    is_single_sample = feature_vector.ndim == 1
    
    # Reshape a 1D vector to a 2D batch of one for uniform processing.
    if is_single_sample:
        feature_vector = feature_vector.reshape(1, -1)
    
    # Validate that the input is a 2D tensor.
    if feature_vector.ndim != 2:
        _LOGGER.error("Input feature_vector must be a 1D or 2D array/tensor.")
        raise ValueError()
    
    results: dict[str,Any] = dict()
    for handler in handlers:
        # validation
        if handler.target_ids is None:
            _LOGGER.error("All inference handlers must have a 'target_ids' attribute.")
            raise AttributeError()
        if handler.task != MLTaskKeys.REGRESSION:
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_ids[0]}' is for '{handler.task}', only single target regression tasks are supported.")
            raise ValueError()
            
        # inference
        if output == "numpy":
            # This path returns NumPy arrays or standard Python scalars
            numpy_result = handler.predict_batch_numpy(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, convert the 1-element array to a Python scalar
                results[handler.target_ids[0]] = numpy_result.item()
            else:
                # For a batch, return the full NumPy array of predictions
                results[handler.target_ids[0]] = numpy_result

        else:  # output == "torch"
            # This path returns PyTorch tensors on the model's device
            torch_result = handler.predict_batch(feature_vector)[PyTorchInferenceKeys.PREDICTIONS]
            if is_single_sample:
                # For a single sample, return the 0-dim tensor
                results[handler.target_ids[0]] = torch_result[0]
            else:
                # For a batch, return the full tensor of predictions
                results[handler.target_ids[0]] = torch_result

    return results


def multi_inference_classification(
    handlers: list[DragonInferenceHandler], 
    feature_vector: Union[np.ndarray, torch.Tensor], 
    output: Literal["numpy","torch"]="numpy"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Performs classification inference on a single sample or a batch.

    This function iterates through a list of DragonInferenceHandler objects,
    each configured for a different classification target. It returns two
    dictionaries: one for the predicted labels and one for the probabilities.

    The function adapts its behavior based on the input dimensions:
    - 1D input: The dictionaries map target ID to a single label and a single probability array.
    - 2D input: The dictionaries map target ID to an array of labels and an array of probability arrays.

    Args:
        handlers (list[DragonInferenceHandler]): A list of initialized inference handlers. Each must have a unique `target_id` and be configured
            with `task="classification"`.
        feature_vector (Union[np.ndarray, torch.Tensor]): An input sample (1D)
            or a batch of samples (2D) for prediction.
        output (Literal["numpy", "torch"], optional): The desired format for the
            output predictions.

    Returns:
        (tuple[dict[str, Any], dict[str, Any]]): A tuple containing two dictionaries:
        1.  A dictionary mapping `target_id` to the predicted label(s).
        2.  A dictionary mapping `target_id` to the prediction probabilities.

    Raises:
        AttributeError: If any handler in the list is missing a `target_id`.
        ValueError: If any handler's `task` is not 'classification' or if the input `feature_vector` is not 1D or 2D.
    """
    # Store if the original input was a single sample
    is_single_sample = feature_vector.ndim == 1
    
    # Reshape a 1D vector to a 2D batch of one for uniform processing
    if is_single_sample:
        feature_vector = feature_vector.reshape(1, -1)
    
    if feature_vector.ndim != 2:
        _LOGGER.error("Input feature_vector must be a 1D or 2D array/tensor.")
        raise ValueError()

    # Initialize two dictionaries for results
    labels_results: dict[str, Any] = dict()
    probs_results: dict[str, Any] = dict()

    for handler in handlers:
        # Validation
        if handler.target_ids is None:
            _LOGGER.error("All inference handlers must have a 'target_id' attribute.")
            raise AttributeError()
        if handler.task not in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTICLASS_CLASSIFICATION]:
            _LOGGER.error(f"Invalid task type: The handler for target_id '{handler.target_ids[0]}' is for '{handler.task}', but this function only supports binary and multiclass classification.")
            raise ValueError()
            
        # Inference
        if output == "numpy":
            # predict_batch_numpy returns a dict of NumPy arrays
            result = handler.predict_batch_numpy(feature_vector)
        else: # torch
            # predict_batch returns a dict of Torch tensors
            result = handler.predict_batch(feature_vector)
        
        labels = result[PyTorchInferenceKeys.LABELS]
        probabilities = result[PyTorchInferenceKeys.PROBABILITIES]
        
        if is_single_sample:
            # For "numpy", convert the single label to a Python int scalar.
            # For "torch", get the 0-dim tensor label.
            if output == "numpy":
                labels_results[handler.target_ids[0]] = labels.item()
            else: # torch
                labels_results[handler.target_ids[0]] = labels[0]
            
            # The probabilities are an array/tensor of values
            probs_results[handler.target_ids[0]] = probabilities[0]
        else:
            labels_results[handler.target_ids[0]] = labels
            probs_results[handler.target_ids[0]] = probabilities
            
    return labels_results, probs_results

