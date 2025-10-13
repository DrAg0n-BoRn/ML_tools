from ._script_info import _script_info
from ._logger import _LOGGER
from .path_manager import make_fullpath, list_files_by_extension
from .keys import EnsembleKeys

from typing import Union, Literal, Dict, Any, Optional, List
from pathlib import Path
import json

import joblib
import numpy as np
# Inference models
import xgboost
import lightgbm


__all__ = [
    "InferenceHandler",
    "model_report"
]


class InferenceHandler:
    """
    Handles loading ensemble models and performing inference for either regression or classification tasks.
    """
    def __init__(self, 
                 models_dir: Union[str,Path], 
                 task: Literal["classification", "regression"],
                 verbose: bool=True) -> None:
        """
        Initializes the handler by loading all models from a directory.

        Args:
            models_dir (Path): The directory containing the saved .joblib model files.
            task ("regression" | "classification"): The type of task the models perform.
        """
        self.models: Dict[str, Any] = dict()
        self.task: str = task
        self.verbose = verbose
        self._feature_names: Optional[List[str]] = None
        
        model_files = list_files_by_extension(directory=models_dir, extension="joblib")
        
        for fname, fpath in model_files.items():
            try:
                full_object: dict
                full_object = _deserialize_object(filepath=fpath, 
                                                 verbose=self.verbose, 
                                                 raise_on_error=True) # type: ignore
                
                model: Any = full_object[EnsembleKeys.MODEL]
                target_name: str = full_object[EnsembleKeys.TARGET]
                feature_names_list: List[str] = full_object[EnsembleKeys.FEATURES]
                
                # Check that feature names match
                if self._feature_names is None:
                    # Store the feature names from the first model loaded.
                    self._feature_names = feature_names_list
                elif self._feature_names != feature_names_list:
                    # Add a warning if subsequent models have different feature names.
                    _LOGGER.warning(f"Mismatched feature names in {fname}. Using feature order from the first model loaded.")
                
                self.models[target_name] = model
                if self.verbose:
                    _LOGGER.info(f"Loaded model for target: {target_name}")

            except Exception:
                _LOGGER.error(f"Failed to load or parse {fname}.")

    @property
    def feature_names(self) -> List[str]:
        """
        Getter for the list of feature names the models expect.
        Returns an empty list if no models were loaded.
        """
        return self._feature_names if self._feature_names is not None else []
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predicts on a single feature vector.

        Args:
            features (np.ndarray): A 1D or 2D NumPy array for a single sample.

        Returns:
            Dict[str, Any]: A dictionary where keys are target names.
                - For regression: The value is the single predicted float.
                - For classification: The value is another dictionary {'label': ..., 'probabilities': ...}.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[0] != 1:
            _LOGGER.error("The 'predict()' method is for a single sample. Use 'predict_batch()' for multiple samples.")
            raise ValueError()

        results: Dict[str, Any] = dict()
        for target_name, model in self.models.items():
            if self.task == "regression":
                prediction = model.predict(features)
                results[target_name] = prediction.item()
            else: # Classification
                label = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                results[target_name] = {EnsembleKeys.CLASSIFICATION_LABEL: label, 
                                        EnsembleKeys.CLASSIFICATION_PROBABILITIES: probabilities}
        
        if self.verbose:
            _LOGGER.info("Inference process complete.")
        return results

    def predict_batch(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predicts on a batch of feature vectors.

        Args:
            features (np.ndarray): A 2D NumPy array where each row is a sample.

        Returns:
            Dict[str, Any]: A dictionary where keys are target names.
                - For regression: The value is a NumPy array of predictions.
                - For classification: The value is another dictionary {'labels': ..., 'probabilities': ...}.
        """
        if features.ndim != 2:
            _LOGGER.error("Input for batch prediction must be a 2D array.")
            raise ValueError()

        results: Dict[str, Any] = dict()
        for target_name, model in self.models.items():
            if self.task == "regression":
                results[target_name] = model.predict(features)
            else: # Classification
                labels = model.predict(features)
                probabilities = model.predict_proba(features)
                results[target_name] = {"labels": labels, "probabilities": probabilities}
                
        if self.verbose:
            _LOGGER.info("Inference process complete.")

        return results


def model_report(
        model_path: Union[str,Path],
        output_dir: Optional[Union[str,Path]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
    """
    Deserializes a model and generates a summary report.

    This function loads a serialized model object (joblib), prints a summary to the
    console (if verbose), and saves a detailed JSON report.

    Args:
        model_path (str): The path to the serialized model file.
        output_dir (str, optional): Directory to save the JSON report.
            If None, it defaults to the same directory as the model file.
        verbose (bool, optional): If True, prints summary information
            to the console. Defaults to True.

    Returns:
        (Dict[str, Any]): A dictionary containing the model metadata.

    Raises:
        FileNotFoundError: If the model_path does not exist.
        KeyError: If the deserialized object is missing required keys from `ModelSaveKeys`.
    """
    # 1. Convert to Path object
    model_p = make_fullpath(model_path)

    # --- 2. Deserialize and Extract Info ---
    try:
        full_object: dict = _deserialize_object(model_p) # type: ignore
        model = full_object[EnsembleKeys.MODEL]
        target = full_object[EnsembleKeys.TARGET]
        features = full_object[EnsembleKeys.FEATURES]
    except FileNotFoundError:
        _LOGGER.error(f"Model file not found at '{model_p}'")
        raise
    except (KeyError, TypeError) as e:
        _LOGGER.error(
            f"The serialized object is missing required keys '{EnsembleKeys.MODEL}', '{EnsembleKeys.TARGET}', '{EnsembleKeys.FEATURES}'"
        )
        raise e

    # --- 3. Print Summary to Console (if verbose) ---
    if verbose:
        print("\n--- 📝 Model Summary ---")
        print(f"Source File:    {model_p.name}")
        print(f"Model Type:     {type(model).__name__}")
        print(f"Target:         {target}")
        print(f"Feature Count:  {len(features)}")
        print("-----------------------")

    # --- 4. Generate JSON Report ---
    report_data = {
        "source_file": model_p.name,
        "model_type": str(type(model)),
        "target_name": target,
        "feature_count": len(features),
        "feature_names": features
    }

    # Determine output path
    output_p = make_fullpath(output_dir, make=True) if output_dir else model_p.parent
    json_filename = model_p.stem + "_info.json"
    json_filepath = output_p / json_filename    

    try:
        with open(json_filepath, 'w') as f:
            json.dump(report_data, f, indent=4)
        if verbose:
            _LOGGER.info(f"JSON report saved to: '{json_filepath}'")
    except PermissionError:
        _LOGGER.exception(f"Permission denied to write JSON report at '{json_filepath}'.")

    # --- 5. Return the extracted data ---
    return report_data


# Local implementation to avoid calling utilities dependencies
def _deserialize_object(filepath: Union[str,Path], verbose: bool=True, raise_on_error: bool=True) -> Optional[Any]:
    """
    Loads a serialized object from a .joblib file.

    Parameters:
        filepath (str | Path): Full path to the serialized .joblib file.

    Returns:
        (Any | None): The deserialized Python object, or None if loading fails.
    """
    true_filepath = make_fullpath(filepath)
    
    try:
        obj = joblib.load(true_filepath)
    except (IOError, OSError, EOFError, TypeError, ValueError) as e:
        _LOGGER.error(f"Failed to deserialize object from '{true_filepath}'.")
        if raise_on_error:
            raise e
        return None
    else:
        if verbose:
            _LOGGER.info(f"Loaded object of type '{type(obj)}'")
        return obj


def info():
    _script_info(__all__)
