import pandas as pd
from pathlib import Path
from typing import Union, Any, Optional, Dict, List, Iterable
import torch
from torch import nn

from .path_manager import make_fullpath, list_subdirectories, list_files_by_extension
from ._script_info import _script_info
from ._logger import _LOGGER
from ._keys import DatasetKeys, PytorchModelArchitectureKeys, PytorchArtifactPathKeys, SHAPKeys, UtilityKeys, PyTorchCheckpointKeys
from .utilities import load_dataframe
from .custom_logger import save_list_strings, custom_logger
from .serde import serialize_object_filename


__all__ = [
    "find_model_artifacts",
    "select_features_by_shap",
    "get_model_parameters",
    "inspect_model_architecture",
    "inspect_pth_file",
    "set_parameter_requires_grad",
    "save_pretrained_transforms"
]


def find_model_artifacts(target_directory: Union[str,Path], load_scaler: bool, verbose: bool=False) -> list[dict[str,Any]]:
    """
    Scans subdirectories to find paths to model weights, target names, feature names, and model architecture. Optionally an scaler path if `load_scaler` is True.

    This function operates on a specific directory structure. It expects the
    `target_directory` to contain one or more subdirectories, where each
    subdirectory represents a single trained model result.

    The expected directory structure for each model is as follows:
    ```
        target_directory
        ├── model_1
        │   ├── *.pth
        │   ├── scaler_*.pth          (Required if `load_scaler` is True)
        │   ├── feature_names.txt
        │   ├── target_names.txt
        │   └── architecture.json
        └── model_2/
            └── ...
    ```

    Args:
        target_directory (str | Path): The path to the root directory that contains model subdirectories.
        load_scaler (bool): If True, the function requires and searches for a scaler file (`.pth`) in each model subdirectory.
        verbose (bool): If True, enables detailed logging during the file paths search process.

    Returns:
        (list[dict[str, Path]]): A list of dictionaries, where each dictionary
            corresponds to a model found in a subdirectory. The dictionary
            maps standardized keys to the absolute paths of the model's
            artifacts (weights, architecture, features, targets, and scaler).
            The scaler path will be `None` if `load_scaler` is False.
    """
    # validate directory
    root_path = make_fullpath(target_directory, enforce="directory")
    
    # store results
    all_artifacts: list[dict] = list()
    
    # find model directories
    result_dirs_dict = list_subdirectories(root_dir=root_path, verbose=verbose)
    for dir_name, dir_path in result_dirs_dict.items():
        # find files
        model_pth_dict = list_files_by_extension(directory=dir_path, extension="pth", verbose=verbose)
        
        # restriction
        if load_scaler:
            if len(model_pth_dict) != 2:
                _LOGGER.error(f"Directory {dir_path} should contain exactly 2 '.pth' files: scaler and weights.")
                raise IOError()
        else:
            if len(model_pth_dict) != 1:
                _LOGGER.error(f"Directory {dir_path} should contain exactly 1 '.pth' file: weights.")
                raise IOError()
        
        ##### Scaler and Weights #####
        scaler_path = None
        weights_path = None
        
        # load weights and scaler if present
        for pth_filename, pth_path in model_pth_dict.items():
            if load_scaler and pth_filename.lower().startswith(DatasetKeys.SCALER_PREFIX):
                scaler_path = pth_path
            else:
                weights_path = pth_path
        
        # validation
        if not weights_path:
            _LOGGER.error(f"Error parsing the model weights path from '{dir_name}'")
            raise IOError()
        
        if load_scaler and not scaler_path:
            _LOGGER.error(f"Error parsing the scaler path from '{dir_name}'")
            raise IOError()
        
        ##### Target and Feature names #####
        target_names_path = None
        feature_names_path = None
        
        # load feature and target names
        model_txt_dict = list_files_by_extension(directory=dir_path, extension="txt", verbose=verbose)
        
        for txt_filename, txt_path in model_txt_dict.items():
            if txt_filename == DatasetKeys.FEATURE_NAMES:
                feature_names_path = txt_path
            elif txt_filename == DatasetKeys.TARGET_NAMES:
                target_names_path = txt_path
        
        # validation
        if not target_names_path or not feature_names_path:
            _LOGGER.error(f"Error parsing features path or targets path from '{dir_name}'")
            raise IOError()
        
        ##### load model architecture path #####
        architecture_path = None
        
        model_json_dict = list_files_by_extension(directory=dir_path, extension="json", verbose=verbose)
        
        for json_filename, json_path in model_json_dict.items():
            if json_filename == PytorchModelArchitectureKeys.SAVENAME:
                architecture_path = json_path
        
        # validation
        if not architecture_path:
            _LOGGER.error(f"Error parsing the model architecture path from '{dir_name}'")
            raise IOError()
        
        ##### Paths dictionary #####
        parsing_dict = {
            PytorchArtifactPathKeys.WEIGHTS_PATH: weights_path,
            PytorchArtifactPathKeys.ARCHITECTURE_PATH: architecture_path,
            PytorchArtifactPathKeys.FEATURES_PATH: feature_names_path,
            PytorchArtifactPathKeys.TARGETS_PATH: target_names_path,
            PytorchArtifactPathKeys.SCALER_PATH: scaler_path
        }
        
        all_artifacts.append(parsing_dict)
    
    return all_artifacts


def select_features_by_shap(
    root_directory: Union[str, Path],
    shap_threshold: float,
    log_feature_names_directory: Optional[Union[str, Path]],
    verbose: bool = True) -> list[str]:
    """
    Scans subdirectories to find SHAP summary CSVs, then extracts feature
    names whose mean absolute SHAP value meets a specified threshold.

    This function is useful for automated feature selection based on feature
    importance scores aggregated from multiple models.

    Args:
        root_directory (str | Path):
            The path to the root directory that contains model subdirectories.
        shap_threshold (float):
            The minimum mean absolute SHAP value for a feature to be included
            in the final list.
        log_feature_names_directory (str | Path | None):
            If given, saves the chosen feature names as a .txt file in this directory.

    Returns:
        list[str]:
            A single, sorted list of unique feature names that meet the
            threshold criteria across all found files.
    """
    if verbose:
        _LOGGER.info(f"Starting feature selection with SHAP threshold >= {shap_threshold}")
    root_path = make_fullpath(root_directory, enforce="directory")

    # --- Step 2: Directory and File Discovery ---
    subdirectories = list_subdirectories(root_dir=root_path, verbose=False)
    
    shap_filename = SHAPKeys.SAVENAME + ".csv"

    valid_csv_paths = []
    for dir_name, dir_path in subdirectories.items():
        expected_path = dir_path / shap_filename
        if expected_path.is_file():
            valid_csv_paths.append(expected_path)
        else:
            _LOGGER.warning(f"No '{shap_filename}' found in subdirectory '{dir_name}'.")
    
    if not valid_csv_paths:
        _LOGGER.error(f"Process halted: No '{shap_filename}' files were found in any subdirectory.")
        return []

    if verbose:
        _LOGGER.info(f"Found {len(valid_csv_paths)} SHAP summary files to process.")

    # --- Step 3: Data Processing and Feature Extraction ---
    master_feature_set = set()
    for csv_path in valid_csv_paths:
        try:
            df, _ = load_dataframe(csv_path, kind="pandas", verbose=False)
            
            # Validate required columns
            required_cols = {SHAPKeys.FEATURE_COLUMN, SHAPKeys.SHAP_VALUE_COLUMN}
            if not required_cols.issubset(df.columns):
                _LOGGER.warning(f"Skipping '{csv_path}': missing required columns.")
                continue

            # Filter by threshold and extract features
            filtered_df = df[df[SHAPKeys.SHAP_VALUE_COLUMN] >= shap_threshold]
            features = filtered_df[SHAPKeys.FEATURE_COLUMN].tolist()
            master_feature_set.update(features)

        except (ValueError, pd.errors.EmptyDataError):
            _LOGGER.warning(f"Skipping '{csv_path}' because it is empty or malformed.")
            continue
        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred while processing '{csv_path}': {e}")
            continue

    # --- Step 4: Finalize and Return ---
    final_features = sorted(list(master_feature_set))
    if verbose:
        _LOGGER.info(f"Selected {len(final_features)} unique features across all files.")
        
    if log_feature_names_directory is not None:
        save_names_path = make_fullpath(log_feature_names_directory, make=True, enforce="directory")
        save_list_strings(list_strings=final_features,
                          directory=save_names_path,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=verbose)
    
    return final_features


def get_model_parameters(model: nn.Module, save_dir: Optional[Union[str,Path]]=None) -> Dict[str, int]:
    """
    Calculates the total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to inspect.
        save_dir: Optional directory to save the output as a JSON file.

    Returns:
        Dict[str, int]: A dictionary containing:
            - "total_params": The total number of parameters.
            - "trainable_params": The number of trainable parameters (where requires_grad=True).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    report = {
        UtilityKeys.TOTAL_PARAMS: total_params,
        UtilityKeys.TRAINABLE_PARAMS: trainable_params
    }
    
    if save_dir is not None:
        output_dir = make_fullpath(save_dir, make=True, enforce="directory")
        custom_logger(data=report,
                      save_directory=output_dir,
                      log_name=UtilityKeys.MODEL_PARAMS_FILE,
                      add_timestamp=False,
                      dict_as="json")

    return report


def inspect_model_architecture(
    model: nn.Module,
    save_dir: Union[str, Path]
) -> None:
    """
    Saves a human-readable text summary of a model's instantiated
    architecture, including parameter counts.

    Args:
        model (nn.Module): The PyTorch model to inspect.
        save_dir (str | Path): Directory to save the text file.
    """
    # --- 1. Validate path ---
    output_dir = make_fullpath(save_dir, make=True, enforce="directory")
    architecture_filename = UtilityKeys.MODEL_ARCHITECTURE_FILE + ".txt"
    filepath = output_dir / architecture_filename

    # --- 2. Get parameter counts from existing function ---
    try:
        params_report = get_model_parameters(model) # Get dict, don't save
        total = params_report.get(UtilityKeys.TOTAL_PARAMS, 'N/A')
        trainable = params_report.get(UtilityKeys.TRAINABLE_PARAMS, 'N/A')
        header = (
            f"Model: {model.__class__.__name__}\n"
            f"Total Parameters: {total:,}\n"
            f"Trainable Parameters: {trainable:,}\n"
            f"{'='*80}\n\n"
        )
    except Exception as e:
        _LOGGER.warning(f"Could not get model parameters: {e}")
        header = f"Model: {model.__class__.__name__}\n{'='*80}\n\n"

    # --- 3. Get architecture string ---
    arch_string = str(model)

    # --- 4. Write to file ---
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(arch_string)
        _LOGGER.info(f"Model architecture summary saved to '{filepath.name}'")
    except Exception as e:
        _LOGGER.error(f"Failed to write model architecture file: {e}")
        raise


def inspect_pth_file(
    pth_path: Union[str, Path],
    save_dir: Union[str, Path],
) -> None:
    """
    Inspects a .pth file (e.g., checkpoint) and saves a human-readable
    JSON summary of its contents.

    Args:
        pth_path (str | Path): The path to the .pth file to inspect.
        save_dir (str | Path): The directory to save the JSON report.

    Returns:
        Dict (str, Any): A dictionary containing the inspection report.

    Raises:
        ValueError: If the .pth file is empty or in an unrecognized format.
    """
    # --- 1. Validate paths ---
    pth_file = make_fullpath(pth_path, enforce="file")
    output_dir = make_fullpath(save_dir, make=True, enforce="directory")
    pth_name = pth_file.stem

    # --- 2. Load data ---
    try:
        # Load onto CPU to avoid GPU memory issues
        loaded_data = torch.load(pth_file, map_location=torch.device('cpu'))
    except Exception as e:
        _LOGGER.error(f"Failed to load .pth file '{pth_file}': {e}")
        raise

    # --- 3. Initialize Report ---
    report = {
        "top_level_type": str(type(loaded_data)),
        "top_level_summary": {},
        "model_state_analysis": None,
        "notes": []
    }

    # --- 4. Parse loaded data ---
    if isinstance(loaded_data, dict):
        # --- Case 1: Loaded data is a dictionary (most common case) ---
        # "main loop" that iterates over *everything* first.
        for key, value in loaded_data.items():
            key_summary = {}
            val_type = str(type(value))
            key_summary["type"] = val_type
            
            if isinstance(value, torch.Tensor):
                key_summary["shape"] = list(value.shape)
                key_summary["dtype"] = str(value.dtype)
            elif isinstance(value, dict):
                key_summary["key_count"] = len(value)
                key_summary["key_preview"] = list(value.keys())[:5]
            elif isinstance(value, (int, float, str, bool)):
                key_summary["value_preview"] = str(value)
            elif isinstance(value, (list, tuple)):
                 key_summary["value_preview"] = str(value)[:100]
            
            report["top_level_summary"][key] = key_summary

        # Now, try to find the model state_dict within the dict
        if PyTorchCheckpointKeys.MODEL_STATE in loaded_data and isinstance(loaded_data[PyTorchCheckpointKeys.MODEL_STATE], dict):
            report["notes"].append(f"Found standard checkpoint key: '{PyTorchCheckpointKeys.MODEL_STATE}'. Analyzing as model state_dict.")
            state_dict = loaded_data[PyTorchCheckpointKeys.MODEL_STATE]
            report["model_state_analysis"] = _generate_weight_report(state_dict)
        
        elif all(isinstance(v, torch.Tensor) for v in loaded_data.values()):
            report["notes"].append("File dictionary contains only tensors. Analyzing entire dictionary as model state_dict.")
            state_dict = loaded_data
            report["model_state_analysis"] = _generate_weight_report(state_dict)
        
        else:
            report["notes"].append("Could not identify a single model state_dict. See top_level_summary for all contents. No detailed weight analysis will be performed.")

    elif isinstance(loaded_data, nn.Module):
        # --- Case 2: Loaded data is a full pickled model ---
        # _LOGGER.warning("Loading a full, pickled nn.Module is not recommended. Inspecting its state_dict().")
        report["notes"].append("File is a full, pickled nn.Module. This is not recommended. Extracting state_dict() for analysis.")
        state_dict = loaded_data.state_dict()
        report["model_state_analysis"] = _generate_weight_report(state_dict)

    else:
        # --- Case 3: Unrecognized format (e.g., single tensor, list) ---
        _LOGGER.error(f"Could not parse .pth file. Loaded data is of type {type(loaded_data)}, not a dict or nn.Module.")
        raise ValueError()

    # --- 5. Save Report ---
    custom_logger(data=report,
                  save_directory=output_dir,
                  log_name=UtilityKeys.PTH_FILE + pth_name,
                  add_timestamp=False,
                  dict_as="json")


def _generate_weight_report(state_dict: dict) -> dict:
    """
    Internal helper to analyze a state_dict and return a structured report.
    
    Args:
        state_dict (dict): The model state_dict to analyze.

    Returns:
        dict: A report containing total parameters and a per-parameter breakdown.
    """
    weight_report = {}
    total_params = 0
    if not isinstance(state_dict, dict):
        _LOGGER.warning(f"Attempted to generate weight report on non-dict type: {type(state_dict)}")
        return {"error": "Input was not a dictionary."}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
             _LOGGER.warning(f"Skipping key '{key}' in state_dict: value is not a tensor (type: {type(tensor)}).")
             weight_report[key] = {
                 "type": str(type(tensor)),
                 "value_preview": str(tensor)[:50] # Show a preview
             }
             continue
        weight_report[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "requires_grad": tensor.requires_grad,
            "num_elements": tensor.numel()
        }
        total_params += tensor.numel()

    return {
        "total_parameters": total_params,
        "parameter_key_count": len(weight_report),
        "parameters": weight_report
    }


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


def info():
    _script_info(__all__)
