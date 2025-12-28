import pandas as pd
from pathlib import Path
from typing import Union, Optional
import torch
from torch import nn

from ..utilities import load_dataframe
from ..IO_tools import save_list_strings, save_json

from ..path_manager import make_fullpath, list_subdirectories
from .._core import get_logger
from ..keys._keys import DatasetKeys, SHAPKeys, UtilityKeys, PyTorchCheckpointKeys


_LOGGER = get_logger("ML Inspection")


__all__ = [
    "get_model_parameters",
    "inspect_model_architecture",
    "inspect_pth_file",
    "select_features_by_shap"
]


def get_model_parameters(model: nn.Module, save_dir: Optional[Union[str,Path]]=None, verbose: int = 3) -> dict[str, int]:
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
        
        save_json(data=report,
                  directory=output_dir,
                  filename=UtilityKeys.MODEL_PARAMS_FILE,
                  verbose=False)
        
        if verbose >= 2:
            _LOGGER.info(f"Model parameters report saved to '{output_dir.name}/{UtilityKeys.MODEL_PARAMS_FILE}.json'")

    return report


def inspect_model_architecture(
    model: nn.Module,
    save_dir: Union[str, Path],
    verbose: int = 3
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
        if verbose >= 1:
            _LOGGER.warning(f"Could not get model parameters: {e}")
        header = f"Model: {model.__class__.__name__}\n{'='*80}\n\n"

    # --- 3. Get architecture string ---
    arch_string = str(model)

    # --- 4. Write to file ---
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(arch_string)
        if verbose >= 2:
            _LOGGER.info(f"Model architecture summary saved to '{filepath.name}'")
    except Exception as e:
        _LOGGER.error(f"Failed to write model architecture file: {e}")
        raise


def inspect_pth_file(
    pth_path: Union[str, Path],
    save_dir: Union[str, Path],
    verbose: int = 3
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
            report["model_state_analysis"] = _generate_weight_report(state_dict, verbose=verbose)
        
        elif all(isinstance(v, torch.Tensor) for v in loaded_data.values()):
            report["notes"].append("File dictionary contains only tensors. Analyzing entire dictionary as model state_dict.")
            state_dict = loaded_data
            report["model_state_analysis"] = _generate_weight_report(state_dict, verbose=verbose)
        
        else:
            report["notes"].append("Could not identify a single model state_dict. See top_level_summary for all contents. No detailed weight analysis will be performed.")

    elif isinstance(loaded_data, nn.Module):
        # --- Case 2: Loaded data is a full pickled model ---
        # _LOGGER.warning("Loading a full, pickled nn.Module is not recommended. Inspecting its state_dict().")
        report["notes"].append("File is a full, pickled nn.Module. This is not recommended. Extracting state_dict() for analysis.")
        state_dict = loaded_data.state_dict()
        report["model_state_analysis"] = _generate_weight_report(state_dict, verbose=verbose)

    else:
        # --- Case 3: Unrecognized format (e.g., single tensor, list) ---
        _LOGGER.error(f"Could not parse .pth file. Loaded data is of type {type(loaded_data)}, not a dict or nn.Module.")
        raise ValueError()

    # --- 5. Save Report ---    
    save_json(data=report,
              directory=output_dir,
              filename=UtilityKeys.PTH_FILE + pth_name,
              verbose=False)
    
    if verbose >= 2:
        _LOGGER.info(f".pth file inspection report saved to '{output_dir.name}/{UtilityKeys.PTH_FILE + pth_name}.json'")


def _generate_weight_report(state_dict: dict, verbose: int = 3) -> dict:
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
        if verbose >= 1:
            _LOGGER.warning(f"Attempted to generate weight report on non-dict type: {type(state_dict)}")
        return {"error": "Input was not a dictionary."}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
             if verbose >= 1:
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


def select_features_by_shap(
    root_directory: Union[str, Path],
    shap_threshold: float,
    log_feature_names_directory: Optional[Union[str, Path]],
    verbose: int = 3) -> list[str]:
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
    if verbose >= 2:
        _LOGGER.info(f"Starting feature selection with SHAP threshold >= {shap_threshold}")
    root_path = make_fullpath(root_directory, enforce="directory")

    # --- Step 2: Directory and File Discovery ---
    subdirectories = list_subdirectories(root_dir=root_path, verbose=False, raise_on_empty=True)
    
    shap_filename = SHAPKeys.SAVENAME + ".csv"

    valid_csv_paths = []
    for dir_name, dir_path in subdirectories.items():
        expected_path = dir_path / shap_filename
        if expected_path.is_file():
            valid_csv_paths.append(expected_path)
        else:
            if verbose >= 1:
                _LOGGER.warning(f"No '{shap_filename}' found in subdirectory '{dir_name}'.")
    
    if not valid_csv_paths:
        _LOGGER.error(f"Process halted: No '{shap_filename}' files were found in any subdirectory.")
        return []

    if verbose >= 3:
        _LOGGER.info(f"Found {len(valid_csv_paths)} SHAP summary files to process.")

    # --- Step 3: Data Processing and Feature Extraction ---
    master_feature_set = set()
    for csv_path in valid_csv_paths:
        try:
            df, _ = load_dataframe(csv_path, kind="pandas", verbose=False)
            
            # Validate required columns
            required_cols = {SHAPKeys.FEATURE_COLUMN, SHAPKeys.SHAP_VALUE_COLUMN}
            if not required_cols.issubset(df.columns):
                if verbose >= 1:
                    _LOGGER.warning(f"Skipping '{csv_path}': missing required columns.")
                continue

            # Filter by threshold and extract features
            filtered_df = df[df[SHAPKeys.SHAP_VALUE_COLUMN] >= shap_threshold]
            features = filtered_df[SHAPKeys.FEATURE_COLUMN].tolist()
            master_feature_set.update(features)

        except (ValueError, pd.errors.EmptyDataError):
            if verbose >= 1:
                _LOGGER.warning(f"Skipping '{csv_path}' because it is empty or malformed.")
            continue
        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred while processing '{csv_path}': {e}")
            continue

    # --- Step 4: Finalize and Return ---
    final_features = sorted(list(master_feature_set))
    if verbose >= 2:
        _LOGGER.info(f"Selected {len(final_features)} unique features across all files.")
        
    if log_feature_names_directory is not None:
        save_names_path = make_fullpath(log_feature_names_directory, make=True, enforce="directory")
        save_list_strings(list_strings=final_features,
                          directory=save_names_path,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=False)
    
    return final_features

