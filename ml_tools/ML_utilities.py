import pandas as pd
from pathlib import Path
from typing import Union, Any, Optional

from .path_manager import make_fullpath, list_subdirectories, list_files_by_extension
from ._script_info import _script_info
from ._logger import _LOGGER
from .keys import DatasetKeys, PytorchModelArchitectureKeys, PytorchArtifactPathKeys, SHAPKeys
from .utilities import load_dataframe
from .custom_logger import save_list_strings


__all__ = [
    "find_model_artifacts",
    "select_features_by_shap"
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


def info():
    _script_info(__all__)
