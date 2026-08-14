import pandas as pd
from pathlib import Path
from typing import Union, Optional

from ..utilities import load_dataframe
from ..IO_tools import save_list_strings

from ..path_manager import make_fullpath, list_subdirectories
from .._core import get_logger
from ..keys._keys import DatasetKeys, SHAPKeys


_LOGGER = get_logger("SHAP Inspection")


__all__ = [
    "select_features_by_shap"
]


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
