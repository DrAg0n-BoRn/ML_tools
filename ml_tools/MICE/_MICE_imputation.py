import pandas as pd
import miceforest as mf
from pathlib import Path
from typing import Optional, Union

from ..utilities import load_dataframe
from ..math_utilities import threshold_binary_values

from ..path_manager import make_fullpath, list_csv_paths
from .._core import get_logger

from ._dragon_mice import (
    _save_imputed_datasets,
    get_convergence_diagnostic,
    get_imputed_distributions,
)

_LOGGER = get_logger("MICE")


__all__ = [
    "apply_mice",
    "run_mice_pipeline",
]


def apply_mice(df: pd.DataFrame, df_name: str, binary_columns: Optional[list[str]]=None, resulting_datasets: int=1, iterations: int=20, random_state: int=101):
    
    # Initialize kernel with number of imputed datasets to generate
    kernel = mf.ImputationKernel(
        data=df,
        num_datasets=resulting_datasets,
        random_state=random_state
    )
    
    _LOGGER.info("➡️ MICE imputation running...")
    
    # Perform MICE with n iterations per dataset
    kernel.mice(iterations)
    
    # Retrieve the imputed datasets 
    imputed_datasets = [kernel.complete_data(dataset=i) for i in range(resulting_datasets)]
    
    if imputed_datasets is None or len(imputed_datasets) == 0:
        _LOGGER.error("No imputed datasets were generated. Check the MICE process.")
        raise ValueError()
    
    # threshold binary columns
    if binary_columns is not None:
        invalid_binary_columns = set(binary_columns) - set(df.columns)
        if invalid_binary_columns:
            _LOGGER.warning(f"These 'binary columns' are not in the dataset:")
            for invalid_binary_col in invalid_binary_columns:
                print(f"  - {invalid_binary_col}")
        valid_binary_columns = [col for col in binary_columns if col not in invalid_binary_columns]
        for imputed_df in imputed_datasets:
            for binary_column_name in valid_binary_columns:
                imputed_df[binary_column_name] = threshold_binary_values(imputed_df[binary_column_name]) # type: ignore
            
    if resulting_datasets == 1:
        imputed_dataset_names = [f"{df_name}_MICE"]
    else:
        imputed_dataset_names = [f"{df_name}_MICE_{i+1}" for i in range(resulting_datasets)]
    
    # Ensure indexes match
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        assert imputed_df.shape[0] == df.shape[0], f"❌ Row count mismatch in dataset {subname}" # type: ignore
        assert all(imputed_df.index == df.index), f"❌ Index mismatch in dataset {subname}" # type: ignore
    # print("✅ All imputed datasets match the original DataFrame indexes.")
    
    _LOGGER.info("MICE imputation complete.")
    
    return kernel, imputed_datasets, imputed_dataset_names


#Get names of features that had missing values before imputation
def _get_na_column_names(df: pd.DataFrame):
    return [col for col in df.columns if df[col].isna().any()]


def run_mice_pipeline(df_path_or_dir: Union[str,Path], target_columns: list[str], 
                      save_datasets_dir: Union[str,Path], save_metrics_dir: Union[str,Path], 
                      binary_columns: Optional[list[str]]=None,
                      resulting_datasets: int=1, 
                      iterations: int=20, 
                      random_state: int=101):
    """
    DEPRECATED: Use DragonMICE class instead.
    
    Call functions in sequence for each dataset in the provided path or directory:
        1. Load dataframe
        2. Apply MICE
        3. Save imputed dataset(s)
        4. Save convergence metrics
        5. Save distribution metrics
        
    Target columns must be skipped from the imputation. Binary columns will be thresholded after imputation.
    """
    # Check paths
    save_datasets_path = make_fullpath(save_datasets_dir, make=True)
    save_metrics_path = make_fullpath(save_metrics_dir, make=True)
    
    input_path = make_fullpath(df_path_or_dir)
    if input_path.is_file():
        all_file_paths = [input_path]
    else:
        all_file_paths = list(list_csv_paths(input_path, raise_on_empty=True).values())
    
    for df_path in all_file_paths:
        df: pd.DataFrame
        df, df_name = load_dataframe(df_path=df_path, kind="pandas") # type: ignore
        
        df, df_targets = _skip_targets(df, target_columns)
        
        kernel, imputed_datasets, imputed_dataset_names = apply_mice(df=df, df_name=df_name, binary_columns=binary_columns, resulting_datasets=resulting_datasets, iterations=iterations, random_state=random_state)
        
        _save_imputed_datasets(save_dir=save_datasets_path, imputed_datasets=imputed_datasets, df_targets=df_targets, imputed_dataset_names=imputed_dataset_names)
        
        imputed_column_names = _get_na_column_names(df=df)
        
        get_convergence_diagnostic(kernel=kernel, imputed_dataset_names=imputed_dataset_names, column_names=imputed_column_names, root_dir=save_metrics_path)
        
        get_imputed_distributions(kernel=kernel, df_name=df_name, root_dir=save_metrics_path, column_names=imputed_column_names)


def _skip_targets(df: pd.DataFrame, target_cols: list[str]):
    valid_targets = [col for col in target_cols if col in df.columns]
    df_targets = df[valid_targets]
    df_feats = df.drop(columns=valid_targets)
    return df_feats, df_targets


