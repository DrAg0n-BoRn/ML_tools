import pandas as pd
from typing import Optional
from sklearn.neighbors import LocalOutlierFactor

from .._core import get_logger

from ._helper import _prepare_numeric_data, _prepare_mixed_data


_LOGGER = get_logger("Local Outlier Factor")


__all__ = [
    "local_outlier_factor_continuous",
    "local_outlier_factor"
]


def local_outlier_factor_continuous(
    df_features: pd.DataFrame,
    ignore_columns: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
    n_neighbors: int = 20,
    leaf_size: int = 30,
    shadow_median_imputation: bool = True,
    verbose: int = 2
) -> pd.Series:
    """
    ## Continuous Features Only
    
    Detects outliers using the Local Outlier Factor (LOF) algorithm. LOF identifies outliers based on the local density of data points, making it effective for datasets with varying densities and non-linear relationships.
    
    Does not handle missing values natively, so rows with NaNs in the selected columns will be ignored for fitting if shadow_median_imputation is False.

    Args:
        df_features (pd.DataFrame): The input dataset containing features for outlier detection.
        ignore_columns (list[str] | None): Columns to exclude early from outlier detection.
        columns (list[str] | None): Specific columns to use. If None, uses all numeric columns.
        n_neighbors (int): Number of neighbors to use by default for k-neighbors queries.
        leaf_size (int): Leaf size passed to BallTree or KDTree.
        shadow_median_imputation (bool): Temporarily fill missing values with the median for fitting the model.
        verbose (int): Controls the verbosity of the model.

    Returns:
        pd.Series: A boolean mask aligned with the input DataFrame index. True indicates an outlier.
    """
    data_clean = _prepare_numeric_data(df=df_features, 
                                       ignore_columns=ignore_columns,
                                       columns=columns, 
                                       verbose=verbose, 
                                       shadow_median_imputation=shadow_median_imputation)
    
    if data_clean.empty:
        _LOGGER.error("All rows contain NaN values. Cannot fit LOF.")
        return pd.Series(False, index=df_features.index)
    
    # LOF needs n_neighbors to be less than the number of samples
    actual_neighbors = min(n_neighbors, data_clean.shape[0] - 1)
    if actual_neighbors < n_neighbors and verbose >= 1:
        _LOGGER.warning(f"n_neighbors ({n_neighbors}) reduced to {actual_neighbors} due to sample size.")

    if verbose >= 2:
        _LOGGER.info(f"Fitting Local Outlier Factor on {data_clean.shape[0]} rows and {data_clean.shape[1]} features.")
    
    model = LocalOutlierFactor(
        n_neighbors=actual_neighbors,
        leaf_size=leaf_size,
        contamination="auto",
        n_jobs=-1
    )
    
    # Returns 1 for inliers, -1 for outliers
    preds = model.fit_predict(data_clean)
    
    outlier_mask = (preds == -1)
    
    result = pd.Series(False, index=df_features.index)
    result.loc[data_clean.index] = outlier_mask
    
    if verbose >= 2:
        _LOGGER.info(f"Local Outlier Factor detected {result.sum()} outlier samples.")
        
    return result


def local_outlier_factor(
    df_features: pd.DataFrame,
    ignore_columns: Optional[list[str]] = None,
    continuous_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    n_neighbors: int = 20,
    leaf_size: int = 30,
    shadow_median_imputation: bool = True,
    verbose: int = 2
) -> pd.Series:
    """
    Detects outliers using the Local Outlier Factor (LOF) algorithm, supporting mixed data types.
    Categorical features are handled automatically via frequency encoding.

    Args:
        df_features (pd.DataFrame): The input dataset containing features for outlier detection.
        ignore_columns (list[str] | None): Columns to exclude early from outlier detection.
        continuous_columns (list[str] | None): Specific continuous columns to use. If None, uses all numeric columns.
        categorical_columns (list[str] | None): Specific categorical columns to use. If None, uses all non-numeric columns.
        n_neighbors (int): Number of neighbors to use by default for k-neighbors queries.
        leaf_size (int): Leaf size passed to BallTree or KDTree.
        shadow_median_imputation (bool): Temporarily fill missing values with the median for fitting the model.
        verbose (int): Controls the verbosity of the model.

    Returns:
        pd.Series: A boolean mask aligned with the input DataFrame index. True indicates an outlier.
        
    Note:
    - If any column is specified in one parameter (e.g., continuous_columns), but the other parameter is None, the function will not automatically infer the remaining columns. 
    It will only use the specified columns for outlier detection.
    """
    data_clean = _prepare_mixed_data(
        df=df_features, 
        continuous_columns=continuous_columns, 
        categorical_columns=categorical_columns,
        ignore_columns=ignore_columns,
        verbose=verbose, 
        shadow_median_imputation=shadow_median_imputation
    )
    
    if data_clean.empty:
        _LOGGER.error("All rows contain NaN values after processing. Cannot fit LOF.")
        return pd.Series(False, index=df_features.index)
    
    actual_neighbors = min(n_neighbors, data_clean.shape[0] - 1)
    if actual_neighbors < n_neighbors and verbose >= 1:
        _LOGGER.warning(f"n_neighbors ({n_neighbors}) reduced to {actual_neighbors} due to sample size.")

    if verbose >= 2:
        _LOGGER.info(f"Fitting Categorical Local Outlier Factor on {data_clean.shape[0]} rows and {data_clean.shape[1]} features.")
    
    model = LocalOutlierFactor(
        n_neighbors=actual_neighbors,
        leaf_size=leaf_size,
        contamination="auto",
        n_jobs=-1
    )
    
    preds = model.fit_predict(data_clean)
    outlier_mask = (preds == -1)
    
    result = pd.Series(False, index=df_features.index)
    result.loc[data_clean.index] = outlier_mask
    
    if verbose >= 2:
        _LOGGER.info(f"Categorical Local Outlier Factor detected {result.sum()} outlier samples.")
        
    return result
