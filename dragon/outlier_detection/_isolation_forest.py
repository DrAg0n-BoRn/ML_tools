import pandas as pd
from typing import Optional
from sklearn.ensemble import IsolationForest

from .._core import get_logger

from ._helper import _prepare_numeric_data, _prepare_mixed_data


_LOGGER = get_logger("Isolation Forest")


__all__ = [
    "isolation_forest_continuous",
    "isolation_forest"
]


def isolation_forest_continuous(
    df_features: pd.DataFrame,
    ignore_columns: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    shadow_median_imputation: bool = True,
    verbose: int = 2
) -> pd.Series:
    """
    ## Continuous Features Only
    
    Detects outliers using the Isolation Forest algorithm. Works well for high-dimensional data and does not assume any distribution.
    
    Does not handle missing values natively, so rows with NaNs in the selected columns will be ignored for fitting if shadow_median_imputation is False.

    Args:
        df_features (pd.DataFrame): The input dataset containing features for outlier detection.
        ignore_columns (list[str] | None): Columns to exclude early from outlier detection.
        columns (list[str] | None): Specific columns to use. If None, uses all numeric columns.
        n_estimators (int): The number of base estimators in the ensemble.
        random_state (int): Seed for reproducibility.
        shadow_median_imputation (bool): Temporarily fill missing values with the median for fitting the model.
        verbose (int): Controls the verbosity.
        
    Returns:
        pd.Series: A boolean mask aligned with the input DataFrame index. True indicates an outlier.
    """
    data_clean = _prepare_numeric_data(df=df_features, 
                                       ignore_columns=ignore_columns,
                                       columns=columns, 
                                       verbose=verbose, 
                                       shadow_median_imputation=shadow_median_imputation)
    
    if data_clean.empty:
        _LOGGER.error("All rows contain NaN values. Cannot fit Isolation Forest.")
        return pd.Series(False, index=df_features.index)
    
    if verbose >= 2:
        _LOGGER.info(f"Fitting Isolation Forest on {data_clean.shape[0]} rows and {data_clean.shape[1]} features.")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto", 
        random_state=random_state,
        n_jobs=-1
    )
    
    # Returns 1 for inliers, -1 for outliers
    preds = model.fit_predict(data_clean)
    
    # Convert to boolean mask (True == Outlier)
    outlier_mask = (preds == -1)
    
    # Realign with original dataframe index (fill dropped NaN rows with False)
    result = pd.Series(False, index=df_features.index)
    result.loc[data_clean.index] = outlier_mask
    
    if verbose >= 2:
        _LOGGER.info(f"Isolation Forest detected {result.sum()} outlier samples.")
    
    return result


def isolation_forest(
    df_features: pd.DataFrame,
    ignore_columns: Optional[list[str]] = None,
    continuous_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    shadow_median_imputation: bool = True,
    verbose: int = 2
) -> pd.Series:
    """
    Detects outliers using Isolation Forest, supporting both continuous and categorical features.
    Categorical features are handled automatically via frequency encoding.

    Args:
        df_features (pd.DataFrame): The input dataset containing features for outlier detection.
        ignore_columns (list[str] | None): Columns to exclude early from outlier detection.
        continuous_columns (list[str] | None): Specific continuous columns to use. If None, uses all numeric columns.
        categorical_columns (list[str] | None): Specific categorical columns to use. If None, uses all non-numeric columns.
        n_estimators (int): The number of base estimators in the ensemble.
        random_state (int): Seed for reproducibility.
        shadow_median_imputation (bool): Temporarily fill missing continuous values with the median.
        verbose (int): Controls the verbosity.
        
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
        _LOGGER.error("All rows contain NaN values after processing. Cannot fit Isolation Forest.")
        return pd.Series(False, index=df_features.index)
    
    if verbose >= 2:
        _LOGGER.info(f"Fitting Categorical Isolation Forest on {data_clean.shape[0]} rows and {data_clean.shape[1]} features.")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto", 
        random_state=random_state,
        n_jobs=-1
    )
    
    preds = model.fit_predict(data_clean)
    outlier_mask = (preds == -1)
    
    result = pd.Series(False, index=df_features.index)
    result.loc[data_clean.index] = outlier_mask
    
    if verbose >= 2:
        _LOGGER.info(f"Categorical Isolation Forest detected {result.sum()} outlier samples.")
    
    return result
