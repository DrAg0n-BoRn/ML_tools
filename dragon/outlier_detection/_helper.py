import pandas as pd
from typing import Optional


from .._core import get_logger


_LOGGER = get_logger("Outlier Detection")


def _prepare_numeric_data(df: pd.DataFrame, 
                          columns: Optional[list[str]] = None,
                          ignore_columns: Optional[list[str]] = None, 
                          verbose: int = 2, 
                          shadow_median_imputation: bool = True) -> pd.DataFrame:
    """Helper function to extract and clean numeric data for outlier detection."""
    # copy data
    data = df.copy()
    
    #validate columns
    if columns is not None:
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            _LOGGER.error(f"Specified columns not found in DataFrame: {missing_cols}")
            raise ValueError()
        
    if ignore_columns is not None:
        missing_ignore_cols = [col for col in ignore_columns if col not in data.columns]
        if missing_ignore_cols:
            _LOGGER.error(f"Specified ignore_columns not found in DataFrame: {missing_ignore_cols}")
            raise ValueError()
        
        if verbose >= 2:
            _LOGGER.info(f"Ignoring specified columns: {ignore_columns}")
        data = data.drop(columns=ignore_columns)
    
    
    if columns:
        data = data[columns]
        # Ensure selected columns are numeric
        non_numeric = data.select_dtypes(exclude='number').columns
        if not non_numeric.empty:
            if verbose >= 1:
                _LOGGER.warning(f"Ignoring non-numeric columns provided: {non_numeric.tolist()}")
            data = data.select_dtypes(include='number')
    else:
        data = data.select_dtypes(include='number')
    
    if data.empty:
        raise ValueError("No numeric columns available for outlier detection.")
    
    if shadow_median_imputation:
        if verbose >= 3:
            _LOGGER.info("Using median imputation for missing values.")
        data = data.fillna(data.median())

    # Safety drop in case a column was 100% NaNs or if shadow_median_imputation is False
    data_clean = data.dropna()

    return data_clean


def _prepare_mixed_data(
    df: pd.DataFrame, 
    ignore_columns: Optional[list[str]] = None,
    continuous_columns: Optional[list[str]] = None, 
    categorical_columns: Optional[list[str]] = None,
    verbose: int = 2, 
    shadow_median_imputation: bool = True
) -> pd.DataFrame:
    """Helper function to extract, clean, and frequency-encode mixed data for outlier detection."""
    # copy data
    data = df.copy()
    
    #validate columns
    all_columns = set(data.columns)
    specified_columns = set((continuous_columns or []) + (categorical_columns or []))
    if specified_columns and not specified_columns.issubset(all_columns):
        missing_cols = specified_columns - all_columns
        _LOGGER.error(f"Specified columns not found in DataFrame: {missing_cols}")
        raise ValueError()
    
    if ignore_columns is not None:
        missing_ignore_cols = set(ignore_columns) - all_columns
        if missing_ignore_cols:
            _LOGGER.error(f"Specified ignore_columns not found in DataFrame: {missing_ignore_cols}")
            raise ValueError()
        
        if verbose >= 2:
            _LOGGER.info(f"Ignoring specified columns: {ignore_columns}")
        data = data.drop(columns=ignore_columns)
    
    # --- INFERENCE LOGIC ---
    # If both are None, infer everything based on data types
    if continuous_columns is None and categorical_columns is None:
        continuous_columns = data.select_dtypes(include='number').columns.tolist()
        categorical_columns = data.select_dtypes(exclude='number').columns.tolist()
    else:
        # If the user provided one list, assume the other is empty unless specified
        continuous_columns = continuous_columns if continuous_columns is not None else []
        categorical_columns = categorical_columns if categorical_columns is not None else []
        
    if not continuous_columns and not categorical_columns:
        _LOGGER.error("No valid columns available for outlier detection.")
        raise ValueError()
        
    # Isolate only the required columns
    data = data[continuous_columns + categorical_columns]
    
    # Process continuous columns
    if continuous_columns:
        non_numeric = data[continuous_columns].select_dtypes(exclude='number').columns
        if not non_numeric.empty:
            if verbose >= 1:
                _LOGGER.warning(f"Coercing non-numeric continuous columns to NaN: {non_numeric.tolist()}")
            data[non_numeric] = data[non_numeric].apply(pd.to_numeric, errors='coerce')
            
        if shadow_median_imputation:
            if verbose >= 3:
                _LOGGER.info("Using median imputation for missing continuous values.")
            data[continuous_columns] = data[continuous_columns].fillna(data[continuous_columns].median())

    # Process categorical columns (Frequency Encoding)
    if categorical_columns:
        if verbose >= 2:
            _LOGGER.info(f"Applying frequency encoding to {len(categorical_columns)} categorical columns.")
            
        # --- CATEGORICAL NAN HANDLING ---
        if shadow_median_imputation:
            # Treat NaN as a distinct structural category so it gets a frequency
            data[categorical_columns] = data[categorical_columns].fillna("__MISSING-DATA__")
            
        for col in categorical_columns:
            # dropna=True ensures that if shadow_median_imputation is False, 
            # actual NaNs map to NaN and get dropped by the final data.dropna()
            freq_mapping = data[col].value_counts(normalize=True, dropna=True)
            data[col] = data[col].map(freq_mapping)

    # Final safety drop for any remaining NaNs 
    data_clean = data.dropna()

    return data_clean
