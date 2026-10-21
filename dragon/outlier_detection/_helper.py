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
        # Ensure selected columns are numeric by coercing them
        non_numeric = data.select_dtypes(exclude='number').columns
        if not non_numeric.empty:
            if verbose >= 1:
                _LOGGER.warning(f"Coercing non-numeric columns to NaN: {non_numeric.tolist()}")
            data[non_numeric] = data[non_numeric].apply(pd.to_numeric, errors='coerce')
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
    treat_as_continuous: Optional[list[str]] = None, 
    treat_as_categorical: Optional[list[str]] = None,
    verbose: int = 2, 
    shadow_median_imputation: bool = True
) -> pd.DataFrame:
    """Helper function to extract, clean, and frequency-encode mixed data for outlier detection."""
    # copy data
    data = df.copy()
    
    #validate columns
    all_columns = set(data.columns)
    
    treat_as_continuous = treat_as_continuous or []
    treat_as_categorical = treat_as_categorical or []
    
    specified_columns = set(treat_as_continuous + treat_as_categorical)
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
    # Determine which columns are left to be inferred
    remaining_cols = [c for c in data.columns if c not in treat_as_continuous and c not in treat_as_categorical]
    
    inferred_continuous = data[remaining_cols].select_dtypes(include='number').columns.tolist()
    inferred_categorical = data[remaining_cols].select_dtypes(exclude='number').columns.tolist()
    
    final_continuous = treat_as_continuous + inferred_continuous
    final_categorical = treat_as_categorical + inferred_categorical
        
    if not final_continuous and not final_categorical:
        _LOGGER.error("No valid columns available for outlier detection.")
        raise ValueError()
        
    # Isolate only the required columns
    data = data[final_continuous + final_categorical]
    
    # Process continuous columns
    if final_continuous:
        non_numeric = data[final_continuous].select_dtypes(exclude='number').columns
        if not non_numeric.empty:
            if verbose >= 1:
                _LOGGER.warning(f"Coercing non-numeric continuous columns to NaN: {non_numeric.tolist()}")
            data[non_numeric] = data[non_numeric].apply(pd.to_numeric, errors='coerce')
            
        if shadow_median_imputation:
            if verbose >= 3:
                _LOGGER.info("Using median imputation for missing continuous values.")
            data[final_continuous] = data[final_continuous].fillna(data[final_continuous].median())

    # Process categorical columns (Frequency Encoding)
    if final_categorical:
        if verbose >= 2:
            _LOGGER.info(f"Applying frequency encoding to {len(final_categorical)} categorical columns.")
            
        # --- CATEGORICAL NAN HANDLING ---
        if shadow_median_imputation:
            # Treat NaN as a distinct structural category so it gets a frequency
            data[final_categorical] = data[final_categorical].fillna("__MISSING-DATA__")
            
        for col in final_categorical:
            # dropna=True ensures that if shadow_median_imputation is False, 
            # actual NaNs map to NaN and get dropped by the final data.dropna()
            freq_mapping = data[col].value_counts(normalize=True, dropna=True)
            data[col] = data[col].map(freq_mapping)

    # Final safety drop for any remaining NaNs 
    data_clean = data.dropna()

    return data_clean
