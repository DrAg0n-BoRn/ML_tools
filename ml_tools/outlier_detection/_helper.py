import pandas as pd
from typing import Optional


from .._core import get_logger


_LOGGER = get_logger("Outlier Detection")


def _prepare_numeric_data(df: pd.DataFrame, columns: Optional[list[str]] = None, verbose: int = 2, shadow_median_imputation: bool = True) -> pd.DataFrame:
    """Helper function to extract and clean numeric data for outlier detection."""
    if columns:
        data = df[columns].copy()
        # Ensure selected columns are numeric
        non_numeric = data.select_dtypes(exclude='number').columns
        if not non_numeric.empty:
            if verbose >= 1:
                _LOGGER.warning(f"Ignoring non-numeric columns provided: {non_numeric.tolist()}")
            data = data.select_dtypes(include='number')
    else:
        data = df.select_dtypes(include='number')
    
    if data.empty:
        raise ValueError("No numeric columns available for outlier detection.")
    
    if shadow_median_imputation:
        if verbose >= 3:
            _LOGGER.info("Using median imputation for missing values.")
        data = data.fillna(data.median())

    # Safety drop in case a column was 100% NaNs or if shadow_median_imputation is False
    data_clean = data.dropna()

    return data_clean
