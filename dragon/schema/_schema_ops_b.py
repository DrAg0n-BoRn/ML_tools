import pandas as pd
from typing import Literal

from .._core import get_logger

from ._feature_schema import FeatureSchema


_LOGGER = get_logger("Schema Ops")


__all__ = [
    "validate_schema_match",
    "extract_continuous_features",
    "extract_categorical_features",
    "enforce_schema_dtypes",
]


def validate_schema_match(df: pd.DataFrame, 
                          schema: FeatureSchema, 
                          allow_extra_cols: bool = True) -> bool:
    """
    Validates whether the input DataFrame contains the features defined in the schema
    and verifies that the column order matches the schema.
    
    Args:
        df (pd.DataFrame): The input DataFrame to validate.
        schema (FeatureSchema): The schema to validate against.
        allow_extra_cols (bool): If True, validation will not fail if the DataFrame contains columns 
                                  not present in the schema (target columns, etc.).
                                  
    Returns:
        bool: True if validation passes, False otherwise.
    """
    missing_features = [col for col in schema.feature_names if col not in df.columns]
    
    if missing_features:
        _LOGGER.error(f"Validation failed. Missing required features: {missing_features}")
        return False
        
    if allow_extra_cols:
        extra_features = [col for col in df.columns if col not in schema.feature_names]
        if extra_features:
            _LOGGER.error(f"Validation failed. Extra columns found: {extra_features}")
            return False
        
    # Validate column order (the relative order of schema features must match)
    df_schema_cols = [col for col in df.columns if col in schema.feature_names]
    if df_schema_cols != list(schema.feature_names):
        _LOGGER.error("Validation failed. Column order does not match the schema.")
        return False
            
    return True


def extract_continuous_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the continuous features defined in the schema.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        schema (FeatureSchema): The schema defining feature types.
        
    Returns:
        pd.DataFrame: A new DataFrame containing only the continuous features.
    """
    available_cols = [col for col in schema.continuous_feature_names if col in df.columns]
    return df[available_cols].copy()


def extract_categorical_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the categorical features defined in the schema.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        schema (FeatureSchema): The schema defining feature types.
        
    Returns:
        pd.DataFrame: A new DataFrame containing only the categorical features.
    """
    available_cols = [col for col in schema.categorical_feature_names if col in df.columns]
    return df[available_cols].copy()


def enforce_schema_dtypes(
    df: pd.DataFrame, 
    schema: FeatureSchema, 
    categorical_as: Literal["int", "str"] = "int"
) -> pd.DataFrame:
    """
    Explicitly casts continuous features to numeric and categorical features to 
    either int or str, preventing silent downstream type errors.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        schema (FeatureSchema): The schema defining feature types.
        categorical_as (Literal["int", "str"]): Target dtype for categorical features.
        
    Returns:
        pd.DataFrame: A new DataFrame with enforced data types.
    """
    df_out = df.copy()
    
    for col in schema.continuous_feature_names:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
            
    for col in schema.categorical_feature_names:
        if col in df_out.columns:
            if categorical_as == "int":
                # Using Int64 allows for safe integer casting even if NaNs are present
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype("Int64")
            elif categorical_as == "str":
                df_out[col] = df_out[col].astype(str)
            else:
                _LOGGER.error(f"Invalid categorical_as value: {categorical_as}. Must be 'int' or 'str'.")
                raise ValueError()
                
    return df_out
