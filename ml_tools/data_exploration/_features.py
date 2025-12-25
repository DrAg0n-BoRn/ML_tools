import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
from typing import Any, Optional, Union
import re

from .._core import get_logger


_LOGGER = get_logger("Data Exploration: Feature Ops")


__all__ = [
    "split_features_targets",
    "split_continuous_binary",
    "split_continuous_categorical_targets",
    "encode_categorical_features",
    "reconstruct_one_hot",
    "reconstruct_binary",
    "reconstruct_multibinary",
]


def split_features_targets(df: pd.DataFrame, targets: list[str]):
    """
    Splits a DataFrame's columns into features and targets.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the dataset.
        targets (list[str]): List of column names to be treated as target variables.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Features dataframe.
            - pd.DataFrame: Targets dataframe.

    Prints:
        - Shape of the original dataframe.
        - Shape of the features dataframe.
        - Shape of the targets dataframe.
    """
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        _LOGGER.error(f"Target columns not found in DataFrame: {missing_targets}")
        raise ValueError()

    # 2. Perform the split
    df_targets = df[targets]
    df_features = df.drop(columns=targets)
    
    # 3. Print summary
    print(f"Original shape: {df.shape}\nFeatures shape: {df_features.shape}\nTargets shape: {df_targets.shape}")
    
    return df_features, df_targets


def split_continuous_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into two DataFrames: one with continuous columns, one with binary columns.
    Normalize binary values like 0.0/1.0 to 0/1 if detected.

    Parameters:
        df (pd.DataFrame): Input DataFrame with only numeric columns.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): (continuous_columns_df, binary_columns_df)

    Raises:
        TypeError: If any column is not numeric.
    """
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        _LOGGER.error("All columns must be numeric (int or float).")
        raise TypeError()

    binary_cols = []
    continuous_cols = []

    for col in df.columns:
        series = df[col]
        unique_values = set(series[~series.isna()].unique())

        if unique_values.issubset({0, 1}):
            binary_cols.append(col)
        elif unique_values.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: 0 if x == 0.0 else (1 if x == 1.0 else x))
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    binary_cols.sort()

    df_cont = df[continuous_cols]
    df_bin = df[binary_cols]

    print(f"Continuous columns shape: {df_cont.shape}")
    print(f"Binary columns shape: {df_bin.shape}")

    return df_cont, df_bin # type: ignore


def split_continuous_categorical_targets(
    df: pd.DataFrame, 
    categorical_cols: list[str], 
    target_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into three subsets: Continuous, Categorical, and Targets.
    
    Logic:
    1. Categorical and Target columns are explicitly provided.
    2. Continuous columns are inferred (All columns - Categorical - Targets).
    3. Continuous columns should be numeric and have more than 2 unique values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list[str]): List of categorical column names.
        target_cols (list[str]): List of target column names.

    Returns:
        Tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): 
            (df_continuous, df_categorical, df_targets)
    """
    # Set operations to find inferred continuous columns
    all_cols = set(df.columns)
    cat_set = set(categorical_cols)
    tgt_set = set(target_cols)

    # Basic input validation
    missing_cat = cat_set - all_cols
    missing_tgt = tgt_set - all_cols
    if missing_cat:
        _LOGGER.error(f"Categorical columns not found in DataFrame: {missing_cat}")
        raise ValueError()
    if missing_tgt:
        _LOGGER.error(f"Target columns not found in DataFrame: {missing_tgt}")
        raise ValueError()

    # Identify continuous columns
    inferred_continuous = list(all_cols - cat_set - tgt_set)
    inferred_continuous.sort() # Ensure deterministic order

    # Validate inferred continuous columns
    for col in inferred_continuous:
        series = df[col]
        
        # Check 1: Must be numeric
        if not is_numeric_dtype(series):
            _LOGGER.warning(f"Column '{col}' was inferred as continuous but is not numeric (dtype: {series.dtype}).")
        
        # Check 2: Must have > 2 unique values (cardinality check)
        # We drop NA to count actual unique values
        unique_count = series.dropna().nunique()
        if unique_count <= 2:
            _LOGGER.warning(f"Column '{col}' was inferred as continuous but has only {unique_count} unique value(s). It might be binary or constant.")

    # Split DataFrames
    df_continuous = df[inferred_continuous]
    df_categorical = df[categorical_cols] # Preserve user order
    df_targets = df[target_cols] # Preserve user order

    _LOGGER.info(
        f"Split complete.\n"
        f"  - Continuous: {df_continuous.shape}\n"
        f"  - Categorical: {df_categorical.shape}\n"
        f"  - Targets: {df_targets.shape}"
    )

    return df_continuous, df_categorical, df_targets


def encode_categorical_features(
    df_categorical: pd.DataFrame,
    encode_nulls: bool,
    null_label: str = "Other",
    verbose: int = 1
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Encodes all columns in the provided DataFrame as categorical features using Label Encoding.
    
    This function generates a unique integer mapping for the values in each column.

    Args:
        df_categorical (pd.DataFrame): DataFrame containing ONLY the categorical columns to encode.
        encode_nulls (bool): 
            - If True, Nulls (NaN/None) are encoded as a distinct category (0). Real categories will start from 1.
            - If False, Nulls are left as NaN. Real categories start from 0.
        null_label (str): Label used for the null category in the returned mapping if `encode_nulls` is True.
        verbose (int): 
            - 0: Error level only.
            - 1: Info and Warning levels.
            - 2: Debug/Print everything (includes per-column summary).

    Returns:
        Tuple (Dataframe, Dict):
            - pd.DataFrame: A new DataFrame with all columns encoded as integers.
            - Dict[str, Dict[str, int]]: A dictionary where keys are column names and values are the value-to-integer mappings.
    """
    df_encoded = df_categorical.copy()
    mappings: dict[str, dict[str, int]] = {}
    
    cols_to_process = df_encoded.columns.tolist()

    if verbose >= 1:
        _LOGGER.info(f"Encoding {len(cols_to_process)} categorical column(s).")

    for col_name in cols_to_process:
        has_nulls = df_encoded[col_name].isnull().any()
        
        # Get unique values (excluding nulls) to determine categories
        # Sorting ensures deterministic integer assignment
        raw_unique_values = df_encoded[col_name].dropna().unique()
        categories = sorted([str(cat) for cat in raw_unique_values])
        
        # --- Check for constant columns ---
        # Note: If encode_nulls=True and we have nulls, it's effectively binary (Null vs Value), so we keep it.
        is_effectively_binary = encode_nulls and has_nulls
        if len(categories) <= 1 and not is_effectively_binary:
            if verbose >= 1:
                _LOGGER.warning(f"Column '{col_name}' has only {len(categories)} unique non-null value(s).")

        # --- Encoding Logic ---
        if encode_nulls and has_nulls:
            # Mode A: Encode Nulls. 
            # Null -> 0
            # Categories -> 1, 2, 3...
            
            mapping = {category: i + 1 for i, category in enumerate(categories)}
            
            # 1. Map existing non-null values (cast to str first to match mapping keys)
            mapped_series = df_encoded[col_name].astype(str).map(mapping)
            
            # 2. Fill NaNs with 0
            df_encoded[col_name] = mapped_series.fillna(0).astype(int)
            
            # --- Handle Mapping Dict Collision ---
            current_null_label = null_label
            if current_null_label in mapping:
                current_null_label = "__NULL__"
                if verbose >= 1:
                    _LOGGER.warning(f"Collision in '{col_name}': '{null_label}' is a real category. Using '{current_null_label}' for nulls.")

            # Add null key to user mapping
            user_mapping = {**mapping, current_null_label: 0}
            mappings[col_name] = user_mapping
            
        else:
            # Mode B: Ignore Nulls (preserve them as NaN) or No Nulls exist.
            # Categories -> 0, 1, 2...
            
            mapping = {category: i for i, category in enumerate(categories)}
            
            # Map values. 
            # Note: map() on a Series with NaNs will result in NaNs for those positions.
            # use 'Int64' (capital I) to handle Integers with <NA> values cleanly.
            df_encoded[col_name] = df_encoded[col_name].astype(str).map(mapping).astype("Int64")
            
            mappings[col_name] = mapping

        if verbose >= 2:
            cardinality = len(mappings[col_name])
            print(f"  - Encoded '{col_name}': {cardinality} categories.")

    return df_encoded, mappings


def reconstruct_one_hot(
    df: pd.DataFrame,
    features_to_reconstruct: list[Union[str, tuple[str, Optional[str]]]],
    separator: str = '_',
    baseline_category_name: Optional[str] = "Other",
    drop_original: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reconstructs original categorical columns from a one-hot encoded DataFrame.

    This function identifies groups of one-hot encoded columns based on a common
    prefix (base feature name) and a separator. It then collapses each group
    into a single column containing the categorical value.

    Args:
        df (pd.DataFrame): 
            The input DataFrame with one-hot encoded columns.
        features_to_reconstruct (List[str | Tuple[str, str | None]]):
            A list defining the features to reconstruct. This list can contain:
            
            - A string: (e.g., "Color")
              This reconstructs the feature 'Color' and assumes all-zero rows represent the baseline category ("Other" by default).
            - A tuple: (e.g., ("Pet", "Dog"))
              This reconstructs 'Pet' and maps all-zero rows to the baseline category "Dog".
            - A tuple with None: (e.g., ("Size", None))
              This reconstructs 'Size' and maps all-zero rows to the NaN value.
            Example:
            [
                "Mood",                      # All-zeros -> "Other"
                ("Color", "Red"),            # All-zeros -> "Red"
                ("Size", None)               # All-zeros -> NaN
            ]
        separator (str): 
            The character separating the base name from the categorical value in 
            the column names (e.g., '_' in 'B_a').
        baseline_category_name (str | None):
            The baseline category name to use by default if it is not explicitly provided.
        drop_original (bool): 
            If True, the original one-hot encoded columns will be dropped from 
            the returned DataFrame.

    Returns:
        pd.DataFrame: 
            A new DataFrame with the specified one-hot encoded features 
            reconstructed into single categorical columns.
    
    <br>
    
    ## Note: 
    
    This function is designed to be robust, but users should be aware of two key edge cases:

    1.  **Ambiguous Base Feature Prefixes**: If `base_feature_names` list contains names where one is a prefix of another (e.g., `['feat', 'feat_ext']`), the order is critical. The function will match columns greedily. To avoid incorrect grouping, always list the **most specific base names first** (e.g., `['feat_ext', 'feat']`).

    2.  **Malformed One-Hot Data**: If a row contains multiple `1`s within the same feature group (e.g., both `B_a` and `B_c` are `1`), the function will not raise an error. It uses `.idxmax()`, which returns the first column that contains the maximum value. This means it will silently select the first category it encounters and ignore the others, potentially masking an upstream data issue.
    """
    if not isinstance(df, pd.DataFrame):
        _LOGGER.error("Input must be a pandas DataFrame.")
        raise TypeError()
    
    if not (baseline_category_name is None or isinstance(baseline_category_name, str)):
        _LOGGER.error("The baseline_category must be None or a string.")
        raise TypeError()

    new_df = df.copy()
    all_ohe_cols_to_drop = []
    reconstructed_count = 0
    
    # --- 1. Parse and validate the reconstruction config ---
    # This normalizes the input into a clean {base_name: baseline_val} dict
    reconstruction_config: dict[str, Optional[str]] = {}
    try:
        for item in features_to_reconstruct:
            if isinstance(item, str):
                # Case 1: "Color"
                base_name = item
                baseline_val = baseline_category_name
            elif isinstance(item, tuple) and len(item) == 2:
                # Case 2: ("Pet", "dog") or ("Size", None)
                base_name, baseline_val = item
                if not (isinstance(base_name, str) and (isinstance(baseline_val, str) or baseline_val is None)):
                    _LOGGER.error(f"Invalid tuple format for '{item}'. Must be (str, str|None).")
                    raise ValueError()
            else:
                _LOGGER.error(f"Invalid item '{item}'. Must be str or (str, str|None) tuple.")
                raise ValueError()
            
            if base_name in reconstruction_config and verbose:
                _LOGGER.warning(f"Duplicate entry for '{base_name}' found. Using the last provided configuration.")
            
            reconstruction_config[base_name] = baseline_val
    
    except Exception as e:
        _LOGGER.error(f"Failed to parse 'features_to_reconstruct' argument: {e}")
        raise ValueError("Invalid configuration for 'features_to_reconstruct'.") from e
    
    _LOGGER.info(f"Attempting to reconstruct {len(reconstruction_config)} one-hot encoded feature(s).")
    
    # Main logic
    for base_name, baseline_category in reconstruction_config.items():
        # Regex to find all columns belonging to this base feature.
        pattern = f"^{re.escape(base_name)}{re.escape(separator)}"
        
        # Find matching columns
        ohe_cols = [col for col in df.columns if re.match(pattern, col)]

        if not ohe_cols:
            _LOGGER.warning(f"No one-hot encoded columns found for base feature '{base_name}'. Skipping.")
            continue

        # For each row, find the column name with the maximum value (which is 1)
        reconstructed_series = new_df[ohe_cols].idxmax(axis=1) # type: ignore

        # Extract the categorical value (the suffix) from the column name
        # Use n=1 in split to handle cases where the category itself might contain the separator
        new_column_values = reconstructed_series.str.split(separator, n=1).str[1] # type: ignore
        
        # Handle rows where all OHE columns were 0 (e.g., original value was NaN or a dropped baseline).
        all_zero_mask = new_df[ohe_cols].sum(axis=1) == 0 # type: ignore
        
        if baseline_category is not None:
            # A baseline category was provided
            new_column_values.loc[all_zero_mask] = baseline_category
        else:
            # No baseline provided: assign NaN
            new_column_values.loc[all_zero_mask] = np.nan # type: ignore
            
        if verbose:
            print(f"  - Mapped 'all-zero' rows for '{base_name}' to baseline: '{baseline_category}'.")

        # Assign the new reconstructed column to the DataFrame
        new_df[base_name] = new_column_values
        
        all_ohe_cols_to_drop.extend(ohe_cols)
        reconstructed_count += 1
        if verbose:
            print(f"  - Reconstructed '{base_name}' from {len(ohe_cols)} columns.")

    # Cleanup
    if drop_original and all_ohe_cols_to_drop:
        # Drop the original OHE columns, ensuring no duplicates in the drop list
        unique_cols_to_drop = list(set(all_ohe_cols_to_drop))
        new_df.drop(columns=unique_cols_to_drop, inplace=True)
        _LOGGER.info(f"Dropped {len(unique_cols_to_drop)} original one-hot encoded columns.")

    _LOGGER.info(f"Successfully reconstructed {reconstructed_count} feature(s).")

    return new_df


def reconstruct_binary(
    df: pd.DataFrame,
    reconstruction_map: dict[str, tuple[str, Any, Any]],
    drop_original: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reconstructs new categorical columns from existing binary (0/1) columns.

    Used to reverse a binary encoding by mapping 0 and 1 back to
    descriptive categorical labels.

    Args:
        df (pd.DataFrame):
            The input DataFrame.
        reconstruction_map (Dict[str, Tuple[str, Any, Any]]):
            A dictionary defining the reconstructions.
            Format:
            { "new_col_name": ("source_col_name", "label_for_0", "label_for_1") }
            Example:
            {
                "Sex": ("Sex_male", "Female", "Male"),
                "Smoker": ("Is_Smoker", "No", "Yes")
            }
        drop_original (bool):
            If True, the original binary source columns (e.g., "Sex_male")
            will be dropped from the returned DataFrame.
        verbose (bool):
            If True, prints the details of each reconstruction.

    Returns:
        pd.DataFrame:
            A new DataFrame with the reconstructed categorical columns.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `reconstruction_map` is not a dictionary or a
                    configuration is invalid (e.g., column name collision).

    Notes:
        - The function operates on a copy of the DataFrame.
        - Rows with `NaN` in the source column will have `NaN` in the
          new column.
        - Values in the source column other than 0 or 1 (e.g., 2) will
          result in `NaN` in the new column.
    """
    if not isinstance(df, pd.DataFrame):
        _LOGGER.error("Input must be a pandas DataFrame.")
        raise TypeError()

    if not isinstance(reconstruction_map, dict):
        _LOGGER.error("`reconstruction_map` must be a dictionary with the required format.")
        raise ValueError()

    new_df = df.copy()
    source_cols_to_drop: list[str] = []
    reconstructed_count = 0

    _LOGGER.info(f"Attempting to reconstruct {len(reconstruction_map)} binary feature(s).")

    for new_col_name, config in reconstruction_map.items():
        
        # --- 1. Validation ---
        if not (isinstance(config, tuple) and len(config) == 3):
            _LOGGER.error(f"Config for '{new_col_name}' is invalid. Must be a 3-item tuple. Skipping.")
            raise ValueError()

        source_col, label_for_0, label_for_1 = config

        if source_col not in new_df.columns:
            _LOGGER.error(f"Source column '{source_col}' for new column '{new_col_name}' not found. Skipping.")
            raise ValueError()
        
        if new_col_name in new_df.columns and new_col_name != source_col and verbose:
            _LOGGER.warning(f"New column '{new_col_name}' already exists and will be overwritten.")

        # --- 2. Reconstruction ---
        mapping_dict = {0: label_for_0, 1: label_for_1}
        new_df[new_col_name] = new_df[source_col].map(mapping_dict)

        # --- 3. Logging/Tracking ---
        # Only mark source for dropping if it's NOT the same as the new column
        if source_col != new_col_name:
            source_cols_to_drop.append(source_col)

        reconstructed_count += 1
        if verbose:
            print(f"  - Reconstructed '{new_col_name}' from '{source_col}' (0='{label_for_0}', 1='{label_for_1}').")

    # --- 4. Cleanup ---
    if drop_original and source_cols_to_drop:
        unique_cols_to_drop = list(set(source_cols_to_drop))
        new_df.drop(columns=unique_cols_to_drop, inplace=True)
        _LOGGER.info(f"Dropped {len(unique_cols_to_drop)} original binary source column(s).")

    _LOGGER.info(f"Successfully reconstructed {reconstructed_count} feature(s).")

    return new_df


def reconstruct_multibinary(
    df: pd.DataFrame,
    pattern: str,
    pos_label: str = "Yes",
    neg_label: str = "No",
    case_sensitive: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """
    Identifies binary columns matching a regex pattern and converts their numeric 
    values (0/1) into categorical string labels (e.g., "No"/"Yes").

    This allows mass-labeling of binary features so they are treated as proper 
    categorical variables with meaningful keys during subsequent encoding steps.

    Args:
        df (pd.DataFrame): The input DataFrame.
        pattern (str): Regex pattern to identify the group of binary columns.
        pos_label (str): The label to assign to 1 or True (default "Yes").
        neg_label (str): The label to assign to 0 or False (default "No").
        case_sensitive (bool): If True, regex matching is case-sensitive.
        verbose (bool): If True, prints a summary of the operation.

    Returns:
        Tuple(pd.DataFrame, List[str]): 
            - A new DataFrame with the matched columns converted to Strings.
            - A list of the column names that were modified.
    """
    if not isinstance(df, pd.DataFrame):
        _LOGGER.error("Input must be a pandas DataFrame.")
        raise TypeError()

    new_df = df.copy()

    # 1. Find columns matching the regex
    mask = new_df.columns.str.contains(pattern, case=case_sensitive, regex=True)
    target_columns = new_df.columns[mask].to_list()

    if not target_columns:
        _LOGGER.warning(f"No columns found matching pattern '{pattern}'. Returning original DataFrame.")
        return new_df, list()

    # 2. Define robust mapping (handles ints, floats, and booleans)
    # Note: Any value not in this map will become NaN
    mapping_dict = {
        0: neg_label, 
        0.0: neg_label, 
        False: neg_label,
        1: pos_label, 
        1.0: pos_label, 
        True: pos_label
    }

    converted_count = 0
    
    # 3. Apply mapping
    for col in target_columns:
        # Check if column is numeric or boolean before attempting map to avoid destroying existing strings
        if is_numeric_dtype(new_df[col]) or is_object_dtype(new_df[col]):
            # We cast to object implicitly by mapping to strings
            new_df[col] = new_df[col].map(mapping_dict)
            converted_count += 1

    if verbose:
        _LOGGER.info(f"Reconstructed {converted_count} binary columns matching '{pattern}'.")

    return new_df, target_columns
