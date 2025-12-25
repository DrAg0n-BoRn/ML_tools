from ..schema import FeatureSchema
from ..ML_inference import DragonInferenceHandler

from ..keys._keys import MLTaskKeys, ChainKeys
from .._core import get_logger


_LOGGER = get_logger("Schema Updater")


__all__ = [
    "derive_next_step_schema",
]


def derive_next_step_schema(
    current_schema: FeatureSchema, 
    handler: DragonInferenceHandler, 
    verbose: bool = True
) -> FeatureSchema:
    """
    Creates the FeatureSchema for the NEXT step in the chain by appending the current handler's predictions as new features.
    
    Args:
        current_schema (FeatureSchema): The current FeatureSchema.
        handler (DragonInferenceHandler): The inference handler of the model trained using the current schema.
    
    Returns:
        FeatureSchema: An updated schema including new predicted features.
    """
    # 1. Determine New Column Names
    # Match logic from _chaining_tools.py
    if handler.target_ids is None:
        _LOGGER.error("Handler target_ids is None; cannot derive schema.")
        raise ValueError()
        
    new_cols = [f"{ChainKeys.CHAIN_PREDICTION_PREFIX}{tid}" for tid in handler.target_ids]
    
    # 2. Base Lists (Convert tuples to lists for mutation)
    new_feature_names = list(current_schema.feature_names) + new_cols
    new_cont_names = list(current_schema.continuous_feature_names)
    new_cat_names = list(current_schema.categorical_feature_names)
    
    # Copy existing maps (handle None case)
    new_cat_index_map = dict(current_schema.categorical_index_map) if current_schema.categorical_index_map else {}
    new_cat_mappings = dict(current_schema.categorical_mappings) if current_schema.categorical_mappings else {}

    # 3. Determine Feature Type based on Task
    is_categorical = False
    cardinality = 0
    
    if handler.task in [MLTaskKeys.BINARY_CLASSIFICATION, MLTaskKeys.MULTILABEL_BINARY_CLASSIFICATION]:
        is_categorical = True
        cardinality = 2
        
    elif handler.task == MLTaskKeys.MULTICLASS_CLASSIFICATION:
        is_categorical = True
        # We rely on the class map to know the 'vocabulary' size
        if handler._class_map is None:
            _LOGGER.error("Handler class_map is None, cannot determine cardinality for multiclass classification model.")
            raise ValueError()
        cardinality = len(handler._class_map)
        
    # 4. Append New Metadata
    current_total_feats = len(current_schema.feature_names)
    
    for i, col_name in enumerate(new_cols):
        # Calculate the absolute index of this new column
        # If we had 10 features (0-9), the new one is at index 10 + i
        new_index = current_total_feats + i
        
        if is_categorical:
            new_cat_names.append(col_name)
            
            # A. Update Cardinality for Embeddings
            new_cat_index_map[new_index] = cardinality
            
            # B. Create Identity Mapping (Dummy Encoding)
            # Maps string representation of int back to the int. 
            identity_map = {str(k): k for k in range(cardinality)}
            new_cat_mappings[col_name] = identity_map
        else:
            # Regression / Multitarget Regression
            new_cont_names.append(col_name)
    
    if verbose:
        _LOGGER.info(f"Derived next step schema with {len(new_feature_names)} features:\n    {len(new_cont_names)} continuous\n    {len(new_cat_names)} categorical")

    # 5. Return New Immutable Schema
    return FeatureSchema(
        feature_names=tuple(new_feature_names),
        continuous_feature_names=tuple(new_cont_names),
        categorical_feature_names=tuple(new_cat_names),
        categorical_index_map=new_cat_index_map if new_cat_index_map else None,
        categorical_mappings=new_cat_mappings if new_cat_mappings else None
    )
