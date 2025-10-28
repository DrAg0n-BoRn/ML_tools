from typing import NamedTuple, Tuple, Optional, Dict

class FeatureSchema(NamedTuple):
    """Holds the final, definitive schema for the model pipeline."""
    
    # The final, ordered list of all feature names
    feature_names: Tuple[str, ...]
    
    # List of all continuous feature names
    continuous_feature_names: Tuple[str, ...]
    
    # List of all categorical feature names
    categorical_feature_names: Tuple[str, ...]
    
    # Map of {column_index: cardinality} for categorical features
    categorical_index_map: Optional[Dict[int, int]]
    
    # The original string-to-int mappings (e.g., {'color': {'red': 0, 'blue': 1}})
    categorical_mappings: Optional[Dict[str, Dict[str, int]]]
