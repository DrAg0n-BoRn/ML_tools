from typing import Any, Optional


__all__ = ["prepare_schema_from_json"]


def prepare_schema_from_json(data: dict[str, Any]) -> dict[str, Any]:
    """
    Processes a raw dictionary (loaded from JSON) into the clean arguments 
    required to instantiate a FeatureSchema.

    Performs the following restorations:
    1. Converts list fields back to tuples.
    2. Converts string keys in 'categorical_index_map' back to integers.

    Args:
        data (dict): The raw dictionary from a JSON file (e.g. from 'schema_dict').

    Returns:
        dict: A dictionary of kwargs ready to be unpacked into FeatureSchema(**kwargs).
    """
    # 1. Restore Tuples (JSON loads them as lists)
    feature_names = tuple(data.get("feature_names", []))
    cont_names = tuple(data.get("continuous_feature_names", []))
    cat_names = tuple(data.get("categorical_feature_names", []))

    # 2. Restore Integer Keys for categorical_index_map
    raw_map = data.get("categorical_index_map")
    cat_index_map: Optional[dict[int, int]] = None
    if raw_map is not None:
        # JSON keys are always strings; convert back to int
        cat_index_map = {int(k): v for k, v in raw_map.items()}

    # 3. Mappings (keys are strings, no conversion needed)
    cat_mappings = data.get("categorical_mappings", None)

    return {
        "feature_names": feature_names,
        "continuous_feature_names": cont_names,
        "categorical_feature_names": cat_names,
        "categorical_index_map": cat_index_map,
        "categorical_mappings": cat_mappings
    }