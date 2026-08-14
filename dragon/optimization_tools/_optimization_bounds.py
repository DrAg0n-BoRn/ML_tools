from typing import Union, Any, Optional
from pathlib import Path

from ..schema import FeatureSchema
from ..IO_tools import save_json, load_json

from ..keys._keys import OptimizationToolsKeys
from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("Optimization Bounds")


__all__ = [
    "make_continuous_bounds_template",
    "load_continuous_bounds_template",
    "create_optimization_bounds",
    "parse_lower_upper_bounds",
]


def make_continuous_bounds_template(
    directory: Union[str, Path],
    feature_schema: FeatureSchema,
    default_bounds: tuple[float, float] = (0, 1)
) -> None:
    """
    Creates a JSON template for manual entry of continuous feature optimization bounds.

    The resulting file maps each continuous feature name to a [min, max] list 
    populated with `default_bounds`. Edit the values in this file before using.

    Args:
        directory (str | Path): The directory where the template will be saved.
        feature_schema (FeatureSchema): The loaded schema containing feature definitions.
        default_bounds (Tuple[float, float]): Default (min, max) values to populate the template.
    """
    # validate directory path
    dir_path = make_fullpath(directory, make=True, enforce="directory")
    
    # 1. Check if continuous features exist
    if not feature_schema.continuous_feature_names:
        _LOGGER.warning("No continuous features found in FeatureSchema. Skipping bounds template generation.")
        return

    # 2. Construct the dictionary: {feature_name: [min, max]}
    bounds_map = {
        name: list(default_bounds)
        for name in feature_schema.continuous_feature_names
    }
    
    # use a fixed key for the filename
    filename = OptimizationToolsKeys.OPTIMIZATION_BOUNDS_FILENAME + ".json"

    # 3. Save to JSON using the IO tool
    save_json(
        data=bounds_map,
        directory=dir_path,
        filename=filename,
        verbose=False
    )
    
    _LOGGER.info(f"💾 Continuous bounds template saved to: '{dir_path.name}/{filename}'")
    

def load_continuous_bounds_template(directory: Union[str, Path]) -> dict[str, list[float]]:
    """
    Loads the continuous feature bounds template from JSON. Expected filename: `optimization_bounds.json`.

    Args:
        directory (str | Path): The directory where the template is located.

    Returns:
        Dictionary (Dict[str, List[float]]): A dictionary mapping feature names to [min, max] bounds.
    """
    dir_path = make_fullpath(directory, enforce="directory")
    full_path = dir_path / (OptimizationToolsKeys.OPTIMIZATION_BOUNDS_FILENAME + ".json")
    
    bounds_map = load_json(
        file_path=full_path,
        expected_type='dict',
        verbose=False
    )
    
    # validate loaded data
    if not all(
            isinstance(v, list) and     # Check type
            len(v) == 2 and     # Check length
            all(isinstance(i, (int, float)) for i in v) # Check contents are numbers
            for v in bounds_map.values()
        ):
        _LOGGER.error(f"Invalid format in bounds template at '{full_path}'. Each value must be a list of [min, max].")
        raise ValueError()
    
    _LOGGER.info(f"Continuous bounds template loaded from: '{dir_path.name}'")
    
    return bounds_map


def create_optimization_bounds(
    schema: FeatureSchema,
    continuous_bounds_map: Union[dict[str, tuple[float, float]], dict[str, list[float]]],
    start_at_zero: bool = True
) -> tuple[list[float], list[float]]:
    """
    Generates the lower and upper bounds lists for the optimizer from a FeatureSchema.

    This helper function automates the creation of unbiased bounds for
    categorical features and combines them with user-defined bounds for
    continuous features, using the schema as the single source of truth
    for feature order and type.

    Args:
        schema (FeatureSchema):
            The definitive schema object created by 
            `data_exploration.finalize_feature_schema()`.
        continuous_bounds_map (Dict[str, Tuple[float, float]], Dict[str, List[float]]):
            A dictionary mapping the *name* of each **continuous** feature
            to its (min_bound, max_bound).
        start_at_zero (bool):
            - If True, assumes categorical encoding is [0, 1, ..., k-1].
              Bounds will be set as [-0.5, k - 0.5].
            - If False, assumes encoding is [1, 2, ..., k].
              Bounds will be set as [0.5, k + 0.5].

    Returns:
        Tuple[List[float], List[float]]:
            A tuple containing two lists: (lower_bounds, upper_bounds).

    Raises:
        ValueError: If a feature is missing from `continuous_bounds_map`
                    or if a feature name in the map is not a
                    continuous feature according to the schema.
    """
    # validate length in the continuous_bounds_map values
    for name, bounds in continuous_bounds_map.items():
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            _LOGGER.error(f"Bounds for feature '{name}' must be a list or tuple of length 2 (min, max). Found: {bounds}")
            raise ValueError()
    
    # 1. Get feature names and map from schema
    feature_names = schema.feature_names
    categorical_index_map = schema.categorical_index_map
    total_features = len(feature_names)

    if total_features <= 0:
        _LOGGER.error("Schema contains no features.")
        raise ValueError()
        
    _LOGGER.info(f"Generating bounds for {total_features} total features...")

    # 2. Initialize bound lists
    lower_bounds: list[Optional[float]] = [None] * total_features
    upper_bounds: list[Optional[float]] = [None] * total_features

    # 3. Populate categorical bounds (Index-based)
    if categorical_index_map:
        for index, cardinality in categorical_index_map.items():
            if not (0 <= index < total_features):
                _LOGGER.error(f"Categorical index {index} is out of range for the {total_features} features.")
                raise ValueError()
                
            if start_at_zero:
                # Rule for [0, k-1]: bounds are [-0.5, k - 0.5]
                low = -0.5
                high = float(cardinality) - 0.5
            else:
                # Rule for [1, k]: bounds are [0.5, k + 0.5]
                low = 0.5
                high = float(cardinality) + 0.5
                
            lower_bounds[index] = low
            upper_bounds[index] = high
        
        _LOGGER.info(f"Automatically set bounds for {len(categorical_index_map)} categorical features.")
    else:
        _LOGGER.info("No categorical features found in schema.")

    # 4. Populate continuous bounds (Name-based)
    # Use schema.continuous_feature_names for robust checking
    continuous_names_set = set(schema.continuous_feature_names)
    
    if continuous_names_set != set(continuous_bounds_map.keys()):
        missing_in_map = continuous_names_set - set(continuous_bounds_map.keys())
        if missing_in_map:
            _LOGGER.error(f"The following continuous features are missing from 'continuous_bounds_map': {list(missing_in_map)}")
        
        extra_in_map = set(continuous_bounds_map.keys()) - continuous_names_set
        if extra_in_map:
            _LOGGER.error(f"The following features in 'continuous_bounds_map' are not defined as continuous in the schema: {list(extra_in_map)}")
            
        raise ValueError("Mismatch between 'continuous_bounds_map' and schema's continuous features.")

    count_continuous = 0
    for name, (low, high) in continuous_bounds_map.items():
        # Map name to its index in the *feature-only* list
        # This is guaranteed to be correct by the schema
        index = feature_names.index(name)

        if lower_bounds[index] is not None:
            # This should be impossible if schema is correct, but good to check
            _LOGGER.error(f"Schema conflict: Feature '{name}' (at index {index}) is defined as both continuous and categorical.")
            raise ValueError()

        lower_bounds[index] = float(low)
        upper_bounds[index] = float(high)
        count_continuous += 1
        
    _LOGGER.info(f"Manually set bounds for {count_continuous} continuous features.")

    # 5. Final Validation (all Nones should be filled)
    if None in lower_bounds:
        missing_indices = [i for i, b in enumerate(lower_bounds) if b is None]
        missing_names = [feature_names[i] for i in missing_indices]
        _LOGGER.error(f"Failed to create all bounds. This indicates an internal logic error. Missing: {missing_names}")
        raise RuntimeError("Internal error: Not all bounds were populated.")
    
    # Cast to float lists, as 'None' sentinels are gone
    return (
        [float(b) for b in lower_bounds],  # type: ignore
        [float(b) for b in upper_bounds] # type: ignore
    )


def parse_lower_upper_bounds(source: dict[str,tuple[Any,Any]]):
    """
    Parse lower and upper boundaries, returning 2 lists:
    
    `lower_bounds`, `upper_bounds`
    """
    lower = [low[0] for low in source.values()]
    upper = [up[1] for up in source.values()]
    
    return lower, upper

