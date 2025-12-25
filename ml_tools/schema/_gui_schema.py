from typing import Union, Any
from pathlib import Path
import json

from ..path_manager import make_fullpath

from ..keys._keys import SchemaKeys
from .._core import get_logger

from ._feature_schema import FeatureSchema


_LOGGER = get_logger("GUISchema")


__all__ = [
    "create_guischema_template",
    "make_multibinary_groups",
]


def create_guischema_template(
    directory: Union[str, Path],
    feature_schema: FeatureSchema,
    targets: list[str],
    continuous_ranges: dict[str, tuple[float, float]],
    multibinary_groups: Union[dict[str, list[str]], None] = None,
) -> None:
    """
    Generates a 'GUISchema.json' boilerplate file based on the Model FeatureSchema.
    
    The generated JSON contains entries with empty "gui_name" fields for manual mapping.
    Leave 'gui_name' empty to use auto-formatted Title Case.
    
    Args:
        directory (str | Path): Where to save the json file.
        feature_schema (FeatureSchema): The source FeatureSchema object.
        targets (list[str]): List of target names as used in the ML pipeline.
        continuous_ranges (Dict[str, Tuple[float, float]]): Dict {model_name: (min, max)}.
        multibinary_groups (Dict[str, list[str]] | None): Optional Dict {GUI_Group_Name: [model_col_1, model_col_2]}.
                            Used to group binary columns into a single multi-select list.
    """
    dir_path = make_fullpath(directory, make=True, enforce="directory")
        
    schema = feature_schema
    output_data: dict[str, Any] = {
        SchemaKeys.TARGETS: [],
        SchemaKeys.CONTINUOUS: [],
        SchemaKeys.BINARY: [],
        SchemaKeys.MULTIBINARY: {}, # Structure: GroupName: [{model: x, gui: ""}]
        SchemaKeys.CATEGORICAL: []
    }

    # Track handled columns to prevent duplicates in binary/categorical
    handled_cols = set()

    # 1. Targets
    for t in targets:
        output_data[SchemaKeys.TARGETS].append({
            SchemaKeys.MODEL_NAME: t,
            SchemaKeys.GUI_NAME: "" # User to fill
        })

    # 2. Continuous
    # Validate ranges against schema
    schema_cont_set = set(schema.continuous_feature_names)
    for name, min_max in continuous_ranges.items():
        if name in schema_cont_set:
            output_data[SchemaKeys.CONTINUOUS].append({
                SchemaKeys.MODEL_NAME: name,
                SchemaKeys.GUI_NAME: "",
                SchemaKeys.MIN_VALUE: min_max[0],
                SchemaKeys.MAX_VALUE: min_max[1]
            })
            handled_cols.add(name)
        else:
            _LOGGER.warning(f"GUISchema: Provided range for '{name}', but it is not in FeatureSchema continuous list.")

    # 3. Multi-Binary Groups
    if multibinary_groups:
        # Check for validity within the generic feature list
        all_feats = set(schema.feature_names)
        
        for group_name, cols in multibinary_groups.items():
            # Validation: Groups cannot be empty
            if not cols:
                # warn and skip
                _LOGGER.warning(f"GUISchema: Multi-binary group '{group_name}' is empty and will be skipped.")
                continue

            group_options = []
            for col in cols:
                # Validation: Columns must exist in schema
                if col not in all_feats:
                    # warn and skip
                    _LOGGER.warning(f"GUISchema: Multi-binary column '{col}' in group '{group_name}' not found in FeatureSchema. Skipping.")
                    continue
                # else, add to group
                group_options.append({
                    SchemaKeys.MODEL_NAME: col,
                    SchemaKeys.GUI_NAME: "" 
                })
                handled_cols.add(col)
            output_data[SchemaKeys.MULTIBINARY][group_name] = group_options

    # 4. Binary & Categorical (Derived from Schema Mappings)
    if schema.categorical_mappings:
        for name, mapping in schema.categorical_mappings.items():
            if name in handled_cols:
                continue
            
            # Heuristic: Cardinality 2 = Binary, >2 = Categorical
            if len(mapping) == 2:
                output_data[SchemaKeys.BINARY].append({
                    SchemaKeys.MODEL_NAME: name,
                    SchemaKeys.GUI_NAME: "" # User to fill
                })
            else:
                # For categorical, we also allow renaming the specific options
                options_with_names = {k: "" for k in mapping.keys()} # Default gui_option = model_option
                
                output_data[SchemaKeys.CATEGORICAL].append({
                    SchemaKeys.MODEL_NAME: name,
                    SchemaKeys.GUI_NAME: "", # User to fill feature name
                    SchemaKeys.MAPPING: mapping, # Original mapping
                    SchemaKeys.OPTIONAL_LABELS: options_with_names # User can edit keys here
                })

    save_path = dir_path / SchemaKeys.GUI_SCHEMA_FILENAME
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        _LOGGER.info(f"GUISchema template generated at: '{dir_path.name}/{SchemaKeys.GUI_SCHEMA_FILENAME}'")
    except IOError as e:
        _LOGGER.error(f"Failed to save GUISchema template: {e}")


def make_multibinary_groups(
    feature_schema: FeatureSchema,
    group_prefixes: list[str],
    separator: str = "_"
) -> dict[str, list[str]]:
    """
    Helper to automate creating the multibinary_groups dictionary for create_guischema_template.

    Iterates through provided prefixes and groups categorical features that contain
    the pattern '{prefix}{separator}'.

    Args:
        feature_schema: The loaded FeatureSchema containing categorical feature names.
        group_prefixes: A list of group prefixes to search for.
        separator: The separator used in Multibinary Encoding (default '_').

    Returns:
        Dict[str, list[str]]: A dictionary mapping group names to their found column names.
    """
    groups: dict[str, list[str]] = {}
    
    # check that categorical features exist
    if not feature_schema.categorical_feature_names:
        _LOGGER.error("FeatureSchema has no categorical features defined.")
        raise ValueError()
    
    # validate separator
    if not separator or not isinstance(separator, str):
        _LOGGER.error(f"Invalid separator '{separator}' of type {type(separator)}.")
        raise ValueError()

    for prefix in group_prefixes:
        if not prefix or not isinstance(prefix, str):
            _LOGGER.error(f"Invalid prefix '{prefix}' of type {type(prefix)}.")
            raise ValueError()
        
        search_term = f"{prefix}{separator}"
        
        # check if substring exists in the column name. must begin with prefix+separator
        cols = [
            name for name in feature_schema.categorical_feature_names
            if name.startswith(search_term)
        ]

        if cols:
            groups[prefix] = cols
        else:
            _LOGGER.warning(f"No columns found for group '{prefix}' using search term '{search_term}'")
            
    # log resulting groups
    _LOGGER.info(f"Multibinary groups created: {list(groups.keys())}")

    return groups

