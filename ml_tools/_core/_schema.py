from typing import NamedTuple, Tuple, Optional, Dict, Union, Any
from pathlib import Path
import json

from ._custom_logger import save_list_strings
from ._keys import DatasetKeys, SchemaKeys
from ._logger import get_logger
from ._path_manager import make_fullpath
from ._script_info import _script_info


_LOGGER = get_logger("FeatureSchema")


__all__ = [
    "FeatureSchema",
    "create_guischema_template",
]


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
    
    # Map string-to-int category values (e.g., {'color': {'red': 0, 'blue': 1}})
    categorical_mappings: Optional[Dict[str, Dict[str, int]]]
    
    def to_json(self, directory: Union[str, Path], verbose: bool = True) -> None:
        """
        Saves the schema as 'FeatureSchema.json' to the provided directory. 
        
        Handles conversion of Tuple->List and IntKeys->StrKeys automatically.
        """
        # validate path
        dir_path = make_fullpath(directory, enforce="directory")
        file_path = dir_path / SchemaKeys.SCHEMA_FILENAME
        
        try:
            # Convert named tuple to dict
            data = self._asdict()
            
            # Write to disk
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            if verbose:
                _LOGGER.info(f"FeatureSchema saved to '{dir_path.name}/{SchemaKeys.SCHEMA_FILENAME}'")
                
        except (IOError, TypeError) as e:
            _LOGGER.error(f"Failed to save FeatureSchema to JSON: {e}")
            raise e
        
    @classmethod
    def from_json(cls, directory: Union[str, Path], verbose: bool = True) -> 'FeatureSchema':
        """
        Loads a 'FeatureSchema.json' from the provided directory.
        
        Restores Tuples from Lists and Integer Keys from Strings.
        """
        # validate directory
        dir_path = make_fullpath(directory, enforce="directory")
        file_path = dir_path / SchemaKeys.SCHEMA_FILENAME
        
        if not file_path.exists():
            _LOGGER.error(f"FeatureSchema file not found at '{directory}'")
            raise FileNotFoundError()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)
            
            # 1. Restore Tuples (JSON loads them as lists)
            feature_names = tuple(data.get("feature_names", []))
            cont_names = tuple(data.get("continuous_feature_names", []))
            cat_names = tuple(data.get("categorical_feature_names", []))

            # 2. Restore Integer Keys for categorical_index_map
            raw_map = data.get("categorical_index_map")
            cat_index_map: Optional[Dict[int, int]] = None
            if raw_map is not None:
                cat_index_map = {int(k): v for k, v in raw_map.items()}

            # 3. Mappings (keys are strings, no conversion needed)
            cat_mappings = data.get("categorical_mappings")

            schema = cls(
                feature_names=feature_names,
                continuous_feature_names=cont_names,
                categorical_feature_names=cat_names,
                categorical_index_map=cat_index_map,
                categorical_mappings=cat_mappings
            )

            if verbose:
                _LOGGER.info(f"FeatureSchema loaded from '{dir_path.name}'")

            return schema

        except (IOError, ValueError, KeyError) as e:
            _LOGGER.error(f"Failed to load FeatureSchema from '{dir_path}': {e}")
            raise e

    def _save_helper(self, artifact: Tuple[str, ...], directory: Union[str,Path], filename: str, verbose: bool):
        to_save = list(artifact)
        
        # empty check
        if not to_save:
            _LOGGER.warning(f"Skipping save for '{filename}': The feature list is empty.")
            return
        
        save_list_strings(list_strings=to_save,
                          directory=directory,
                          filename=filename,
                          verbose=verbose)

    def save_all_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves all feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.feature_names,
                          directory=directory,
                          filename=DatasetKeys.FEATURE_NAMES,
                          verbose=verbose)
        
    def save_continuous_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves continuous feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.continuous_feature_names,
                          directory=directory,
                          filename=DatasetKeys.CONTINUOUS_NAMES,
                          verbose=verbose)
    
    def save_categorical_features(self, directory: Union[str,Path], verbose: bool=True):
        """
        Saves categorical feature names to a text file.

        Args:
            directory: The directory where the file will be saved.
            verbose: If True, prints a confirmation message upon saving.
        """
        self._save_helper(artifact=self.categorical_feature_names,
                          directory=directory,
                          filename=DatasetKeys.CATEGORICAL_NAMES,
                          verbose=verbose)
        
    def save_artifacts(self, directory: Union[str,Path]):
        """
        Saves feature names, categorical feature names, continuous feature names to separate text files.
        """
        self.save_all_features(directory=directory, verbose=True)
        self.save_continuous_features(directory=directory, verbose=True)
        self.save_categorical_features(directory=directory, verbose=True)
        
    def __repr__(self) -> str:
        """Returns a concise representation of the schema's contents."""
        total = len(self.feature_names)
        cont = len(self.continuous_feature_names)
        cat = len(self.categorical_feature_names)
        index_map = self.categorical_index_map is not None
        cat_map = self.categorical_mappings is not None
        return (
            f"FeatureSchema(total={total}, continuous={cont}, categorical={cat}, index_map={index_map}, categorical_map={cat_map})"
        )


def create_guischema_template(
    directory: Union[str, Path],
    feature_schema: FeatureSchema,
    targets: list[str],
    continuous_ranges: Dict[str, Tuple[float, float]],
    multibinary_groups: Union[Dict[str, list[str]], None] = None,
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
    output_data: Dict[str, Any] = {
        "targets": [],
        "continuous": [],
        "binary": [],
        "multi_binary": {}, # Structure: GroupName: [{model: x, gui: ""}]
        "categorical": []
    }

    # Track handled columns to prevent duplicates in binary/categorical
    handled_cols = set()

    # 1. Targets
    for t in targets:
        output_data["targets"].append({
            "model_name": t,
            "gui_name": "" # User to fill
        })

    # 2. Continuous
    # Validate ranges against schema
    schema_cont_set = set(schema.continuous_feature_names)
    for name, min_max in continuous_ranges.items():
        if name in schema_cont_set:
            output_data["continuous"].append({
                "model_name": name,
                "gui_name": "",
                "min": min_max[0],
                "max": min_max[1]
            })
            handled_cols.add(name)
        else:
            _LOGGER.warning(f"GUISchema: Provided range for '{name}', but it is not in FeatureSchema continuous list.")

    # 3. Multi-Binary Groups
    if multibinary_groups:
            # Check for validity within the generic feature list
            all_feats = set(schema.feature_names)
            
            for group_name, cols in multibinary_groups.items():
                group_options = []
                for col in cols:
                    if col not in all_feats:
                        _LOGGER.warning(f"GUISchema: Multi-binary column '{col}' not found in FeatureSchema.")
                    
                    group_options.append({
                        "model_name": col,
                        "gui_name": "" 
                    })
                    handled_cols.add(col)
                output_data["multi_binary"][group_name] = group_options

    # 4. Binary & Categorical (Derived from Schema Mappings)
    if schema.categorical_mappings:
        for name, mapping in schema.categorical_mappings.items():
            if name in handled_cols:
                continue
            
            # Heuristic: Cardinality 2 = Binary, >2 = Categorical
            if len(mapping) == 2:
                output_data["binary"].append({
                    "model_name": name,
                    "gui_name": "" # User to fill
                })
            else:
                # For categorical, we also allow renaming the specific options
                options_with_names = {k: "" for k in mapping.keys()} # Default gui_option = model_option
                
                output_data["categorical"].append({
                    "model_name": name,
                    "gui_name": "", # User to fill feature name
                    "mapping": mapping, # Preserved for reference
                    "option_labels": options_with_names # User can edit keys here
                })

    save_path = dir_path / SchemaKeys.GUI_SCHEMA_FILENAME
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        _LOGGER.info(f"GUISchema template generated at: '{dir_path.name}/{SchemaKeys.GUI_SCHEMA_FILENAME}'")
    except IOError as e:
        _LOGGER.error(f"Failed to save GUISchema template: {e}")


def info():
    _script_info(__all__)
