from pathlib import Path
from typing import Union

from ..IO_tools import load_list_strings
from ..schema import FeatureSchema

from ..path_manager import make_fullpath, list_subdirectories, list_files_by_extension
from .._core import get_logger
from ..keys._keys import DatasetKeys, PytorchModelArchitectureKeys, PytorchArtifactPathKeys


_LOGGER = get_logger("ArtifactFinder")


__all__ = [
    "DragonArtifactFinder",
    "find_model_artifacts_multi",
]


class DragonArtifactFinder:
    """
    Finds, processes, and returns model training artifacts from a target directory.
    
    The expected directory structure is:
    
    ```
        directory
        ├── *.pth
        ├── scaler_*.pth          (Required if `load_scaler` is True)
        ├── feature_names.txt
        ├── target_names.txt
        ├── architecture.json
        └── FeatureSchema.json     (Required if `load_schema` is True)
    ```
    """
    def __init__(self, 
                 directory: Union[str, Path], 
                 load_scaler: bool, 
                 load_schema: bool,
                 strict: bool=False,
                 verbose: bool=True) -> None:
        """
        Args:
            directory (str | Path): The path to the directory that contains training artifacts.
            load_scaler (bool): If True, requires and searches for a scaler file `scaler_*.pth`.
            load_schema (bool): If True, requires and searches for a FeatureSchema file `FeatureSchema.json`.
            strict (bool): If True, raises an error if any artifact is missing. If False, returns None for missing artifacts silently.
            verbose (bool): Displays the missing artifacts in the directory or a success message.
        """
        # validate directory
        dir_path = make_fullpath(directory, enforce="directory")
        
        parsing_dict = _find_model_artifacts(target_directory=dir_path, load_scaler=load_scaler, verbose=False, strict=strict)
        
        self._weights_path = parsing_dict[PytorchArtifactPathKeys.WEIGHTS_PATH]
        self._feature_names_path = parsing_dict[PytorchArtifactPathKeys.FEATURES_PATH]
        self._target_names_path = parsing_dict[PytorchArtifactPathKeys.TARGETS_PATH]
        self._model_architecture_path = parsing_dict[PytorchArtifactPathKeys.ARCHITECTURE_PATH]
        self._scaler_path = None
        self._schema = None
        self._strict = strict
        
        if load_scaler:
            self._scaler_path = parsing_dict[PytorchArtifactPathKeys.SCALER_PATH]
            
        if load_schema:
            try:
                self._schema = FeatureSchema.from_json(directory=dir_path)
            except Exception:
                if strict:
                    # FeatureSchema logs its own error details
                    # _LOGGER.error(f"Failed to load FeatureSchema from '{dir_path.name}': {e}")
                    raise FileNotFoundError()
                else:
                    # _LOGGER.warning(f"Could not load FeatureSchema from '{dir_path.name}': {e}")
                    self._schema = None

        # Process feature names
        if self._feature_names_path is not None:
            self._feature_names = self._process_text(self._feature_names_path)
        else:
            self._feature_names = None
        # Process target names
        if self._target_names_path is not None:
            self._target_names = self._process_text(self._target_names_path)
        else:
            self._target_names = None
            
        if verbose:
            # log missing artifacts
            missing_artifacts = []
            if self._feature_names is None:
                missing_artifacts.append("Feature Names")
            if self._target_names is None:
                missing_artifacts.append("Target Names")
            if self._weights_path is None:
                missing_artifacts.append("Weights File")
            if self._model_architecture_path is None:
                missing_artifacts.append("Model Architecture File")
            if load_scaler and self._scaler_path is None:
                missing_artifacts.append("Scaler File")
            if load_schema and self._schema is None:
                missing_artifacts.append("FeatureSchema File")
            
            if missing_artifacts:
                _LOGGER.warning(f"Missing artifacts in '{dir_path.name}': {', '.join(missing_artifacts)}.")
            else:
                _LOGGER.info(f"All artifacts successfully loaded from '{dir_path.name}'.")

    def _process_text(self, text_file_path: Path):
        list_strings = load_list_strings(text_file=text_file_path, verbose=False)
        return list_strings
    
    @property
    def feature_names(self) -> Union[list[str], None]:
        """Returns the feature names as a list of strings."""
        if self._strict and not self._feature_names:
            _LOGGER.error("No feature names loaded for Strict mode.")
            raise ValueError()
        return self._feature_names
    
    @property
    def target_names(self) -> Union[list[str], None]:
        """Returns the target names as a list of strings."""
        if self._strict and not self._target_names:
            _LOGGER.error("No target names loaded for Strict mode.")
            raise ValueError()
        return self._target_names
    
    @property
    def weights_path(self) -> Union[Path, None]:
        """Returns the path to the state dictionary pth file."""
        if self._strict and self._weights_path is None:
            _LOGGER.error("No weights file loaded for Strict mode.")
            raise ValueError()
        return self._weights_path
    
    @property
    def model_architecture_path(self) -> Union[Path, None]:
        """Returns the path to the model architecture json file."""
        if self._strict and self._model_architecture_path is None:
            _LOGGER.error("No model architecture file loaded for Strict mode.")
            raise ValueError()
        return self._model_architecture_path
    
    @property
    def scaler_path(self) -> Union[Path, None]:
        """Returns the path to the scaler file."""
        if self._strict and self._scaler_path is None:
            _LOGGER.error("No scaler file loaded for Strict mode.")
            raise ValueError()
        else:
            return self._scaler_path
        
    @property
    def feature_schema(self) -> Union[FeatureSchema, None]:
        """Returns the FeatureSchema object."""
        if self._strict and self._schema is None:
            _LOGGER.error("No FeatureSchema loaded for Strict mode.")
            raise ValueError()
        else:
            return self._schema
        
    def __repr__(self) -> str:
        dir_name = self._weights_path.parent.name if self._weights_path else "Unknown"
        n_features = len(self._feature_names) if self._feature_names else "None"
        n_targets = len(self._target_names) if self._target_names else "None"
        scaler_status = self._scaler_path.name if self._scaler_path else "None"
        schema_status = "Loaded" if self._schema else "None"
        
        return (
            f"{self.__class__.__name__}\n"
            f"    directory='{dir_name}'\n"
            f"    weights='{self._weights_path.name if self._weights_path else 'None'}'\n"
            f"    architecture='{self._model_architecture_path.name if self._model_architecture_path else 'None'}'\n"
            f"    scaler='{scaler_status}'\n"
            f"    schema='{schema_status}'\n"
            f"    features={n_features}\n" 
            f"    targets={n_targets}"
        )


def _find_model_artifacts(target_directory: Union[str,Path], load_scaler: bool, verbose: bool=True, strict:bool=True) -> dict[str, Union[Path, None]]:
    """
    Scans a directory to find paths to model weights, target names, feature names, and model architecture. Optionally an scaler path if `load_scaler` is True.
    
    The expected directory structure is as follows:
    
    ```
        target_directory
        ├── *.pth
        ├── scaler_*.pth          (Required if `load_scaler` is True)
        ├── feature_names.txt
        ├── target_names.txt
        └── architecture.json
    ```
    
    Args:
        target_directory (str | Path): The path to the directory that contains training artifacts.
        load_scaler (bool): If True, the function requires and searches for a scaler file `scaler_*.pth`.
        verbose (bool): If True, enables detailed logging during the search process.
        strict (bool): If True, raises errors on missing files. If False, returns None for missing files.
    """
    # validate directory
    dir_path = make_fullpath(target_directory, enforce="directory")
    dir_name = dir_path.name
    
    # find files
    model_pth_dict = list_files_by_extension(directory=dir_path, extension="pth", verbose=False, raise_on_empty=False)
    
    if not model_pth_dict:
        pth_msg=f"No '.pth' files found in directory: {dir_name}."
        if strict:
            _LOGGER.error(pth_msg)
            raise IOError()
        else:
            if verbose:
                _LOGGER.warning(pth_msg)
            model_pth_dict = None
    
    # restriction
    if model_pth_dict is not None:
        valid_count = False
        msg = ""
        
        if load_scaler:
            if len(model_pth_dict) == 2:
                valid_count = True
            else:
                msg = f"Directory '{dir_name}' should contain exactly 2 '.pth' files: scaler and weights. Found {len(model_pth_dict)}."
        else:
            if len(model_pth_dict) == 1:
                valid_count = True
            else:
                msg = f"Directory '{dir_name}' should contain exactly 1 '.pth' file for weights. Found {len(model_pth_dict)}."
        
        # Respect strict mode for count mismatch
        if not valid_count:
            if strict:
                _LOGGER.error(msg)
                raise IOError()
            else:
                if verbose:
                    _LOGGER.warning(msg)
                # Invalidate dictionary
                model_pth_dict = None
    
    ##### Scaler and Weights #####
    scaler_path = None
    weights_path = None
    
    # load weights and scaler if present
    if model_pth_dict is not None:
        for pth_filename, pth_path in model_pth_dict.items():
            if load_scaler and pth_filename.lower().startswith(DatasetKeys.SCALER_PREFIX):
                scaler_path = pth_path
            else:
                weights_path = pth_path
    
    # validation
    if not weights_path and strict:
        _LOGGER.error(f"Error parsing the model weights path from '{dir_name}'")
        raise IOError()
    
    if strict and load_scaler and not scaler_path:
        _LOGGER.error(f"Error parsing the scaler path from '{dir_name}'")
        raise IOError()
    
    ##### Target and Feature names #####
    target_names_path = None
    feature_names_path = None
    
    # load feature and target names
    model_txt_dict = list_files_by_extension(directory=dir_path, extension="txt", verbose=False, raise_on_empty=False)
    
    # if the directory has no txt files, the loop is skipped
    for txt_filename, txt_path in model_txt_dict.items():
        if txt_filename == DatasetKeys.FEATURE_NAMES:
            feature_names_path = txt_path
        elif txt_filename == DatasetKeys.TARGET_NAMES:
            target_names_path = txt_path
    
    # validation per case
    if strict and not target_names_path:
        _LOGGER.error(f"Error parsing the target names path from '{dir_name}'")
        raise IOError()
    elif verbose and not target_names_path:
        _LOGGER.warning(f"Target names file not found in '{dir_name}'.")
    
    if strict and not feature_names_path:
        _LOGGER.error(f"Error parsing the feature names path from '{dir_name}'")
        raise IOError()
    elif verbose and not feature_names_path:
        _LOGGER.warning(f"Feature names file not found in '{dir_name}'.")

    ##### load model architecture path #####
    architecture_path = None
    
    model_json_dict = list_files_by_extension(directory=dir_path, extension="json", verbose=False, raise_on_empty=False)
    
    # if the directory has no json files, the loop is skipped
    for json_filename, json_path in model_json_dict.items():
        if json_filename == PytorchModelArchitectureKeys.SAVENAME:
            architecture_path = json_path
    
    # validation
    if strict and not architecture_path:
        _LOGGER.error(f"Error parsing the model architecture path from '{dir_name}'")
        raise IOError()
    elif verbose and not architecture_path:
        _LOGGER.warning(f"Model architecture file not found in '{dir_name}'.")
    
    ##### Paths dictionary #####
    parsing_dict = {
        PytorchArtifactPathKeys.WEIGHTS_PATH: weights_path,
        PytorchArtifactPathKeys.ARCHITECTURE_PATH: architecture_path,
        PytorchArtifactPathKeys.FEATURES_PATH: feature_names_path,
        PytorchArtifactPathKeys.TARGETS_PATH: target_names_path,
    }
    
    if load_scaler:
        parsing_dict[PytorchArtifactPathKeys.SCALER_PATH] = scaler_path
    
    return parsing_dict


def find_model_artifacts_multi(target_directory: Union[str,Path], load_scaler: bool, verbose: bool=False) -> list[dict[str, Path]]:
    """
    Scans subdirectories to find paths to model weights, target names, feature names, and model architecture. Optionally an scaler path if `load_scaler` is True.

    This function operates on a specific directory structure. It expects the
    `target_directory` to contain one or more subdirectories, where each
    subdirectory represents a single trained model result.
    
    This function works using a strict mode, meaning that it will raise errors if
    any required artifacts are missing in a model's subdirectory.

    The expected directory structure for each model is as follows:
    ```
        target_directory
        ├── model_1
        │   ├── *.pth
        │   ├── scaler_*.pth          (Required if `load_scaler` is True)
        │   ├── feature_names.txt
        │   ├── target_names.txt
        │   └── architecture.json
        └── model_2/
            └── ...
    ```

    Args:
        target_directory (str | Path): The path to the root directory that contains model subdirectories.
        load_scaler (bool): If True, the function requires and searches for a scaler file (`.pth`) in each model subdirectory.
        verbose (bool): If True, enables detailed logging during the file paths search process.

    Returns:
        (list[dict[str, Path]]): A list of dictionaries, where each dictionary
            corresponds to a model found in a subdirectory. The dictionary
            maps standardized keys to the absolute paths of the model's
            artifacts (weights, architecture, features, targets, and scaler).
    """
    # validate directory
    root_path = make_fullpath(target_directory, enforce="directory")
    
    # store results
    all_artifacts: list[dict[str, Path]] = list()
    
    # find model directories
    result_dirs_dict = list_subdirectories(root_dir=root_path, verbose=verbose, raise_on_empty=True)
    for _dir_name, dir_path in result_dirs_dict.items():
        
        parsing_dict = _find_model_artifacts(target_directory=dir_path,
                                            load_scaler=load_scaler,
                                            verbose=verbose,
                                            strict=True)
        
        # parsing_dict is guaranteed to have all required paths due to strict=True
        all_artifacts.append(parsing_dict)  # type: ignore
    
    return all_artifacts

