import configparser
from pathlib import Path
import traceback
import FreeSimpleGUI as sg
from functools import wraps
from typing import Any, Dict, Tuple, List, Literal, Union, Optional, Callable
from .utilities import _script_info
import numpy as np
from .logger import _LOGGER


__all__ = [
    "ConfigManager", 
    "GUIFactory",
    "catch_exceptions", 
    "FeatureMaster",
    "GUIHandler" 
]

# --- Configuration Management ---
class _SectionProxy:
    """A helper class to represent a section of the .ini file as an object."""
    def __init__(self, parser: configparser.ConfigParser, section_name: str):
        for option, value in parser.items(section_name):
            setattr(self, option.lower(), self._process_value(value))

    def _process_value(self, value_str: str) -> Any:
        """Automatically converts string values to appropriate types."""
        # Handle None
        if value_str is None or value_str.lower() == 'none':
            return None
        # Handle Booleans
        if value_str.lower() in ['true', 'yes', 'on']:
            return True
        if value_str.lower() in ['false', 'no', 'off']:
            return False
        # Handle Integers
        try:
            return int(value_str)
        except ValueError:
            pass
        # Handle Floats
        try:
            return float(value_str)
        except ValueError:
            pass
        # Handle 'width,height' tuples
        if ',' in value_str:
            try:
                return tuple(map(int, value_str.split(",")))
            except (ValueError, TypeError):
                pass
        # Fallback to the original string
        return value_str

class ConfigManager:
    """
    Loads a .ini file and provides access to its values as object attributes.
    Includes a method to generate a default configuration template.
    """
    def __init__(self, config_path: str | Path):
        """
        Initializes the ConfigManager and dynamically creates attributes
        based on the .ini file's sections and options.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        parser = configparser.ConfigParser(comment_prefixes=('#', ';'), inline_comment_prefixes=('#', ';'))
        parser.read(config_path)

        for section in parser.sections():
            setattr(self, section.lower(), _SectionProxy(parser, section))

    @staticmethod
    def generate_template(file_path: str | Path, force_overwrite: bool = False):
        """
        Generates a complete, commented .ini template file that works with the GUIFactory.

        Args:
            file_path (str | Path): The path where the .ini file will be saved.
            force_overwrite (bool): If True, overwrites the file if it already exists.
        """
        path = Path(file_path)
        if path.exists() and not force_overwrite:
            _LOGGER.warning(f"⚠️ Configuration file already exists at {path}. Aborting.")
            return

        config = configparser.ConfigParser()

        config['General'] = {
            '; The overall theme for the GUI. Find more at https://www.pysimplegui.org/en/latest/call%20reference/#themes-automatic-coloring-of-elements': '',
            'theme': 'LightGreen6',
            '; Default font for the application.': '',
            'font_family': 'Helvetica',
            '; Title of the main window.': '',
            'window_title': 'My Application',
            '; Can the user resize the window? (true/false)': '',
            'resizable_window': 'false',
            '; Optional minimum window size (width,height). Leave blank for no minimum.': '',
            'min_size': '800,600',
            '; Optional maximum window size (width,height). Leave blank for no maximum.': '',
            'max_size': ''
        }
        config['Layout'] = {
            '; Default size for continuous input boxes (width,height in characters/rows).': '',
            'input_size_cont': '16,1',
            '; Default size for combo/binary boxes (width,height in characters/rows).': '',
            'input_size_binary': '14,1',
            '; Size for multiselect listboxes (width,height in characters/rows).': '',
            'input_size_multi': '14,4',
            '; Default size for buttons (width,height in characters/rows).': '',
            'button_size': '15,2'
        }
        config['Fonts'] = {
            '; Font settings. Style can be "bold", "italic", "underline", or a combination.': '',
            'label_size': '11',
            'label_style': 'bold',
            'range_size': '9',
            'range_style': '',
            'button_size': '14',
            'button_style': 'bold',
            'frame_size': '14',
            'frame_style': ''
        }
        config['Colors'] = {
            '; Use standard hex codes (e.g., #FFFFFF) or color names (e.g., white).': '',
            '; Color for the text inside a disabled target/output box.': '',
            'target_text': '#0000D0',
            '; Background color for a disabled target/output box.': '',
            'target_background': '#E0E0E0',
            '; Color for the text on a button.': '',
            'button_text': '#FFFFFF',
            '; Background color for a button.': '',
            'button_background': '#3c8a7e',
            '; Background color when the mouse is over a button.': '',
            'button_background_hover': '#5499C7'
        }
        config['Meta'] = {
            '; Optional application version, displayed in the window title.': '',
            'version': '1.0.0'
        }

        with open(path, 'w') as configfile:
            config.write(configfile)
        _LOGGER.info(f"📝 Successfully generated config template at: '{path}'")


# --- GUI Factory ---
class GUIFactory:
    """
    Builds styled FreeSimpleGUI elements and layouts using a "building block"
    approach, driven by a ConfigManager instance.
    """
    def __init__(self, config: ConfigManager):
        """
        Initializes the factory with a configuration object.
        """
        self.config = config
        sg.theme(self.config.general.theme) # type: ignore
        sg.set_options(font=(self.config.general.font_family, 12)) # type: ignore

    # --- Atomic Element Generators ---
    def make_button(self, text: str, key: str, **kwargs) -> sg.Button:
        """
        Creates a single, styled action button.

        Args:
            text (str): The text displayed on the button.
            key (str): The key for the button element.
            **kwargs: Override default styles or add other sg.Button parameters
                      (e.g., `tooltip='Click me'`, `disabled=True`).
        """
        cfg = self.config
        font = (cfg.fonts.font_family, cfg.fonts.button_size, cfg.fonts.button_style) # type: ignore
        
        style_args = {
            "size": cfg.layout.button_size, # type: ignore
            "font": font,
            "button_color": (cfg.colors.button_text, cfg.colors.button_background), # type: ignore
            "mouseover_colors": (cfg.colors.button_text, cfg.colors.button_background_hover), # type: ignore
            "border_width": 0,
            **kwargs
        }
        return sg.Button(text.title(), key=key, **style_args)

    def make_frame(self, title: str, layout: List[List[Union[sg.Element, sg.Column]]], **kwargs) -> sg.Frame:
        """
        Creates a styled frame around a given layout.

        Args:
            title (str): The title displayed on the frame's border.
            layout (list): The layout to enclose within the frame.
            **kwargs: Override default styles or add other sg.Frame parameters
                      (e.g., `title_color='red'`, `relief=sg.RELIEF_SUNKEN`).
        """
        cfg = self.config
        font = (cfg.fonts.font_family, cfg.fonts.frame_size) # type: ignore
        
        style_args = {
            "font": font,
            "expand_x": True,
            "background_color": sg.theme_background_color(),
            **kwargs
        }
        return sg.Frame(title, layout, **style_args)

    # --- General-Purpose Layout Generators ---
    def generate_continuous_layout(
        self,
        data_dict: Dict[str, Union[Tuple[Union[int,float,None], Union[int,float,None]],List[Union[int,float,None]]]],
        is_target: bool = False,
        layout_mode: Literal["grid", "row"] = 'grid',
        features_per_column: int = 4
    ) -> List[List[sg.Column]]:
        """
        Generates a layout for continuous features or targets.

        Args:
            data_dict (dict): Keys are feature names, values are (min, max) tuples.
            is_target (bool): If True, creates disabled inputs for displaying results.
            layout_mode (str): 'grid' for a multi-row grid layout, or 'row' for a single horizontal row.
            features_per_column (int): Number of features per column when `layout_mode` is 'grid'.

        Returns:
            A list of lists of sg.Column elements, ready to be used in a window layout.
        """
        cfg = self.config
        bg_color = sg.theme_background_color()
        label_font = (cfg.fonts.font_family, cfg.fonts.label_size, cfg.fonts.label_style) # type: ignore
        
        columns = []
        for name, value in data_dict.items():
            if value is None:
                raise ValueError(f"Feature '{name}' was assigned a 'None' value.")
            elif len(value) != 2:
                raise ValueError(f"Feature '{name}' must provide exactly 2 values.")
            else:
                val_min, val_max = value
            key = name
            default_text = "" if is_target else str(val_max)
            
            label = sg.Text(name, font=label_font, background_color=bg_color, key=f"_text_{name}")
            
            input_style = {"size": cfg.layout.input_size_cont, "justification": "center"} # type: ignore
            if is_target:
                input_style["text_color"] = cfg.colors.target_text # type: ignore
                input_style["disabled_readonly_background_color"] = cfg.colors.target_background # type: ignore
            
            element = sg.Input(default_text, key=key, disabled=is_target, **input_style)
            
            if is_target:
                layout = [[label], [element]]
            else:
                range_font = (cfg.fonts.font_family, cfg.fonts.range_size) # type: ignore
                range_text = sg.Text(f"Range: {int(val_min)}-{int(val_max)}", font=range_font, background_color=bg_color) # type: ignore
                layout = [[label], [element], [range_text]]
            
            # each feature is wrapped as a column element
            layout.append([sg.Text(" ", font=(cfg.fonts.font_family, 2), background_color=bg_color)]) # type: ignore
            columns.append(sg.Column(layout, background_color=bg_color))

        if layout_mode == 'row':
            return [columns] # A single row containing all columns
        
        # Default to 'grid' layout
        return [columns[i:i + features_per_column] for i in range(0, len(columns), features_per_column)]

    def generate_combo_layout(
        self,
        data_dict: Dict[str, Union[List[Any],Tuple[Any,...]]],
        layout_mode: Literal["grid", "row"] = 'grid',
        features_per_column: int = 4
    ) -> List[List[sg.Column]]:
        """
        Generates a layout for categorical or binary features using Combo boxes.

        Args:
            data_dict (dict): Keys are feature names, values are lists of options.
            layout_mode (str): 'grid' for a multi-row grid layout, or 'row' for a single horizontal row.
            features_per_column (int): Number of features per column when `layout_mode` is 'grid'.

        Returns:
            A list of lists of sg.Column elements, ready to be used in a window layout.
        """
        cfg = self.config
        bg_color = sg.theme_background_color()
        label_font = (cfg.fonts.font_family, cfg.fonts.label_size, cfg.fonts.label_style) # type: ignore

        columns = []
        for name, values in data_dict.items():
            label = sg.Text(name, font=label_font, background_color=bg_color, key=f"_text_{name}")
            element = sg.Combo(
                values, default_value=values[0], key=name,
                size=cfg.layout.input_size_binary, readonly=True # type: ignore
            )
            layout = [[label], [element]]
            layout.append([sg.Text(" ", font=(cfg.fonts.font_family, 2), background_color=bg_color)]) # type: ignore
            # each feature is wrapped in a Column element
            columns.append(sg.Column(layout, background_color=bg_color))

        if layout_mode == 'row':
            return [columns] # A single row containing all columns
            
        # Default to 'grid' layout
        return [columns[i:i + features_per_column] for i in range(0, len(columns), features_per_column)]
    
    def generate_multiselect_layout(
        self,
        data_dict: Dict[str, Union[List[Any], Tuple[Any, ...]]],
        layout_mode: Literal["grid", "row"] = 'grid',
        features_per_column: int = 4
    ) -> List[List[sg.Column]]:
        """
        Generates a layout for features using Listbox elements for multiple selections.

        This allows the user to select zero or more options from a list without
        being able to input custom text.

        Args:
            data_dict (dict): Keys are feature names, values are lists of options.
            layout_mode (str): 'grid' for a multi-row grid layout, or 'row' for a single horizontal row.
            features_per_column (int): Number of features per column when `layout_mode` is 'grid'.

        Returns:
            A list of lists of sg.Column elements, ready to be used in a window layout.
        """
        cfg = self.config
        bg_color = sg.theme_background_color()
        label_font = (cfg.fonts.font_family, cfg.fonts.label_size, cfg.fonts.label_style) # type: ignore

        columns = []
        for name, values in data_dict.items():
            label = sg.Text(name, font=label_font, background_color=bg_color, key=f"_text_{name}")

            # Use sg.Listbox for multiple selections.
            element = sg.Listbox(
                values,
                key=name,
                select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                size=cfg.layout.input_size_multi, # type: ignore
                no_scrollbar=False
            )
            # -------------------

            layout = [[label], [element]]
            # Add a small spacer for consistent vertical alignment.
            layout.append([sg.Text(" ", font=(cfg.fonts.font_family, 2), background_color=bg_color)]) # type: ignore

            # Each feature is wrapped in a Column element for proper alignment.
            columns.append(sg.Column(layout, background_color=bg_color))

        if layout_mode == 'row':
            return [columns]  # A single row containing all columns

        # Default to 'grid' layout
        return [columns[i:i + features_per_column] for i in range(0, len(columns), features_per_column)]

    # --- Window Creation ---
    def create_window(self, title: str, layout: List[List[sg.Element]], **kwargs) -> sg.Window:
        """
        Creates and finalizes the main application window.

        Args:
            title (str): The title for the window.
            layout (list): The final, assembled layout for the window.
            **kwargs: Additional arguments to pass to the sg.Window constructor
                      (e.g., `location=(100, 100)`, `keep_on_top=True`).
        """
        cfg = self.config.general # type: ignore
        version = getattr(self.config.meta, 'version', None) # type: ignore
        full_title = f"{title} v{version}" if version else title

        window_args = {
            "resizable": cfg.resizable_window,
            "finalize": True,
            "background_color": sg.theme_background_color(),
            **kwargs
        }
        window = sg.Window(full_title, layout, **window_args)
        
        if cfg.min_size: window.TKroot.minsize(*cfg.min_size)
        if cfg.max_size: window.TKroot.maxsize(*cfg.max_size)
        
        return window


# --- Exception Handling Decorator ---
def catch_exceptions(show_popup: bool = True):
    """
    A decorator that wraps a function in a try-except block.
    If an exception occurs, it's caught and displayed in a popup window.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Format the full traceback to give detailed error info
                error_msg = traceback.format_exc()
                if show_popup:
                    sg.popup_error("An error occurred:", error_msg, title="Error")
                else:
                    # Fallback for non-GUI contexts or if popup is disabled
                    _LOGGER.error(error_msg)
        return wrapper
    return decorator


# --- Feature Handler ---
class FeatureMaster:
    """
    Manages and organizes feature definitions for a machine learning model.

    This class serves as a centralized registry for all features and targets 
    used by a model. It is designed to bridge the gap between a user-facing 
    application (like a GUI) and the underlying model's data representation.

    It takes various types of features (continuous, binary, one-hot encoded, 
    categorical) and targets, processing them into two key formats:
    1.  A mapping from a user-friendly "GUI name" to the corresponding "model name" 
        used in the dataset or model training.
    2.  A structure containing the acceptable values or ranges for each feature, 
        suitable for populating GUI elements like sliders, dropdowns, or checkboxes.

    By separating the GUI representation from the model's internal logic, this 
    class simplifies the process of building user interfaces for model interaction 
    and ensures that user input is correctly formatted. At least one type of 
    feature must be provided upon initialization.

    Properties are available to access the processed mappings and GUI-ready values 
    for each feature type.
    """
    def __init__(self,
                 targets: Dict[str, str],
                 continuous_features: Optional[Dict[str, Tuple[str, float, float]]] = None,
                 binary_features: Optional[Dict[str, str]] = None,
                 multi_binary_features: Optional[Dict[str, Dict[str, str]]] = None,
                 one_hot_features: Optional[Dict[str, Dict[str, str]]] = None,
                 categorical_features: Optional[List[Tuple[str, str, Dict[str, int]]]] = None) -> None:
        """
        Initializes the FeatureMaster instance by processing feature and target definitions.

        This constructor creates internal mappings to translate between GUI-friendly names and model-specific feature names. It also
        prepares data structures needed to populate UI components.

        Args:
            targets (Dict[str, str]):
                A dictionary defining the model's target variables.
                -   **key** (str): The name to be displayed in the GUI.
                -   **value** (str): The corresponding column name in the model's dataset.
            
            continuous_features (Dict[str, Tuple[str, float, float]]):
                A dictionary for continuous numerical features.
                -   **key** (str): The name to be displayed in the GUI (e.g., for a slider).
                -   **value** (Tuple[str, float, float]): A tuple containing:
                    -   `[0]` (str): The model's internal feature name.
                    -   `[1]` (float): The minimum allowed value (inclusive).
                    -   `[2]` (float): The maximum allowed value (inclusive).
            
            binary_features (Dict[str, str]):
                A dictionary for binary (True/False) features.
                -   **key** (str): The name to be displayed in the GUI (e.g., for a checkbox).
                -   **value** (str): The model's internal feature name.
                
            multi_binary_features (Dict[str, Dict[str, str]]):
                A dictionary for features where multiple binary-like options can be
                selected at once (e.g., from a multi-select listbox).
                -   **key** (str): The name for the group to be displayed in the GUI.
                -   **value** (Dict[str, str]): A nested dictionary where:
                    -   key (str): The user-selectable option.
                    -   value (str): The corresponding model's internal feature name.

            one_hot_features (Dict[str, Dict[str, str]]):
                A dictionary for features that will be one-hot encoded from a single
                categorical input.
                -   **key** (str): The name for the group to be displayed in the GUI (e.g., 
                    for a dropdown menu).
                -   **value** (Dict[str, str]): A nested dictionary where:
                    -   key (str): The user-selectable option (e.g., 'Category A').
                    -   value (str): The corresponding model column name.

            categorical_features (List[Tuple[str, str, Dict[str, int]]]):
                A list for ordinal or label-encoded categorical features.
                -   **Each element is a tuple** containing:
                    -   `[0]` (str): The name to be displayed in the GUI (e.g., for a 
                        dropdown menu).
                    -   `[1]` (str): The model's internal feature name.
                    -   `[2]` (Dict[str, int]): A dictionary mapping the user-selectable 
                        options to their corresponding integer values.
        """
        # Validation
        if continuous_features is None and binary_features is None and one_hot_features is None and categorical_features is None and multi_binary_features is None:
            raise ValueError("No features provided.")
        
        # Targets
        self._targets_values = self._handle_targets(targets)
        self._targets_mapping = targets
        
        # continuous features
        if continuous_features is not None:
            self._continuous_values, self._continuous_mapping = self._handle_continuous_features(continuous_features)
            self.has_continuous = True
        else:
            self._continuous_values, self._continuous_mapping = None, None
            self.has_continuous = False
            
        # binary features
        if binary_features is not None:
            self._binary_values = self._handle_binary_features(binary_features)
            self._binary_mapping = binary_features
            self.has_binary = True
        else:
            self._binary_values, self._binary_mapping = None, None
            self.has_binary = False
            
        # multi-binary features
        if multi_binary_features is not None:
            self._multi_binary_values = self._handle_multi_binary_features(multi_binary_features)
            self._multi_binary_mapping = multi_binary_features
            self.has_multi_binary = True
        else:
            self._multi_binary_values, self._multi_binary_mapping = None, None
            self.has_multi_binary = False
        
        # one-hot features
        if one_hot_features is not None:
            self._one_hot_values = self._handle_one_hot_features(one_hot_features)
            self._one_hot_mapping = one_hot_features
            self.has_one_hot = True
        else:
            self._one_hot_values, self._one_hot_mapping = None, None
            self.has_one_hot = False
        
        # categorical features
        if categorical_features is not None:
            self._categorical_values, self._categorical_mapping = self._handle_categorical_features(categorical_features)
            self.has_categorical = True
        else:
            self._categorical_values, self._categorical_mapping = None, None
            self.has_categorical = False
            
        # all features attribute
        self._all_features = self._get_all_gui_features()
        
    def _handle_targets(self, targets: Dict[str, str]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[None,None]] = {gui_key: (None, None) for gui_key in targets.keys()}
        # Map GUI name to Model name (same as input)
        return gui_values
        
    def _handle_continuous_features(self, continuous_features: Dict[str, Tuple[str, float, float]]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[float,float]] = {gui_key: (tuple_values[1], tuple_values[2]) for gui_key, tuple_values in continuous_features.items()}
        # Map GUI name to Model name
        gui_to_model: dict[str,str] = {gui_key: tuple_values[0] for gui_key, tuple_values in continuous_features.items()}
        return gui_values, gui_to_model
    
    def _handle_binary_features(self, binary_features: Dict[str, str]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[Literal["False"],Literal["True"]]] = {gui_key: ("False", "True") for gui_key in binary_features.keys()}
        # Map GUI name to Model name (same as input)
        return gui_values
    
    def _handle_multi_binary_features(self, multi_binary_features: Dict[str, Dict[str, str]]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[str,...]] = {
            gui_key: tuple(nested_dict.keys()) 
            for gui_key, nested_dict in multi_binary_features.items()}
        # Map GUI name to Model name and preserve internal mapping (same as input)
        return gui_values

    def _handle_one_hot_features(self, one_hot_features: Dict[str, Dict[str,str]]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[str,...]] = {gui_key: tuple(nested_dict.keys()) for gui_key, nested_dict in one_hot_features.items()}
        # Map GUI name to Model name and preserve internal mapping (same as input)
        return gui_values
        
    def _handle_categorical_features(self, categorical_features: List[Tuple[str, str, Dict[str, int]]]):
        # Make dictionary GUI name: range values
        gui_values: dict[str, tuple[str,...]] = {gui_key: tuple(gui_options.keys()) for gui_key, _, gui_options in categorical_features}
        # Map GUI name to Model name and preserve internal mapping
        gui_to_model: dict[str, tuple[str, dict[str, int]]] = {gui_key: (model_key, gui_options) for gui_key, model_key, gui_options in categorical_features}
        return gui_values, gui_to_model
    
    def _get_all_gui_features(self) -> dict[str,Any]:
        all_dict: dict[str,Any] = dict()
        # Add all feature GUI keys
        if self._continuous_mapping is not None:
            all_dict.update(self._continuous_mapping)
        if self._binary_mapping is not None:
            all_dict.update(self._binary_mapping)
        if self._multi_binary_mapping is not None:
            all_dict.update(self._multi_binary_mapping)
        if self._one_hot_mapping is not None:
            all_dict.update(self._one_hot_mapping)
        if self._categorical_mapping is not None:
            all_dict.update(self._categorical_mapping)
        return all_dict
    
    @property
    def all_features(self):
        """
        A merged dictionary of all feature mappings.
        
        The value type varies based on the feature type (str, dict, or tuple).
        
        Structure:
            Dict[str, Any]
        """
        return self._all_features
    
    @property
    def targets(self):
        """
        The mapping for target variables from GUI name to model name.
        
        Structure: 
            Dict[str, str]
        """
        return self._targets_mapping
    
    @property
    def targets_gui(self):
        """
        The GUI value structure for targets.
        
        Structure: 
            Dict[str, Tuple[None, None]]
        """
        return self._targets_values
    
    @property
    def continuous(self):
        """
        The mapping for continuous features from GUI name to model name.
        
        Structure: 
            Dict[str, str]
        """
        if self._continuous_mapping is not None:
            return self._continuous_mapping
    
    @property
    def continuous_gui(self):
        """
        The GUI value ranges (min, max) for continuous features.
        
        Structure: 
            Dict[str, Tuple[float, float]]
        """
        if self._continuous_values is not None:
            return self._continuous_values
    
    @property
    def binary(self):
        """
        The mapping for binary features from GUI name to model name.
        
        Structure: 
            Dict[str, str]
        """
        if self._binary_mapping is not None:
            return self._binary_mapping
        
    @property
    def binary_gui(self):
        """
        The GUI options ('False', 'True') for binary features.
        
        Structure: 
            Dict[str, Tuple['False', 'True']]
        """
        if self._binary_values is not None:
            return self._binary_values
        
    @property
    def multi_binary(self):
        """
        The mapping for multi-binary features.
        
        Structure: 
            {"GUI NAME": {"GUI OPTION 1": "model_column"}}
        """
        if self._multi_binary_mapping is not None:
            return self._multi_binary_mapping
        
    @property
    def multi_binary_gui(self):
        """
        The GUI options for multi-binary feature groups.
        
        Structure: 
            Dict[str, Tuple[str, ...]]
        """
        if self._multi_binary_values is not None:
            return self._multi_binary_values

    @property
    def one_hot(self):
        """
        The mapping for one-hot encoded features.
        
        {"GUI NAME": {"GUI OPTION 1": "model_column"}}
        
        Structure: 
            Dict[str, Dict[str, str]]
        """
        if self._one_hot_mapping is not None:
            return self._one_hot_mapping
        
    @property
    def one_hot_gui(self):
        """
        The GUI options for one-hot encoded feature groups.
        
        Structure: 
            Dict[str, Tuple[str, ...]]
        """
        if self._one_hot_values is not None:
            return self._one_hot_values

    @property
    def categorical(self):
        """
        The mapping for categorical features.
        
        {"GUI NAME": ("model_column", {"GUI OPTION 1": column_value})}
        
        Structure: 
            Dict[str, Tuple[str, Dict[str, int]]]
        """
        if self._categorical_mapping is not None:
            return self._categorical_mapping
        
    @property
    def categorical_gui(self):
        """
        The GUI options for categorical features.
        
        Structure: 
            Dict[str, Tuple[str, ...]]
        """
        if self._categorical_values is not None:
            return self._categorical_values


# --- GUI-Model API ---
class GUIHandler:
    """
    Translates data between a GUI and a machine learning model.
    
    This class acts as the primary interface between a user-facing application
    (FreeSimpleGUI) and the model's expected data format. It uses a `FeatureMaster` instance to correctly process
    and encode user inputs.

    Its main responsibilities are:
    1.  To take raw values from GUI elements and, using the definitions from
        `FeatureMaster`, convert them into a single, ordered `numpy.ndarray`
        that can be fed directly into a model for inference.
    2.  To take the results of a model's inference and update the
        corresponding target fields in the GUI to display the prediction.

    This handler ensures a clean separation of concerns, where the GUI is
    only responsible for presentation, and the model sees correctly formatted numerical data.
    """
    def __init__(self, feature_handler: FeatureMaster, model_expected_features: list[str]) -> None:
        """
        Initializes the GUIHandler.

        Args:
            feature_handler (FeatureMaster):
                An initialized instance of the `FeatureMaster` class. This object
                contains all the necessary mappings and definitions for the model's
                features and targets.
            model_expected_features (list[str]):
                A list of strings specifying the exact names of the features the
                machine learning model expects in its input vector. The **order**
                of features in this list is critical, as it dictates the final
                column order of the output numpy array.

        Raises:
            TypeError: If `model_expected_features` is not a list or if any of its elements are not strings.
        """
        if not isinstance(model_expected_features, list):
            raise TypeError("Input 'model_expected_features' must be a list.")
        if not all(isinstance(col, str) for col in model_expected_features):
            raise TypeError("All elements in the 'model_expected_features' must be strings.")
        
        # Model expected features
        self.model_expected_features = tuple(model_expected_features)
        # Feature master instance
        self.master = feature_handler

    def _process_continuous(self, gui_feature: str, chosen_value: Any) -> Tuple[str,float]:
        """
        Maps GUI name to model expected name and casts the value to float.
        """
        try:
            model_name = self.master.continuous[gui_feature] # type: ignore
            float_value = float(chosen_value)
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature}' defined as continuous.")
            raise e
        except (ValueError, TypeError) as e2:
            _LOGGER.error(f"Invalid number conversion for '{chosen_value}' of '{gui_feature}'.")
            raise e2
        else:
            return model_name, float_value
        
    def _process_binary(self, gui_feature: str, chosen_value: str) -> Tuple[str,int]:
        """
        Maps GUI name to model expected name and casts the value to binary (0,1).
        """
        try:
            model_name = self.master.binary[gui_feature] # type: ignore
            binary_mapping_keys = self.master.binary_gui[gui_feature] # type: ignore
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature}' defined as binary.")
            raise e
        else:
            mapping_dict = {
                binary_mapping_keys[0]: 0,
                binary_mapping_keys[1]: 1
            }
            result = mapping_dict[chosen_value]
            return model_name, result
    
    def _process_multi_binary(self, gui_feature: str, chosen_values: list[str]) -> dict[str, int]:
        """
        Maps GUI names to model expected names and casts values to multi-binary encoding.

        For a given feature group, this sets all selected options to 1 and all
        unselected options to 0.
        """
        try:
            # Get the mapping for the group
            multi_binary_mapping = self.master.multi_binary[gui_feature] # type: ignore
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature}' defined as multi-binary.")
            raise e
        else:
            # Start with all possible features for this group set to 0 (unselected)
            results = {model_key: 0 for model_key in multi_binary_mapping.values()}
            # Set the features for the chosen options to 1
            for chosen_option in chosen_values:
                model_name = multi_binary_mapping[chosen_option]
                results[model_name] = 1

            return results
        
    def _process_one_hot(self, gui_feature: str, chosen_value: str) -> Dict[str,int]:
        """
        Maps GUI names to model expected names and casts values to one-hot encoding.
        """
        try:
            one_hot_mapping = self.master.one_hot[gui_feature] # type: ignore
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature}' defined as one-hot.")
            raise e
        else:
            mapped_chosen_value = one_hot_mapping[chosen_value]
            # base results mapped to 0
            results = {model_key: 0 for model_key in one_hot_mapping.values()}
            # update chosen value
            results[mapped_chosen_value] = 1
            return results
        
    def _process_categorical(self, gui_feature: str, chosen_value: str) -> Tuple[str,int]:
        """
        Maps GUI name to model expected name and casts the value to a categorical number.
        """
        try:
            categorical_tuple = self.master.categorical[gui_feature] # type: ignore
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature}' defined as categorical.")
            raise e
        else:
            model_name = categorical_tuple[0]
            categorical_mapping = categorical_tuple[1]
            result = categorical_mapping[chosen_value]
            return model_name, result
        
    def update_target_fields(self, window: sg.Window, inference_results: Dict[str, Any]):
        """
        Updates the GUI's target fields with inference results.

        Args:
            window (sg.Window): The application's window object.
            inference_results (dict): A dictionary where keys are target names (as used by the model) and values are the predicted results to update.
        """
        # Target values to update
        gui_targets_values = {gui_key: inference_results[model_key] for gui_key, model_key in self.master.targets.items()}
        
        # Update window
        for gui_key, result in gui_targets_values.items():
            # Format numbers to 2 decimal places, leave other types as-is
            display_value = f"{result:.2f}" if isinstance(result, (int, float)) else result
            window[gui_key].update(display_value) # type: ignore
            
    def _call_subprocess(self, window_values: Dict[str,Any], master_feature: Dict[str,str], processor: Callable) -> Dict[str, Union[float,int]]:
        processed_features_subset: Dict[str, Union[float,int]] = dict()
        
        for gui_name in master_feature.keys():
            chosen_value = window_values.get(gui_name)
            # value validation
            if chosen_value is None or str(chosen_value) == '':
                raise ValueError(f"GUI input '{gui_name}' is missing a value.")
            # process value
            raw_result = processor(gui_name, chosen_value)
            if isinstance(raw_result, tuple):
                model_name, result = raw_result
                processed_features_subset[model_name] = result
            elif isinstance(raw_result, dict):
                processed_features_subset.update(raw_result)
            else:
                raise TypeError(f"Processor returned an unrecognized type: {type(raw_result)}")
        
        return processed_features_subset

    def process_features(self,  window_values: Dict[str, Any]) -> np.ndarray:
        """
        Translates GUI values to a model-expected input array, returning a 1D numpy array.
        """
        # Stage 1: Process GUI inputs into a dictionary
        processed_features: Dict[str, Union[float,int]] = {}
        
        if self.master.has_continuous:
            processed_subset = self._call_subprocess(window_values=window_values,
                                                     master_feature=self.master.continuous, # type: ignore
                                                     processor=self._process_continuous)
            processed_features.update(processed_subset)
        
        if self.master.has_binary:
            processed_subset = self._call_subprocess(window_values=window_values,
                                                     master_feature=self.master.binary, # type: ignore
                                                     processor=self._process_binary)
            processed_features.update(processed_subset)
            
        if self.master.has_multi_binary:
            processed_subset = self._call_subprocess(window_values=window_values,
                                                     master_feature=self.master.multi_binary, # type: ignore
                                                     processor=self._process_multi_binary)
            processed_features.update(processed_subset)
        
        if self.master.has_one_hot:
            processed_subset = self._call_subprocess(window_values=window_values,
                                                     master_feature=self.master.one_hot, # type: ignore
                                                     processor=self._process_one_hot)
            processed_features.update(processed_subset)
            
        if self.master.has_categorical:
            processed_subset = self._call_subprocess(window_values=window_values,
                                                     master_feature=self.master.categorical, # type: ignore
                                                     processor=self._process_categorical)
            processed_features.update(processed_subset)

        # Stage 2: Assemble the final vector using the model's required order
        final_vector: List[float] = list()
        
        try:
            for feature_name in self.model_expected_features:
                final_vector.append(processed_features[feature_name])
        except KeyError as e:
            raise RuntimeError(f"Configuration Error: Implemented methods failed to generate the required model feature: '{e}'")
        
        return np.array(final_vector, dtype=np.float32)

def info():
    _script_info(__all__)
