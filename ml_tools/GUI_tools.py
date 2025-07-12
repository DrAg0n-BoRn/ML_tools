import configparser
from pathlib import Path
import traceback
import FreeSimpleGUI as sg
from functools import wraps
from typing import Any, Dict, Tuple, List, Literal, Union, Any, Optional
from .utilities import _script_info
import numpy as np
from .logger import _LOGGER
from abc import ABC, abstractmethod


__all__ = [
    "ConfigManager", 
    "GUIFactory",
    "catch_exceptions", 
    "BaseFeatureHandler", 
    "update_target_fields"
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
            '; Default size for continuous input boxes (width,height in characters).': '',
            'input_size_cont': '16,1',
            '; Default size for combo/binary boxes (width,height in characters).': '',
            'input_size_binary': '14,1',
            '; Default size for buttons (width,height in characters).': '',
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
        data_dict: Dict[str, Optional[Tuple[Union[int,float], Union[int,float]]]],
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
                val_min, val_max = None, None
                if not is_target:
                    raise ValueError(f"Feature '{name}' was assigned a 'None' value. It is not defined as a target.")
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
        data_dict: Dict[str, List[Any]],
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


# --- Inference Helper ---
class BaseFeatureHandler(ABC):
    """
    An abstract base class that defines the template for preparing a model input feature vector to perform inference, from GUI inputs.

    A subclass must implement the `gui_input_map` property and the `process_categorical` method.
    """
    def __init__(self, expected_columns_in_order: list[str]):
        """
        Validates and stores the feature names in the order the model expects.
        
        Args:
            expected_columns_in_order (List[str]): A list of strings with the feature names in the correct order.
        """
        # --- Validation Logic ---
        if not isinstance(expected_columns_in_order, list):
            raise TypeError("Input 'expected_columns_in_order' must be a list.")
            
        if not all(isinstance(col, str) for col in expected_columns_in_order):
            raise TypeError("All elements in the 'expected_columns_in_order' list must be strings.")
        # -----------------------
        
        self._model_feature_order = expected_columns_in_order
        
    @property
    @abstractmethod
    def gui_input_map(self) -> Dict[str, Literal["continuous","categorical"]]:
        """
        Must be implemented by the subclass.

        Should return a dictionary mapping each GUI input name to its type ('continuous' or 'categorical').
        
        _Example:_
        ```python
        {
            'Temperature': 'continuous', 
            'Material Type': 'categorical'
        }
        ```
        """
        pass
    
    @property
    @abstractmethod
    def map_gui_to_real(self) -> Dict[str,str]:
        """
        Must be implemented by the subclass.

        Should return a dictionary mapping each GUI continuous feature name to its expected model feature name.
        
        _Example:_
        ```python
        {
            'Temperature (K)': 'temperature_k', 
            'Pressure (Pa)': 'pressure_pa'
        }
        ```
        """
        pass

    @abstractmethod
    def process_categorical(self, gui_feature_name: str, chosen_value: Any) -> Dict[str, float]:
        """
        Must be implemented by the subclass.

        Should take a GUI categorical feature name and its chosen value, and return a dictionary mapping the one-hot-encoded/binary real feature names to their
        float values (as expected by the inference model).
        
        _Example:_
        ```python        
        # GUI input: "Material Type"
        # GUI values: "Steel", "Aluminum", "Titanium"
        {
            "is_steel": 0, 
            "is_aluminum": 1,
            "is_titanium": 0,
        }
        ```
        """
        pass
    
    def _process_continuous(self, gui_feature_name: str, chosen_value: Any) -> Tuple[str, float]:
        """
        Maps GUI names to model expected names and casts the value to float.
        
        Should not be overridden by subclasses.
        """
        try:
            real_name = self.map_gui_to_real[gui_feature_name]
            float_value = float(chosen_value)
        except KeyError as e:
            _LOGGER.error(f"No matching name for '{gui_feature_name}'. Check the 'map_gui_to_real' implementation.")
            raise e
        except (ValueError, TypeError) as e2:
            _LOGGER.error(f"Invalid number conversion for '{chosen_value}' of '{gui_feature_name}'.")
            raise e2
        else:
            return real_name, float_value
    
    def __call__(self, window_values: Dict[str, Any]) -> np.ndarray:
        """
        Performs the full vector preparation, returning a 1D numpy array.
        
        Should not be overridden by subclasses.
        """
        # Stage 1: Process GUI inputs into a dictionary
        processed_features: Dict[str, float] = {}
        for gui_name, feature_type in self.gui_input_map.items():
            chosen_value = window_values.get(gui_name)
            
            # value validation
            if chosen_value is None or str(chosen_value) == '':
                raise ValueError(f"GUI input '{gui_name}' is missing a value.")

            # process continuous
            if feature_type == 'continuous':
                mapped_name, float_value = self._process_continuous(gui_name, chosen_value)
                processed_features[mapped_name] = float_value
            
            # process categorical
            elif feature_type == 'categorical':
                feature_dict = self.process_categorical(gui_name, chosen_value)
                processed_features.update(feature_dict)

        # Stage 2: Assemble the final vector using the model's required order
        final_vector: List[float] = []
        
        try:
            for feature_name in self._model_feature_order:
                final_vector.append(processed_features[feature_name])
        except KeyError as e:
            raise RuntimeError(
                f"Configuration Error: Implemented methods failed to generate "
                f"the required model feature: '{e}'"
                f"Check the gui_input_map and process_categorical logic."
            )
            
        return np.array(final_vector, dtype=np.float32)


def update_target_fields(window: sg.Window, results_dict: Dict[str, Any]):
    """
    Updates the GUI's target fields with inference results.

    Args:
        window (sg.Window): The application's window object.
        results_dict (dict): A dictionary where keys are target element-keys and values are the predicted results to update.
    """
    for target_name, result in results_dict.items():
        # Format numbers to 2 decimal places, leave other types as-is
        display_value = f"{result:.2f}" if isinstance(result, (int, float)) else result
        window[target_name].update(display_value) # type: ignore


def info():
    _script_info(__all__)
