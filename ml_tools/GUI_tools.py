import configparser
from pathlib import Path
from typing import Optional, Callable, Any
import traceback
import FreeSimpleGUI as sg
from functools import wraps
from typing import Any, Dict, Tuple, List
from .utilities import _script_info
import numpy as np
from .logger import _LOGGER


__all__ = [
    "PathManager", 
    "ConfigManager", 
    "GUIFactory",
    "catch_exceptions", 
    "prepare_feature_vector", 
    "update_target_fields"
]


# --- Path Management ---
class PathManager:
    """
    Manages paths for a Python application, supporting both development mode and bundled mode via Briefcase.
    """
    def __init__(self, anchor_file: str):
        """
        Initializes the PathManager. The package name is automatically inferred
        from the parent directory of the anchor file.

        Args:
            anchor_file (str): The absolute path to a file within the project's
                               package, typically `__file__` from a module inside
                               that package (paths.py).

        Note:
            This inference assumes that the anchor file's parent directory
            has the same name as the package (e.g., `.../src/my_app/paths.py`).
            This is a standard and recommended project structure.
        """
        resolved_anchor_path = Path(anchor_file).resolve()
        self.package_name = resolved_anchor_path.parent.name
        self._is_bundled, self._resource_path_func = self._check_bundle_status()

        if self._is_bundled:
            # In a Briefcase bundle, resource_path gives an absolute path
            # to the resource directory.
            self.package_root = self._resource_path_func(self.package_name, "") # type: ignore
        else:
            # In development mode, the package root is the directory
            # containing the anchor file.
            self.package_root = resolved_anchor_path.parent

    def _check_bundle_status(self) -> tuple[bool, Optional[Callable]]:
        """Checks if the app is running in a bundled environment."""
        try:
            # This is the function Briefcase provides in a bundled app
            from briefcase.platforms.base import resource_path # type: ignore
            return True, resource_path
        except ImportError:
            return False, None

    def get_path(self, relative_path: str | Path) -> Path:
        """
        Gets the absolute path for a given resource file or directory
        relative to the package root.

        Args:
            relative_path (str | Path): The path relative to the package root (e.g., 'helpers/icon.png').

        Returns:
            Path: The absolute path to the resource.
        """
        if self._is_bundled:
            # Briefcase's resource_path handles resolving the path within the app bundle 
            return self._resource_path_func(self.package_name, str(relative_path)) # type: ignore
        else:
            # In dev mode, join package root with the relative path.
            return self.package_root / relative_path


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

    def make_frame(self, title: str, layout: List[List[sg.Element]], **kwargs) -> sg.Frame:
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
        data_dict: Dict[str, Tuple[float, float]],
        is_target: bool = False,
        layout_mode: str = 'grid',
        columns_per_row: int = 4
    ) -> List[List[sg.Column]]:
        """
        Generates a layout for continuous features or targets.

        Args:
            data_dict (dict): Keys are feature names, values are (min, max) tuples.
            is_target (bool): If True, creates disabled inputs for displaying results.
            layout_mode (str): 'grid' for a multi-row grid layout, or 'row' for a single horizontal row.
            columns_per_row (int): Number of feature columns per row when layout_mode is 'grid'.

        Returns:
            A list of lists of sg.Column elements, ready to be used in a window layout.
        """
        cfg = self.config
        bg_color = sg.theme_background_color()
        label_font = (cfg.fonts.font_family, cfg.fonts.label_size, cfg.fonts.label_style) # type: ignore
        
        columns = []
        for name, (val_min, val_max) in data_dict.items():
            key = f"TARGET_{name}" if is_target else name
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
                range_text = sg.Text(f"Range: {int(val_min)}-{int(val_max)}", font=range_font, background_color=bg_color)
                layout = [[label], [element], [range_text]]
            
            layout.append([sg.Text(" ", font=(cfg.fonts.font_family, 2), background_color=bg_color)]) # type: ignore
            columns.append(sg.Column(layout, background_color=bg_color))

        if layout_mode == 'row':
            return [columns] # A single row containing all columns
        
        # Default to 'grid' layout
        return [columns[i:i + columns_per_row] for i in range(0, len(columns), columns_per_row)]

    def generate_combo_layout(
        self,
        data_dict: Dict[str, List[Any]],
        layout_mode: str = 'grid',
        columns_per_row: int = 4
    ) -> List[List[sg.Column]]:
        """
        Generates a layout for categorical or binary features using Combo boxes.

        Args:
            data_dict (dict): Keys are feature names, values are lists of options.
            layout_mode (str): 'grid' for a multi-row grid layout, or 'row' for a single horizontal row.
            columns_per_row (int): Number of feature columns per row when layout_mode is 'grid'.

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
            columns.append(sg.Column(layout, background_color=bg_color))

        if layout_mode == 'row':
            return [columns] # A single row containing all columns
            
        # Default to 'grid' layout
        return [columns[i:i + columns_per_row] for i in range(0, len(columns), columns_per_row)]

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


# --- Inference Helpers ---
def _default_categorical_processor(feature_name: str, chosen_value: Any) -> List[float]:
    """
    Default processor for binary 'True'/'False' strings.
    Returns a list containing a single float.
    """
    return [1.0] if str(chosen_value) == 'True' else [0.0]

def prepare_feature_vector(
    values: Dict[str, Any],
    feature_order: List[str],
    continuous_features: List[str],
    categorical_features: List[str],
    categorical_processor: Optional[Callable[[str, Any], List[float]]] = None
) -> np.ndarray:
    """
    Validates and converts GUI values into a numpy array for a model.
    This function supports label encoding and one-hot encoding via the processor.

    Args:
        values (dict): The values dictionary from a `window.read()` call.
        feature_order (list): A list of all feature names that have a GUI element.
                              For one-hot encoding, this should be the name of the
                              single GUI element (e.g., 'material_type'), not the
                              expanded feature names (e.g., 'material_is_steel').
        continuous_features (list): A list of names for continuous features.
        categorical_features (list): A list of names for categorical features.
        categorical_processor (callable, optional): A function to process categorical
            values. It should accept (feature_name, chosen_value) and return a
            list of floats (e.g., [1.0] for label encoding, [0.0, 1.0, 0.0] for one-hot).
            If None, a default 'True'/'False' processor is used.

    Returns:
        A 1D numpy array ready for model inference.
    """
    processed_values: List[float] = []
    
    # Use the provided processor or the default one
    processor = categorical_processor or _default_categorical_processor
    
    # Create sets for faster lookups
    cont_set = set(continuous_features)
    cat_set = set(categorical_features)

    for name in feature_order:
        chosen_value = values.get(name)
        
        if chosen_value is None or chosen_value == '':
            raise ValueError(f"Feature '{name}' is missing a value.")

        if name in cont_set:
            try:
                processed_values.append(float(chosen_value))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid input for '{name}'. Please enter a valid number.")
        
        elif name in cat_set:
            # The processor returns a list of values (one for label, multiple for one-hot)
            numeric_values = processor(name, chosen_value)
            processed_values.extend(numeric_values)
            
    return np.array(processed_values, dtype=np.float32)


def update_target_fields(window: sg.Window, results_dict: Dict[str, Any]):
    """
    Updates the GUI's target fields with inference results.

    Args:
        window (sg.Window): The application's window object.
        results_dict (dict): A dictionary where keys are target key names (including 'TARGET_' prefix if necessary) and values are the predicted results.
    """
    for target_name, result in results_dict.items():
        # Format numbers to 2 decimal places, leave other types as-is
        display_value = f"{result:.2f}" if isinstance(result, (int, float)) else result
        window[target_name].update(display_value)


def info():
    _script_info(__all__)
