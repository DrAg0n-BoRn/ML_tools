import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

from .._core import get_logger


_LOGGER = get_logger("Color Validator")


def get_valid_matplotlib_color(color: str) -> str:
    """Validates a single color string for Matplotlib."""
    if mcolors.is_color_like(color):
        return color
        
    _LOGGER.warning(
        f"Matplotlib color '{color}' is not valid. Defaulting to 'tab:blue'."
    )
    return "tab:blue"


def get_valid_seaborn_color(color: str) -> str:
    """
    Validates a single color string for Seaborn.
    Note: Seaborn relies on Matplotlib's engine for single colors.
    """
    if mcolors.is_color_like(color):
        return color
        
    _LOGGER.warning(
        f"Seaborn color '{color}' is not valid. Defaulting to 'tab:blue'."
    )
    return "tab:blue"


def get_valid_matplotlib_cmap(cmap: str) -> str:
    """
    Validates a colormap string for Matplotlib.
    Falls back to 'viridis' because Seaborn-specific strings (like 'husl') 
    are not registered in Matplotlib.
    """
    # Check against Matplotlib's registered colormap strings
    if cmap in plt.colormaps():
        return cmap
        
    _LOGGER.warning(
        f"Matplotlib colormap '{cmap}' is not valid. Defaulting to 'viridis'."
    )
    return "viridis"


def get_valid_seaborn_cmap(cmap: str) -> str:
    """
    Validates a colormap/palette string for Seaborn.
    Falls back to 'husl'.
    """
    try:
        # Validate by attempting to generate the Seaborn palette
        sns.color_palette(cmap)
        return cmap
    except ValueError:
        # Catch the exception raised by an invalid palette string
        _LOGGER.warning(
            f"Seaborn colormap '{cmap}' is not valid. Defaulting to 'husl'."
        )
        return "husl"
    