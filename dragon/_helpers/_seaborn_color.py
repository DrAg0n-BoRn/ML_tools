import seaborn as sns

from .._core import get_logger

_LOGGER = get_logger("Seaborn Palette")


def _get_consistent_palette(
    keys: list[str], 
    palette_name: str = "tab10"
) -> dict[str, tuple]:
    """
    Generates a consistent color palette mapping for a list of unique keys.
    Validates the requested palette and falls back to a default if invalid.
    
    This guarantees that the same target (e.g., dataset name) or category 
    always receives the exact same color across different plots.
    
    Args:
        keys (list[str]): A list of unique identifiers (e.g., dataset names, column categories).
        palette_name (str): The name of the matplotlib/seaborn color palette to use.
        
    Returns:
        dict[str, tuple]: A dictionary mapping each key to an RGB color tuple.
    """
    # Ensure keys are unique while preserving their original order
    unique_keys = list(dict.fromkeys(keys))
    n_colors = len(unique_keys)
    
    DEFAULT_PALETTE = "tab10"  # Fallback palette that should always be valid in Seaborn/Matplotlib
    
    try:
        # Try to generate the requested palette
        colors = sns.color_palette(palette_name, n_colors=n_colors)
    except ValueError:
        # Catch the exception raised by an invalid palette string
        _LOGGER.warning(
            f"Palette '{palette_name}' is not valid. Defaulting to '{DEFAULT_PALETTE}'."
        )
        # Generate the fallback palette
        colors = sns.color_palette(DEFAULT_PALETTE, n_colors=n_colors)
        
    # Create and return the mapping dictionary
    return dict(zip(unique_keys, colors))
