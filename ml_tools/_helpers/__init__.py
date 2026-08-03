from ._abbr_wrap import (
    check_and_abbreviate_name,
    wrap_text
)

from ._seaborn_color import (
    _get_consistent_palette
)

from ._validate_colors import (
    get_valid_matplotlib_color,
    get_valid_seaborn_color,
    get_valid_matplotlib_cmap,
    get_valid_seaborn_cmap,
)

__all__ = [
    "check_and_abbreviate_name",
    "wrap_text",
    "_get_consistent_palette",
    "get_valid_matplotlib_color",
    "get_valid_seaborn_color",
    "get_valid_matplotlib_cmap",
    "get_valid_seaborn_cmap",
]

