import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import platform

from .._core import get_logger


_LOGGER = get_logger("Plot Fonts")


__all__ = [
    "configure_cjk_fonts"
]


def configure_cjk_fonts(verbose: bool = True) -> None:
    """
    Configures Matplotlib to support CJK (Chinese, Japanese, Korean) characters 
    by detecting and setting an appropriate font based on the Operating System.

    Args:
        verbose (bool): If True, prints which font was selected.
    """
    system = platform.system()
    candidates = []

    # Define OS-specific priority lists for Simplified Chinese
    if system == "Darwin":  # macOS
        # PingFang SC is the standard modern UI font for Simplified Chinese on macOS
        candidates = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
    
    elif system == "Windows":
        # SimHei is the standard legacy; Microsoft YaHei is the modern interface font
        candidates = ["SimHei", "Microsoft YaHei", "SimSun"]
        
    else:  # Linux and others
        # Noto Sans CJK SC is standard on many server distros; WenQuanYi is a common fallback
        candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]

    # Global Fallbacks: Append these to the end regardless of OS to be safe
    candidates.extend(["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"])

    # Get set of available system font names
    system_fonts = {f.name for f in fm.fontManager.ttflist}

    found_font = None
    for font in candidates:
        if font in system_fonts:
            found_font = font
            break

    if found_font:
        # Prepend the found font to the existing sans-serif list
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
        
        # Fix negative sign display which is often broken in CJK fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        if verbose:
            _LOGGER.info(f"Matplotlib configured to use CJK font ({system}): '{found_font}'")
    else:
        if verbose:
            _LOGGER.warning(f"No suitable Simplified Chinese fonts found for {system}. Text may not render correctly.")

