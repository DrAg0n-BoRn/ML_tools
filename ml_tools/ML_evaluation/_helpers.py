from ..keys._keys import _EvaluationConfig
from ..path_manager import sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Metrics Helper")


def check_and_abbreviate_name(name: str) -> str:
    """
    Checks if a name exceeds the NAME_LIMIT. If it does, creates an abbreviation 
    (initials of words) or truncates it if the abbreviation is empty.
    
    Args:
        name (str): The original label or target name.
        
    Returns:
        str: The potentially abbreviated name.
    """
    limit = _EvaluationConfig.NAME_LIMIT
    
    # Strip whitespace
    name = name.strip()
    
    if len(name) <= limit:
        return name
        
    # Attempt abbreviation: First letter of each word (split by space or underscore)
    parts = [w for w in name.replace("_", " ").split() if w]
    abbr = "".join(p[0].upper() for p in parts)
    
    # Keep only alphanumeric characters
    abbr = "".join(ch for ch in abbr if ch.isalnum())
    
    # Fallback if abbreviation failed or is empty
    if not abbr:
        sanitized = sanitize_filename(name)
        abbr = sanitized[:limit]
        
    _LOGGER.warning(f"Label '{name}' is too long. Abbreviating to '{abbr}'.")
    return abbr
