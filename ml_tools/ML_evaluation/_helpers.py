from ..keys._keys import _EvaluationConfig
from ..path_manager import sanitize_filename
from .._core import get_logger
import re


_LOGGER = get_logger("Metrics Helper")


def check_and_abbreviate_name(name: str) -> str:
    """
    Checks if a name exceeds the NAME_LIMIT. If it does, creates an abbreviation 
    by removing vowels from the words (keeping the first letter). If the result 
    still exceeds the limit, it falls back to initials, and finally truncates.
    
    Args:
        name (str): The original label or target name.
        
    Returns:
        str: The potentially abbreviated name.
    """
    limit = _EvaluationConfig.NAME_LIMIT

    name = name.strip()
    
    if len(name) <= limit:
        return name

    parts = [w for w in re.split(r'[\s_\-/]+', name) if w]
    
    # Attempt 1: Remove delimiters and TitleCase (PascalCase)
    abbr = "".join(p.capitalize() for p in parts if p.isalnum())
    if abbr and len(abbr) <= limit:
        return abbr

    # Attempt 2: Prefix abbreviation (First 3 letters of each word, TitleCase)
    abbr = "".join(p[:3].capitalize() for p in parts if p.isalnum())
    if abbr and len(abbr) <= limit:
        return abbr

    # Attempt 3: Vowel-stripping
    abbr_words = []
    for p in parts:
        if not p:
            continue
        first_char = p[0]
        rest = "".join(ch for ch in p[1:] if ch.isalnum() and ch.lower() not in 'aeiou')
        
        if first_char.isalnum():
            abbr_words.append(first_char.upper() + rest)
        elif rest:
            abbr_words.append(rest[0].upper() + rest[1:])
            
    abbr = "".join(abbr_words)
    if abbr and len(abbr) <= limit:
        return abbr
    
    # Attempt 4: Fallback to initials
    abbr = "".join(p[0].upper() for p in parts if p and p[0].isalnum())
    if abbr and len(abbr) <= limit:
        return abbr
        
    # Attempt 5: Safe fallback - Truncate
    sanitized = sanitize_filename(name)
    abbr = sanitized[:limit]
    
    # Warn if we use the last resort fallback
    _LOGGER.warning(f"Label '{name}' is too long. Abbreviating to '{abbr}'.")
    return abbr
