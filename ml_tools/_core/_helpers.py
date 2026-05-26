import re
import textwrap

from ..keys._keys import _EvaluationConfig



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
    
    # Attempt 2.1: Prefix abbreviation (First 2 letters of each word, TitleCase)
    abbr = "".join(p[:2].capitalize() for p in parts if p.isalnum())
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
    abbr = name[:limit]
    
    return abbr


def wrap_text(text: str, width: int=_EvaluationConfig.NAME_LIMIT, break_char: str = "\n") -> str:
    """
    Wraps text to a specified width while keeping trailing numbers attached to their base words.
    Use break_char="<br>" for Plotly and break_char="\n" for Matplotlib.
    """
    clean_text = str(text).strip()
    
    # Protect underscores or spaces immediately followed by a digit 
    # by temporarily replacing them with a null character (\x00)
    clean_text = re.sub(r'[ _](?=\d)', '\x00', clean_text)
    
    # Replace remaining standard separators with spaces so textwrap can find logical break points
    clean_text = clean_text.replace('_', ' ')
    
    # Wrap the text. break_long_words=False prevents aggressively chopping words in half.
    wrapped_list = textwrap.wrap(clean_text, width=width, break_long_words=False)
    
    # Restore the protected characters as standard spaces
    wrapped_list = [line.replace('\x00', ' ') for line in wrapped_list]
    
    return break_char.join(wrapped_list)
