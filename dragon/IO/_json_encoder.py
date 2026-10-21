import json
from datetime import date, datetime
from uuid import UUID
from pathlib import Path
from enum import Enum


__all__ = [
    "TrulyRobustEncoder"
]


class TrulyRobustEncoder(json.JSONEncoder):
    """
    A robust JSON encoder that natively handles common problematic types 
    while preventing crashes on unknown objects.
    """
    def default(self, o):
        # 1. Explicitly handle common non-serializable types first
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return o.value
            
        # 2. Handle the specific `type` case from your original code
        if isinstance(o, type):
            return o.__name__  # Returns 'int' instead of the messy "<class 'int'>"

        # 3. Attempt standard serialization
        try:
            return super().default(o)
        except TypeError:
            # 4. Absolute last resort for unknown custom objects
            return repr(o)
