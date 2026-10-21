from typing import Any
from pathlib import Path
from collections.abc import Mapping

from ..schema import FeatureSchema


__all__ = [    
    "_BaseModelParams",
]


class _BaseModelParams(Mapping):
    """
    [PRIVATE] Base class for model parameter configs.
    
    Inherits from Mapping to behave like a dictionary, enabling
    `**params` unpacking directly into model constructors.
    """
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __or__(self, other) -> dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(self) | dict(other)
        return NotImplemented
    
    def __ror__(self, other) -> dict[str, Any]:
        """Allows merging with other Mappings using the | operator."""
        if isinstance(other, Mapping):
            return dict(other) | dict(self)
        return NotImplemented

    def __repr__(self) -> str:
        """Returns a formatted multi-line string representation."""
        class_name = self.__class__.__name__
        # Format parameters for clean logging
        params = []
        for k, v in self.__dict__.items():
            # If value is huge (like FeatureSchema), use its own repr
            val_str = repr(v)
            params.append(f"  {k}={val_str}")
            
        params_str = ",\n".join(params)
        return f"{class_name}(\n{params_str}\n)"
    
    def to_log(self) -> dict[str, Any]:
        """        
        Safely converts complex types (like FeatureSchema) to their string 
        representation for cleaner JSON logging.
        """
        clean_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FeatureSchema):
                # Force the repr() string, otherwise json.dump treats it as a list
                clean_dict[k] = repr(v)
            elif isinstance(v, Path):
                # JSON cannot serialize Path objects, convert to string
                clean_dict[k] = str(v)
            else:
                clean_dict[k] = v
        return clean_dict
