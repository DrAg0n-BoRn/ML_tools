from typing import Any, Callable

class _classproperty:
    """Descriptor that evaluates a method as a property on a class."""
    def __init__(self, fget: Callable[[Any], str]):
        self.fget = fget

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> str:
        if owner is None:
            owner = type(instance)
        return self.fget(owner)


class ClassNameMixin:
    @_classproperty
    def class_name(cls: Any) -> str:
        """Returns the name of the class."""
        return cls.__name__
