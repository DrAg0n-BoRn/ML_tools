
class ClassNameMixin:
    @property
    def class_name(self) -> str:
        """Returns the name of the class."""
        return self.__class__.__name__
