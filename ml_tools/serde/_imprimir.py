from .._core import _imprimir_disponibles

_GRUPOS = [
    "serialize_object_filename",
    "serialize_object",
    "deserialize_object",
]

def info():
    _imprimir_disponibles(_GRUPOS)
