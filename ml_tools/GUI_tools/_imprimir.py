from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonGUIConfig", 
    "DragonGUIFactory",
    "DragonFeatureMaster",
    "DragonGUIHandler",
    "catch_exceptions"
]

def info():
    _imprimir_disponibles(_GRUPOS)
