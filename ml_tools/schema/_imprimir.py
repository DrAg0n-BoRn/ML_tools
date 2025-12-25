from .._core import _imprimir_disponibles

_GRUPOS = [
    "FeatureSchema",
    "create_guischema_template",
    "make_multibinary_groups",
]

def info():
    _imprimir_disponibles(_GRUPOS)
