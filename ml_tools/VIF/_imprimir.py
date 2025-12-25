from .._core import _imprimir_disponibles

_GRUPOS = [
    "compute_vif",
    "drop_vif_based",
    "compute_vif_multi"
]

def info():
    _imprimir_disponibles(_GRUPOS)
