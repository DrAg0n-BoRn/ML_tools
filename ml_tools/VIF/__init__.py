from ._VIF_factor import (
    compute_vif,
    drop_vif_based,
    compute_vif_multi,
)

from .._core import _imprimir_disponibles


__all__ = [
    "compute_vif",
    "drop_vif_based",
    "compute_vif_multi"
]


def info():
    _imprimir_disponibles(__all__)
