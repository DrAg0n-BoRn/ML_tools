from ._plot_fonts import (
    configure_cjk_fonts
)

from .._core import _imprimir_disponibles


__all__ = [
    "configure_cjk_fonts"
]


def info():
    _imprimir_disponibles(__all__)
