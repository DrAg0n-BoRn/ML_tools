from ._isolation_forest import (
    isolation_forest
)

from ._lof import (
    local_outlier_factor
)

from ._handler import (
    clip_outliers_single,
    clip_outliers_multi,
    drop_outliers_rule,
    drop_outliers_mask,
    replace_outliers_mask
)


from .._core import _imprimir_disponibles


__all__ = [
    "isolation_forest",
    "local_outlier_factor",
    "clip_outliers_single",
    "clip_outliers_multi",
    "drop_outliers_rule",
    "drop_outliers_mask",
    "replace_outliers_mask",
]


def info():
    _imprimir_disponibles(__all__)
