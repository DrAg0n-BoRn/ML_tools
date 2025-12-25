from .._core import _imprimir_disponibles

_GRUPOS = [
    "ObjectiveFunction",
    "multiple_objective_functions_from_dir",
    "run_pso"
]

def info():
    _imprimir_disponibles(_GRUPOS)
