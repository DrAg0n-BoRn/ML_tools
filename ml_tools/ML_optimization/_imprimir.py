from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonParetoOptimizer",
    "DragonOptimizer",
    # manual optimization tools
    "FitnessEvaluator",
    "create_pytorch_problem",
    "run_optimization",
]

def info():
    _imprimir_disponibles(_GRUPOS)
