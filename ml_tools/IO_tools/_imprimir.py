from .._core import _imprimir_disponibles

_GRUPOS = [
    "custom_logger",
    "train_logger",
    "save_json",
    "load_json",
    "save_list_strings",
    "load_list_strings",
    "compare_lists"
]

def info():
    _imprimir_disponibles(_GRUPOS)
