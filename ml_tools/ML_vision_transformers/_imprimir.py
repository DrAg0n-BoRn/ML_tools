from .._core import _imprimir_disponibles

_GRUPOS = [
    # Custom Transforms
    "ResizeAspectFill",
    "LetterboxResize",
    "HistogramEqualization",
    "RandomHistogramEqualization",
    # Offline Augmentation
    "create_offline_augmentations",
]

def info():
    _imprimir_disponibles(_GRUPOS)
