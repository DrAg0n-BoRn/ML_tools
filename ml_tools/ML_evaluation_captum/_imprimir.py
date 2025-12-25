from .._core import _imprimir_disponibles

_GRUPOS = [
    "captum_feature_importance", 
    "captum_image_heatmap",
    "captum_segmentation_heatmap"
]

def info():
    _imprimir_disponibles(_GRUPOS)
