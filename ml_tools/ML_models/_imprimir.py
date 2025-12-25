from .._core import _imprimir_disponibles

_GRUPOS = [
    # MLP and Attention Models
    "DragonMLP",
    "DragonAttentionMLP",
    "DragonMultiHeadAttentionNet",
    # Tabular Transformer Model
    "DragonTabularTransformer",
    # Advanced Models
    "DragonGateModel",
    "DragonNodeModel",
    "DragonAutoInt",
    "DragonTabNet",
]

def info():
    _imprimir_disponibles(_GRUPOS)
