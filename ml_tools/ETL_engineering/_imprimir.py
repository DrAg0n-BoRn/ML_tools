from .._core import _imprimir_disponibles

_GRUPOS = [
    "DragonTransformRecipe",
    "DragonProcessor",
    "BinaryTransformer",
    "MultiBinaryDummifier",
    "AutoDummifier",
    "KeywordDummifier",
    "NumberExtractor",
    "MultiNumberExtractor",
    "TemperatureExtractor",
    "MultiTemperatureExtractor",
    "RatioCalculator",
    "TriRatioCalculator",
    "CategoryMapper",
    "RegexMapper",
    "ValueBinner",
    "DateFeatureExtractor",
    "MolecularFormulaTransformer"
]

def info():
    _imprimir_disponibles(_GRUPOS)
