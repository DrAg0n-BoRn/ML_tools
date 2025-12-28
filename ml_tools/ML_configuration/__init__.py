from ._metrics import (
    FormatRegressionMetrics,
    FormatMultiTargetRegressionMetrics,
    FormatBinaryClassificationMetrics,
    FormatMultiClassClassificationMetrics,
    FormatBinaryImageClassificationMetrics,
    FormatMultiClassImageClassificationMetrics,
    FormatMultiLabelBinaryClassificationMetrics,
    FormatBinarySegmentationMetrics,
    FormatMultiClassSegmentationMetrics,
    FormatSequenceValueMetrics,
    FormatSequenceSequenceMetrics,
)

from ._finalize import (
    FinalizeBinaryClassification,
    FinalizeBinarySegmentation,
    FinalizeBinaryImageClassification,
    FinalizeMultiClassClassification,
    FinalizeMultiClassImageClassification,
    FinalizeMultiClassSegmentation,
    FinalizeMultiLabelBinaryClassification,
    FinalizeMultiTargetRegression,
    FinalizeRegression,
    FinalizeObjectDetection,
    FinalizeSequenceSequencePrediction,
    FinalizeSequenceValuePrediction,
)

from ._models import (
    DragonMLPParams,
    DragonAttentionMLPParams,
    DragonMultiHeadAttentionNetParams,
    DragonTabularTransformerParams,
    DragonGateParams,
    DragonNodeParams,
    DragonTabNetParams,
    DragonAutoIntParams,
)

from ._training import (
    DragonTrainingConfig,
    DragonParetoConfig,
    DragonOptimizerConfig
)

from .._core import _imprimir_disponibles


__all__ = [
    # --- Metrics Formats ---
    "FormatRegressionMetrics",
    "FormatMultiTargetRegressionMetrics",
    "FormatBinaryClassificationMetrics",
    "FormatMultiClassClassificationMetrics",
    "FormatBinaryImageClassificationMetrics",
    "FormatMultiClassImageClassificationMetrics",
    "FormatMultiLabelBinaryClassificationMetrics",
    "FormatBinarySegmentationMetrics",
    "FormatMultiClassSegmentationMetrics",
    "FormatSequenceValueMetrics",
    "FormatSequenceSequenceMetrics",
    
    # --- Finalize Configs ---
    "FinalizeBinaryClassification",
    "FinalizeBinarySegmentation",
    "FinalizeBinaryImageClassification",
    "FinalizeMultiClassClassification",
    "FinalizeMultiClassImageClassification",
    "FinalizeMultiClassSegmentation",
    "FinalizeMultiLabelBinaryClassification",
    "FinalizeMultiTargetRegression",
    "FinalizeRegression",
    "FinalizeObjectDetection",
    "FinalizeSequenceSequencePrediction",
    "FinalizeSequenceValuePrediction",
    
    # --- Model Parameter Configs ---
    "DragonMLPParams",
    "DragonAttentionMLPParams",
    "DragonMultiHeadAttentionNetParams",
    "DragonTabularTransformerParams",
    "DragonGateParams",
    "DragonNodeParams",
    "DragonTabNetParams",
    "DragonAutoIntParams",
    
    # --- Training Config ---
    "DragonTrainingConfig",
    "DragonParetoConfig",
    "DragonOptimizerConfig",
]


def info():
    _imprimir_disponibles(__all__)
