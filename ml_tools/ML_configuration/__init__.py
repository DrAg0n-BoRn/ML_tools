from ._config_metrics import (
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
    FormatAutoencoderMetrics,
    FormatTabularDiffusionMetrics
)

from ._config_finalize import (
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
    FinalizeAutoencoder,
    FinalizeTabularDiffusion
)

from ._config_models import (
    DragonMLPParams,
    DragonAttentionMLPParams,
    DragonMultiHeadAttentionNetParams,
    DragonTabularTransformerParams,
    DragonGateParams,
    DragonNodeParams,
    DragonTabNetParams,
    DragonAutoIntParams,
    DragonAutoencoderParams,
    DragonAutoencoderV2Params,
    DragonDiTParams,
    DragonDiTV2Params,
    DragonSequenceLSTMParams,
    DragonResNetParams,
    DragonEfficientNetParams,
    DragonVGGParams,
    DragonFCNParams,
    DragonDeepLabv3Params,
    DragonFastRCNNParams,
)

from ._config_training import (
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
    "FormatAutoencoderMetrics",
    "FormatTabularDiffusionMetrics",
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
    "FinalizeAutoencoder",
    "FinalizeTabularDiffusion",
    # --- Model Parameter Configs ---
    "DragonMLPParams",
    "DragonAttentionMLPParams",
    "DragonMultiHeadAttentionNetParams",
    "DragonTabularTransformerParams",
    "DragonGateParams",
    "DragonNodeParams",
    "DragonTabNetParams",
    "DragonAutoIntParams",
    "DragonAutoencoderParams",
    "DragonAutoencoderV2Params",
    "DragonDiTParams",
    "DragonDiTV2Params",
    "DragonSequenceLSTMParams",
    "DragonResNetParams",
    "DragonEfficientNetParams",
    "DragonVGGParams",
    "DragonFCNParams",
    "DragonDeepLabv3Params",
    "DragonFastRCNNParams",

    # --- Training Config ---
    "DragonTrainingConfig",
    "DragonParetoConfig",
    "DragonOptimizerConfig",
]


def info():
    _imprimir_disponibles(__all__)
