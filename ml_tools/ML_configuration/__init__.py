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
    FormatAutoregressiveSequenceValueMetrics,
    FormatAutoregressiveSequenceSequenceMetrics,
    FormatExogenousSequenceValueMetrics,
    FormatExogenousSequenceSequenceMetrics,
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
    FinalizeAutoregressiveSequenceSequence,
    FinalizeAutoregressiveSequenceValue,
    FinalizeExogenousSequenceSequence,
    FinalizeExogenousSequenceValue,
    FinalizeAutoencoder,
    FinalizeTabularDiffusion
)

from ._config_training import (
    DragonTrainingConfig,
)

from ._config_optimization import (
    DragonParetoConfig,
    DragonOptimizerConfig
)

from ._config_checkpoint import (
    DragonCheckpointConfig
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
    "FormatAutoregressiveSequenceValueMetrics",
    "FormatAutoregressiveSequenceSequenceMetrics",
    "FormatExogenousSequenceValueMetrics",
    "FormatExogenousSequenceSequenceMetrics",
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
    "FinalizeAutoregressiveSequenceSequence",
    "FinalizeAutoregressiveSequenceValue",
    "FinalizeExogenousSequenceSequence",
    "FinalizeExogenousSequenceValue",
    "FinalizeAutoencoder",
    "FinalizeTabularDiffusion",

    # --- Training Config ---
    "DragonTrainingConfig",
    # --- Optimization Config ---
    "DragonParetoConfig",
    "DragonOptimizerConfig",
    # --- Checkpoint Config ---
    "DragonCheckpointConfig"
]


def info():
    _imprimir_disponibles(__all__)
