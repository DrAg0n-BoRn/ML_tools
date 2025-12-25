from .._core import _imprimir_disponibles

_GRUPOS = [
    "summarize_dataframe",
    "show_null_columns",
    "drop_constant_columns",
    "drop_rows_with_missing_data",
    "drop_columns_with_missing_data",
    "drop_macro",
    "clean_column_names",
    "plot_value_distributions",
    "split_features_targets", 
    "split_continuous_binary", 
    "split_continuous_categorical_targets",
    "clip_outliers_single", 
    "clip_outliers_multi",
    "drop_outlier_samples",
    "plot_continuous_vs_target",
    "plot_categorical_vs_target",
    "plot_correlation_heatmap", 
    "encode_categorical_features",
    "finalize_feature_schema",
    "apply_feature_schema",
    "match_and_filter_columns_by_regex",
    "standardize_percentages",
    "reconstruct_one_hot",
    "reconstruct_binary",
    "reconstruct_multibinary",
]

def info():
    _imprimir_disponibles(_GRUPOS)
