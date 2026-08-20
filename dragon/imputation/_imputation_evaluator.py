import pandas as pd
import numpy as np
from typing import Union, Optional, Literal
from pathlib import Path
from scipy.stats import wasserstein_distance, ks_2samp

from ..schema._feature_schema import FeatureSchema
from ..utilities import load_dataframe, save_dataframe_filename
from ..data_exploration import plot_value_distributions_multi, show_null_columns

from .._core import get_logger


_LOGGER = get_logger("Imputation Evaluator")


__all__ = ["DragonImputationEvaluator"]


class DragonImputationEvaluator:
    """
    Evaluates the results of missing data imputation by comparing the original 
    DataFrame against the imputed DataFrame.
    """
    def __init__(
        self,
        df_original: Union[pd.DataFrame, Path, str],
        df_imputed: Union[pd.DataFrame, Path, str],
        categorical_schema: Union[FeatureSchema, list[int], list[str], None] = None
    ):
        """
        Initializes the DragonImputationEvaluator with the original and imputed DataFrames.
        
        Validates that the datasets have identical shapes and columns. It also parses 
        the provided categorical_schema to distinguish between continuous and categorical features, 
        and identifies which columns had missing values that were subsequently imputed.
        
        Args:
            df_original (Union[pd.DataFrame, Path, str]): The original DataFrame with missing values.
            df_imputed (Union[pd.DataFrame, Path, str]): The DataFrame after imputation, which should have the same shape and columns as df_original.
            categorical_schema (Union[FeatureSchema, list[int], list[str], None]): Schema indicating which features are categorical. Can be: 
                - FeatureSchema object
                - list of categorical column indices
                - list of categorical column names
                - None (if no categorical columns are present)
        """
        if isinstance(df_original, (Path, str)):
            df_original_validated, _ = load_dataframe(df_original, kind="pandas", verbose=False)
        else:
            df_original_validated = df_original.copy()
            
        if isinstance(df_imputed, (Path, str)):
            df_imputed_validated, _ = load_dataframe(df_imputed, kind="pandas", verbose=False)
        else:
            df_imputed_validated = df_imputed.copy()
        
        self.df_original = df_original_validated
        self.df_imputed = df_imputed_validated
        
        
        # 1. Structural Validation
        if self.df_original.shape != self.df_imputed.shape:
            _LOGGER.error("Original and Imputed DataFrames must have the exact same shape.")
            raise ValueError()
            
        if not self.df_original.columns.equals(self.df_imputed.columns):
            _LOGGER.error("Original and Imputed DataFrames must have the exact same columns.")
            raise ValueError()
        
        # report loaded DataFrame shapes
        _LOGGER.info(f"Loaded DataFrames:\n\t- Original DataFrame shape: {self.df_original.shape}\n\t- Imputed DataFrame shape: {self.df_imputed.shape}")

        # 2. Parse Categorical Schema and Categorical Features
        self.categorical_cols = self._validate_schema(categorical_schema)
        self.continuous_cols = [
            c for c in self.df_original.columns if c not in self.categorical_cols
        ]

        # 3. State Tracking
        # Boolean mask where True indicates a missing value in the original data
        self.missing_mask = self.df_original.isnull()
        
        # Identify columns that actually had missing values imputed
        self.imputed_cols: list[str] = self.missing_mask.columns[self.missing_mask.any()].tolist()

    def _validate_schema(self, cat_schema: Union[FeatureSchema, list[int], list[str], None]) -> list[str]:
        """
        Parses the provided categorical schema and returns a list of categorical column names.
        Also validates that categorical columns in the imputed dataset are properly formatted.
        """
        cols = self.df_original.columns.tolist()
        
        if cat_schema is None:
            return []
            
        categorical_names = []
        
        # Handle FeatureSchema object
        if isinstance(cat_schema, FeatureSchema):
            categorical_names = list(cat_schema.categorical_feature_names)
            
        # Handle lists (integers or strings)
        elif isinstance(cat_schema, list):
            if len(cat_schema) == 0:
                return []
            
            if all(isinstance(x, int) for x in cat_schema):
                try:
                    categorical_names: list[str] = [cols[i] for i in cat_schema] # type: ignore
                except IndexError:
                    _LOGGER.error("One or more column indices in the provided 'categorical_schema are out of bounds.")
                    raise ValueError()
                    
            elif all(isinstance(x, str) for x in cat_schema):
                missing_cols = [c for c in cat_schema if c not in cols]
                if missing_cols:
                    _LOGGER.error(f"Categorical columns declared in the 'categorical_schema' not found in DataFrame: {missing_cols}")
                    raise ValueError()
                categorical_names: list[str] = cat_schema  # type: ignore
            else:
                _LOGGER.error("Categorical schema list must contain either all integers or all strings.")
                raise TypeError()
        else:
            _LOGGER.error("Categorical schema must be a FeatureSchema, list of ints, list of strings, or None.")
            raise TypeError()

        # Validation: check if categorical columns in imputed df are integers where expected
        for cat_col in categorical_names:
            if not pd.api.types.is_integer_dtype(self.df_imputed[cat_col]):
                _LOGGER.warning(
                    f"Categorical column '{cat_col}' in imputed DataFrame is not an integer type. "
                    f"Current type: {self.df_imputed[cat_col].dtype}. Consider rounding/casting."
                )

        return categorical_names

    def plot_missing_overview(
        self,
        save_dir: Union[str, Path],
        plot_filename: str = "Original Dataset",
        use_all_columns: bool = False,
        round_digits: int = 2
    ) -> pd.DataFrame:
        """
        Plots and summarizes the missing data profile of the original dataset.
        
        Args:
            save_dir (Union[str, Path]): Directory to save the plot.
            plot_filename (str): Title for the plot, also used for the saved plot filename.
            use_all_columns (bool): If True, include all columns in the summary; otherwise, only columns with missing values.
            round_digits (int): Number of decimal places to round percentages in the summary.
        
        Returns:
            pd.DataFrame: A DataFrame summarizing missing values in each column.
        
        """
        return show_null_columns(
            df=self.df_original,
            round_digits=round_digits,
            plot_to_dir=save_dir,
            plot_filename=plot_filename,
            use_all_columns=use_all_columns
        )

    def plot_overall_distributions(
        self,
        save_dir: Union[str, Path],
        columns: Optional[list[str]] = None,
        only_imputed_columns: bool = True,
        max_categories: int = 20,
        font_scaling: float = 1.5,
        mode: Literal["count", "percentage"] = "percentage",
        palette: str = "tab10"
    ) -> None:
        """
        Compares the total distribution of original data against the full imputed dataset.
        
        Args:
            save_dir (str | Path): Directory path to save the plots.
            columns (list[str] | None): Specific columns to plot. If None, all columns are considered. Overrides `only_imputed_columns` if provided.
            only_imputed_columns (bool): If True, only columns that had missing values imputed will be plotted.
            max_categories (int): The maximum number of unique categories a categorical feature can have to be plotted.
            font_scaling (float): Scaling factor for all fonts in the generated plots.
            mode (Literal["count", "percentage"]): Whether to plot absolute counts or relative percentages.
            palette (str): The name of the matplotlib/seaborn color palette to use for differentiating categories.
        
        <br>
    
        ### [Seaborn Color Palettes](https://www.practicalpythonfordatascience.com/ap_seaborn_palette)
        """
        target_cols = self._get_target_columns(columns, only_imputed_columns)
        if not target_cols:
            _LOGGER.warning("No columns to plot for overall distribution comparison.")
            return

        named_dfs = {
            "Original": self.df_original[target_cols],
            "Imputed": self.df_imputed[target_cols]
        }

        plot_value_distributions_multi(
            named_dataframes=named_dfs,
            save_dir=save_dir,
            max_categories=max_categories,
            font_scaling=font_scaling,
            mode=mode,
            palette=palette
        )

    def plot_imputed_vs_observed(
        self,
        save_dir: Union[str, Path],
        columns: Optional[list[str]] = None,
        max_categories: int = 20,
        font_scaling: float = 1.5,
        mode: Literal["count", "percentage"] = "percentage",
        palette: str = "tab10"
    ) -> None:
        """
        Compares the observed (naturally present) values against the newly imputed 
        (synthesized) values exclusively.
        
        Args:
            save_dir (str | Path): Directory path to save the plots.
            columns (list[str] | None): Specific columns to plot. If None, all columns with missing values are considered.
            max_categories (int): The maximum number of unique categories a categorical feature can have to be plotted.
            font_scaling (float): Scaling factor for all fonts in the generated plots.
            mode (Literal["count", "percentage"]): Whether to plot absolute counts or relative percentages.
            palette (str): The name of the matplotlib/seaborn color palette to use for differentiating categories.
        
        <br>
        
        ### [Seaborn Color Palettes](https://www.practicalpythonfordatascience.com/ap_seaborn_palette)
        """
        target_cols = self._get_target_columns(columns, only_imputed_columns=True)
        if not target_cols:
            _LOGGER.warning("No columns with missing values found to compare observed vs imputed.")
            return

        # df_original naturally contains only the observed values
        df_observed = self.df_original[target_cols]
        
        # Mask df_imputed to keep only the values that were originally missing
        df_imputed_only = self.df_imputed[target_cols].where(self.missing_mask[target_cols], np.nan)

        named_dfs = {
            "Observed": df_observed,
            "Imputed Values Only": df_imputed_only
        }

        plot_value_distributions_multi(
            named_dataframes=named_dfs,
            save_dir=save_dir,
            max_categories=max_categories,
            font_scaling=font_scaling,
            mode=mode,
            palette=palette
        )

    def _get_target_columns(
        self, 
        columns: Optional[list[str]], 
        only_imputed_columns: bool
    ) -> list[str]:
        """
        Helper method to resolve which columns should be included in plotting.
        """
        if columns is not None:
            invalid_cols = [c for c in columns if c not in self.df_original.columns]
            if invalid_cols:
                _LOGGER.error(f"Columns not found in DataFrame: {invalid_cols}")
                raise ValueError()
            return columns
            
        if only_imputed_columns:
            return self.imputed_cols
            
        return self.df_original.columns.tolist()
    
    def evaluate_continuous(self, save_dir: Union[Path, str]) -> pd.DataFrame:
        """
        Calculates distribution and statistical metrics comparing the observed 
        values against the imputed values for continuous columns.
        
        Args:
            save_dir (Union[Path, str]): Directory to save the metrics DataFrame as a CSV file.
        
        Returns:
            pd.DataFrame: A DataFrame containing metrics for each continuous feature.
        """
        metrics = []
        target_cols = [c for c in self.imputed_cols if c in self.continuous_cols]
        
        if not target_cols:
            _LOGGER.warning("No continuous columns with missing values found for metric calculation. Returning empty DataFrame.")
            return pd.DataFrame(columns=[
                "Feature", "Observed Mean", "Imputed Mean", "Mean Shift", 
                "Observed Std", "Imputed Std", "Std Shift", 
                "Wasserstein Distance", "KS Statistic", "KS p-value"
            ]).set_index("Feature")
        
        for col in target_cols:
            obs_vals = self.df_original[col].dropna()
            imp_vals = self.df_imputed.loc[self.missing_mask[col], col]
            
            if obs_vals.empty or imp_vals.empty:
                continue
                
            obs_mean, imp_mean = obs_vals.mean(), imp_vals.mean()
            obs_std, imp_std = obs_vals.std(), imp_vals.std()
            
            # Wasserstein Distance (Earth Mover's Distance)
            wd = wasserstein_distance(obs_vals, imp_vals)
            
            # Kolmogorov-Smirnov Test (Distribution similarity)
            ks_stat, p_value = ks_2samp(obs_vals, imp_vals)
            
            metrics.append({
                "Feature": col,
                "Observed Mean": obs_mean,
                "Imputed Mean": imp_mean,
                "Mean Shift": abs(obs_mean - imp_mean),
                "Observed Std": obs_std,
                "Imputed Std": imp_std,
                "Std Shift": abs(obs_std - imp_std),
                "Wasserstein Distance": wd,
                "KS Statistic": ks_stat,
                "KS p-value": p_value
            })
        
        df_metrics = pd.DataFrame(metrics)
        
        try:
            save_dataframe_filename(
                df=df_metrics,
                save_dir=save_dir,
                filename="continuous_metrics_imputation.csv",
                verbose=1
            )
        except Exception as e:
            _LOGGER.error(f"Failed to save continuous metrics DataFrame: {e}")
        else:
            _LOGGER.info(f"Continuous metrics DataFrame saved successfully to '{save_dir}'.")
            
        return df_metrics.set_index("Feature")

    def evaluate_categorical(self, save_dir: Union[Path, str]) -> pd.DataFrame:
        """
        Calculates distribution metrics comparing the observed values against 
        the imputed values for categorical columns.
        
        Args:
            save_dir (Union[Path, str]): Directory to save the metrics DataFrame as a CSV file.
        
        Returns:
            pd.DataFrame: A DataFrame containing metrics for each categorical feature.
        """
        metrics = []
        target_cols = [c for c in self.imputed_cols if c in self.categorical_cols]
        
        if not target_cols:
            _LOGGER.warning("No categorical columns with missing values found for metric calculation. Returning empty DataFrame.")
            return pd.DataFrame(columns=[
                "Feature", "Total Categories", "Total Variation Distance"
            ]).set_index("Feature")
        
        for col in target_cols:
            obs_vals = self.df_original[col].dropna()
            imp_vals = self.df_imputed.loc[self.missing_mask[col], col]
            
            if obs_vals.empty or imp_vals.empty:
                continue
                
            obs_freq = obs_vals.value_counts(normalize=True)
            imp_freq = imp_vals.value_counts(normalize=True)
            
            # Align indices using pandas native align method
            obs_freq, imp_freq = obs_freq.align(imp_freq, fill_value=0)
            
            # Total Variation Distance (TVD)
            tvd = 0.5 * np.sum(np.abs(obs_freq - imp_freq))
            
            metrics.append({
                "Feature": col,
                "Total Categories": len(obs_freq),
                "Total Variation Distance": tvd
            })
        
        df_metrics = pd.DataFrame(metrics)
        
        try:
            save_dataframe_filename(
                df=df_metrics,
                save_dir=save_dir,
                filename="categorical_metrics_imputation.csv",
                verbose=1
            )
        except Exception as e:
            _LOGGER.error(f"Failed to save categorical metrics DataFrame: {e}")
        else:
            _LOGGER.info(f"Categorical metrics DataFrame saved successfully to '{save_dir}'.")
            
        return df_metrics.set_index("Feature")

    def __repr__(self) -> str:
        """Returns a concise string representation of the evaluator's state."""
        total_cols = len(self.df_original.columns)
        n_imputed = len(self.imputed_cols)
        n_cat = len(self.categorical_cols)
        n_cont = len(self.continuous_cols)
        return (
            f"DragonImputationEvaluator("
            f"total_features={total_cols}, "
            f"imputed_features={n_imputed}, "
            f"continuous={n_cont}, "
            f"categorical={n_cat})"
        )
