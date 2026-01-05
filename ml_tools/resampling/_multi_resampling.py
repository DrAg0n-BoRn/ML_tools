import polars as pl
import pandas as pd
import numpy as np
from typing import Union, Optional

from .._core import get_logger

from ._base_resampler import _DragonBaseResampler


_LOGGER = get_logger("DragonMultiResampler")


__all__ = [
    "DragonMultiResampler",
]


class DragonMultiResampler(_DragonBaseResampler):
    """
    A robust resampler for multi-label binary classification tasks using Polars.
    
    It provides methods to downsample "all-negative" rows and balance the dataset
    based on unique label combinations (Powerset).
    """
    def __init__(self, 
                 target_columns: list[str], 
                 return_pandas: bool = False,
                 seed: int = 42):
        """
        Args:
            target_columns (List[str]): The list of binary target column names.
            return_pandas (bool): Whether to return results as pandas DataFrame.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(return_pandas=return_pandas, seed=seed)
        self.targets = target_columns

    def downsample_all_negatives(self, 
                                 df: Union[pd.DataFrame, pl.DataFrame], 
                                 negative_ratio: float = 1.0,
                                 verbose: int = 2) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Downsamples rows where ALL target columns are 0 ("background" class).
        
        Args:
            df (pd.DataFrame | pl.DataFrame): Input DataFrame.
            negative_ratio (float): Ratio of negatives to positives to retain.
            verbose (int): Verbosity level for logging.
            
        Returns:
            Dataframe: Resampled DataFrame.
        """
        df_pl = self._convert_to_polars(df)
        
        # 1. Identify "All Negative" vs "Has Signal"
        fold_expr = pl.sum_horizontal(pl.col(self.targets)).cast(pl.UInt32)
        
        df_pos = df_pl.filter(fold_expr > 0)
        df_neg = df_pl.filter(fold_expr == 0)

        n_pos = df_pos.height
        n_neg_original = df_neg.height

        if n_pos == 0:
            if verbose >= 1:
                _LOGGER.warning("No positive cases found in any label. Returning original DataFrame.")
            return self._process_return(df_pl, shuffle=False)

        # 2. Calculate target count for negatives
        target_n_neg = int(n_pos * negative_ratio)
        
        # 3. Sample if necessary
        if n_neg_original > target_n_neg:
            if verbose >= 2:
                _LOGGER.info(f"📉 Downsampling 'All-Negative' rows from {n_neg_original} to {target_n_neg}")
            
            # Here we use standard sampling because we are not grouping
            df_neg_sampled = df_neg.sample(n=target_n_neg, seed=self.seed, with_replacement=False)
            df_resampled = pl.concat([df_pos, df_neg_sampled])
            
            return self._process_return(df_resampled)
        else:
            if verbose >= 1:
                _LOGGER.warning(f"Negative count ({n_neg_original}) is already below target ({target_n_neg}). No downsampling applied.")
            return self._process_return(df_pl, shuffle=False)

    def balance_powerset(self, 
                         df: Union[pd.DataFrame, pl.DataFrame], 
                         max_samples_per_combination: Optional[int] = None,
                         quantile_limit: float = 0.90,
                         verbose: int = 2) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Groups data by unique label combinations (Powerset) and downsamples 
        majority combinations.
        
        Args:
            df (pd.DataFrame | pl.DataFrame): Input DataFrame.
            max_samples_per_combination (int | None): Fixed cap per combination.
                If None, uses quantile_limit to determine cap.
            quantile_limit (float): Quantile to determine cap if max_samples_per_combination is None.
            verbose (int): Verbosity level for logging.
            
        Returns:
            Dataframe: Resampled DataFrame.
        """
        df_pl = self._convert_to_polars(df)

        # 1. Create a hash/structural representation of the targets for grouping
        df_lazy = df_pl.lazy().with_columns(
            pl.concat_list(pl.col(self.targets)).alias("_powerset_key")
        )

        # 2. Calculate frequencies
        # We need to collect partially to calculate the quantile cap
        combo_counts = df_lazy.group_by("_powerset_key").len().collect()
        
        # Determine the Cap
        if max_samples_per_combination is None:
            # Handle potential None from quantile (satisfies linter)
            q_val = combo_counts["len"].quantile(quantile_limit)
            
            if q_val is None:
                if verbose >= 1:
                    _LOGGER.warning("Data empty or insufficient to calculate quantile. Returning original.")
                return self._process_return(df_pl, shuffle=False)
            
            cap_size = int(q_val)
            
            if verbose >= 3:
                _LOGGER.info(f"📊 Auto-calculated Powerset Cap: {cap_size} samples (based on {quantile_limit} quantile).")
        else:
            cap_size = max_samples_per_combination

        # 3. Apply Stratified Sampling / Capping (Randomized)
        df_balanced = (
            df_lazy
            .filter(
                pl.int_range(0, pl.len())
                .shuffle(seed=self.seed)
                .over("_powerset_key") 
                < cap_size
            )
            .drop("_powerset_key")
            .collect()
        )
        
        if verbose >= 2:
            original_count = df_pl.height
            new_count = df_balanced.height
            _LOGGER.info(f"⚖️ Powerset Balancing: Reduced from {original_count} to {new_count} rows.")
            
        return self._process_return(df_balanced)

    def describe_balance(self, df: Union[pd.DataFrame, pl.DataFrame], top_n: int = 10) -> None:
        df_pl = self._convert_to_polars(df)
        total_rows = df_pl.height
        
        message_1 = f"\n📊 --- Target Balance Report ({total_rows} samples) ---\n🎯 Multi-Targets: {len(self.targets)} columns"
        
        # A. Individual Label Counts
        sums = df_pl.select([
            pl.sum(col).alias(col) for col in self.targets
        ]).transpose(include_header=True, header_name="Label", column_names=["Count"])
        
        sums = sums.with_columns(
            (pl.col("Count") / total_rows * 100).round(2).alias("Percentage(%)")
        ).sort("Count", descending=True)
        
        message_1 += "\n🔹 Individual Label Frequencies:"
        
        # B. Powerset (Combination) Counts    
        message_2 = f"🔹 Top {top_n} Label Combinations (Powerset):"
        
        combo_stats = (
            df_pl.group_by(self.targets)
            .len(name="Count")
            .sort("Count", descending=True)
            .with_columns(
                (pl.col("Count") / total_rows * 100).round(2).alias("Percentage(%)")
            )
        )
        
        _LOGGER.info(f"{message_1}\n{sums.head(top_n)}\n{message_2}\n{combo_stats.head(top_n)}")
