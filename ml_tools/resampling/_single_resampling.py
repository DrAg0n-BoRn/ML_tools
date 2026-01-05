import polars as pl
import pandas as pd
import numpy as np
from typing import Union

from .._core import get_logger

from ._base_resampler import _DragonBaseResampler


_LOGGER = get_logger("DragonResampler")


__all__ = [
    "DragonResampler",
]


class DragonResampler(_DragonBaseResampler):
    """
    A resampler for Single-Target Classification tasks (Binary or Multiclass).

    It balances classes by downsampling majority classes relative to the size of the minority class.
    """
    def __init__(self, 
                 target_column: str, 
                 return_pandas: bool = False,
                 seed: int = 42):
        """
        Args:
            target_column (str): The name of the single target column.
            return_pandas (bool): Whether to return results as pandas DataFrame.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(return_pandas=return_pandas, seed=seed)
        self.target = target_column

    def balance_classes(self, 
                        df: Union[pd.DataFrame, pl.DataFrame], 
                        majority_ratio: float = 1.0,
                        verbose: int = 2) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Downsamples all classes to match the minority class size (scaled by a ratio).
        """
        df_pl = self._convert_to_polars(df)
        
        # 1. Calculate Class Counts
        counts = df_pl.group_by(self.target).len().sort("len")
        
        if counts.height == 0:
            _LOGGER.error("DataFrame is empty or target column missing.")
            return self._process_return(df_pl, shuffle=False)

        # 2. Identify Statistics
        min_val = counts["len"].min()
        max_val = counts["len"].max()

        if min_val is None or max_val is None:
            _LOGGER.error("Failed to calculate class statistics (unexpected None).")
            raise ValueError()

        minority_count: int = min_val  # type: ignore
        majority_count: int = max_val  # type: ignore
        
        # Calculate the cap
        cap_size = int(minority_count * majority_ratio)
        
        if verbose >= 3:
            _LOGGER.info(f"📊 Class Distribution:\n{counts}")
            _LOGGER.info(f"🎯 Strategy: Cap majorities at {cap_size}")

        # Optimization: If data is already balanced enough
        if majority_count <= cap_size:
            if verbose >= 2:
                _LOGGER.info("Data is already within the requested balance ratio.")
            return self._process_return(df_pl, shuffle=False)

        # 3. Apply Downsampling (Randomized)
        # We generate a random range index per group and filter by it.
        # This ensures we pick a random subset, not the first N rows.
        df_balanced = (
            df_pl.lazy()
            .filter(
                pl.int_range(0, pl.len())
                .shuffle(seed=self.seed)
                .over(self.target) 
                < cap_size
            )
            .collect()
        )

        if verbose >= 2:
            reduced_count = df_balanced.height
            _LOGGER.info(f"⚖️ Balancing Complete: {df_pl.height} -> {reduced_count} rows.")
            
        return self._process_return(df_balanced)

    def describe_balance(self, df: Union[pd.DataFrame, pl.DataFrame], top_n: int = 10) -> None:
        df_pl = self._convert_to_polars(df)
        total_rows = df_pl.height
        
        message = f"\n📊 --- Target Balance Report ({total_rows} samples) ---\n🎯 Single Target: '{self.target}'"
        
        stats = (
            df_pl.group_by(self.target)
            .len(name="Count")
            .sort("Count", descending=True)
            .with_columns(
                (pl.col("Count") / total_rows * 100).round(2).alias("Percentage(%)")
            )
        )
        
        _LOGGER.info(f"{message}\n{stats.head(top_n)}")
