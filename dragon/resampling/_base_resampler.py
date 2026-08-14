import polars as pl
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod


__all__ = ["_DragonBaseResampler"]


class _DragonBaseResampler(ABC):
    """
    Base class for Dragon resamplers handling common I/O and state.
    """
    def __init__(self, 
                 return_pandas: bool = False, 
                 seed: int = 42):
        self.return_pandas = return_pandas
        self.seed = seed

    def _convert_to_polars(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """Standardizes input to Polars DataFrame."""
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        return df
    
    def _convert_to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """Converts Polars DataFrame back to Pandas."""
        return df.to_pandas(use_pyarrow_extension_array=False)

    def _process_return(self, df: pl.DataFrame, shuffle: bool = True) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Finalizes the DataFrame:
        1. Global Shuffle (optional but recommended for ML).
        2. Conversion to Pandas (if requested).
        """
        if shuffle:
            # Random shuffle of the final dataset
            df = df.sample(fraction=1.0, seed=self.seed, with_replacement=False)
        
        if self.return_pandas:
            return self._convert_to_pandas(df)
        return df
    
    @abstractmethod
    def describe_balance(self, df: Union[pd.DataFrame, pl.DataFrame], top_n: int = 10) -> None:
        """
        Prints a statistical summary of the target distribution.
        """
        pass
