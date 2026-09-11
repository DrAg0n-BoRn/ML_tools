import pandas as pd
import numpy as np
from typing import Any, Optional, Union

from .._core import get_logger


_LOGGER = get_logger("Faulty Entry Manager")


__all__ = [
    "DragonFaultyEntryManager"
]


# Acceptable scalar types for search and replacement
ScalarValue = Union[int, float, str, bool, None]

class DragonFaultyEntryManager:
    """
    Manages the identification, tracking, and remediation of faulty entries 
    within a pandas DataFrame.
    
    Provides a ledger-based approach to locate specific cell values, apply fixes, 
    or drop invalid rows and columns while maintaining synchronized state tracking.
    
    Works on single-level indexed DataFrames with unique indices to ensure accurate tracking of faulty entries.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the manager with a copy of the target DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to manage and clean. Must have a unique index.
        """
        # check that the input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            _LOGGER.error("Input is not a pandas DataFrame.")
            raise TypeError()
        # check that the DataFrame has a single-level index and that it is unique
        if isinstance(df.index, pd.MultiIndex):
            _LOGGER.error("DataFrame must have a single-level index, not a MultiIndex.")
            raise ValueError()
        if not df.index.is_unique:
            _LOGGER.error("DataFrame index must be unique to accurately track faulty entries.")
            raise ValueError()
        
        self._df = df.copy()
        self._original_dtypes = self._df.dtypes.to_dict()
        self._current_targets: Optional[dict[str, ScalarValue]] = None
        
        # Map now stores: FaultyEntry ID -> (Original_Index, Column_Name)
        self._current_id_map: dict[int, tuple[Any, str]] = {}
        self._active_faulty_ids: set[int] = set()

    def _validate_dtypes(self, col: str, replacement_val: ScalarValue) -> None:
        """
        Validates the data type of the replacement value against the original column's data type.
        Logs warnings if there is a potential type mismatch that could lead to unintended behavior.
        """
        original_type = self._original_dtypes[col]
        
        # Warn if inserting a string into a numeric column
        if pd.api.types.is_numeric_dtype(original_type) and isinstance(replacement_val, str):
            _LOGGER.warning(
                f"Type mismatch: Replacing numeric value in '{col}' with string '{replacement_val}'. This may upcast the column to object."
            )
        
        # Warn if inserting a numeric/boolean value into an object/string column
        elif pd.api.types.is_object_dtype(original_type) or pd.api.types.is_string_dtype(original_type):
            if isinstance(replacement_val, bool):
                _LOGGER.warning(
                    f"Mixed types warning: Inserting boolean value {replacement_val} into object/string column '{col}'. This can cause issues with downstream string operations."
                )
            elif isinstance(replacement_val, (int, float)):
                _LOGGER.warning(
                    f"Mixed types warning: Inserting numeric value {replacement_val} into object/string column '{col}'. This can cause issues with downstream string operations."
                )
    
    def _get_current_ledger(self) -> pd.DataFrame:
        """
        Returns the current ledger of active faulty entries.
        """        
        if not self._active_faulty_ids:
            _LOGGER.info("🗿 No active faulty entries remaining.")
            return pd.DataFrame(columns=[
                'FaultyEntry ID', 'Original_Index', 'Column', 'Current_Value'
            ]).set_index('FaultyEntry ID')
            
        faulty_records = []
        for fid in sorted(self._active_faulty_ids):
            idx, col = self._current_id_map[fid]
            faulty_records.append({
                'FaultyEntry ID': fid,
                'Original_Index': idx,
                'Column': col,
                'Current_Value': self._df.at[idx, col]
            })
        
        return pd.DataFrame(faulty_records).set_index('FaultyEntry ID')
    
    def find(self, targets: dict[str, ScalarValue]) -> pd.DataFrame:
        """
        Locates occurrences of target values within specified DataFrame columns 
        and registers them as active faulty entries.
        
        Args:
            targets (dict[str, ScalarValue]): A dictionary mapping column names 
                to target values to search for.
                    - Int
                    - Float
                    - String
                    - Boolean
                    - None (to locate NaN values)
                
        Returns:
            pd.DataFrame: A ledger DataFrame containing all located active faulty entries.
        """
        if not targets:
            _LOGGER.error("Targets dictionary cannot be empty.")
            raise ValueError()
        
        dict_copy = targets.copy()
        
        # validate that all specified columns exist in the DataFrame
        for col in dict_copy.keys():
            if col not in self._df.columns:
                _LOGGER.warning(f"Column '{col}' not found in DataFrame. Skipping this target.")
                dict_copy.pop(col)
        
        self._current_targets = dict_copy
        self._current_id_map.clear()
        self._active_faulty_ids.clear()
        
        current_id = 1
        
        for col, val in self._current_targets.items():
            if pd.isna(val):
                mask = self._df[col].isna()
            else:
                # fillna(False) prevents ValueError when checking against pandas nullable dtypes
                mask = (self._df[col] == val).fillna(False)
            
            matching_indices = self._df[mask].index
            
            for idx in matching_indices:
                self._current_id_map[current_id] = (idx, col)
                self._active_faulty_ids.add(current_id)
                current_id += 1
        
        return self._get_current_ledger()

    def apply_fix(self, fixes: dict[int, ScalarValue]) -> pd.DataFrame:
        """
        Applies replacement values to specific active faulty entries by their `FaultyEntry ID`.
        
        Args:
            fixes (dict[int, ScalarValue]): A dictionary mapping FaultyEntry IDs 
                to their new replacement values.
                
        Returns:
            pd.DataFrame: The updated ledger of remaining active faulty entries.
        """
        if not self._active_faulty_ids:
            _LOGGER.error("No active faulty entries to fix. Call 'find()' first.")
            raise RuntimeError()

        success_count = 0
        for fid, replacement_val in fixes.items():
            if fid not in self._active_faulty_ids:
                _LOGGER.warning(f"FaultyEntry ID {fid} is not active. Skipping.")
                continue
                
            # Extract specific cell coordinates directly from the map
            orig_idx, col = self._current_id_map[fid]
            
            # Validate dtype compatibility and apply fix
            self._validate_dtypes(col, replacement_val)
            self._df.at[orig_idx, col] = replacement_val
            
            # Deactivate ID so it cannot be fixed again
            self._active_faulty_ids.remove(fid)
            success_count += 1
            
        _LOGGER.info(f"🛠️ Successfully applied {success_count} fixes.")
        return self._get_current_ledger()

    def drop_rows(self, faulty_ids: list[int]) -> pd.DataFrame:
        """
        Drops rows associated with the specified faulty entry IDs and synchronizes the active entries ledger.
        
        Args:
            faulty_ids (list[int]): A list of active faulty entry IDs whose rows should be dropped.
                
        Returns:
            pd.DataFrame: The updated ledger of remaining active faulty entries.
        """
        if not self._active_faulty_ids:
            _LOGGER.error("No active faulty entries found. Call 'find()' first.")
            raise RuntimeError()
        
        valid_ids = [fid for fid in faulty_ids if fid in self._active_faulty_ids]
        
        if not valid_ids:
            _LOGGER.warning("None of the provided IDs are currently active or valid.")
            return self._get_current_ledger()
            
        # Find all unique original indices associated with the provided IDs
        indices_to_drop = set(self._current_id_map[fid][0] for fid in valid_ids)
        
        self._df = self._df.drop(index=list(indices_to_drop))
        
        # Cross-synchronization: deactivate ALL IDs that fall within the dropped rows
        ids_to_remove = [
            fid for fid in self._active_faulty_ids 
            if self._current_id_map[fid][0] in indices_to_drop
        ]
        for fid in ids_to_remove:
            self._active_faulty_ids.remove(fid)
            
        _LOGGER.info(f"🗑️ Successfully dropped {len(indices_to_drop)} rows.")
        return self._get_current_ledger()

    def drop_columns(self, columns: list[str]) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame and synchronizes the active entries ledger.
        
        Args:
            columns (list[str]): A list of column names to drop.
                
        Returns:
            pd.DataFrame: The updated ledger of remaining active faulty entries.
        """
        if self._current_targets is None:
            _LOGGER.error("No active targets found. Call 'find()' first.")
            raise RuntimeError()
        
        valid_cols = [col for col in columns if col in self._current_targets]
        
        if not valid_cols:
            _LOGGER.warning("None of the provided columns were targeted in the last 'find()' call.")
            return self._get_current_ledger()
            
        cols_to_drop = [col for col in valid_cols if col in self._df.columns]
        
        if not cols_to_drop:
            _LOGGER.warning("None of the provided columns exist in the DataFrame.")
            return self._get_current_ledger()
            
        self._df = self._df.drop(columns=cols_to_drop)
        
        # Cross-synchronization: deactivate ALL IDs that fall within the dropped columns
        ids_to_remove = [
            fid for fid in self._active_faulty_ids 
            if self._current_id_map[fid][1] in cols_to_drop
        ]
        for fid in ids_to_remove:
            self._active_faulty_ids.remove(fid)
            
        for col in cols_to_drop:
            del self._original_dtypes[col]
            del self._current_targets[col]
            
        _LOGGER.info(f"🗑️ Successfully dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        return self._get_current_ledger()
    
    def return_df(self) -> pd.DataFrame:
        """
        Returns a copy of the current state of the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: A copy of the current cleaned DataFrame.
        """
        _LOGGER.info(f"➡️ Returning the current state of the cleaned DataFrame with shape {self._df.shape}.")
        return self._df.copy()
