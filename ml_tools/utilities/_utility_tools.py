import pandas as pd
from pathlib import Path
from typing import Literal, Union, Iterator

from ..path_manager import make_fullpath, list_csv_paths
from .._core import get_logger

from ._utility_save_load import load_dataframe, save_dataframe_filename


_LOGGER = get_logger("Utility Tools")


__all__ = [
    "merge_dataframes",
    "distribute_dataset_by_target",
    "train_dataset_orchestrator",
    "train_dataset_yielder"
]


def merge_dataframes(
    *dfs: pd.DataFrame,
    reset_index: bool = False,
    direction: Literal["horizontal", "vertical"] = "horizontal",
    verbose: bool=True
) -> pd.DataFrame:
    """
    Merges multiple DataFrames either horizontally or vertically.

    Parameters:
        *dfs (pd.DataFrame): Variable number of DataFrames to merge.
        reset_index (bool): Whether to reset index in the final merged DataFrame.
        direction (["horizontal" | "vertical"]):
            - "horizontal": Merge on index, adding columns.
            - "vertical": Append rows; all DataFrames must have identical columns.

    Returns:
        pd.DataFrame: A single merged DataFrame.

    Raises:
        ValueError:
            - If fewer than 2 DataFrames are provided.
            - If indexes do not match for horizontal merge.
            - If column names or order differ for vertical merge.
    """
    if len(dfs) < 2:
        raise ValueError("❌ At least 2 DataFrames must be provided.")
    
    if verbose:
        for i, df in enumerate(dfs, start=1):
            print(f"➡️ DataFrame {i} shape: {df.shape}")
    

    if direction == "horizontal":
        reference_index = dfs[0].index
        for i, df in enumerate(dfs, start=1):
            if not df.index.equals(reference_index):
                raise ValueError(f"❌ Indexes do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=1)

    elif direction == "vertical":
        reference_columns = dfs[0].columns
        for i, df in enumerate(dfs, start=1):
            if not df.columns.equals(reference_columns):
                raise ValueError(f"❌ Column names/order do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=0)

    else:
        _LOGGER.error(f"Invalid merge direction: {direction}")
        raise ValueError()

    if reset_index:
        merged_df = merged_df.reset_index(drop=True)
    
    if verbose:
        _LOGGER.info(f"Merged DataFrame shape: {merged_df.shape}")

    return merged_df


def distribute_dataset_by_target(
    df_or_path: Union[pd.DataFrame, str, Path],
    target_columns: list[str],
    verbose: bool = False
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yields cleaned DataFrames for each target column, where rows with missing
    target values are removed. The target column is placed at the end.

    Parameters
    ----------
    df_or_path : [pd.DataFrame | str | Path]
        Dataframe or path to Dataframe with all feature and target columns ready to split and train a model.
    target_columns : List[str]
        List of target column names to generate per-target DataFrames.
    verbose: bool
        Whether to print info for each yielded dataset.

    Yields
    ------
    Tuple[str, pd.DataFrame]
        * Target name.
        * Pandas DataFrame.
    """
    # Validate path or dataframe
    if isinstance(df_or_path, str) or isinstance(df_or_path, Path):
        df_path = make_fullpath(df_or_path)
        df, _ = load_dataframe(df_path)
    else:
        df = df_or_path
    
    valid_targets = [col for col in df.columns if col in target_columns]
    feature_columns = [col for col in df.columns if col not in valid_targets]

    for target in valid_targets:
        subset = df[feature_columns + [target]].dropna(subset=[target]) # type: ignore
        if verbose:
            print(f"Target: '{target}' - Dataframe shape: {subset.shape}")
        yield target, subset


def train_dataset_orchestrator(list_of_dirs: list[Union[str,Path]], 
                               target_columns: list[str], 
                               save_dir: Union[str,Path],
                               safe_mode: bool=False):
    """
    Orchestrates the creation of single-target datasets from multiple directories each with a variable number of CSV datasets.

    This function iterates through a list of directories, finds all CSV files,
    and splits each dataframe based on the provided target columns. Each resulting
    single-target dataframe is then saved to a specified directory.

    Parameters
    ----------
    list_of_dirs : list[str | Path]
        A list of directory paths where the source CSV files are located.
    target_columns : list[str]
        A list of column names to be used as targets for splitting the datasets.
    save_dir : str | Path
        The directory where the newly created single-target datasets will be saved.
    safe_mode : bool
        If True, prefixes the saved filename with the source directory name to prevent overwriting files with the same name from different sources.
    """
    all_dir_paths: list[Path] = list()
    for dir in list_of_dirs:
        dir_path = make_fullpath(dir)
        if not dir_path.is_dir():
            _LOGGER.error(f"'{dir}' is not a directory.")
            raise IOError()
        all_dir_paths.append(dir_path)
    
    # main loop
    total_saved = 0
    for df_dir in all_dir_paths:
        for df_name, df_path in list_csv_paths(df_dir).items():
            try:
                for target_name, df in distribute_dataset_by_target(df_or_path=df_path, target_columns=target_columns, verbose=False):
                    if safe_mode:
                        filename = df_dir.name + '_' + target_name + '_' + df_name
                    else:
                        filename = target_name + '_' + df_name
                    save_dataframe_filename(df=df, save_dir=save_dir, filename=filename)
                    total_saved += 1
            except Exception as e:
                _LOGGER.error(f"Failed to process file '{df_path}'. Reason: {e}")
                continue 

    _LOGGER.info(f"{total_saved} single-target datasets were created.")


def train_dataset_yielder(
    df: pd.DataFrame,
    target_cols: list[str]
) -> Iterator[tuple[pd.DataFrame, pd.Series, list[str], str]]:
    """ 
    Yields one tuple at a time:
        (features_dataframe, target_series, feature_names, target_name)

    Skips any target columns not found in the DataFrame.
    """
    # Determine which target columns actually exist in the DataFrame
    valid_targets = [col for col in target_cols if col in df.columns]

    # Features = all columns excluding valid target columns
    df_features = df.drop(columns=valid_targets)
    feature_names = df_features.columns.to_list()

    for target_col in valid_targets:
        df_target = df[target_col]
        yield (df_features, df_target, feature_names, target_col)

