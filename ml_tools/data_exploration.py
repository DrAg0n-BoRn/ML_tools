import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from IPython import get_ipython
from IPython.display import clear_output
import time
from typing import Union, Literal, Dict, Tuple, Optional
import os
import sys
import textwrap
from utilities import sanitize_filename


# Keep track of all available functions, show using `info()`
__all__ = ["load_dataframe",
           "summarize_dataframe",
           "drop_rows_with_missing_data",
           "split_features_targets", 
           "show_null_columns",
           "drop_columns_with_missing_data",
           "split_continuous_binary",
           "plot_correlation_heatmap",
           "check_value_distributions",
           "plot_value_distributions",
           "clip_outliers_single",
           "clip_outliers_multi",
           "merge_dataframes",
           "save_dataframe",
           "compute_vif",
           "drop_vif_based"]


def load_dataframe(df_path: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file.

    Args:
        df_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(df_path, encoding='utf-8')
    print(f"DataFrame shape {df.shape}")
    return df


def summarize_dataframe(df: pd.DataFrame, round_digits: int = 2):
    """
    Returns a summary DataFrame with data types, non-null counts, number of unique values,
    missing value percentage, and basic statistics for each column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Decimal places to round numerical statistics.

    Returns:
        pd.DataFrame: Summary table.
    """
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Unique Values': df.nunique(),
        'Missing %': (df.isnull().mean() * 100).round(round_digits)
    })

    # For numeric columns, add summary statistics
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        summary_numeric = df[numeric_cols].describe().T[
            ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        ].round(round_digits)
        summary = summary.join(summary_numeric, how='left')

    print(f"Shape: {df.shape}")
    return summary


def show_null_columns(df: pd.DataFrame, round_digits: int = 2):
    """
    Displays a table of columns with missing values, showing both the count and
    percentage of missing entries per column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Number of decimal places for the percentage.
        
    Returns:
        pd.DataFrame: A DataFrame summarizing missing values in each column.
    """
    null_counts = df.isnull().sum()
    null_percent = df.isnull().mean() * 100

    # Filter only columns with at least one null
    mask = null_counts > 0
    null_summary = pd.DataFrame({
        'Missing Count': null_counts[mask],
        'Missing %': null_percent[mask].round(round_digits)
    })

    # Sort by descending percentage of missing values
    null_summary = null_summary.sort_values(by='Missing %', ascending=False)
    # print(null_summary)
    return null_summary


def drop_rows_with_missing_data(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Drops rows with more than `threshold` fraction of missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Fraction of missing values above which rows are dropped.

    Returns:
        pd.DataFrame: A new DataFrame without the dropped rows.
    """
    missing_fraction = df.isnull().mean(axis=1)
    rows_to_drop = missing_fraction[missing_fraction > threshold].index

    if len(rows_to_drop) > 0:
        print(f"Dropping {len(rows_to_drop)} rows with more than {threshold*100:.0f}% missing data.")
    else:
        print(f"No rows have more than {threshold*100:.0f}% missing data.")

    return df.drop(index=rows_to_drop)


def split_features_targets(df: pd.DataFrame, targets: list[str]):
    """
    Splits a DataFrame's columns into features and targets.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the dataset.
        targets (list[str]): List of column names to be treated as target variables.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Targets dataframe.
            - pd.DataFrame: Features dataframe.

    Prints:
        - Shape of the original dataframe.
        - Shape of the targets dataframe.
        - Shape of the features dataframe.
    """
    df_targets = df[targets]
    df_features = df.drop(columns=targets)
    print(f"Original shape: {df.shape}\nTargets shape: {df_targets.shape}\nFeatures shape: {df_features.shape}")
    return df_targets, df_features


def split_continuous_binary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into two DataFrames: one with continuous columns, one with binary columns.
    Normalize binary values like 0.0/1.0 to 0/1 if detected.

    Parameters:
        df (pd.DataFrame): Input DataFrame with only numeric columns.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): (continuous_columns_df, binary_columns_df)

    Raises:
        TypeError: If any column is not numeric.
    """
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric (int or float).")

    binary_cols = []
    continuous_cols = []

    for col in df.columns:
        series = df[col]
        unique_values = set(series[~series.isna()].unique())

        if unique_values.issubset({0, 1}):
            binary_cols.append(col)
        elif unique_values.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: 0 if x == 0.0 else (1 if x == 1.0 else x))
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    binary_cols.sort()

    df_cont = df[continuous_cols]
    df_bin = df[binary_cols]

    print(f"Continuous columns shape: {df_cont.shape}")
    print(f"Binary columns shape: {df_bin.shape}")

    return df_cont, df_bin # type: ignore

    
def drop_columns_with_missing_data(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Drops columns with more than `threshold` fraction of missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Fraction of missing values above which columns are dropped.

    Returns:
        pd.DataFrame: A new DataFrame without the dropped columns.
    """
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index

    if len(cols_to_drop) > 0:
        print(f"Dropping columns with more than {threshold*100:.0f}% missing data:")
        print(list(cols_to_drop))
    else:
        print(f"No columns have more than {threshold*100:.0f}% missing data.")

    return df.drop(columns=cols_to_drop)


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Union[str, None] = None, method: Literal["pearson", "kendall", "spearman"]="pearson", plot_title: str="Correlation Heatmap"):
    """
    Plots a heatmap of pairwise correlations between numeric features in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | None): If provided, the heatmap will be saved to this directory as a svg file.
        plot_title: To make different plots, or overwrite existing ones.
        method (str): Correlation method to use. Must be one of:
            - 'pearson' (default): measures linear correlation (assumes normally distributed data),
            - 'kendall': rank correlation (non-parametric),
            - 'spearman': monotonic relationship (non-parametric).

    Notes:
        - Only numeric columns are included.
        - Annotations are disabled if there are more than 20 features.
        - Missing values are handled via pairwise complete observations.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("No numeric columns found. Heatmap not generated.")
        return
    
    corr = numeric_df.corr(method=method)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot setup
    size = max(10, numeric_df.shape[1])
    plt.figure(figsize=(size, size * 0.8))

    annot_bool = numeric_df.shape[1] <= 20
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot_bool,
        cmap='coolwarm',
        fmt=".2f",
        cbar_kws={"shrink": 0.8}
    )
    
    # sanitize the plot title
    plot_title = sanitize_filename(plot_title)

    plt.title(plot_title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, plot_title + ".svg")
        plt.savefig(full_path, bbox_inches="tight", format='svg')
        print(f"Saved correlation heatmap to: {full_path}")
    
    plt.show()
    plt.close()


def check_value_distributions(df: pd.DataFrame, view_frequencies: bool=True, bin_threshold: int=10, skip_cols_with_key: Union[str, None]=None):
    """
    Analyzes value counts for each column in a DataFrame, optionally plots distributions, 
    and saves them as .png files in the specified directory.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        view_frequencies (bool): Print relative frequencies instead of value counts.
        bin_threshold (int): Threshold of unique values to start using bins.
        skip_cols_with_key (str | None): Skip column names containing the key. If None, don't skip any column.
    
    Notes:
        - Binning is adaptive: if quantile binning results in ≤ 2 unique bins, raw values are used instead.
    """
    # cherrypick columns
    if skip_cols_with_key is not None:
        columns = [col for col in df.columns if skip_cols_with_key not in col]
    else:
        columns = df.columns.to_list()
    
    for col in columns:
        if _is_notebook():
            clear_output(wait=False)
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > bin_threshold:
            bins_number = 10
            binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
            while binned.nunique() <= 2:
                bins_number -= 1
                binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
                if bins_number <= 2:
                    break
            
            if binned.nunique() <= 2:
                view_std = df[col].value_counts(ascending=False)
            else:
                view_std = binned.value_counts(sort=False)
            
        else:
            view_std = df[col].value_counts(ascending=False)

        view_std.name = col

        # unlikely scenario where the series is empty
        if view_std.sum() == 0:
            view_freq = view_std
        else:
            view_freq = view_std / view_std.sum()
        # view_freq = df[col].value_counts(normalize=True, bins=10)  # relative percentages
        view_freq.name = col

        # Print value counts
        print(view_freq if view_frequencies else view_std)
        
        time.sleep(1)
        user_input_ = input("Press enter to continue")


def plot_value_distributions(df: pd.DataFrame, save_dir: str, bin_threshold: int=10, skip_cols_with_key: Union[str, None]=None):
    """
    Plots and saves the value distributions for all (or selected) columns in a DataFrame, 
    with adaptive binning for numerical columns when appropriate.

    For each column both raw counts and relative frequencies are computed and plotted.

    Plots are saved as PNG files under two subdirectories in `save_dir`:
    - "Distribution_Counts" for absolute counts.
    - "Distribution_Frequency" for relative frequencies.

    Args:
        df (pd.DataFrame): The input DataFrame whose columns are to be analyzed.
        save_dir (str): Directory path where the plots will be saved. Will be created if it does not exist.
        bin_threshold (int): Minimum number of unique values required to trigger binning
            for numerical columns.
        skip_cols_with_key (str | None): If provided, any column whose name contains this
            substring will be excluded from analysis.

    Notes:
        - Binning is adaptive: if quantile binning results in ≤ 2 unique bins, raw values are used instead.
        - All non-alphanumeric characters in column names are sanitized for safe file naming.
        - Colormap is automatically adapted based on the number of categories or bins.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)    
    
    dict_to_plot_std = dict()
    dict_to_plot_freq = dict()
    
    # cherrypick columns
    if skip_cols_with_key is not None:
        columns = [col for col in df.columns if skip_cols_with_key not in col]
    else:
        columns = df.columns.to_list()
    
    saved_plots = 0
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > bin_threshold:
            bins_number = 10
            binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
            while binned.nunique() <= 2:
                bins_number -= 1
                binned = pd.qcut(df[col], q=bins_number, duplicates='drop')
                if bins_number <= 2:
                    break
            
            if binned.nunique() <= 2:
                view_std = df[col].value_counts(sort=False).sort_index()
            else:
                view_std = binned.value_counts(sort=False)
            
        else:
            view_std = df[col].value_counts(sort=False).sort_index()

        # unlikely scenario where the series is empty
        if view_std.sum() == 0:
            view_freq = view_std
        else:
            view_freq = 100 * view_std / view_std.sum() # Percentage
        # view_freq = df[col].value_counts(normalize=True, bins=10)  # relative percentages
        
        if save_dir:
            dict_to_plot_std[col] = dict(view_std)
            dict_to_plot_freq[col] = dict(view_freq)
            saved_plots += 1
    
    # plot helper
    def _plot_helper(dict_: dict, target_dir: str, ylabel: Literal["Frequency", "Counts"], base_fontsize: int=12):
        for col, data in dict_.items():
            safe_col = sanitize_filename(col)
            
            if isinstance(list(data.keys())[0], pd.Interval):
                labels = [str(interval) for interval in data.keys()]
            else:
                labels = data.keys()
                
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab20.colors if len(data) <= 20 else plt.cm.viridis(np.linspace(0, 1, len(data)))
                
            plt.bar(labels, data.values(), color=colors[:len(data)], alpha=0.85)
            plt.xlabel("Values", fontsize=base_fontsize)
            plt.ylabel(ylabel, fontsize=base_fontsize)
            plt.title(f"Value Distribution for '{col}'", fontsize=base_fontsize+2)
            plt.xticks(rotation=45, ha='right', fontsize=base_fontsize-2)
            plt.yticks(fontsize=base_fontsize-2)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.gca().set_facecolor('#f9f9f9')
            plt.tight_layout()
            
            plot_path = os.path.join(target_dir, f"{safe_col}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
    
    # Save plots
    freq_dir = os.path.join(save_dir, "Distribution_Frequency")
    std_dir = os.path.join(save_dir, "Distribution_Counts")
    os.makedirs(freq_dir, exist_ok=True)
    os.makedirs(std_dir, exist_ok=True)
    _plot_helper(dict_=dict_to_plot_std, target_dir=std_dir, ylabel="Counts")
    _plot_helper(dict_=dict_to_plot_freq, target_dir=freq_dir, ylabel="Frequency")

    print(f"Saved {saved_plots} plot(s)")


def clip_outliers_single(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float
) -> Union[pd.DataFrame, None]:
    """
    Clips values in the specified numeric column to the range [min_val, max_val],
    and returns a new DataFrame where the original column is replaced by the clipped version.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to clip.
        min_val (float): Minimum allowable value; values below are clipped to this.
        max_val (float): Maximum allowable value; values above are clipped to this.

    Returns:
        pd.DataFrame: A new DataFrame with the specified column clipped in place.
        
        None: if a problem with the dataframe column occurred.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' must be numeric.")
        return None

    new_df = df.copy(deep=True)
    new_df[column] = new_df[column].clip(lower=min_val, upper=max_val)

    print(f"Column '{column}' clipped to range [{min_val}, {max_val}].")
    return new_df


def clip_outliers_multi(
    df: pd.DataFrame,
    clip_dict: Dict[str, Tuple[Union[int, float], Union[int, float]]],
    verbose: bool=False
) -> pd.DataFrame:
    """
    Clips values in multiple specified numeric columns to given [min, max] ranges,
    updating values (deep copy) and skipping invalid entries.

    Args:
        df (pd.DataFrame): The input DataFrame.
        clip_dict (dict): A dictionary where keys are column names and values are (min_val, max_val) tuples.
        verbose (bool): prints clipped range for each column.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns clipped.

    Notes:
        - Invalid specifications (missing column, non-numeric type, wrong tuple length)
          will be reported but skipped.
    """
    new_df = df.copy()
    skipped_columns = []
    clipped_columns = 0

    for col, bounds in clip_dict.items():
        try:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric.")

            if not (isinstance(bounds, tuple) and len(bounds) == 2):
                raise ValueError(f"Bounds for '{col}' must be a tuple of (min, max).")

            min_val, max_val = bounds
            new_df[col] = new_df[col].clip(lower=min_val, upper=max_val)
            if verbose:
                print(f"Clipped '{col}' to range [{min_val}, {max_val}].")
            clipped_columns += 1

        except Exception as e:
            skipped_columns.append((col, str(e)))
            continue
        
    print(f"Clipped {clipped_columns} columns.")

    if skipped_columns:
        print("\n⚠️ Skipped columns:")
        for col, msg in skipped_columns:
            print(f" - {col}: {msg}")

    return new_df


def merge_dataframes(
    *dfs: pd.DataFrame,
    reset_index: bool = False,
    direction: Literal["horizontal", "vertical"] = "horizontal"
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
        raise ValueError("At least 2 DataFrames must be provided.")
    
    for i, df in enumerate(dfs, start=1):
        print(f"DataFrame {i} shape: {df.shape}")
    

    if direction == "horizontal":
        reference_index = dfs[0].index
        for i, df in enumerate(dfs, start=1):
            if not df.index.equals(reference_index):
                raise ValueError(f"Indexes do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=1)

    elif direction == "vertical":
        reference_columns = dfs[0].columns
        for i, df in enumerate(dfs, start=1):
            if not df.columns.equals(reference_columns):
                raise ValueError(f"Column names/order do not match: Dataset 1 and Dataset {i}.")
        merged_df = pd.concat(dfs, axis=0)

    else:
        raise ValueError(f"Invalid merge direction: {direction}")

    if reset_index:
        merged_df = merged_df.reset_index(drop=True)

    print(f"Merged DataFrame shape: {merged_df.shape}")

    return merged_df


def save_dataframe(df: pd.DataFrame, save_dir: str, filename: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
        df: pandas.DataFrame to save
        save_dir: str, directory where the CSV file will be saved.
        filename: str, CSV filename, extension will be added if missing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filename = sanitize_filename(filename)
    
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    output_path = os.path.join(save_dir, filename)
        
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved file: '{filename}'")


def compute_vif(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    ignore_cols: Optional[list[str]] = None,
    plot: bool = True,
    save_dir: Union[str, None] = None
) -> pd.DataFrame:
    """
    Computes Variance Inflation Factors (VIF) for numeric features, optionally plots and saves the results.
    
    There cannot be empty values in the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list[str] | None): Optional list of column names to evaluate. Defaults to all numeric columns.
        ignore_cols (list[str] | None): Optional list of column names to ignore.
        plot (bool): Whether to display a barplot of VIF values.
        save_dir (str | None): Directory to save the plot as SVG. If None, plot is not saved.

    Returns:
        pd.DataFrame: DataFrame with features and corresponding VIF values, sorted descending.
    
    NOTE:
    **Variance Inflation Factor (VIF)** quantifies the degree of multicollinearity among features in a dataset. 
    A VIF value indicates how much the variance of a regression coefficient is inflated due to linear dependence with other features. 
    A VIF of 1 suggests no correlation, values between 1 and 5 indicate moderate correlation, and values greater than 10 typically signal high multicollinearity, which may distort model interpretation and degrade performance.
        
    """
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()
    
    if ignore_cols is not None:
        missing = set(ignore_cols) - set(features)
        if missing:
            raise ValueError(f"The following 'columns to ignore' are not in the Dataframe:\n{missing}")
        features = [f for f in features if f not in ignore_cols]

    X = df[features].copy()
    X = add_constant(X, has_constant='add')

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Drop the constant column
    vif_data = vif_data[vif_data["feature"] != "const"]
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True) # type: ignore

    # Add color coding based on thresholds
    def vif_color(v: float) -> str:
        if v > 10:
            return "red"
        elif v > 5:
            return "gold"
        else:
            return "green"

    vif_data["color"] = vif_data["VIF"].apply(vif_color)

    # Plot
    if plot or save_dir:
        plt.figure(figsize=(10, 6))
        bars = plt.barh(
            vif_data["feature"], 
            vif_data["VIF"], 
            color=vif_data["color"], 
            edgecolor='black'
        )
        plt.title("Variance Inflation Factor (VIF) per Feature")
        plt.xlabel("VIF")
        plt.axvline(x=5, color='gold', linestyle='--', label='VIF = 5')
        plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
        plt.legend(loc='lower right')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "VIF_plot.svg")
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"Saved VIF plot to: {save_path}")

        if plot:
            plt.show()
        plt.close()

    return vif_data.drop(columns="color")


def drop_vif_based(df: pd.DataFrame, vif_df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Drops features from the original DataFrame based on their VIF values exceeding a given threshold.

    Args:
        df (pd.DataFrame): Original DataFrame containing the features.
        vif_df (pd.DataFrame): DataFrame with 'feature' and 'VIF' columns as returned by `compute_vif()`.
        threshold (float): VIF threshold above which features will be dropped.

    Returns:
        pd.DataFrame: A new DataFrame with high-VIF features removed.
    """
    # Ensure expected structure
    if 'feature' not in vif_df.columns or 'VIF' not in vif_df.columns:
        raise ValueError("`vif_df` must contain 'feature' and 'VIF' columns.")
    
    # Identify features to drop
    to_drop = vif_df[vif_df["VIF"] > threshold]["feature"].tolist()
    print(f"Dropping {len(to_drop)} feature(s) with VIF > {threshold}: {to_drop}")

    return df.drop(columns=to_drop, errors="ignore")


def _is_notebook():
    return get_ipython() is not None


def info(full_info: bool=True):
    """
    List available functions and their descriptions.
    """
    print("Available functions for data exploration:")
    if full_info:
        module = sys.modules[__name__]
        for name in __all__:
            obj = getattr(module, name, None)
            if callable(obj):
                doc = obj.__doc__ or "No docstring provided."
                formatted_doc = textwrap.indent(textwrap.dedent(doc.strip()), prefix="    ")
                print(f"\n{name}:\n{formatted_doc}")
    else:
        for i, name in enumerate(__all__, start=1):
            print(f"{i} - {name}")


if __name__ == "__main__":
    info()
