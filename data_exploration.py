import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time
from typing import Union, Literal, Dict, Tuple
import os


def get_features_targets(df_path: str, targets: list[str]):
    """
    Reads a CSV file and separates its columns into features and targets.

    Args:
        df_path (str): Path to the CSV file containing the dataset.
        targets (list[str]): List of column names to be treated as target variables.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Full dataset.
            - pd.DataFrame: Target variables dataset.
            - pd.DataFrame: Feature variables dataset.

    Prints:
        - Shape of the original dataframe.
        - Shape of the targets dataframe.
        - Shape of the features dataframe.
    """
    df = pd.read_csv(df_path, encoding='utf-8')
    df_targets = df[targets]
    df_features = df.drop(columns=targets)
    print(f"Original shape: {df.shape}\nTargets shape: {df_targets.shape}\nFeatures shape: {df_features.shape}")
    return df, df_targets, df_features


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

    print(summary)
    return summary


def show_null_columns(df: pd.DataFrame, round_digits: int = 2):
    """
    Displays a table of columns with missing values, showing both the count and
    percentage of missing entries per column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        round_digits (int): Number of decimal places for the percentage.
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
    print(null_summary)
    
    
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


def check_value_distributions(df: pd.DataFrame, save_dir: Union[str, None]=None, view_frequencies: bool=False, plot_values_threshold: int=50):
    """
    Analyzes value counts for each column in a DataFrame, optionally plots distributions, 
    and saves them as .png files in the specified directory.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        save_dir (str | None): The directory where plots will be saved.
        view_frequencies (bool): Visualize relative frequencies instead of value counts.
        plot_values_threshold (int): Threshold of unique values to skip plot.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)    
    
    dict_to_plot = dict()
    
    for col in df.columns:
        if view_frequencies:
            view = df[col].value_counts(normalize=True)  # percentages
        else:
            view = df[col].value_counts(ascending=False)
        print(view)
        time.sleep(1)
        
        if save_dir:
            if view.size > plot_values_threshold:
                print(f"'{col}' has {view.size} unique values — skipping plot.")
                continue
            
            user_input = input(f"Plot value distribution for '{col}'? (y/N): ")
            if user_input.lower() in ["y", "yes"]:
                dict_to_plot[col] = dict(view)
        else:
            user_input = input("Press enter")
            
        clear_output(wait=False)
    
    # plot and save
    saved_plots = list()
    if dict_to_plot and save_dir:
        for col, data in dict_to_plot.items():
            plt.bar(data.keys(), data.values(), color="skyblue", alpha=0.7)
            ylabel = "Frequency" if view_frequencies else "Counts"
            plt.xlabel("Values")
            plt.ylabel(ylabel)
            plt.title(f"Value Distribution for '{col}'")
            plt.xticks(rotation=45)

            # Save plot
            plot_path = os.path.join(save_dir, f"{col}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close figure to free memory
            
            saved_plots.append(col)
            
    if saved_plots:
        clear_output(wait=False)
        print(f"Saved {len(saved_plots)} plot(s):")
        print(f"{saved_plots}")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Union[str, None] = None, method: Literal["pearson", "kendall", "spearman"]="pearson"):
    """
    Plots a heatmap of pairwise correlations between numeric features in a DataFrame.

    Only numeric columns are considered.

    Args:
        df (pd.DataFrame): The input dataset.
        save_dir (str | None): If provided, the heatmap will be saved to this directory as a svg file.
        method (str): Correlation method to use. Must be one of:
            - 'pearson' (default): measures linear correlation (assumes normally distributed data),
            - 'kendall': rank correlation (non-parametric),
            - 'spearman': monotonic relationship (non-parametric).

    Notes:
        - If the number of numeric features exceeds 20, value annotations inside the heatmap will be disabled for readability.
        - The size of the figure is scaled automatically based on the number of numeric features.
        - Missing values are handled internally by `pandas.DataFrame.corr` using pairwise complete observations.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("No numeric columns found. Heatmap not generated.")
        return
    
    size = max(10, numeric_df.shape[1])
    plt.figure(figsize=(size, size * 0.8))
    
    corr = numeric_df.corr(method=method)
    annot_bool = numeric_df.shape[1] <= 20
    sns.heatmap(corr, annot=annot_bool, cmap='coolwarm', fmt=".2f")

    plt.title("Feature Correlation Heatmap")
    
    plt.show()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, "Feature Correlation Heatmap.svg")
        plt.savefig(save_dir, bbox_inches="tight", format='svg')
        print(f"Saved correlation heatmap to: {full_path}")
    
    plt.close()
    

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
    clip_dict: Dict[str, Tuple[Union[int, float], Union[int, float]]]
) -> pd.DataFrame:
    """
    Clips values in multiple specified numeric columns to given [min, max] ranges,
    updating values (deep copy) and skipping invalid entries.

    Args:
        df (pd.DataFrame): The input DataFrame.
        clip_dict (dict): A dictionary where keys are column names and values are (min_val, max_val) tuples.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns clipped.

    Notes:
        - Invalid specifications (missing column, non-numeric type, wrong tuple length)
          will be reported but skipped.
    """
    new_df = df.copy()
    skipped_columns = []

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
            print(f"Clipped '{col}' to range [{min_val}, {max_val}].")

        except Exception as e:
            skipped_columns.append((col, str(e)))
            continue

    if skipped_columns:
        print("\n⚠️ Some columns were skipped due to errors:")
        for col, msg in skipped_columns:
            print(f" - {col}: {msg}")

    return new_df
