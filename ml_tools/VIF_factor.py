
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
from pathlib import Path
from .utilities import sanitize_filename, yield_dataframes_from_dir, save_dataframe, _script_info, make_fullpath
from .logger import _LOGGER


__all__ = [
    "compute_vif",
    "drop_vif_based",
    "compute_vif_multi"
]


def compute_vif(
    df: pd.DataFrame,
    use_columns: Optional[list[str]] = None,
    ignore_columns: Optional[list[str]] = None,
    max_features_to_plot: int = 20,
    save_dir: Optional[Union[str,Path]] = None,
    filename: Optional[str] = None,
    fontsize: int = 14,
    show_plot: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Computes Variance Inflation Factors (VIF) for numeric columns in a DataFrame. Optionally, generates a bar plot of VIF values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        use_columns (list[str] | None): Optional list of columns to include. Defaults to all numeric columns.
        ignore_columns (list[str] | None): Optional list of columns to exclude from the VIF computation. Skipped if `use_columns` is provided.
        max_features_to_plot (int): Adjust the number of features shown in the plot.
        save_dir (str | Path | None): Directory to save the plot as SVG. If None, the plot is not saved.
        filename (str | None): Optional filename for saving the plot. Defaults to "VIF_plot.svg".
        fontsize (int): Base fontsize to scale title and labels on the plot.
        show_plot (bool): Display plot.

    Returns:
        pd.DataFrame: DataFrame with features and their corresponding VIF values.
        
    NOTE:
    **Variance Inflation Factor (VIF)** quantifies the degree of multicollinearity among features in a dataset. 
    A VIF value indicates how much the variance of a regression coefficient is inflated due to linear dependence with other features. 
    A VIF of 1 suggests no correlation, values between 1 and 5 indicate moderate correlation, and values greater than 10 typically signal high multicollinearity, which may distort model interpretation and degrade performance.
    """
    ground_truth_cols = df.columns.to_list()
    if use_columns is None:
        sanitized_columns = df.select_dtypes(include='number').columns.tolist()
        missing_features = set(ground_truth_cols) - set(sanitized_columns)
        if missing_features and verbose:
            _LOGGER.warning(f"⚠️ These columns are not Numeric:\n{missing_features}")
    else:
        sanitized_columns = list()
        for feature in use_columns:
            if feature not in ground_truth_cols:
                if verbose:
                    _LOGGER.warning(f"⚠️ The provided column '{feature}' is not in the DataFrame.")
            else:
                sanitized_columns.append(feature)
    
    if ignore_columns is not None and use_columns is None:
        missing_ignore = set(ignore_columns) - set(ground_truth_cols)
        if missing_ignore and verbose:
            _LOGGER.warning(f"⚠️ Warning: The following 'columns to ignore' are not in the Dataframe:\n{missing_ignore}")
        sanitized_columns = [f for f in sanitized_columns if f not in ignore_columns]

    X = df[sanitized_columns].copy()
    X = add_constant(X, has_constant='add')

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns # type: ignore
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1]) # type: ignore
        ]

    # Replace infinite values (perfect multicollinearity)
    vif_data["VIF"] = vif_data["VIF"].replace([np.inf, -np.inf], 999.0)

    # Drop the constant column
    vif_data = vif_data[vif_data["feature"] != "const"]

    # Add color coding
    def vif_color(v: float) -> str:
        if v >= 10:
            return "red"
        elif v >= 5:
            return "gold"
        else:
            return "green"

    vif_data["color"] = vif_data["VIF"].apply(vif_color)

    # Sort by VIF descending
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

    # Filter for plotting
    plot_data = vif_data.head(max_features_to_plot)
    
    if save_dir or show_plot:
        if not plot_data.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(
                plot_data["feature"],
                plot_data["VIF"],
                color=plot_data["color"],
                edgecolor='black'
            )
            plt.title("Variance Inflation Factor (VIF) per Feature", fontsize=fontsize+1)
            plt.xlabel("VIF value", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.axvline(x=5, color='gold', linestyle='--', label='VIF = 5')
            plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
            plt.xlim(0, 12)
            plt.legend(loc='lower right', fontsize=fontsize-1)
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()

            if save_dir:
                save_path = make_fullpath(save_dir, make=True)
                if filename is None:
                    filename = "VIF_plot.svg"
                else:
                    filename = sanitize_filename(filename)
                    filename = "VIF_" + filename
                    if not filename.endswith(".svg"):
                        filename += ".svg"
                full_save_path = save_path / filename
                plt.savefig(full_save_path, format='svg', bbox_inches='tight')
                print(f"\tSaved VIF plot: '{filename}'")
            
            if show_plot:
                plt.show()
            plt.close()

    return vif_data.drop(columns="color")


def drop_vif_based(df: pd.DataFrame, vif_df: pd.DataFrame, threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops columns from the original DataFrame based on their VIF values exceeding a given threshold.

    Args:
        df (pd.DataFrame): Original DataFrame containing the columns to test.
        vif_df (pd.DataFrame): DataFrame with 'feature' and 'VIF' columns as returned by `compute_vif()`.
        threshold (float): VIF threshold above which columns will be dropped.

    Returns:
        (tuple[pd.DataFrame, list[str]]): 
            - A new DataFrame with high-VIF columns removed.
            - A list with dropped column names.
    """
    # Ensure expected structure
    if 'feature' not in vif_df.columns or 'VIF' not in vif_df.columns:
        raise ValueError("`vif_df` must contain 'feature' and 'VIF' columns.")
    
    # Identify features to drop
    to_drop = vif_df[vif_df["VIF"] > threshold]["feature"].tolist()
    _LOGGER.info(f"🗑️ Dropping {len(to_drop)} column(s) with VIF > {threshold}: {to_drop}")
    
    result_df = df.drop(columns=to_drop)
    
    if result_df.empty:
        _LOGGER.warning(f"⚠️ All columns were dropped.")

    return result_df, to_drop


def compute_vif_multi(input_directory: Union[str, Path],
                      output_plot_directory: Union[str, Path],
                      output_dataset_directory: Optional[Union[str, Path]] = None,
                      use_columns: Optional[list[str]] = None,
                      ignore_columns: Optional[list[str]] = None,
                      max_features_to_plot: int = 20,
                      fontsize: int = 14):
    """
    Computes Variance Inflation Factors (VIF) for numeric columns in a directory with CSV files (loaded as pandas DataFrames). No plots or warnings will be displayed inline.
    Generates a bar plot of VIF values. Optionally drops columns with VIF >= 10 and saves as a new CSV file.

    Args:
        input_directory (str | Path): Target directory with CSV files able to be loaded as DataFrame.
        output_plot_directory (str | Path): Save plots to this directory.
        output_dataset_directory (str | Path | None): If provided, saves new CSV files to this directory.
        use_columns (list[str] | None): Optional list of columns to include. Defaults to all numeric columns.
        ignore_columns (list[str] | None): Optional list of columns to exclude from the VIF computation. Skipped if `use_columns` is provided.
        max_features_to_plot (int): Adjust the number of features shown in the plot.
        fontsize (int): Base fontsize to scale title and labels on hte plot.
        
    NOTE:
    **Variance Inflation Factor (VIF)** quantifies the degree of multicollinearity among features in a dataset. 
    A VIF value indicates how much the variance of a regression coefficient is inflated due to linear dependence with other features. 
    A VIF of 1 suggests no correlation, values between 1 and 5 indicate moderate correlation, and values greater than 10 typically signal high multicollinearity, which may distort model interpretation and degrade performance.
    """
    if output_dataset_directory is not None:
        output_dataset_path = make_fullpath(output_dataset_directory, make=True)
    else:
        output_dataset_path = None
    
    for df, df_name in yield_dataframes_from_dir(datasets_dir=input_directory):
        vif_dataframe = compute_vif(df=df,
                            use_columns=use_columns,
                            ignore_columns=ignore_columns,
                            max_features_to_plot=max_features_to_plot,
                            fontsize=fontsize,
                            save_dir=output_plot_directory,
                            filename=df_name,
                            show_plot=False,
                            verbose=False)
        
        if output_dataset_path is not None:
            new_filename = df_name + '_VIF'
            result_df, dropped_cols = drop_vif_based(df=df, vif_df=vif_dataframe)
            
            if len(dropped_cols) > 0:
                save_dataframe(df=result_df, save_dir=output_dataset_path, filename=new_filename)


def info():
    _script_info(__all__)
