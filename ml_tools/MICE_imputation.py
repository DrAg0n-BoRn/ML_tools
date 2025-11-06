import pandas as pd
import miceforest as mf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plotnine import ggplot, labs, theme, element_blank # type: ignore
from typing import Optional, Union

from .utilities import load_dataframe, merge_dataframes, save_dataframe_filename
from .math_utilities import threshold_binary_values, discretize_categorical_values
from .path_manager import sanitize_filename, make_fullpath, list_csv_paths
from ._logger import _LOGGER
from ._script_info import _script_info
from ._schema import FeatureSchema


__all__ = [
    "DragonMICE",
    "apply_mice",
    "save_imputed_datasets",
    "get_convergence_diagnostic",
    "get_imputed_distributions",
    "run_mice_pipeline",
]


def apply_mice(df: pd.DataFrame, df_name: str, binary_columns: Optional[list[str]]=None, resulting_datasets: int=1, iterations: int=20, random_state: int=101):
    
    # Initialize kernel with number of imputed datasets to generate
    kernel = mf.ImputationKernel(
        data=df,
        num_datasets=resulting_datasets,
        random_state=random_state
    )
    
    _LOGGER.info("➡️ MICE imputation running...")
    
    # Perform MICE with n iterations per dataset
    kernel.mice(iterations)
    
    # Retrieve the imputed datasets 
    imputed_datasets = [kernel.complete_data(dataset=i) for i in range(resulting_datasets)]
    
    if imputed_datasets is None or len(imputed_datasets) == 0:
        _LOGGER.error("No imputed datasets were generated. Check the MICE process.")
        raise ValueError()
    
    # threshold binary columns
    if binary_columns is not None:
        invalid_binary_columns = set(binary_columns) - set(df.columns)
        if invalid_binary_columns:
            _LOGGER.warning(f"These 'binary columns' are not in the dataset:")
            for invalid_binary_col in invalid_binary_columns:
                print(f"  - {invalid_binary_col}")
        valid_binary_columns = [col for col in binary_columns if col not in invalid_binary_columns]
        for imputed_df in imputed_datasets:
            for binary_column_name in valid_binary_columns:
                imputed_df[binary_column_name] = threshold_binary_values(imputed_df[binary_column_name]) # type: ignore
            
    if resulting_datasets == 1:
        imputed_dataset_names = [f"{df_name}_MICE"]
    else:
        imputed_dataset_names = [f"{df_name}_MICE_{i+1}" for i in range(resulting_datasets)]
    
    # Ensure indexes match
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        assert imputed_df.shape[0] == df.shape[0], f"❌ Row count mismatch in dataset {subname}" # type: ignore
        assert all(imputed_df.index == df.index), f"❌ Index mismatch in dataset {subname}" # type: ignore
    # print("✅ All imputed datasets match the original DataFrame indexes.")
    
    _LOGGER.info("MICE imputation complete.")
    
    return kernel, imputed_datasets, imputed_dataset_names


def save_imputed_datasets(save_dir: Union[str, Path], imputed_datasets: list, df_targets: pd.DataFrame, imputed_dataset_names: list[str]):
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        merged_df = merge_dataframes(imputed_df, df_targets, direction="horizontal", verbose=False)
        save_dataframe_filename(df=merged_df, save_dir=save_dir, filename=subname)


#Get names of features that had missing values before imputation
def _get_na_column_names(df: pd.DataFrame):
    return [col for col in df.columns if df[col].isna().any()]


#Convergence diagnostic
def get_convergence_diagnostic(kernel: mf.ImputationKernel, imputed_dataset_names: list[str], column_names: list[str], root_dir: Union[str,Path], fontsize: int=16):
    """
    Generate and save convergence diagnostic plots for imputed variables.

    Parameters:
    - kernel: Trained miceforest.ImputationKernel.
    - imputed_dataset_names: Names assigned to each imputed dataset.
    - column_names: List of feature names to track over iterations.
    - root_dir: Directory to save convergence plots.
    """
    # get number of iterations used
    iterations_cap = kernel.iteration_count()
    dataset_count = kernel.num_datasets
    
    if dataset_count != len(imputed_dataset_names):
        _LOGGER.error(f"Expected {dataset_count} names in imputed_dataset_names, got {len(imputed_dataset_names)}")
        raise ValueError()
    
    # Check path
    root_path = make_fullpath(root_dir, make=True)
    
    # Styling parameters
    label_font = {'size': fontsize, 'weight': 'bold'}
    
    # iterate over each imputed dataset
    for dataset_id, imputed_dataset_name in zip(range(dataset_count), imputed_dataset_names):
        #Check directory for current dataset
        dataset_file_dir = f"Convergence_Metrics_{imputed_dataset_name}"
        local_save_dir = make_fullpath(input_path=root_path / dataset_file_dir, make=True)
        
        for feature_name in column_names:
            means_per_iteration = []
            for iteration in range(iterations_cap):
                current_imputed = kernel.complete_data(dataset=dataset_id, iteration=iteration)
                means_per_iteration.append(np.mean(current_imputed[feature_name])) # type: ignore
                
            plt.figure(figsize=(10, 8))
            plt.plot(means_per_iteration, marker='o')
            plt.xlabel("Iteration", **label_font)
            plt.ylabel("Mean of Imputed Values", **label_font)
            plt.title(f"Mean Convergence for '{feature_name}'", **label_font)
            
            # Adjust plot display for the X axis
            _ticks = np.arange(iterations_cap)
            _labels = np.arange(1, iterations_cap + 1)
            plt.xticks(ticks=_ticks, labels=_labels) # type: ignore
            plt.grid(True)
            
            feature_save_name = sanitize_filename(feature_name)
            feature_save_name = feature_save_name + ".svg"
            save_path = local_save_dir / feature_save_name
            plt.savefig(save_path, bbox_inches='tight', format="svg")
            plt.close()
            
        _LOGGER.info(f"{dataset_file_dir} process completed.")


# Imputed distributions
def get_imputed_distributions(kernel: mf.ImputationKernel, df_name: str, root_dir: Union[str, Path], column_names: list[str], one_plot: bool=False, fontsize: int=14):
    ''' 
    It works using miceforest's authors implementation of the method `.plot_imputed_distributions()`.
    
    Set `one_plot=True` to save a single image including all feature distribution plots instead.
    '''
    # Check path
    root_path = make_fullpath(root_dir, make=True)

    local_dir_name = f"Distribution_Metrics_{df_name}_imputed"
    local_save_dir = make_fullpath(root_path / local_dir_name, make=True)
    
    # Styling parameters
    legend_kwargs = {'frameon': True, 'facecolor': 'white', 'framealpha': 0.8}
    label_font = {'size': fontsize, 'weight': 'bold'}

    def _process_figure(fig, filename: str):
        """Helper function to add labels and legends to a figure"""
        
        if not isinstance(fig, ggplot):
            _LOGGER.error(f"Expected a plotnine.ggplot object, received {type(fig)}.")
            raise TypeError()
        
        # Edit labels and title
        fig = fig + theme(
                plot_title=element_blank(),  # removes labs(title=...)
                strip_text=element_blank()   # removes facet_wrap labels
            )
        
        fig = fig + labs(y="", x="")
        
        # Render to matplotlib figure
        fig = fig.draw()
        
        if not hasattr(fig, 'axes') or len(fig.axes) == 0:
            _LOGGER.error("Rendered figure has no axes to modify.")
            raise RuntimeError()
        
        if filename == "Combined_Distributions":
            custom_xlabel = "Feature Values"
        else:
            custom_xlabel = filename
        
        for ax in fig.axes:            
            # Set axis labels
            ax.set_xlabel(custom_xlabel, **label_font)
            ax.set_ylabel('Distribution', **label_font)
            
            # Add legend based on line colors
            lines = ax.get_lines()
            if len(lines) >= 1:
                lines[0].set_label('Original Data')
                if len(lines) > 1:
                    lines[1].set_label('Imputed Data')
                ax.legend(**legend_kwargs)
                
        # Adjust layout and save
        # fig.tight_layout()
        # fig.subplots_adjust(bottom=0.2, left=0.2)  # Optional, depending on overflow
        
        # sanitize savename
        feature_save_name = sanitize_filename(filename)
        feature_save_name = feature_save_name + ".svg"
        new_save_path = local_save_dir / feature_save_name
        
        fig.savefig(
            new_save_path,
            format='svg',
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close(fig)
    
    if one_plot:
        # Generate combined plot
        fig = kernel.plot_imputed_distributions(variables=column_names)
        _process_figure(fig, "Combined_Distributions")
        # Generate individual plots per feature
    else:
        for feature in column_names:
            fig = kernel.plot_imputed_distributions(variables=[feature])
            _process_figure(fig, feature)

    _LOGGER.info(f"{local_dir_name} completed.")


def run_mice_pipeline(df_path_or_dir: Union[str,Path], target_columns: list[str], 
                      save_datasets_dir: Union[str,Path], save_metrics_dir: Union[str,Path], 
                      binary_columns: Optional[list[str]]=None,
                      resulting_datasets: int=1, 
                      iterations: int=20, 
                      random_state: int=101):
    """
    Call functions in sequence for each dataset in the provided path or directory:
        1. Load dataframe
        2. Apply MICE
        3. Save imputed dataset(s)
        4. Save convergence metrics
        5. Save distribution metrics
        
    Target columns must be skipped from the imputation. Binary columns will be thresholded after imputation.
    """
    # Check paths
    save_datasets_path = make_fullpath(save_datasets_dir, make=True)
    save_metrics_path = make_fullpath(save_metrics_dir, make=True)
    
    input_path = make_fullpath(df_path_or_dir)
    if input_path.is_file():
        all_file_paths = [input_path]
    else:
        all_file_paths = list(list_csv_paths(input_path).values())
    
    for df_path in all_file_paths:
        df: pd.DataFrame
        df, df_name = load_dataframe(df_path=df_path, kind="pandas") # type: ignore
        
        df, df_targets = _skip_targets(df, target_columns)
        
        kernel, imputed_datasets, imputed_dataset_names = apply_mice(df=df, df_name=df_name, binary_columns=binary_columns, resulting_datasets=resulting_datasets, iterations=iterations, random_state=random_state)
        
        save_imputed_datasets(save_dir=save_datasets_path, imputed_datasets=imputed_datasets, df_targets=df_targets, imputed_dataset_names=imputed_dataset_names)
        
        imputed_column_names = _get_na_column_names(df=df)
        
        get_convergence_diagnostic(kernel=kernel, imputed_dataset_names=imputed_dataset_names, column_names=imputed_column_names, root_dir=save_metrics_path)
        
        get_imputed_distributions(kernel=kernel, df_name=df_name, root_dir=save_metrics_path, column_names=imputed_column_names)


def _skip_targets(df: pd.DataFrame, target_cols: list[str]):
    valid_targets = [col for col in target_cols if col in df.columns]
    df_targets = df[valid_targets]
    df_feats = df.drop(columns=valid_targets)
    return df_feats, df_targets


# modern implementation
class DragonMICE:
    """
    A modern MICE imputation pipeline that uses a FeatureSchema
    to correctly discretize categorical features after imputation.
    """
    def __init__(self, 
                 schema: FeatureSchema,
                 iterations: int=20,
                 resulting_datasets: int = 1,
                 random_state: int = 101):
        
        self.schema = schema
        self.random_state = random_state
        self.iterations = iterations
        self.resulting_datasets = resulting_datasets
        
        # --- Store schema info ---
        
        # 1. Categorical info
        if not self.schema.categorical_index_map:
            _LOGGER.warning("FeatureSchema has no 'categorical_index_map'. No discretization will be applied.")
            self.cat_info = {}
        else:
            self.cat_info = self.schema.categorical_index_map
            
        # 2. Ordered feature names (critical for index mapping)
        self.ordered_features = list(self.schema.feature_names)
        
        # 3. Names of categorical features
        self.categorical_features = list(self.schema.categorical_feature_names)

        _LOGGER.info(f"DragonMICE initialized. Found {len(self.cat_info)} categorical features to discretize.")
        
    def _post_process(self, imputed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies schema-based discretization to a completed dataframe.
        
        This method works around the behavior of `discretize_categorical_values`
        (which returns a full int32 array) by:
        1. Calling it on the full, ordered feature array.
        2. Extracting *only* the valid discretized categorical columns.
        3. Updating the original float dataframe with these integer values.
        """
        # If no categorical features are defined, return the df as-is.
        if not self.cat_info:
            return imputed_df

        try:
            # 1. Ensure DataFrame columns match the schema order
            # This is critical for the index-based categorical_info
            df_ordered: pd.DataFrame = imputed_df[self.ordered_features] # type: ignore
            
            # 2. Convert to NumPy array
            array_ordered = df_ordered.to_numpy()

            # 3. Apply discretization utility (which returns a full int32 array)
            # This array has *correct* categorical values but *truncated* continuous values.
            discretized_array_int32 = discretize_categorical_values(
                array_ordered,
                self.cat_info,
                start_at_zero=True  # Assuming 0-based indexing
            )

            # 4. Create a new DF from the int32 array, keeping the categorical columns.
            df_discretized_cats = pd.DataFrame(
                discretized_array_int32,
                columns=self.ordered_features,
                index=df_ordered.index  # <-- Critical: align index
            )[self.categorical_features] # <-- Select only cat features

            # 5. "Rejoin": Start with a fresh copy of the *original* imputed DF (which has correct continuous floats).
            final_df = df_ordered.copy()
            
            # 6. Use .update() to "paste" the integer categorical values
            # over the old float categorical values. Continuous floats are unaffected.
            final_df.update(df_discretized_cats)
            
            return final_df

        except Exception as e:
            _LOGGER.error(f"Failed during post-processing discretization:\n\tInput DF shape: {imputed_df.shape}\n\tSchema features: {len(self.ordered_features)}\n\tCategorical info keys: {list(self.cat_info.keys())}\n{e}")
            raise
        
    def _run_mice(self, 
                  df: pd.DataFrame, 
                  df_name: str) -> tuple[mf.ImputationKernel, list[pd.DataFrame], list[str]]:
        """
        Runs the MICE kernel and applies schema-based post-processing.
        
        Parameters:
            df (pd.DataFrame): The input dataframe *with NaNs*. Should only contain feature columns.
            df_name (str): The base name for the dataset.

        Returns:
            tuple[mf.ImputationKernel, list[pd.DataFrame], list[str]]:
                - The trained MICE kernel
                - A list of imputed and processed DataFrames
                - A list of names for the new DataFrames
        """
        # Ensure input df only contains features from the schema and is in the correct order.
        try:
            df_feats = df[self.ordered_features]
        except KeyError as e:
            _LOGGER.error(f"Input DataFrame is missing required schema columns: {e}")
            raise
        
        # 1. Initialize kernel
        kernel = mf.ImputationKernel(
            data=df_feats,
            num_datasets=self.resulting_datasets,
            random_state=self.random_state
        )
        
        _LOGGER.info("➡️ Schema-based MICE imputation running...")
        
        # 2. Perform MICE
        kernel.mice(self.iterations)
        
        # 3. Retrieve, process, and collect datasets
        imputed_datasets = []
        for i in range(self.resulting_datasets):
            # complete_data returns a pd.DataFrame
            completed_df = kernel.complete_data(dataset=i)
            
            # Apply our new discretization and ordering
            processed_df = self._post_process(completed_df)
            imputed_datasets.append(processed_df)

        if not imputed_datasets:
            _LOGGER.error("No imputed datasets were generated.")
            raise ValueError()

        # 4. Generate names
        if self.resulting_datasets == 1:
            imputed_dataset_names = [f"{df_name}_MICE"]
        else:
            imputed_dataset_names = [f"{df_name}_MICE_{i+1}" for i in range(self.resulting_datasets)]
        
        # 5. Validate indexes 
        for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
            assert imputed_df.shape[0] == df.shape[0], f"❌ Row count mismatch in dataset {subname}"
            assert all(imputed_df.index == df.index), f"❌ Index mismatch in dataset {subname}"
        
        _LOGGER.info("Schema-based MICE imputation complete.")
        
        return kernel, imputed_datasets, imputed_dataset_names
        
    def run_pipeline(self, 
                     df_path_or_dir: Union[str,Path],
                     save_datasets_dir: Union[str,Path], 
                     save_metrics_dir: Union[str,Path],
                     ):
        """
        Runs the complete MICE imputation pipeline.

        This method automates the entire workflow:
            1. Loads data from a CSV file path or a directory with CSV files.
            2. Separates features and targets based on the `FeatureSchema`.
            3. Runs the MICE algorithm on the feature set.
            4. Applies schema-based post-processing to discretize categorical features.
            5. Saves the final, processed, and imputed dataset(s) (re-joined with targets) to `save_datasets_dir`.
            6. Generates and saves convergence and distribution plots for all imputed columns to `save_metrics_dir`.

        Parameters
        ----------
        df_path_or_dir :[str,Path]
            Path to a single CSV file or a directory containing multiple CSV files to impute.
        save_datasets_dir : [str,Path]
            Directory where the final imputed and processed dataset(s) will be saved as CSVs.
        save_metrics_dir : [str,Path]
            Directory where convergence and distribution plots will be saved.
        """
        # Check paths
        save_datasets_path = make_fullpath(save_datasets_dir, make=True)
        save_metrics_path = make_fullpath(save_metrics_dir, make=True)
        
        input_path = make_fullpath(df_path_or_dir)
        if input_path.is_file():
            all_file_paths = [input_path]
        else:
            all_file_paths = list(list_csv_paths(input_path).values())
        
        for df_path in all_file_paths:
            
            df, df_name = load_dataframe(df_path=df_path, kind="pandas")
            
            df_features: pd.DataFrame = df[self.schema.feature_names] # type: ignore
            df_targets = df.drop(columns=self.schema.feature_names)
            
            imputed_column_names = _get_na_column_names(df=df_features)
            
            kernel, imputed_datasets, imputed_dataset_names = self._run_mice(df=df_features, df_name=df_name)
            
            save_imputed_datasets(save_dir=save_datasets_path, imputed_datasets=imputed_datasets, df_targets=df_targets, imputed_dataset_names=imputed_dataset_names)
            
            get_convergence_diagnostic(kernel=kernel, imputed_dataset_names=imputed_dataset_names, column_names=imputed_column_names, root_dir=save_metrics_path)
            
            get_imputed_distributions(kernel=kernel, df_name=df_name, root_dir=save_metrics_path, column_names=imputed_column_names)


def info():
    _script_info(__all__)
