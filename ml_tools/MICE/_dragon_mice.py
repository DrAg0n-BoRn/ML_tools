import pandas as pd
from pathlib import Path
from typing import Union
import miceforest as mf # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from plotnine import ggplot, labs, theme, element_blank # type: ignore

from ..utilities import load_dataframe, merge_dataframes, save_dataframe_filename
from ..schema import FeatureSchema

from ..math_utilities import discretize_categorical_values
from ..path_manager import make_fullpath, sanitize_filename
from .._core import get_logger


_LOGGER = get_logger("Dragon MICE")


__all__ = [
    "DragonMICE",
    "get_convergence_diagnostic",
    "get_imputed_distributions",
]


class DragonMICE:
    """
    A modern MICE imputation pipeline that uses a FeatureSchema
    to correctly discretize categorical features after imputation.
    
    Optionally supports Target Imputation.
    """
    def __init__(
        self,
        input_df_or_path: Union[pd.DataFrame, str, Path],
        schema: FeatureSchema,
        random_state: int = 101
    ):
        """
        Initializes the DragonMICE imputation pipeline with a given dataset and schema.
        
        Args:
            input_df_or_path (pd.DataFrame | str | Path): Input dataset as a pandas DataFrame or a path to a CSV file.
            schema (FeatureSchema): A FeatureSchema instance that defines the features, including categorical features and their mappings.
            random_state (int): Seed for reproducibility of the MICE imputation process.
        """
        # Validation
        if not isinstance(schema, FeatureSchema):
            _LOGGER.error(f"Expected a FeatureSchema instance for 'schema', got {type(schema)}")
            raise TypeError()
        
        self._schema = schema
        self._random_state = random_state
        
        # --- Handle Input DataFrame or Path ---
        if isinstance(input_df_or_path, pd.DataFrame):
            self._df = input_df_or_path.copy()
            self._df_name = "unnamed_dataset"
        elif isinstance(input_df_or_path, (str, Path)):
            self._df, self._df_name = load_dataframe(df_path=input_df_or_path, kind="pandas")
        else:
            _LOGGER.error("Input must be a pandas DataFrame, string, or pathlib.Path.")
            raise TypeError()

        # --- Store schema info ---
        
        # 1. Categorical info
        if not self._schema.categorical_index_map:
            _LOGGER.warning("FeatureSchema has no 'categorical_index_map'. No discretization will be applied.")
            self._cat_info = {}
        else:
            self._cat_info = self._schema.categorical_index_map
            
        # 2. Ordered feature names (critical for index mapping)
        # Convert to list immediately to avoid Pandas Tuple indexing errors
        self._ordered_features = list(self._schema.feature_names)
        
        # 3. Names of categorical features
        self._categorical_features = list(self._schema.categorical_feature_names)

        _LOGGER.info(f"DragonMICE initialized for input '{self._df_name}'. Found {len(self._cat_info)} categorical features to discretize.")

    @property
    def schema(self) -> FeatureSchema:
        """Exposes the used FeatureSchema as read-only for inspection/logging purposes."""
        return self._schema
    
    def _prepare_data(self, impute_targets: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Splits the dataframe based on target imputation configuration."""
        if impute_targets:
            df_input = self._df
            df_targets_to_save = pd.DataFrame(index=self._df.index) 
        else:
            feature_cols = list(self._schema.feature_names)
            
            if not set(feature_cols).issubset(self._df.columns):
                missing = set(feature_cols) - set(self._df.columns)
                _LOGGER.error(f"Dataset '{self._df_name}' is missing schema features: {missing}")
                raise KeyError()

            df_input = self._df[feature_cols]
            df_targets_to_save = self._df.drop(columns=feature_cols)
        
        imputed_column_names = [col for col in df_input.columns if df_input[col].isna().any()]
        return df_input, df_targets_to_save, imputed_column_names

    def _post_process(self, imputed_df: pd.DataFrame) -> pd.DataFrame:
        """Applies schema-based discretization to a completed dataframe."""
        if not self._cat_info:
            return imputed_df

        try:
            df_schema_features = imputed_df[self._ordered_features]
            array_ordered = df_schema_features.to_numpy()

            discretized_array_int32 = discretize_categorical_values(
                array_ordered,
                self._cat_info,
                start_at_zero=True 
            )

            df_discretized_full = pd.DataFrame(
                discretized_array_int32,
                columns=self._ordered_features,
                index=df_schema_features.index 
            )

            df_discretized_cats = df_discretized_full[self._categorical_features]
            final_df = imputed_df.copy()
            final_df.update(df_discretized_cats)
            
            return final_df

        except Exception as e:
            _LOGGER.error(f"Failed during post-processing discretization:\n\tSchema features: {len(self._ordered_features)}\n{e}")
            raise
        
    def _run_mice(
        self, 
        df_input: pd.DataFrame, 
        impute_targets: bool, 
        iterations: int, 
        resulting_datasets: int
    ) -> tuple[mf.ImputationKernel, list[pd.DataFrame], list[str]]:
        """Runs the MICE kernel and applies schema-based post-processing."""
        kernel = mf.ImputationKernel(
            data=df_input,
            num_datasets=resulting_datasets,
            random_state=self._random_state
        )
        
        message = "➡️ Schema-based MICE imputation running"
        if impute_targets:
            message += " (Targets included)"
        
        _LOGGER.info(message)
        
        try:
            kernel.mice(iterations)
        except Exception as e:
            _LOGGER.error(f"MICE imputation failed during execution: {e}")
            raise
        
        imputed_datasets = []
        for i in range(resulting_datasets):
            completed_df = kernel.complete_data(dataset=i)
            
            if completed_df is None:
                _LOGGER.error(f"Failed to retrieve completed dataset {i}.")
                raise ValueError()
            
            processed_df = self._post_process(completed_df)
            imputed_datasets.append(processed_df)

        if not imputed_datasets:
            _LOGGER.error("No imputed datasets were generated.")
            raise ValueError()

        if resulting_datasets == 1:
            imputed_dataset_names = [f"{self._df_name}_MICE"]
        else:
            imputed_dataset_names = [f"{self._df_name}_MICE_{i+1}" for i in range(resulting_datasets)]
        
        for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
            if imputed_df.shape[0] != self._df.shape[0]:
                _LOGGER.error(f"Row count mismatch in dataset {subname}")
                raise ValueError()
            if not all(imputed_df.index == self._df.index):
                _LOGGER.error(f"Index mismatch in dataset {subname}")
                raise ValueError()
        
        _LOGGER.info("⬅️ Schema-based MICE imputation complete.")
        
        return kernel, imputed_datasets, imputed_dataset_names

    def run_pipeline(
        self,
        save_metrics_dir: Union[str, Path],
        impute_targets: bool = False,
        iterations: int = 30
    ) -> pd.DataFrame:
        """
        Runs the MICE imputation pipeline to generate a single imputed dataframe.
        Saves convergence and distribution metrics to the specified directory.
        
        Args:
            save_metrics_dir (str | Path): Directory to save convergence and distribution metrics.
            impute_targets (bool): Whether to include target columns in the imputation process.
            iterations (int): Number of MICE iterations to perform.
            
        Returns:
            pd.DataFrame: The final imputed dataframe with schema-based discretization applied.
        """
        save_metrics_path = make_fullpath(save_metrics_dir, make=True, enforce="directory")
        
        df_input, df_targets_to_save, imputed_column_names = self._prepare_data(impute_targets=impute_targets)
        
        kernel, imputed_datasets, imputed_dataset_names = self._run_mice(
            df_input=df_input,
            impute_targets=impute_targets,
            iterations=iterations,
            resulting_datasets=1
        )
        
        get_convergence_diagnostic(
            kernel=kernel, 
            imputed_dataset_names=imputed_dataset_names, 
            column_names=imputed_column_names, 
            root_dir=save_metrics_path
        )
        
        get_imputed_distributions(
            kernel=kernel, 
            df_name=self._df_name, 
            root_dir=save_metrics_path, 
            column_names=imputed_column_names
        )
        
        final_df = merge_dataframes(
            imputed_datasets[0], 
            df_targets_to_save, 
            direction="horizontal", 
            verbose=False
        )
        
        return final_df

    def run_pipeline_multi(
        self,
        save_datasets_dir: Union[str, Path],
        save_metrics_dir: Union[str, Path],
        resulting_datasets: int = 5,
        impute_targets: bool = False,
        iterations: int = 30
    ) -> None:
        """
        Runs the MICE imputation pipeline to generate multiple imputed dataframes.
        Saves the resulting datasets and metrics to the specified directories.
        
        Args:
            save_datasets_dir (str | Path): Directory to save the resulting imputed datasets.
            save_metrics_dir (str | Path): Directory to save convergence and distribution metrics.
            resulting_datasets (int): Number of imputed datasets to generate.
            impute_targets (bool): Whether to include target columns in the imputation process.
            iterations (int): Number of MICE iterations to perform.
        """
        save_datasets_path = make_fullpath(save_datasets_dir, make=True, enforce="directory")
        save_metrics_path = make_fullpath(save_metrics_dir, make=True, enforce="directory")
        
        df_input, df_targets_to_save, imputed_column_names = self._prepare_data(impute_targets=impute_targets)
        
        kernel, imputed_datasets, imputed_dataset_names = self._run_mice(
            df_input=df_input,
            impute_targets=impute_targets,
            iterations=iterations,
            resulting_datasets=resulting_datasets
        )
        
        _save_imputed_datasets(
            save_dir=save_datasets_path, 
            imputed_datasets=imputed_datasets, 
            df_targets=df_targets_to_save, 
            imputed_dataset_names=imputed_dataset_names
        )
        
        get_convergence_diagnostic(
            kernel=kernel, 
            imputed_dataset_names=imputed_dataset_names, 
            column_names=imputed_column_names, 
            root_dir=save_metrics_path
        )
        
        get_imputed_distributions(
            kernel=kernel, 
            df_name=self._df_name, 
            root_dir=save_metrics_path, 
            column_names=imputed_column_names
        )


def _save_imputed_datasets(save_dir: Union[str, Path], imputed_datasets: list, df_targets: pd.DataFrame, imputed_dataset_names: list[str]):
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        merged_df = merge_dataframes(imputed_df, df_targets, direction="horizontal", verbose=False)
        save_dataframe_filename(df=merged_df, save_dir=save_dir, filename=subname, verbose=2)


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
        dataset_file_dir = f"Convergence_Metrics_{imputed_dataset_name}"
        local_save_dir = make_fullpath(input_path=root_path / dataset_file_dir, make=True)
        
        # 1. Pre-calculate means for all features across all iterations
        # Structure: {feature_name: [mean_iter_0, mean_iter_1, ...]}
        history = {col: [] for col in column_names}
        
        for iteration in range(iterations_cap):
            # Resolve dataset ONLY ONCE per iteration
            current_imputed = kernel.complete_data(dataset=dataset_id, iteration=iteration)
            
            for col in column_names:
                # Fast lookup
                val = np.mean(current_imputed[col])
                history[col].append(val)

        # 2. Plotting loop
        for feature_name, means_per_iteration in history.items():
            plt.figure(figsize=(10, 8))
            plt.plot(means_per_iteration, marker='o')
            plt.xlabel("Iteration", **label_font)
            plt.ylabel("Mean of Imputed Values", **label_font)
            plt.title(f"Mean Convergence for '{feature_name}'", **label_font)
            
            _ticks = np.arange(iterations_cap)
            _labels = np.arange(1, iterations_cap + 1)
            plt.xticks(ticks=_ticks, labels=_labels)
            plt.grid(True)
            
            feature_save_name = sanitize_filename(feature_name) + ".svg"
            save_path = local_save_dir / feature_save_name
            plt.savefig(save_path, bbox_inches='tight', format="svg")
            plt.close()
            
    _LOGGER.info(f"📉 MICE Convergence diagnostics complete.")


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

    _LOGGER.info(f"📊 MICE Imputed distributions complete.")
    
