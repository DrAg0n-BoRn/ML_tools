import pandas as pd
import miceforest as mf
import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import load_dataframe, list_csv_paths


def apply_mice(df: pd.DataFrame, df_name: str, resulting_datasets: int=1, iterations: int=20, random_state: int=101):
    
    # Initialize kernel with number of imputed datasets to generate
    kernel = mf.ImputationKernel(
        data=df,
        datasets=resulting_datasets,
        random_state=random_state
    )
    
    # Perform MICE with n iterations per dataset
    kernel.mice(iterations)
    
    # Retrieve the imputed datasets 
    imputed_datasets = [kernel.complete_data(dataset=i) for i in range(resulting_datasets)]
    
    if resulting_datasets == 1:
        imputed_dataset_names = [f"{df_name}_imputed"]
    else:
        imputed_dataset_names = [f"{df_name}_imputed_{i+1}" for i in range(resulting_datasets)]
    
    # Ensure indexes match
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        assert imputed_df.shape[0] == df.shape[0], f"Row count mismatch in dataset {subname}"
        assert all(imputed_df.index == df.index), f"Index mismatch in dataset {subname}"
    # print("✅ All imputed datasets match the original DataFrame indexes.")
    
    return kernel, imputed_datasets, imputed_dataset_names


def save_imputed_datasets(save_dir: str, imputed_datasets: list, imputed_dataset_names: list[str]):
    # Check path
    os.makedirs(save_dir, exist_ok=True)
    
    for imputed_df, subname in zip(imputed_datasets, imputed_dataset_names):
        output_path = os.path.join(save_dir, subname + ".csv")
        imputed_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\tSaved {subname} with shape {imputed_df.shape}")
        

#Get names of features that had missing values before imputation
def get_na_column_names(df: pd.DataFrame):
    return [col for col in df.columns if df[col].isna().any()]


#Convergence diagnostic
def get_convergence_diagnostic(kernel: mf.ImputationKernel, imputed_dataset_names: list[str], column_names: list[str], root_dir: str):
    # get number of iterations used
    iterations_cap = kernel.iteration_count()
    
    # Check path
    os.makedirs(root_dir, exist_ok=True)
    
    # iterate over each imputed dataset
    for dataset_id, imputed_dataset_name in zip(range(kernel.num_datasets), imputed_dataset_names):
        #Check directory for current dataset
        dataset_file_dir = f"Convergence_Metrics_{imputed_dataset_name}"
        local_save_dir = os.path.join(root_dir, dataset_file_dir)
        if not os.path.isdir(local_save_dir):
            os.makedirs(local_save_dir)
        
        for feature_name in column_names:
            means_per_iteration = []
            for iteration in range(iterations_cap):
                current_imputed = kernel.complete_data(dataset=dataset_id, iteration=iteration)
                means_per_iteration.append(np.mean(current_imputed[feature_name]))

            plt.plot(means_per_iteration, marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Mean of Imputed Values")
            plt.title(f"Mean Convergence for '{feature_name}'")
            
            # Adjust plot display for the X axis
            _ticks = np.arange(iterations_cap)
            _labels = np.arange(1, iterations_cap + 1)
            plt.xticks(ticks=_ticks, labels=_labels)
            
            save_path = os.path.join(local_save_dir, feature_name + ".svg")
            plt.savefig(save_path, bbox_inches='tight', format="svg")
            plt.close()
            
        print(f"{dataset_file_dir} completed.")


# Imputed distributions
def get_imputed_distributions(kernel: mf.ImputationKernel, df_name: str, root_dir: str, column_names: list[str], one_plot: bool=False, fontsize: int=18):
    ''' 
    It works using miceforest's authors implementation of the method `.plot_imputed_distributions()`.
    
    Set `one_plot=True` to save a single image including all feature distribution plots instead.
    '''
    # Check path
    os.makedirs(root_dir, exist_ok=True)
    local_save_dir = os.path.join(root_dir, f"Distribution_Metrics_{df_name}")
    if not os.path.isdir(local_save_dir):
        os.makedirs(local_save_dir)
    
    # Styling parameters
    legend_kwargs = {'frameon': True, 'facecolor': 'white', 'framealpha': 0.8}
    label_font = {'size': fontsize, 'weight': 'bold'}

    def _process_figure(fig, filename):
        """Helper function to add labels and legends to a figure"""
        for ax in fig.axes:
            # Set axis labels
            ax.set_xlabel('Value', **label_font)
            ax.set_ylabel('Density', **label_font)
            
            # Add legend based on line colors
            lines = ax.get_lines()
            if len(lines) >= 1:
                lines[0].set_label('Original Data')
                if len(lines) > 1:
                    lines[1].set_label('Imputed Data')
                ax.legend(**legend_kwargs)
                
        # Adjust layout and save
        fig.tight_layout()
        fig.savefig(
            os.path.join(local_save_dir, filename),
            format='svg',
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

    if one_plot:
        # Generate combined plot
        fig = kernel.plot_imputed_distributions(variables=column_names)
        _process_figure(fig, "Combined_Distributions.svg")
        # Generate individual plots per feature
    else:
        for feature in column_names:
            fig = kernel.plot_imputed_distributions(variables=[feature])
            _process_figure(fig, f"{feature}.svg")

    print("Imputed distributions saved successfully.")


def run_mice_pipeline(df_path_or_dir: str, save_datasets_dir: str, save_metrics_dir: str, resulting_datasets: int=1, iterations: int=20, random_state: int=101):
    """
    Call functions in sequence for each dataset in the provided path or directory:
        1. Load dataframe
        2. Apply MICE
        3. Save imputed dataset(s)
        4. Save convergence metrics
        5. Save distribution metrics
    """
    # Check paths
    os.makedirs(save_datasets_dir, exist_ok=True)
    os.makedirs(save_metrics_dir, exist_ok=True)
    
    if os.path.isfile(df_path_or_dir):
        all_file_paths = [df_path_or_dir]
    elif os.path.isdir(df_path_or_dir):
        all_file_paths, _ = list_csv_paths(df_path_or_dir)
    else:
        raise ValueError(f"Invalid path or directory: {df_path_or_dir}")
    
    for df_path in all_file_paths:
        df, df_name = load_dataframe(df_path=df_path)
        
        kernel, imputed_datasets, imputed_dataset_names = apply_mice(df=df, df_name=df_name, resulting_datasets=resulting_datasets, iterations=iterations, random_state=random_state)
        
        save_imputed_datasets(save_dir=save_datasets_dir, imputed_datasets=imputed_datasets, imputed_dataset_names=imputed_dataset_names)
        
        imputed_column_names = get_na_column_names(df=df)
        
        get_convergence_diagnostic(kernel=kernel, imputed_dataset_names=imputed_dataset_names, column_names=imputed_column_names, root_dir=save_metrics_dir)
        
        get_imputed_distributions(kernel=kernel, df_name=df_name, root_dir=save_metrics_dir, column_names=imputed_column_names)
