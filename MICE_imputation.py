import pandas as pd
import miceforest as mf
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def load_dataframe(df_path: str):
    df = pd.read_csv(df_path, encoding='utf-8')
    print(f"Loaded dataframe shape: {df.shape}")
    return df


def apply_mice(df: pd.DataFrame, resulting_datasets: int=2, iterations: int=20, random_state: int=101):
    
    # Initialize kernel with number of imputed datasets to generate
    kernel = mf.ImputationKernel(
        data=df,
        num_datasets=resulting_datasets,
        random_state=random_state
    )
    
    # Perform MICE with n iterations per dataset
    kernel.mice(iterations)
    
    # Retrieve the imputed datasets 
    imputed_datasets = [kernel.complete_data(dataset=i) for i in range(resulting_datasets)]
    
    # Ensure indexes match
    for i, imputed_df in enumerate(imputed_datasets, start=1):
        assert imputed_df.shape[0] == df.shape[0], f"Row count mismatch in dataset {i}"
        assert all(imputed_df.index == df.index), f"Index mismatch in dataset {i}"
    print("✅ All imputed datasets match the original DataFrame indexes.")
    
    return kernel, imputed_datasets


def save_datasets(save_dir: str, imputed_datasets: list):
    for i, imputed_df in enumerate(imputed_datasets, start=1):
        if i < 10:
            file_name = f"imputed_0{i}.csv"
        else:
            file_name = f"imputed_{i}.csv"
        output_path = os.path.join(save_dir, file_name)
        imputed_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {file_name} with shape {imputed_df.shape}")
        

#Get names of features that had missing values before imputation
def get_na_feature_names(df: pd.DataFrame):
    return [col for col in df.columns if df[col].isna().any()]


#Convergence diagnostic
def get_convergence_diagnostic(kernel: mf.ImputationKernel, feature_names: list[str], save_dir: str):
    # get number of iterations used
    iterations_cap = kernel.iteration_count()
    
    # iterate over each imputed dataset
    for dataset_id in range(kernel.num_datasets):
        #Check directory for current dataset
        dataset_file_dir = f"Convergence Metrics {dataset_id + 1}"
        local_save_dir = os.path.join(save_dir, dataset_file_dir)
        if not os.path.isdir(local_save_dir):
            os.makedirs(local_save_dir)
        
        for feature_name in feature_names:
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
def get_imputed_distributions(kernel: mf.ImputationKernel, save_dir: str, feature_names: Union[list[str], None]=None, individual_plots: bool=True, fontsize: int=22):
    ''' 
    It works using miceforest's authors implementation of the method `.plot_imputed_distributions()`.
    
    Set individual_plots=False to save a single image with all feature distributions instead.
    '''
    # Check path
    os.makedirs(save_dir, exist_ok=True)
    
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
            os.path.join(save_dir, filename),
            format='svg',
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

    if individual_plots and feature_names:
        # Generate individual plots per feature
        for feature in feature_names:
            fig = kernel.plot_imputed_distributions(variables=[feature])
            _process_figure(fig, f"Distribution_{feature}.svg")
    else:
        # Generate combined plot
        fig = kernel.plot_imputed_distributions(variables=feature_names)
        _process_figure(fig, "Combined_Distributions.svg")
    
    print("Imputed distributions saved successfully.")
