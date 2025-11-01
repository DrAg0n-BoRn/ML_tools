import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal, Union, Optional, Dict, Any
from pathlib import Path
import warnings

# --- Third-party imports ---
try:
    from pytorch_tabular.models.common.heads import LinearHeadConfig
    from pytorch_tabular.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        TrainerConfig,
        ExperimentConfig,
    )
    from pytorch_tabular.models import (
        CategoryEmbeddingModelConfig,
        TabNetModelConfig,
        TabTransformerConfig,
        FTTransformerConfig,
        AutoIntConfig,
        NodeConfig,
        GANDALFConfig
    )
    from pytorch_tabular.tabular_model import TabularModel
except ImportError:
    print("----------------------------------------------------------------")
    print("ERROR: `pytorch-tabular` is not installed.")
    print("Please install it to use the models in this script:")
    print('\npip install "dragon-ml-toolbox[py-tab]"')
    print("----------------------------------------------------------------")
    raise

# --- Local ML-Tools imports ---
from ._logger import _LOGGER
from ._script_info import _script_info
from ._schema import FeatureSchema
from .path_manager import make_fullpath, sanitize_filename
from .keys import SHAPKeys
from .ML_datasetmaster import _PytorchDataset
from .ML_evaluation import (
    classification_metrics, 
    regression_metrics
)
from .ML_evaluation_multi import (
    multi_target_regression_metrics, 
    multi_label_classification_metrics
)


__all__ = [
    "PyTabularTrainer"
]


# --- Model Configuration Mapping ---
# Maps a simple string name to the required ModelConfig class
SUPPORTED_MODELS: Dict[str, Any] = {
    "TabNet": TabNetModelConfig,
    "TabTransformer": TabTransformerConfig,
    "FTTransformer": FTTransformerConfig,
    "AutoInt": AutoIntConfig,
    "NODE": NodeConfig,
    "GATE": GANDALFConfig, # Gated Additive Tree Ensemble
    "CategoryEmbedding": CategoryEmbeddingModelConfig, # A basic MLP
}


class PyTabularTrainer:
    """
    A wrapper for models from the `pytorch-tabular` library, designed to be
    compatible with the `dragon-ml-toolbox` ecosystem.
    
    This class acts as a high-level trainer that adapts the `ML_datasetmaster`
    datasets into the format required by `pytorch-tabular` and routes
    evaluation results to the standard `ML_evaluation` functions.
    
    It handles:
    - Automatic `DataConfig` creation from a `FeatureSchema`.
    - Model and Trainer configuration.
    - Training and evaluation.
    - SHAP explanations.
    """
    
    def __init__(self,
                 schema: FeatureSchema,
                 target_names: List[str],
                 kind: Literal["regression", "classification", "multi_target_regression", "multi_label_classification"],
                 model_name: str,
                 model_config_params: Optional[Dict[str, Any]] = None,
                 optimizer_config_params: Optional[Dict[str, Any]] = None,
                 trainer_config_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Model, Data, and Trainer configurations.

        Args:
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            target_names (List[str]): 
                A list of target column names.
            kind (Literal[...]): 
                The type of ML task. This is used to set the `pytorch-tabular`
                task and to route to the correct evaluation function.
            model_name (str): 
                The name of the model to use. Must be one of:
                "TabNet", "TabTransformer", "FTTransformer", "AutoInt",
                "NODE", "GATE", "CategoryEmbedding".
            model_config_params (Dict, optional): 
                Overrides for the chosen model's `ModelConfig`.
                (e.g., `{"n_d": 16, "n_a": 16}` for TabNet).
            optimizer_config_params (Dict, optional): 
                Overrides for the `OptimizerConfig` (e.g., `{"lr": 0.005}`).
            trainer_config_params (Dict, optional): 
                Overrides for the `TrainerConfig` (e.g., `{"max_epochs": 100}`).
        """
        _LOGGER.info(f"Initializing PyTabularTrainer for model: {model_name}")
        
        # --- 1. Store key info ---
        self.schema = schema
        self.target_names = target_names
        self.kind = kind
        self.model_name = model_name
        self._is_fitted = False

        if model_name not in SUPPORTED_MODELS:
            _LOGGER.error(f"Model '{model_name}' is not supported. Choose from: {list(SUPPORTED_MODELS.keys())}")
            raise ValueError(f"Unsupported model: {model_name}")

        # --- 2. Map ML-Tools 'kind' to pytorch-tabular 'task' ---
        if kind == "regression":
            self.task = "regression"
            self._pt_target_names = target_names
        elif kind == "classification":
            self.task = "classification"
            self._pt_target_names = target_names
        elif kind == "multi_target_regression":
            self.task = "multi-label-regression" # pytorch-tabular's name
            self._pt_target_names = target_names
        elif kind == "multi_label_classification":
            self.task = "multi-label-classification"
            self._pt_target_names = target_names
        else:
            _LOGGER.error(f"Unknown task 'kind': {kind}")
            raise ValueError()

        # --- 3. Create DataConfig from FeatureSchema ---
        # Note: pytorch-tabular handles scaling internally
        self.data_config = DataConfig(
            target=self._pt_target_names,
            continuous_cols=list(schema.continuous_feature_names),
            categorical_cols=list(schema.categorical_feature_names),
            continuous_feature_transform="quantile_normal", 
        )

        # --- 4. Create ModelConfig ---
        model_config_class = SUPPORTED_MODELS[model_name]
        
        # Apply user overrides
        if model_config_params is None:
            model_config_params = {}
            
        # Set task in params
        model_config_params["task"] = self.task
        
        # Handle multi-target output for regression
        if self.task == "multi-label-regression":
            # Must configure the model's output head
            if "head" not in model_config_params:
                _LOGGER.info("Configuring model head for multi-target regression.")
                model_config_params["head"] = "LinearHead"
                model_config_params["head_config"] = {
                    "layers": "", # No hidden layers in the head
                    "output_dim": len(self.target_names)
                }

        self.model_config = model_config_class(**model_config_params)

        # --- 5. Create OptimizerConfig ---
        if optimizer_config_params is None:
            optimizer_config_params = {}
        self.optimizer_config = OptimizerConfig(**optimizer_config_params)

        # --- 6. Create TrainerConfig ---
        if trainer_config_params is None:
            trainer_config_params = {}
        
        # Default to GPU if available
        if "accelerator" not in trainer_config_params:
            if torch.cuda.is_available():
                trainer_config_params["accelerator"] = "cuda"
            elif torch.backends.mps.is_available():
                 trainer_config_params["accelerator"] = "mps"
            else:
                 trainer_config_params["accelerator"] = "cpu"
        
        # Set other sensible defaults
        if "checkpoints" not in trainer_config_params:
            trainer_config_params["checkpoints"] = "val_loss"
            trainer_config_params["load_best_at_end"] = True
            
        if "early_stopping" not in trainer_config_params:
            trainer_config_params["early_stopping"] = "val_loss"
        
        self.trainer_config = TrainerConfig(**trainer_config_params)

        # --- 7. Instantiate the TabularModel ---
        self.tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
        )

    def _dataset_to_dataframe(self, dataset: _PytorchDataset) -> pd.DataFrame:
        """Converts an _PytorchDataset back into a pandas DataFrame."""
        try:
            features_np = dataset.features.cpu().numpy()
            labels_np = dataset.labels.cpu().numpy()
            feature_names = dataset.feature_names
            target_names = dataset.target_names
        except Exception as e:
            _LOGGER.error(f"Failed to extract data from provided dataset: {e}")
            raise
            
        # Create features DataFrame
        df = pd.DataFrame(features_np, columns=feature_names)
        
        # Add labels
        if labels_np.ndim == 1:
            df[target_names[0]] = labels_np
        elif labels_np.ndim == 2:
            for i, name in enumerate(target_names):
                df[name] = labels_np[:, i]
        
        return df

    def fit(self, 
            train_dataset: _PytorchDataset, 
            test_dataset: _PytorchDataset, 
            epochs: int = 20, 
            batch_size: int = 10):
        """
        Trains the model using the provided datasets.

        Args:
            train_dataset (_PytorchDataset): The training dataset.
            test_dataset (_PytorchDataset): The validation dataset.
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size.
        """
        _LOGGER.info(f"Converting datasets to pandas DataFrame for {self.model_name}...")
        train_df = self._dataset_to_dataframe(train_dataset)
        test_df = self._dataset_to_dataframe(test_dataset)
        
        _LOGGER.info(f"Starting training for {epochs} epochs...")
        with warnings.catch_warnings():
            # Suppress abundant pytorch-lightning warnings
            warnings.simplefilter("ignore")
            self.tabular_model.fit(
                train=train_df,
                validation=test_df,
                max_epochs=epochs
            )
            
        self._is_fitted = True
        _LOGGER.info("Training complete.")

    def evaluate(self, 
                 save_dir: Union[str, Path], 
                 data: _PytorchDataset, 
                 classification_threshold: float = 0.5):
        """
        Evaluates the model and saves reports using the standard ML_evaluation functions.
        
        Args:
            save_dir (str | Path): Directory to save all reports and plots.
            data (_PytorchDataset): The data to evaluate on.
            classification_threshold (float): Threshold for multi-label tasks.
        """
        if not self._is_fitted:
            _LOGGER.error("Model is not fitted. Call .fit() first.")
            raise RuntimeError()
            
        print("\n--- Model Evaluation (PyTorch-Tabular) ---")
        
        eval_df = self._dataset_to_dataframe(data)
        
        # Get raw predictions from pytorch-tabular
        raw_preds_df = self.tabular_model.predict(
            eval_df, 
            include_input_features=False
        )
        
        # Extract y_true from the dataframe
        y_true = eval_df[self.target_names].to_numpy()
        
        y_pred = None
        y_prob = None
        
        # --- Route based on task kind ---
        
        if self.kind == "regression":
            pred_col_name = f"{self.target_names[0]}_prediction"
            y_pred = raw_preds_df[pred_col_name].to_numpy()
            regression_metrics(y_true.flatten(), y_pred.flatten(), save_dir)
            
        elif self.kind == "classification":
            y_pred = raw_preds_df["prediction"].to_numpy()
            # Get class names from the model's datamodule
            if self.tabular_model.datamodule is None:
                _LOGGER.error("Model's datamodule is not initialized. Cannot extract class names for probabilities.")
                raise RuntimeError("Datamodule not found. Was the model trained or loaded correctly?")
            class_names = self.tabular_model.datamodule.data_config.target_classes[self.target_names[0]]
            prob_cols = [f"{c}_probability" for c in class_names]
            y_prob = raw_preds_df[prob_cols].values
            classification_metrics(save_dir, y_true.flatten(), y_pred, y_prob)
            
        elif self.kind == "multi_target_regression":
            pred_cols = [f"{name}_prediction" for name in self.target_names]
            y_pred = raw_preds_df[pred_cols].to_numpy()
            multi_target_regression_metrics(y_true, y_pred, self.target_names, save_dir)
            
        elif self.kind == "multi_label_classification":
            prob_cols = [f"{name}_probability" for name in self.target_names]
            y_prob = raw_preds_df[prob_cols].to_numpy()
            # y_pred is derived from y_prob
            multi_label_classification_metrics(y_true, y_prob, self.target_names, save_dir, classification_threshold)
            
    def explain(self, 
                save_dir: Union[str, Path], 
                explain_dataset: _PytorchDataset):
        """
        Generates SHAP explanations and saves plots and summary CSVs.
        
        This method uses pytorch-tabular's internal `.explain()` method
        and then formats the output to match the ML_evaluation standard.

        Args:
            save_dir (str | Path): Directory to save all SHAP artifacts.
            explain_dataset (_PytorchDataset): The dataset to explain.
        """
        if not self._is_fitted:
            _LOGGER.error("Model is not fitted. Call .fit() first.")
            raise RuntimeError()

        print(f"\n--- SHAP Value Explanation ({self.model_name}) ---")
        
        explain_df = self._dataset_to_dataframe(explain_dataset)
        
        # We must use the dataframe *without* the target columns for explanation
        feature_df: pd.DataFrame = explain_df[self.schema.feature_names] # type: ignore
        
        # This returns a DataFrame (single-target) or Dict[str, DataFrame]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_output = self.tabular_model.explain(feature_df)
        
        save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
        plt.ioff()
        
        # --- 1. Handle single-target (regression/classification) ---
        if isinstance(shap_output, pd.DataFrame):
            # shap_output is (n_samples, n_features)
            shap_values = shap_output.to_numpy()
            
            # Save Bar Plot
            self._save_shap_plots(
                shap_values=shap_values,
                instances_df=feature_df,
                save_dir=save_dir_path,
                suffix="" # No suffix for single target
            )
            # Save Summary Data
            self._save_shap_csv(
                shap_values=shap_values,
                feature_names=list(self.schema.feature_names),
                save_dir=save_dir_path,
                suffix=""
            )
        
        # --- 2. Handle multi-target ---
        elif isinstance(shap_output, dict):
            for target_name, shap_df in shap_output.items(): # type: ignore
                _LOGGER.info(f"  -> Generating SHAP plots for target: '{target_name}'")
                shap_values = shap_df.values
                sanitized_name = sanitize_filename(target_name)
                
                # Save Bar Plot
                self._save_shap_plots(
                    shap_values=shap_values,
                    instances_df=feature_df,
                    save_dir=save_dir_path,
                    suffix=f"_{sanitized_name}",
                    title_suffix=f" for '{target_name}'"
                )
                # Save Summary Data
                self._save_shap_csv(
                    shap_values=shap_values,
                    feature_names=list(self.schema.feature_names),
                    save_dir=save_dir_path,
                    suffix=f"_{sanitized_name}"
                )

        plt.ion()
        _LOGGER.info(f"All SHAP plots saved to '{save_dir_path.name}'")

    def _save_shap_plots(self, shap_values: np.ndarray, 
                         instances_df: pd.DataFrame, 
                         save_dir: Path, 
                         suffix: str = "",
                         title_suffix: str = ""):
        """Internal helper to save standard SHAP plots."""
        try:
            import shap
        except ImportError:
            _LOGGER.error("`shap` is required for plotting. Please install it: pip install shap")
            return
            
        # Save Bar Plot
        bar_path = save_dir / f"shap_bar_plot{suffix}.svg"
        shap.summary_plot(shap_values, instances_df, plot_type="bar", show=False)
        ax = plt.gca()
        ax.set_xlabel("SHAP Value Impact", labelpad=10)
        plt.title(f"SHAP Feature Importance{title_suffix}")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()

        # Save Dot Plot
        dot_path = save_dir / f"shap_dot_plot{suffix}.svg"
        shap.summary_plot(shap_values, instances_df, plot_type="dot", show=False)
        ax = plt.gca()
        ax.set_xlabel("SHAP Value Impact", labelpad=10)
        if plt.gcf().axes and len(plt.gcf().axes) > 1:
            cb = plt.gcf().axes[-1]
            cb.set_ylabel("", size=1)
        plt.title(f"SHAP Feature Importance{title_suffix}")
        plt.tight_layout()
        plt.savefig(dot_path)
        plt.close()

    def _save_shap_csv(self, shap_values: np.ndarray, 
                       feature_names: List[str], 
                       save_dir: Path, 
                       suffix: str = ""):
        """Internal helper to save standard SHAP summary CSV."""
        
        shap_summary_filename = f"{SHAPKeys.SAVENAME}{suffix}.csv"
        summary_path = save_dir / shap_summary_filename
        
        # Handle multi-class (list of arrays) vs. regression (single array)
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(np.stack(shap_values)).mean(axis=0).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        mean_abs_shap = mean_abs_shap.flatten()
            
        summary_df = pd.DataFrame({
            SHAPKeys.FEATURE_COLUMN: feature_names,
            SHAPKeys.SHAP_VALUE_COLUMN: mean_abs_shap
        }).sort_values(SHAPKeys.SHAP_VALUE_COLUMN, ascending=False)
        
        summary_df.to_csv(summary_path, index=False)

    def save_model(self, directory: Union[str, Path]):
        """
        Saves the entire trained model, configuration, and datamodule
        to a directory.

        Args:
            directory (str | Path): The directory to save the model.
                                    The directory will be created.
        """
        if not self._is_fitted:
            _LOGGER.error("Cannot save a model that has not been fitted.")
            return
            
        save_path = make_fullpath(directory, make=True, enforce="directory")
        self.tabular_model.save_model(str(save_path))
        _LOGGER.info(f"Model saved to '{save_path.name}'")

    @classmethod
    def load_model(cls,
                   directory: Union[str, Path],
                   schema: FeatureSchema,
                   target_names: List[str],
                   kind: Literal["regression", "classification", "multi_target_regression", "multi_label_classification"]
                   ) -> 'PyTabularTrainer':
        """
        Loads a saved model and reconstructs the PyTabularTrainer wrapper.
        
        Note: The schema, target_names, and kind must be provided again
        as they are not serialized by pytorch-tabular.

        Args:
            directory (str | Path): The directory from which to load the model.
            schema (FeatureSchema): The schema used during original training.
            target_names (List[str]): The target names used during original training.
            kind (Literal[...]): The task 'kind' used during original training.

        Returns:
            PyTabularTrainer: A new instance of the trainer with the loaded model.
        """
        load_path = make_fullpath(directory, enforce="directory")
        
        _LOGGER.info(f"Loading model from '{load_path.name}'...")
        
        # Load the internal pytorch-tabular model
        loaded_tabular_model = TabularModel.load_model(str(load_path))
        
        if loaded_tabular_model.model is None:
             _LOGGER.error("Loaded model's internal '.model' attribute is None. Load failed.")
             raise RuntimeError("Loaded model is incomplete.")
        
        model_name = loaded_tabular_model.model._model_name
        
        if model_name.startswith("GANDALF"): # Handle GANDALF's dynamic name
            model_name = "GATE"
        
        # Re-create the wrapper
        wrapper = cls(
            schema=schema,
            target_names=target_names,
            kind=kind,
            model_name=model_name 
            # Configs are already part of the loaded_tabular_model
            # We just need to pass the minimum to the __init__
        )
        
        # Overwrite the un-trained model with the loaded trained model
        wrapper.tabular_model = loaded_tabular_model
        wrapper._is_fitted = True
        
        _LOGGER.info(f"Successfully loaded '{model_name}' model.")
        return wrapper


def info():
    _script_info(__all__)