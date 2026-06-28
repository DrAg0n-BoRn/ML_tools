import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path

from ..schema import FeatureSchema

from ..utilities import load_dataframe
from ..path_manager import sanitize_filename, make_fullpath
from .._core import get_logger


_LOGGER = get_logger("VIF")


__all__ = [
    "DragonVIF",
]


class DragonVIF:
    """
    A modern utility class for computing Variance Inflation Factors (VIF) and 
    Generalized Variance Inflation Factors (GVIF) for mixed-type datasets.

    Leverages a `FeatureSchema` to intelligently route continuous features through 
    standard VIF calculations and categorical features (dummy-encoded) through 
    group-level Adjusted GVIF calculations. This prevents false multicollinearity 
    alarms that commonly occur when evaluating individual dummy variables.
    """
    def __init__(
        self, 
        df_or_path: Union[pd.DataFrame, str, Path], 
        schema: "FeatureSchema",
        dataset_name: Optional[str] = None
    ):
        """
        Initializes the DragonVIF instance.

        Args:
            df_or_path (pd.DataFrame | str |Path): The dataset to analyze, provided as 
                a pandas DataFrame or a file path to be loaded.
            schema (FeatureSchema): The schema defining continuous and categorical features. 
                Features not present in the schema (e.g., target columns) are ignored.
            dataset_name (str | None): A custom name for the dataset. Defaults to the loaded filename or 'Unnamed_Dataset'.
        """
        self.schema = schema
        self.dataset_name = dataset_name or "Unnamed_Dataset"
        self.vif_results: Optional[pd.DataFrame] = None
        
        if isinstance(df_or_path, (str, Path)):
            self.df, loaded_name = load_dataframe(df_or_path, kind="pandas", verbose=False)
            if dataset_name is None:
                self.dataset_name = loaded_name
        else:
            self.df = df_or_path.copy()

    def compute_vif(
        self, 
        use_columns: Optional[list[str]] = None, 
        ignore_columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Computes standard VIF for continuous features and squared Adjusted GVIF for 
        categorical features using the inverse correlation matrix.

        The results are stored internally in `self.vif_results` for downstream plotting 
        or dropping, and are also returned directly.

        Args:
            use_columns (Optional[list[str]]): If provided, computes VIF only for these specified features.
            ignore_columns (Optional[list[str]]): If provided, excludes these features from the computation.

        Returns:
            pd.DataFrame: A DataFrame containing 'feature', 'VIF' (numeric score), and 'type' 
                ('Continuous' or 'Categorical'), sorted by descending VIF values.
        """
        base_cols = list(self.schema.feature_names)
        
        if use_columns:
            base_cols = [c for c in base_cols if c in use_columns]
        if ignore_columns:
            base_cols = [c for c in base_cols if c not in ignore_columns]
            
        available_cols = [c for c in base_cols if c in self.df.columns]
        missing = set(base_cols) - set(self.df.columns)
        if missing:
            _LOGGER.warning(f"Missing specified columns in dataframe: {missing}")

        X = self.df[available_cols].copy()
        
        cat_cols = [c for c in self.schema.categorical_feature_names if c in available_cols]
        cont_cols = [c for c in self.schema.continuous_feature_names if c in available_cols]
        
        if cat_cols:
            X_design = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
        else:
            X_design = X.astype(float)
            
        variances = X_design.var()
        constant_cols = variances[variances == 0].index
        if len(constant_cols) > 0:
            _LOGGER.warning(f"Dropping constant columns from VIF calculation: {list(constant_cols)}")
            X_design = X_design.drop(columns=constant_cols)
        
        # Standardize the design matrix to have mean 0 and variance 1
        X_std = (X_design - X_design.mean()) / X_design.std(ddof=1)
        # Compute the correlation matrix
        R = X_std.corr().values
        # Compute the inverse of the correlation matrix
        try:
            C = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            _LOGGER.warning("Correlation matrix is singular. Using pseudo-inverse.")
            C = np.linalg.pinv(R)
            
        design_cols = X_design.columns.tolist()
        results = []
        
        for feature in available_cols:
            if feature in cont_cols:
                if feature in constant_cols:
                    results.append({"feature": feature, "VIF": np.inf, "type": "Continuous"})
                    continue
                idx = design_cols.index(feature)
                
                # Safeguard continuous VIF against floating-point edge cases
                vif = max(1.0, C[idx, idx])
                results.append({"feature": feature, "VIF": vif, "type": "Continuous"})
                
            elif feature in cat_cols:
                dummy_cols = [c for c in design_cols if str(c).startswith(f"{feature}_")]
                
                # Prevent substring collisions from overlapping categorical feature names
                other_cats = [other for other in cat_cols if other.startswith(f"{feature}_") and other != feature]
                for other in other_cats:
                    dummy_cols = [c for c in dummy_cols if not str(c).startswith(f"{other}_")]
                
                if not dummy_cols:
                     results.append({"feature": feature, "VIF": 1.0, "type": "Categorical"})
                     continue
                     
                idx_list = [design_cols.index(c) for c in dummy_cols]
                
                C_sub = C[np.ix_(idx_list, idx_list)]
                R_sub = R[np.ix_(idx_list, idx_list)]
                
                # Safeguard against precision issues & unstable determinants from pseudo-inverses
                raw_gvif = np.linalg.det(C_sub) * np.linalg.det(R_sub)
                gvif = max(1.0, raw_gvif)
                
                df_cat = len(idx_list)
                
                if df_cat > 0:
                    adjusted_gvif_sq = gvif ** (1 / df_cat)
                else:
                    adjusted_gvif_sq = np.inf
                    
                results.append({"feature": feature, "VIF": adjusted_gvif_sq, "type": "Categorical"})

        vif_data = pd.DataFrame(results)
        vif_data["VIF"] = vif_data["VIF"].replace([np.inf, -np.inf], 999.0)
        vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
        
        self.vif_results = vif_data
        return self.vif_results

    def plot_vif(
        self, 
        save_dir: Union[str, Path], 
        max_features_to_plot: int = 20,
        filename: Optional[str] = None,
        fontsize: int = 16
    ) -> None:
        """
        Generates and saves a horizontal bar plot of the computed VIF/GVIF values.

        Categorizes features by color based on standard multicollinearity thresholds 
        (green < 5, gold >= 5, red >= 10). Must be called after `compute_vif()`.

        Args:
            save_dir (str | Path): Directory where the SVG plot will be saved.
            max_features_to_plot (int): Maximum number of top collinear features to display.
            filename (str | None): Custom filename for the saved plot. If None, uses a default name based on `dataset_name`.
            fontsize (int): Base font size for plot labels and ticks.
        
        Raises:
            ValueError: If `compute_vif()` has not been executed prior to calling this method.
        """
        if self.vif_results is None:
            _LOGGER.error("VIF results not found. Call compute_vif() first.")
            raise ValueError()
            
        plot_data = self.vif_results.head(max_features_to_plot).copy()
        
        if plot_data.empty:
            _LOGGER.warning("No VIF data to plot.")
            return

        def vif_color(v: float) -> str:
            if v >= 10: return "red"
            elif v >= 5: return "gold"
            else: return "green"

        plot_data["color"] = plot_data["VIF"].apply(vif_color)

        plt.figure(figsize=(10, 6))
        plt.barh(
            plot_data["feature"],
            plot_data["VIF"],
            color=plot_data["color"],
            edgecolor='black'
        )
        
        # plt.title(f"Variance Inflation Factor (VIF): {self.dataset_name}", fontsize=fontsize+1)
        plt.title("")
        plt.xlabel("VIF Continuous - Adjusted GVIF² Categorical", fontsize=fontsize)
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
        plt.axvline(x=5, color='gold', linestyle='--', label='VIF = 5')
        plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
        plt.xlim(0, max(12, plot_data["VIF"].max() + 1))
        plt.legend(loc='lower right', fontsize=fontsize-1)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()

        save_path = make_fullpath(save_dir, make=True)
        out_name = filename if filename else f"VIF_{sanitize_filename(self.dataset_name)}.svg"
        
        # Case-insensitive check to prevent '.SVG.svg'
        if not out_name.lower().endswith(".svg"):
            out_name += ".svg"
        
        full_save_path = save_path / out_name
        plt.savefig(full_save_path, format='svg', bbox_inches='tight')
        _LOGGER.info(f"📊 Saved VIF plot: '{out_name}'")
        plt.close()

    def drop_vif_based(
        self, 
        threshold: float = 10.0
    ) -> tuple[pd.DataFrame, "FeatureSchema"]:
        """
        Drops features exceeding the specified VIF threshold and yields a mutually updated 
        DataFrame and FeatureSchema.

        Safely removes the entire feature (including all associated categorical mappings 
        in the schema) without mutating the original input data. Must be called after `compute_vif()`.

        Args:
            threshold (float): The VIF value above which a feature is considered highly collinear 
                and will be removed. Defaults to 10.0 as a common threshold for multicollinearity.

        Returns:
            tuple[pd.DataFrame, FeatureSchema]: 
                - A new pandas DataFrame with the highly collinear features removed.
                - A new FeatureSchema accurately reflecting the remaining features.

        Raises:
            ValueError: If `compute_vif()` has not been executed prior to calling this method.
        """
        if self.vif_results is None:
            _LOGGER.error("VIF results not found. Call compute_vif() first.")
            raise ValueError()
            
        to_drop = self.vif_results[self.vif_results["VIF"] > threshold]["feature"].tolist()
        
        if len(to_drop) > 0:
            _LOGGER.info(f"🗑️ Dropping {len(to_drop)} column(s) with VIF > {threshold}:")
            for dc in to_drop:
                print(f"\t{dc}")
        else:
            _LOGGER.info(f"No columns exceed the VIF threshold of '{threshold}'. Returning original dataframe and schema.")
            # return early with the original dataframe and schema
            return self.df.copy(), self.schema
            
        result_df = self.df.drop(columns=to_drop, errors='ignore')
        
        if result_df.empty:
            _LOGGER.warning("All columns were dropped. Returning original dataframe and schema.")
            # return early with the original dataframe and schema
            return self.df.copy(), self.schema
        
        new_feature_names = tuple(f for f in self.schema.feature_names if f not in to_drop)
        new_cont_names = tuple(f for f in self.schema.continuous_feature_names if f not in to_drop)
        new_cat_names = tuple(f for f in self.schema.categorical_feature_names if f not in to_drop)
        
        new_mappings = None
        if self.schema.categorical_mappings:
            new_mappings = {k: v for k, v in self.schema.categorical_mappings.items() if k not in to_drop}
            
        new_index_map = None
        if self.schema.categorical_index_map is not None and new_mappings is not None:
            new_index_map = {}
            for i, feat in enumerate(new_feature_names):
                if feat in new_cat_names:
                    new_index_map[i] = len(new_mappings[feat])
        
        # make new FeatureSchema instance with updated attributes
        new_schema = FeatureSchema(
            feature_names=new_feature_names,
            continuous_feature_names=new_cont_names,
            categorical_feature_names=new_cat_names,
            categorical_index_map=new_index_map,
            categorical_mappings=new_mappings
        )
        
        return result_df, new_schema
