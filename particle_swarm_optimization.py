import numpy as np
from pyswarm import pso
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler
from typing import Literal, Union, Tuple, Dict
from collections.abc import Sequence
import polars as pl


class ObjectiveFunction():
    """
    Callable objective function designed for optimizing continuous outputs from regression models.

    Parameters
    ----------
    trained_model_path : str
        Path to a serialized model (joblib) compatible with scikit-learn-like `.predict`. Must include a 'model' and a 'scaler'.
    add_noise : bool
        Whether to apply multiplicative noise to the input features during evaluation.
    binary_features : int, default=0
        Number of binary features located at the END of the feature vector. Model should be trained with continuous features first, followed by binary.
    task : Literal, default 'maximization'
        Whether to maximize or minimize the target.
    """
    def __init__(self, trained_model_path: str, add_noise: bool=True, task: Literal["maximization", "minimization"]="maximization", binary_features: int=0) -> None:
        self.binary_features = binary_features
        self.is_hybrid = False if binary_features <= 0 else True
        self.use_noise = add_noise
        self._artifact = joblib.load(trained_model_path)
        self.model = self._artifact['model']
        self.scaler = self._artifact['scaler']
        self.task = task
        self.check_model() # check for classification models
    
    def __call__(self, features_array: np.ndarray) -> float:
        if self.use_noise:
            features_array = self.add_noise(features_array)
        if self.is_hybrid:
            features_array = self._handle_hybrid(features_array)
        
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # scale features as the model expects
        features_array = self.scaler.transform(features_array)
        
        result = self.model.predict(features_array)
        scalar = result.item()
        # pso minimizes by default, so we return the negative value to maximize
        if self.task == "maximization":
            return -scalar
        else:
            return scalar
    
    def add_noise(self, features_array):
        noise_range = np.random.uniform(0.95, 1.05, size=features_array.shape)
        new_feature_values = features_array * noise_range
        return new_feature_values
    
    def _handle_hybrid(self, features_array):
        feat_continuous = features_array[:self.binary_features]
        feat_binary = (features_array[self.binary_features:] > 0.5).astype(int) #threshold binary values
        new_feature_values = np.concatenate([feat_continuous, feat_binary])
        return new_feature_values
    
    def check_model(self):
        if isinstance(self.model, ClassifierMixin) or isinstance(self.model, xgb.XGBClassifier) or isinstance(self.model, lgb.LGBMClassifier):
            raise ValueError(f"[Model Check Failed] ❌\nThe loaded model ({type(self.model).__name__}) is a Classifier.\nOptimization is not suitable for standard classification tasks.")
    
    def __repr__(self):
        return (f"<ObjectiveFunction(model={type(self.model).__name__}, scaler={type(self.scaler).__name__}, use_noise={self.use_noise}, is_hybrid={self.is_hybrid}, task='{self.task}')>")


def _set_boundaries(lower_boundaries: Sequence[float], upper_boundaries: Sequence[float]):
    assert len(lower_boundaries) == len(upper_boundaries), "Lower and upper boundaries must have the same length."
    assert len(lower_boundaries) >= 1, "At least one boundary pair is required."
    lower = np.array(lower_boundaries)
    upper = np.array(upper_boundaries)
    return lower, upper


def _get_feature_names(size: int, names: Union[list[str], None]):
    if names is None:
        return [str(i) for i in range(1, size+1)]
    else:
        assert len(names) == size, "List with feature names do not match the number of features"
        return names
    

def _save_results(*dicts, save_dir: str, target_name: str):
    combined_dict = dict()
    for single_dict in dicts:
        combined_dict.update(single_dict)
        
    full_path = os.path.join(save_dir, f"results_{target_name}.csv")
    pl.DataFrame(combined_dict).write_csv(full_path)


def run_pso(lower_boundaries: Sequence[float], upper_boundaries: Sequence[float], objective_function: ObjectiveFunction,
            save_results_dir: str, target_name: str, 
            feature_names: Union[list[str], None]=None,
            swarm_size: int=100, max_iterations: int=100,
            inequality_constrain_function=None, 
            post_hoc_analysis: Union[int, None]=None) -> Tuple[Dict[str, float | list[float]], Dict[str, float | list[float]]]:
    lower, upper = _set_boundaries(lower_boundaries, upper_boundaries)
    names = _get_feature_names(size=len(lower_boundaries), names=feature_names)
        
    arguments = {
            "func":objective_function,
            "lb": lower,
            "ub": upper,
            "f_ieqcons": inequality_constrain_function,
            "swarmsize": swarm_size,
            "maxiter": max_iterations,
            "processes": 1
    }
    
    if post_hoc_analysis is None:
        best_features, best_target, _particle_positions, _target_values_per_position = pso(**arguments)
        
        # inverse transformation
        best_features = np.array(best_features).reshape(1, -1)
        best_features_real = objective_function.scaler.inverse_transform(best_features).flatten()
        
        # name features
        best_features_named = {name: value for name, value in zip(names, best_features_real)}
        best_target_named = {target_name: best_target}
        
        # save results
        _save_results(best_features_named, best_target_named, save_dir=save_results_dir, target_name=target_name)
        
        return best_features_named, best_target_named
    else:
        all_best_targets = list()
        all_best_features = [[] for _ in range(len(lower_boundaries))]
        for  _ in range(post_hoc_analysis):
            best_features, best_target, _particle_positions, _target_values_per_position = pso(**arguments)
            
            # inverse transformation
            best_features = np.array(best_features).reshape(1, -1)
            best_features_real = objective_function.scaler.inverse_transform(best_features).flatten()
            
            for i, best_feature in enumerate(best_features_real):
                all_best_features[i].append(best_feature)
            all_best_targets.append(best_target)
        
        # name features
        all_best_features_named = {name: list_values for name, list_values in zip(names, all_best_features)}
        all_best_targets_named = {target_name: all_best_targets}
        
        # save results
        _save_results(all_best_features_named, all_best_targets_named, save_dir=save_results_dir, target_name=target_name)
        
        return all_best_features_named, all_best_targets_named # type: ignore
