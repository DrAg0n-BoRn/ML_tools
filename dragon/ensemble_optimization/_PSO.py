import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from typing import Literal, Union, Optional
from copy import deepcopy
import torch
from tqdm import trange
from contextlib import nullcontext

from ..serde import deserialize_object
from ..SQL import DragonSQL

from ..math_utilities import threshold_binary_values, threshold_binary_values_batch
from ..keys._keys import EnsembleKeys
from ..path_manager import sanitize_filename, make_fullpath, list_files_by_extension
from .._core import get_logger


_LOGGER = get_logger("PSO")


"""
DEPRECATED
"""


__all__ = [
    "ObjectiveFunction",
    "multiple_objective_functions_from_dir",
    "run_pso"
]


class ObjectiveFunction():
    """
    Callable objective function designed for optimizing continuous outputs from tree-based regression models.
    
    The target serialized file (joblib) must include a trained tree-based 'model'. Additionally 'feature_names' and 'target_name' will be parsed if present.

    Parameters
    ----------
    trained_model_path : str
        Path to a serialized model (joblib) compatible with scikit-learn-like `.predict`. 
    add_noise : bool
        Whether to apply multiplicative noise to the input features during evaluation.
    task : (Literal["maximization", "minimization"])
        Whether to maximize or minimize the target.
    binary_features : int
        Number of binary features located at the END of the feature vector. Model should be trained with continuous features first, followed by binary.
    """
    def __init__(self, trained_model_path: Union[str, Path], add_noise: bool, task: Literal["maximization", "minimization"], binary_features: int) -> None:
        self.binary_features = binary_features
        self.is_hybrid = False if binary_features <= 0 else True
        self.use_noise = add_noise
        self._artifact = deserialize_object(trained_model_path, verbose=False)
        self.model = self._get_from_artifact(EnsembleKeys.MODEL)
        self.feature_names: Optional[list[str]] = self._get_from_artifact(EnsembleKeys.FEATURES) # type: ignore
        self.target_name: Optional[str] = self._get_from_artifact(EnsembleKeys.TARGET) # type: ignore
        self.task = task
        self.check_model() # check for classification models and None values
        
    def __call__(self, features_array: np.ndarray) -> np.ndarray:
        """
        Batched evaluation for PSO. Accepts 2D array (n_samples, n_features).
        
        Applies optional noise and hybrid binary thresholding.
        
        Returns
        -------
        np.ndarray
            1D array with length n_samples containing predicted target values.
        """
        if features_array.ndim != 2:
            _LOGGER.error(f"Expected 2D array, got shape {features_array.shape}.")
            raise AssertionError()
        
        # Apply noise if enabled
        if self.use_noise:
            features_array = self.add_noise(features_array)
        
        # Apply binary thresholding if enabled
        if self.is_hybrid:
            features_array = threshold_binary_values_batch(features_array, self.binary_features)
        
        # Ensure correct type
        features_array = features_array.astype(np.float32)

        # Evaluate
        result = self.model.predict(features_array) # type: ignore

        # Flip sign if maximizing
        if self.task == "maximization":
            return -result
        return result

    def add_noise(self, features_array: np.ndarray) -> np.ndarray:
        """
        Apply multiplicative noise to input feature batch (2D).
        Binary features (if present) are excluded from noise injection.

        Parameters
        ----------
        features_array : np.ndarray
            Input array of shape (batch_size, n_features)

        Returns
        -------
        np.ndarray
            Noised array of same shape
        """
        if features_array.ndim != 2:
            _LOGGER.error(f"Expected 2D array for batch noise injection, got shape {features_array.shape}.")
            raise AssertionError()

        if self.binary_features > 0:
            split_idx = -self.binary_features
            cont_part = features_array[:, :split_idx]
            bin_part = features_array[:, split_idx:]

            noise = np.random.uniform(0.95, 1.05, size=cont_part.shape)
            cont_noised = cont_part * noise

            return np.hstack([cont_noised, bin_part])
        else:
            noise = np.random.uniform(0.95, 1.05, size=features_array.shape)
            return features_array * noise
    
    def check_model(self):
        if isinstance(self.model, xgb.XGBClassifier) or isinstance(self.model, lgb.LGBMClassifier):
            _LOGGER.error(f"[Model Check Failed]\nThe loaded model ({type(self.model).__name__}) is a Classifier.\nOptimization is not suitable for standard classification tasks.")
            raise ValueError()
        if self.model is None:
            _LOGGER.error("Loaded model is None")
            raise ValueError()

    def _get_from_artifact(self, key: str):
        if self._artifact is None:
            _LOGGER.error("Load model error")
            raise TypeError()
        val = self._artifact.get(key)
        if key == EnsembleKeys.FEATURES:
            result = val if isinstance(val, list) and val else None
        else:
            result = val if val else None
        return result
    
    def __repr__(self):
        return (f"<ObjectiveFunction(model={type(self.model).__name__}, use_noise={self.use_noise}, is_hybrid={self.is_hybrid}, task='{self.task}')>")


def multiple_objective_functions_from_dir(directory: Union[str,Path], add_noise: bool, task: Literal["maximization", "minimization"], binary_features: int):
    """
    Loads multiple objective functions from serialized models in the given directory.

    Each `.joblib` file which is loaded and wrapped as an `ObjectiveFunction` instance. Returns a list of such instances along with their corresponding names.

    Parameters:
        directory (str) : Path to the directory containing `.joblib` files (serialized models).
        add_noise (bool) : Whether to apply multiplicative noise to the input features during evaluation.
        task (Literal["maximization", "minimization"]) : Defines the nature of the optimization task.
        binary_features (int) : Number of binary features expected by each objective function.

    Returns:
        (tuple[list[ObjectiveFunction], list[str]]) : A tuple containing:
            - list of `ObjectiveFunction` instances.
            - list of corresponding filenames.
    """
    objective_functions = list()
    objective_function_names = list()
    for file_name, file_path in list_files_by_extension(directory=directory, extension='joblib', raise_on_empty=True).items():
        current_objective = ObjectiveFunction(trained_model_path=file_path,
                                              add_noise=add_noise,
                                              task=task,
                                              binary_features=binary_features)
        objective_functions.append(current_objective)
        objective_function_names.append(file_name)
    return objective_functions, objective_function_names


def _set_boundaries(lower_boundaries: list[float], upper_boundaries: list[float]):
    assert len(lower_boundaries) == len(upper_boundaries), "Lower and upper boundaries must have the same length."
    assert len(lower_boundaries) >= 1, "At least one boundary pair is required."
    lower = np.array(lower_boundaries)
    upper = np.array(upper_boundaries)
    return lower, upper


def _set_feature_names(size: int, names: Union[list[str], None]):
    if names is None:
        return [str(i) for i in range(1, size+1)]
    else:
        assert len(names) == size, "List with feature names do not match the number of features"
        return names


def _run_single_pso(objective_function: ObjectiveFunction, pso_args: dict, feature_names: list[str], target_name: str, random_state: int, save_format: Literal['csv', 'sqlite', 'both'], csv_path: Path, db_manager: Optional[DragonSQL], db_table_name: str):
    """Helper for a single PSO run that also handles saving."""
    pso_args.update({"seed": random_state})
    
    best_features, best_target, *_ = _pso(**pso_args)
    
    if objective_function.task == "maximization":
        best_target = -best_target
    
    binary_number = objective_function.binary_features
    best_features_threshold = threshold_binary_values(best_features, binary_number)
    
    best_features_named = {name: value for name, value in zip(feature_names, best_features_threshold)}
    best_target_named = {target_name: best_target}
    
    # Save the result using the new helper
    combined_dict = {**best_features_named, **best_target_named}
    _save_result(combined_dict, save_format, csv_path, db_manager, db_table_name)
    
    return best_features_named, best_target_named


def _run_post_hoc_pso(objective_function: ObjectiveFunction, pso_args: dict, feature_names: list[str], target_name: str, repetitions: int, save_format: Literal['csv', 'sqlite', 'both'], csv_path: Path, db_manager: Optional[DragonSQL], db_table_name: str):
    """Helper for post-hoc analysis that saves results incrementally."""
    progress = trange(repetitions, desc="Post-Hoc PSO", unit="run")
    for _ in progress:
        best_features, best_target, *_ = _pso(**pso_args)
        
        if objective_function.task == "maximization":
            best_target = -best_target
        
        binary_number = objective_function.binary_features
        best_features_threshold = threshold_binary_values(best_features, binary_number)
        
        result_dict = {name: value for name, value in zip(feature_names, best_features_threshold)}
        result_dict[target_name] = best_target
        
        # Save each result incrementally
        _save_result(result_dict, save_format, csv_path, db_manager, db_table_name)


def run_pso(lower_boundaries: list[float], 
            upper_boundaries: list[float], 
            objective_function: ObjectiveFunction,
            save_results_dir: Union[str,Path],
            save_format: Literal['csv', 'sqlite', 'both'],
            auto_binary_boundaries: bool=True,
            target_name: Union[str, None]=None, 
            feature_names: Union[list[str], None]=None,
            swarm_size: int=200, 
            max_iterations: int=3000,
            random_state: int=101,
            post_hoc_analysis: Optional[int]=20) -> Optional[tuple[dict[str, float], dict[str, float]]]:
    """
    Executes Particle Swarm Optimization (PSO) to optimize a given objective function and saves the results as a CSV file.

    Parameters
    ----------
    lower_boundaries : list[float]
        Lower bounds for each feature in the search space (as many as features expected by the model).
    upper_boundaries : list[float]
        Upper bounds for each feature in the search space (as many as features expected by the model).
    objective_function : ObjectiveFunction
        A callable object encapsulating a tree-based regression model.
    save_results_dir : str | Path
        Directory path to save the results CSV file.
    save_format : {'csv', 'sqlite', 'both'}
        The format for saving optimization results.
        - 'csv': Saves results to a CSV file.
        - 'sqlite': Saves results to an SQLite database file. ⚠️ If a database exists, new tables will be created using the target name.
        - 'both': Saves results to both formats.
    auto_binary_boundaries : bool
        Use `ObjectiveFunction.binary_features` to append as many binary boundaries as needed to `lower_boundaries` and `upper_boundaries` automatically.
    target_name : str or None, optional
        Name of the target variable. If None, attempts to retrieve from the ObjectiveFunction object.
    feature_names : list[str] or None, optional
        List of feature names. If None, attempts to retrieve from the ObjectiveFunction or generate generic names.
    swarm_size : int
        Number of particles in the swarm.
    max_iterations : int
        Maximum number of iterations for the optimization algorithm.
    post_hoc_analysis : int or None
        If specified, runs the optimization multiple times to perform post hoc analysis. The value indicates the number of repetitions.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]] or None
        - If `post_hoc_analysis` is None, returns two dictionaries containing the
          single best features and the corresponding target value.
        - If `post_hoc_analysis` is active, results are streamed directly to a CSV file
          and this function returns `None`.

    Notes
    -----
    - PSO minimizes the objective function by default; if maximization is desired, it should be handled inside the ObjectiveFunction.
    """

    
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    _LOGGER.info(f"👾 Using device: '{device}'")
    
    # set local deep copies to prevent in place list modification
    local_lower_boundaries = deepcopy(lower_boundaries)
    local_upper_boundaries = deepcopy(upper_boundaries)
    
    # Append binary boundaries
    binary_number = objective_function.binary_features
    if auto_binary_boundaries and binary_number > 0:
        # simplify binary search by constraining range
        local_lower_boundaries.extend([0.45] * binary_number)
        local_upper_boundaries.extend([0.55] * binary_number)
        
    # Set the total length of features
    size_of_features = len(local_lower_boundaries)

    lower, upper = _set_boundaries(local_lower_boundaries, local_upper_boundaries)

    # feature names
    if feature_names is None and objective_function.feature_names is not None:
        feature_names = objective_function.feature_names
    names = _set_feature_names(size=size_of_features, names=feature_names)

    # target name
    if target_name is None and objective_function.target_name is not None:
        target_name = objective_function.target_name
    if target_name is None:
        _LOGGER.error(f"'target' name was not provided and was not found in the .joblib object.")
        raise ValueError()
    
    # --- Setup: Saving Infrastructure ---
    sanitized_target_name = sanitize_filename(target_name)
    save_dir_path = make_fullpath(save_results_dir, make=True, enforce="directory")
    base_filename = f"Optimization_{sanitized_target_name}"
    csv_path = save_dir_path / f"{base_filename}.csv"
    db_path = save_dir_path / "Optimization.db"
    db_table_name = f"{sanitized_target_name}"
    
    if save_format in ['sqlite', 'both']:
        # Dynamically create the schema for the database table
        schema = {name: "REAL" for name in names}
        schema[target_name] = "REAL"
        schema = {"result_id": "INTEGER PRIMARY KEY AUTOINCREMENT", **schema}
        
        # Create table
        with DragonSQL(db_path) as db:
            db.create_table(db_table_name, schema)
        
    pso_arguments = {
            "func":objective_function,
            "lb": lower,
            "ub": upper,
            "device": device,
            "swarmsize": swarm_size,
            "maxiter": max_iterations,
            "particle_output": False,
    }
    
    # --- Dispatcher ---
    # Use a real or dummy context manager to handle the DB connection cleanly
    db_context = DragonSQL(db_path) if save_format in ['sqlite', 'both'] else nullcontext()

    with db_context as db_manager:
        if post_hoc_analysis is None or post_hoc_analysis <= 1:
            # --- Single Run Logic ---
            features_dict, target_dict = _run_single_pso(
                objective_function, pso_arguments, names, target_name, random_state,
                save_format, csv_path, db_manager, db_table_name
            )
            _LOGGER.info(f"Single optimization complete.")
            return features_dict, target_dict
        
        else:
            # --- Post-Hoc Analysis Logic ---
            _LOGGER.info(f"🏁 Starting post-hoc analysis with {post_hoc_analysis} repetitions...")
            _run_post_hoc_pso(
                objective_function, pso_arguments, names, target_name, post_hoc_analysis,
                save_format, csv_path, db_manager, db_table_name
            )
            _LOGGER.info("Post-hoc analysis complete. Results saved.")
            return None


def _save_result(
        result_dict: dict,
        save_format: Literal['csv', 'sqlite', 'both'],
        csv_path: Path,
        db_manager: Optional[DragonSQL] = None,
        db_table_name: Optional[str] = None,
        categorical_mappings: Optional[dict[str, dict[str, int]]] = None
    ):
    """
    Private helper to handle saving a single result to CSV, SQLite, or both.
    
    If `categorical_mappings` is provided, it will reverse-map integer values
    to their string representations before saving.
    """
    # --- Reverse Mapping Logic ---
    # Create a copy to hold the values to be saved
    save_dict = result_dict.copy()
    
    if categorical_mappings:
        for feature_name, mapping in categorical_mappings.items():
            if feature_name in save_dict:
                # Create a reverse map {0: 'Category_A', 1: 'Category_B'}
                reverse_map = {idx: name for name, idx in mapping.items()}
                
                # Get the integer value from the results (e.g., 0)
                int_value = save_dict[feature_name]
                
                # Find the corresponding string (e.g., 'Category_A')
                # Use .get() for safety, defaulting to the original value if not found
                string_value = reverse_map.get(int_value, int_value)
                
                # Update the dictionary that will be saved
                save_dict[feature_name] = string_value
    
    # Save to CSV
    if save_format in ['csv', 'both']:
        df_row = pd.DataFrame([save_dict])
        file_exists = csv_path.exists()
        df_row.to_csv(csv_path, mode='a', index=False, header=not file_exists)

    # Save to SQLite
    if save_format in ['sqlite', 'both']:
        if db_manager and db_table_name:
            db_manager.insert_row(db_table_name, save_dict)
        else:
            _LOGGER.warning("SQLite saving requested but db_manager or table_name not provided.")


def _pso(func: ObjectiveFunction,
         lb: np.ndarray,
         ub: np.ndarray,
         device: torch.device,
         swarmsize: int,
         maxiter: int, 
         omega_start = 0.9, # STARTING inertia weight
         omega_end = 0.4,   # ENDING inertia weight
        #  omega = 0.729,     # Clerc and Kennedy’s constriction coefficient
         phip = 1.49445,    # Clerc and Kennedy’s constriction coefficient
         phig = 1.49445,    # Clerc and Kennedy’s constriction coefficient
         tolerance = 1e-8,
         particle_output=False,
         seed: Optional[int] = None):
    """
    Internal PSO implementation using PyTorch tensors for acceleration on CUDA or MPS devices.

    Parameters
    ----------
    func : callable
        Callable objective function with batched evaluation support. Must accept a 2D NumPy array
        of shape (n_particles, n_features) and return a 1D NumPy array of shape (n_particles,).
    
    lb : np.ndarray
        Lower bounds for each feature (1D array of length n_features).
    
    ub : np.ndarray
        Upper bounds for each feature (1D array of length n_features).

    swarmsize : int
        Number of particles in the swarm (i.e., batch size per iteration).

    maxiter : int
        Number of iterations to perform (i.e., optimization steps).

    omega : float
        Inertia weight controlling velocity retention across iterations.
        - Typical range: [0.4, 0.9]
        - Lower values encourage convergence, higher values promote exploration.
        - The default value (0.729) comes from Clerc & Kennedy's constriction method.

    phip : float
        Cognitive acceleration coefficient.
        - Controls how strongly particles are pulled toward their own best-known positions.
        - Typical range: [0.5, 2.5]
        - Default from Clerc & Kennedy's recommended setting.

    phig : float
        Social acceleration coefficient.
        - Controls how strongly particles are pulled toward the swarm's global best.
        - Typical range: [0.5, 2.5]
        - Default from Clerc & Kennedy's recommended setting.

    particle_output : bool, default=False
        If True, returns the full history of particle positions and objective scores at each iteration.

    seed : int or None, default=None
        Random seed for reproducibility. If None, the random state is not fixed.

    Returns
    -------
    best_position : np.ndarray
        1D array of shape (n_features,) representing the best solution found.
    
    best_score : float
        Objective value at `best_position`.

    history_positions : list[np.ndarray], optional
        Only returned if `particle_output=True`. List of particle positions per iteration.
        Each element has shape (swarmsize, n_features).

    history_scores : list[np.ndarray], optional
        Only returned if `particle_output=True`. List of objective scores per iteration.
        Each element has shape (swarmsize,).
    """
    if seed is not None:
        torch.manual_seed(seed)

    ndim = len(lb)
    lb_t = torch.tensor(lb, dtype=torch.float32, device=device, requires_grad=False)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device, requires_grad=False)
    
    # Initialize positions and velocities
    r = torch.rand((swarmsize, ndim), device=device, requires_grad=False)
    positions = lb_t + r * (ub_t - lb_t)
    velocities = torch.zeros_like(positions, requires_grad=False)

    # Initialize best positions and scores
    personal_best_positions = positions.clone()
    personal_best_scores = torch.full((swarmsize,), float('inf'), device=device, requires_grad=False)

    global_best_score = float('inf')
    global_best_position = torch.zeros(ndim, device=device, requires_grad=False)

    if particle_output:
        history_positions = []
        history_scores = []

    previous_best_score = float('inf')
    progress = trange(maxiter, desc="PSO", unit="iter", leave=True)
    with torch.no_grad():
        for i in progress:
            # Evaluate objective for all particles
            positions_np = positions.detach().cpu().numpy()
            scores_np = func(positions_np)
            scores = torch.tensor(scores_np, device=device, dtype=torch.float32)

            # Update personal bests
            improved = scores < personal_best_scores
            personal_best_scores = torch.where(improved, scores, personal_best_scores)
            personal_best_positions = torch.where(improved[:, None], positions, personal_best_positions)

            # Update global best
            min_score, min_idx = torch.min(personal_best_scores, dim=0)
            if min_score < global_best_score:
                global_best_score = min_score.item()
                global_best_position = personal_best_positions[min_idx].clone()
                
                if abs(previous_best_score - global_best_score) < tolerance:
                    progress.set_description(f"PSO (early stop at iteration {i+1})")
                    break
                previous_best_score = global_best_score

            if particle_output:
                history_positions.append(positions.detach().cpu().numpy())
                history_scores.append(scores_np)
         
            # Linearly decreasing inertia weight
            omega = omega_start - (omega_start - omega_end) * (i / maxiter)

            # Velocity update
            rp = torch.rand((swarmsize, ndim), device=device, requires_grad=False)
            rg = torch.rand((swarmsize, ndim), device=device, requires_grad=False)

            cognitive = phip * rp * (personal_best_positions - positions)
            social = phig * rg * (global_best_position - positions)
            velocities = omega * velocities + cognitive + social

            # Position update
            positions = positions + velocities

            positions = torch.max(positions, lb_t)
            positions = torch.min(positions, ub_t)

    best_position = global_best_position.detach().cpu().numpy()
    best_score = global_best_score

    if particle_output:
        return best_position, best_score, history_positions, history_scores
    else:
        return best_position, best_score

