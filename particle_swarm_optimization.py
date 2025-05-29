import numpy as np
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
from functools import partial


class ObjectiveFunction():
    """
    Callable objective function designed for optimizing continuous outputs from regression models.
    
    The trained model must include a 'model' and a 'scaler'. Additionally 'feature_names' and 'target_name' will be parsed if present.

    Parameters
    ----------
    trained_model_path : str
        Path to a serialized model (joblib) compatible with scikit-learn-like `.predict`. 
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
        self.model = self._get_from_artifact('model')
        self.scaler = self._get_from_artifact('scaler')
        self.feature_names: list[str] = self._get_from_artifact('feature_names') # type: ignore
        self.target_name: str = self._get_from_artifact('target_name') # type: ignore
        self.task = task
        self.check_model() # check for classification models and None values
    
    def __call__(self, features_array: np.ndarray) -> float:
        if self.use_noise:
            features_array = self.add_noise(features_array)
        if self.is_hybrid:
            features_array = self._handle_hybrid(features_array)
        
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # scale features as the model expects
        features_array = self.scaler.transform(features_array) # type: ignore
        
        result = self.model.predict(features_array) # type: ignore
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
        if self.model is None:
            raise ValueError("Loaded model is None")
        if self.scaler is None:
            raise ValueError("Loaded scaler is None")

    def _get_from_artifact(self, key: str):
        val = self._artifact.get(key)
        if key == "feature_names":
            result = val if isinstance(val, list) and val else None
        else:
            result = val if val else None
        return result
    
    def __repr__(self):
        return (f"<ObjectiveFunction(model={type(self.model).__name__}, scaler={type(self.scaler).__name__}, use_noise={self.use_noise}, is_hybrid={self.is_hybrid}, task='{self.task}')>")


def _set_boundaries(lower_boundaries: Sequence[float], upper_boundaries: Sequence[float]):
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
    

def _save_results(*dicts, save_dir: str, target_name: str):
    combined_dict = dict()
    for single_dict in dicts:
        combined_dict.update(single_dict)
        
    full_path = os.path.join(save_dir, f"results_{target_name}.csv")
    pl.DataFrame(combined_dict).write_csv(full_path)


def run_pso(lower_boundaries: Sequence[float], upper_boundaries: Sequence[float], objective_function: ObjectiveFunction,
            save_results_dir: str, 
            target_name: Union[str, None]=None, 
            feature_names: Union[list[str], None]=None,
            swarm_size: int=100, max_iterations: int=100,
            inequality_constrain_function=None, 
            post_hoc_analysis: Union[int, None]=None) -> Tuple[Dict[str, float | list[float]], Dict[str, float | list[float]]]:
    """
    Executes Particle Swarm Optimization (PSO) to optimize a given objective function and saves the results.

    Parameters
    ----------
    lower_boundaries : Sequence[float]
        Lower bounds for each feature in the search space.
    upper_boundaries : Sequence[float]
        Upper bounds for each feature in the search space.
    objective_function : ObjectiveFunction
        A callable object encapsulating a regression model and its scaler.
    save_results_dir : str
        Directory path to save the results CSV file.
    target_name : str or None, optional
        Name of the target variable. If None, attempts to retrieve from the ObjectiveFunction object.
    feature_names : list[str] or None, optional
        List of feature names. If None, attempts to retrieve from the ObjectiveFunction or generate generic names.
    swarm_size : int, default=100
        Number of particles in the swarm.
    max_iterations : int, default=100
        Maximum number of iterations for the optimization algorithm.
    inequality_constrain_function : callable or None, optional
        Optional function defining inequality constraints to be respected by the optimization.
    post_hoc_analysis : int or None, optional
        If specified, runs the optimization multiple times to perform post hoc analysis. The value indicates the number of repetitions.

    Returns
    -------
    Tuple[Dict[str, float | list[float]], Dict[str, float | list[float]]]
        If `post_hoc_analysis` is None, returns two dictionaries:
            - best_features_named: Feature values (after inverse scaling) that yield the best result.
            - best_target_named: Best result obtained for the target variable.

        If `post_hoc_analysis` is an integer, returns two dictionaries:
            - all_best_features_named: Lists of best feature values (after inverse scaling) for each repetition.
            - all_best_targets_named: List of best target values across repetitions.

    Notes
    -----
    - PSO minimizes the objective function by default; if maximization is desired, it should be handled inside the ObjectiveFunction.
    - Feature values are scaled before being passed to the model and inverse-transformed before result saving.
    """
    lower, upper = _set_boundaries(lower_boundaries, upper_boundaries)
    
    # feature names
    if feature_names is None and objective_function.feature_names is not None:
        feature_names = objective_function.feature_names
    names = _set_feature_names(size=len(lower_boundaries), names=feature_names)
        
    # target name
    if target_name is None and objective_function.target_name is not None:
        target_name = objective_function.target_name
    if target_name is None:
        target_name = "Target"
        
    arguments = {
            "func":objective_function,
            "lb": lower,
            "ub": upper,
            "f_ieqcons": inequality_constrain_function,
            "swarmsize": swarm_size,
            "maxiter": max_iterations,
            "processes": 1,
            "particle_output": True
    }
    
    if post_hoc_analysis is None:
        # best_features, best_target = pso(**arguments)
        best_features, best_target, _particle_positions, _target_values_per_position = pso(**arguments)
        
        # inverse transformation
        best_features = np.array(best_features).reshape(1, -1)
        best_features_real = objective_function.scaler.inverse_transform(best_features).flatten() # type: ignore
        
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
            # best_features, best_target = pso(**arguments)
            best_features, best_target, _particle_positions, _target_values_per_position = pso(**arguments)
            
            # inverse transformation
            best_features = np.array(best_features).reshape(1, -1)
            best_features_real = objective_function.scaler.inverse_transform(best_features).flatten() # type: ignore
            
            for i, best_feature in enumerate(best_features_real):
                all_best_features[i].append(best_feature)
            all_best_targets.append(best_target)
        
        # name features
        all_best_features_named = {name: list_values for name, list_values in zip(names, all_best_features)}
        all_best_targets_named = {target_name: all_best_targets}
        
        # save results
        _save_results(all_best_features_named, all_best_targets_named, save_dir=save_results_dir, target_name=target_name)
        
        return all_best_features_named, all_best_targets_named # type: ignore




### SOURCE CODE FOR PSO ###
def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)
    
    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
    
    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])
       
    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg
