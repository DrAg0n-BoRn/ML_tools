# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2025-07-07

### Changed

- ETL_engineering: `ColumnCleaner` now supports sub-string replacements with backreferences, as well as default case insensitivity.

## [3.0.0] - 2025-07-06

### Changed

- "trainer" revamped into the following modules using PyTorch:
    - ML_trainer, Uses the main class `MyTrainer` to train PyTorch models. It imports ML_callbacks and ML_evaluation helpers.
    - ML_callbacks, Includes callbacks to use during model training.
    - ML_evaluation, Helper functions to visualize training and evaluation metrics.
    - ML_tutorial, produces a notebook script with a tutorial on how to use these modules.
    - RNN_forecast, `rnn_forecast()` runs a sequential forecast for a trained RNN-based model.

- "datasetmaster" and "vision_helpers" merged and revamped into the new module "datasetmaster" with the following classes:
    - `DatasetMaker`Creates processed PyTorch datasets from a Pandas DataFrame using a fluent, step-by-step interface.
    - `VisionDatasetMaker` Creates processed PyTorch datasets for computer vision tasks from an image folder directory.
    - `SequenceMaker` Creates windowed PyTorch datasets from time-series data.
    - `ResizeAspectFill` Custom transformation to make an image square.

- Update most scripts to work with the package logger when printing messages, warnings, and errors.

## [2.4.0] - 2025-07-03

### Added

- Optional dependency: `FreeSimpleGUI`
- New Module: "GUI_tools":
    - `PathManager` Manages paths for a Python application, supporting both development mode and bundled mode via Briefcase.
    - `ConfigManager` Loads a .ini file and provides access to its configuration values as object attributes.
    - `GUIFactory` Builds styled FreeSimpleGUI elements and layouts using a "building block" approach, driven by a ConfigManager instance.
    - `catch_exceptions()` A decorator that wraps a function in a try-except block. If an exception occurs, it's caught and displayed in a popup window.
    - `prepare_feature_vector()` Validates and converts GUI values into a numpy array for ML inference.
    - `update_target_fields()` Updates the GUI's target fields with inference results.

## [2.3.0] - 2025-07-02

### Added

- ETL_engineering: 
    - `RegexMapper` A transformer that maps string categories to numerical values based on a dictionary of regular expression patterns.
    - `RatioCalculator` A transformer that parses a string ratio and computes the result of the division.
    - `ColumnCleaner` Cleans and standardizes a single pandas Series based on a dictionary of regex-to-value replacement rules.
    - `DataFrameCleaner` Orchestrates the cleaning of multiple columns in a pandas DataFrame using a nested dictionary of rules and `ColumnCleaner` objects.

- PSO_optimization: `plot_optimal_feature_distributions()` Analyzes optimization results and plots the distribution of optimal values for each feature.

### Changed

- PSO_optimization: `run_pso()` Refactor, add logger, add dynamic inertia weight for faster convergence.

## [2.2.1] - 2025-06-30

### Added

- data_exploration: `standardize_percentages()` Standardizes numeric columns containing mixed-format percentages.
- ETL_engineering: `DataProcessor` now has a comprehensive `__str__`, call it through its method `.inspect()`.

### Fixed

- ETL_engineering: `CategoryMapper` fixed call method.

## [2.2.0] - 2025-06-30

### Fixed

- utilities: docstrings not referencing the new `pathlib.Path` objects handled.

### Added

- New module: "ETL_engineering": Extract, Transform, Load data using Polars backend.
    - `TransformationRecipe` A builder class for creating a data transformation recipe.
    - `DataProcessor` Transforms a Polars DataFrame based on a provided `TransformationRecipe` object.
    - `KeywordDummifier` A configurable transformer that creates one-hot encoded columns based on keyword matching in a Polars Series.
    - `NumberExtractor` A configurable transformer that extracts a single number from a Polars string series using a regular expression.
    - `MultiNumberExtractor` Extracts multiple numbers from a single Polars string column into several new columns.
    - `CategoryMapper` A transformer that maps string categories to specified numerical values using a dictionary.
    - `ValueBinner` A transformer that discretizes a continuous numerical Polars column into a finite number of bins.
    - `DateFeatureExtractor` A one-to-many transformer that extracts multiple numerical features from a Polars date or datetime column.

### Changed

- utilities: 
    - `load_dataframe()` can now load either Pandas or Polars dataframes.
    - `save_dataframe()` can now save either Pandas or Polars dataframes.

## [2.1.0] - 2025-06-26

### Added

- data_exploration: Full compatibility with `pathlib.Path` objects.
- ensemble_learning: Full compatibility with `pathlib.Path` objects.
- logger: Full compatibility with `pathlib.Path` objects.
- MICE_imputation: Full compatibility with `pathlib.Path` objects.
- PSO_optimization: Full compatibility with `pathlib.Path` objects.
- VIF_factor: Full compatibility with `pathlib.Path` objects.
- utilities: 
    - Full compatibility with `pathlib.Path` objects.
    - `make_fullpath()` resolves a string or Path into an absolute Path.
- handle_excel:
    - Full compatibility with `pathlib.Path` objects.
    - `find_excel_files()` returns a list of Excel file Paths in the specified directory.

### Changed

- data_exploration: `drop_rows_with_missing_data()` modified to inspect target columns and feature columns.

## [2.0.0] - 2025-06-25

### Changed

- Renamed module: "particle_swarm_optimization" to "_particle_swarm_optimization", deprecated.
- Moved `Pillow` from optional to base dependencies.
- Required Python version 3.10+.
- Updated README.
- Updated third party licenses.

### Added

- Package dependency: `tqdm>=4.0`
- utilities: `threshold_binary_values_batch` threshold the last binary columns of a 2D NumPy array to binary {0,1} using 0.5 cutoff.
- New module: "PSO_optimization"
- PSO_optimization: 
    - Calculate pso algorithm using PyTorch tensors in the backend.
    - New `ObjectiveFunction` class that suits the new backend.

## [1.4.8] - 2025-06-23

### Changed

- Clean and refactor imports from scripts.

## [1.4.7] - 2025-06-22

### Fixed

- PSO: Using deep copies of boundary values in `run_pso()` to avoid in-place modifications when executing the algorithm in a batch.

## [1.4.6] - 2025-06-22 

### Added

- data_exploration: `match_and_filter_columns_by_regex()` returns a tuple of (filtered DataFrame, matched column names) based on a regex pattern.
- utilities: `list_files_by_extension()` lists all files with the specified extension in a given directory and returns a mapping: filenames (without extensions) to their absolute paths.
- PSO: `multiple_objective_functions_from_dir()` loads multiple objective functions from serialized models in the given directory.

### Fixed

- data_exploration: `plot_correlation_heatmap()` will sanitize the title name only before saving the file.

## [1.4.5] - 2025-06-21

### Added

- utilities: `serialize_object()` and `deserialize_object()` to serialize objects using joblib.

### Changed

- utilities: `threshold_binary_values()` now returns the same data type as the input type.
- ensemble_learning: make use of `serialize_object()` from utilities.
- PSO: make use of `deserialize_object()` from utilities.
- MICE: `apply_mice()` correctly thresholds binary columns after imputation.
- data_exploration: 
    - `distribute_datasets_by_target()`, enhanced functionality and moved to "utilities".
    - `split_features_targets()` return order swapped to (df_features, df_targets).

## [1.4.4] - 2025-06-20

### Added

- data_exploration: `distribute_datasets_by_target()` now has a verbose boolean parameter.
- VIF: `compute_vif()` now has a verbose boolean parameter.

### Fixed

- data_exploration: `drop_columns_with_missing_data()` correctly displays the state of columns with nulls after the drop.

### Changed

- MICE: `apply_mice()` will name imputed datasets with the "_MICE" suffix.
- VIF: `compute_vif_multi()` will name imputed datasets with the "_VIF" suffix.
- ensemble_learning: LightGBM now uses 'gbdt' boosting.
- PSO: `run_pso()` set 1500 iterations as default.

## [1.4.3] - 2025-06-19

### Changed

- data_exploration: `drop_columns_with_missing_data()` has an option to display the state of columns with nulls after the drop.
- utilities: `merge_dataframes()` now accepts a "verbose" boolean.
- MICE: 
    - `run_mice_pipeline()` requires a list of target column names. Targets must be skipped from the imputation process. 
    - `save_imputed_datasets()` requires a DataFrame or Series with the target column(s) to merge before saving.
- ensemble_learning: 
    - `get_models()` replaced by classes `ClassificationTreeModels` and `RegressionTreeModels`.
    - Integrate new classes into the pipeline.
    - `dataset_yielder()` enhanced functionality.

### Added

- data_exploration: an iterator `distribute_datasets_by_target()` that yields a dataframe per target column, dropping rows with missing targets.

## [1.4.2] - 2025-06-19

### Changed

- ensemble_learning: Remove scalers, unnecessary for tree-based methods.
- PSO: 
    - Remove scalers.
    - `ObjectiveFunction`: Add noise only to continuous features.

### Fixed

- PSO: 
    - `run_pso()` correctly flips the sign of the target in the last iteration if "maximization" was used.
    - `run_pso()` correctly saves binary values.

### Added

- utilities: `threshold_binary_values()` accepts a 1D sequence and returns a numpy 1D array.

## [1.4.1] - 2025-06-19

### Changed

- handle_excel: sanitize filenames before saving files.
- data_exploration: `merge_dataframes()` moved to "utilities".
- VIF_factor: 
    - `drop_vif_based()` now returns the names of the dropped columns.
    - `compute_vif_multi()` will not save CSV files if there was no dropped columns.

### Fixed

- ensemble_learning: use sanitized filenames before attempting to save files.
- PSO: 
    - sanitize filenames before saving.
    - fix binary-continuous handling inside the ObjectiveFunction class.
    - Refine `run_pso()` logic.
    - Under testing and **unusable** at the moment.

### Added

- PSO: `run_pso()` can now automatically add lower and upper boundaries for binary features. 
- `info()` for all scripts.

### Deleted

- trainer: deprecated method.

## [1.4.0] - 2025-06-17

### Added

- VIF_factor:
    - `compute_vif()` revamped. Additionally, it now accepts an optional filename when saving a plot, and a maximum number of features to plot.
    - `compute_vif()` will correctly handle perfect multicollinearity and suppress warnings.
    - `compute_vif_multi()` function to automate the process.
- Dependency: `ipywidgets`

### Changed

- MICE_imputation: 
    - keep original feature names for metric plots, sanitize before saving them.
    - Update usage of `list_csv_paths()` with the new return type.
- utilities: 
    - `list_csv_paths()` now returns a dictionary {name, path}. 
    - `save_dataframe()` now warns about and skips empty dataframes.
- data_exploration: 
    - VIF related functions moved to "VIF_factor".
    - `save_dataframe()` moved to "utilities".
- ensemble_learning:
    - `run_pipeline()` renamed to `run_ensemble_pipeline()`.
    - Use a unique "base_fontsize" for plotting functions.
    - `get_shap_values()` outputs bar and dot plots.

### Fixed

- Bugs in the ensemble_learning pipeline.
- Bugs in the MICE_imputation pipeline.

## [1.3.2] - 2025-06-16

### Changed

- MICE Imputation
    - Pin dependency `miceforest>=6.0.0,<7.0.0`.
    - Pin dependency version `lightgbm<=4.5.0`.
    - Pin dependency `plotnine>=0.12,<0.13`.

### Fixed

- MICE Imputation script bugs.

### Added

- Notebook dependencies:
    - "ipykernel"
    - "notebook"
    - "jupyterlab"

## [1.3.1] - 2025-06-16

### Fixed

- Correctly list "imbalanced-learn" as a dependency instead of "imblearn".

## [1.3.0] - 2025-06-16

### Changed

- Revamped base and optional dependencies, only pytorch-related are optional. 
- Set requirement `numpy<2.0` for broad compatibility (including miceforest v6).
- Update README with conda-forge installation.

## [1.2.1] - 2025-06-16

### Deleted

- 'load.dataframe()' from 'data_exploration.py'.

### Changed

- 'yield_dataframes()' from 'ensemble_learning.py'. Uses the 'utilities.yield_dataframes_from_dir()' instead.

## [1.2.0] - 2025-06-15

### Added

- \[full\] option to download all dependencies on installation.

### Fixed

- Bugs in local import paths.

### Changed

- README file to clearly display installation and usage.

## [1.1.6] - 2025-06-13

### Added

- MIT "LICENSE" file.

### Changed

- Update reference to license to adhere to the new format of SPDX license expression.

## [1.1.5] - 2025-06-12

### Added

- Default dependencies for the project: numpy, pandas, matplotlib, scikit-learn.
- url-links in project

### Changed

- README.md example usage.

## [1.1.1] - 2025-06-12

### Fixed

- Incorrect metadata in "pyproject.toml".

## [1.1.0] - 2025-06-12

### Added

- Initial public release.
