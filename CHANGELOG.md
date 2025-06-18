# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.2]

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

## [1.4.1]

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

## [1.4.0]

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

## [1.3.2]

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

## [1.3.1]

### Fixed

- Correctly list "imbalanced-learn" as a dependency instead of "imblearn".

## [1.3.0]

### Changed

- Revamped base and optional dependencies, only pytorch-related are optional. 
- Set requirement `numpy<2.0` for broad compatibility (including miceforest v6).
- Update README with conda-forge installation.

## [1.2.1]

### Deleted

- 'load.dataframe()' from 'data_exploration.py'.

### Changed

- 'yield_dataframes()' from 'ensemble_learning.py'. Uses the 'utilities.yield_dataframes_from_dir()' instead.

## [1.2.0]

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
