# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.2]

### Changed

- MICE Imputation
    - Pin dependency `miceforest>=6.0.0,<7.0.0`.
    - Pin dependency version `lightgbm<=4.5.0`
    - Pin dependency `plotnine>=0.12,<0.13`.

### Fixed

- MICE Imputation script bugs

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
- Set requirement `numpy<2.0` for broad compatibility. (including miceforest v6)
- Update README with conda-forge installation.

## [1.2.1]

### Deleted

- 'load.dataframe()' from 'data_exploration.py'

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

- Incorrect metadata in "pyproject.toml"

## [1.1.0] - 2025-06-12

### Added

- Initial public release
