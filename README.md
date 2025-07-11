# dragon-ml-toolbox

A collection of Python utilities for data science and machine learning, structured as a modular package for easy reuse and installation.

## Features

- Modular scripts for data exploration, logging, machine learning, and more.
- Designed for seamless integration as a Git submodule or installable Python package.

## Installation

**Python 3.10+ recommended.**

### Via PyPI

Install the latest stable release from PyPI:

```bash
pip install dragon-ml-toolbox
```

### Via GitHub (Editable)

Clone the repository and install in editable mode with optional dependencies:

```bash
git clone https://github.com/DrAg0n-BoRn/ML_tools.git
cd ML_tools
pip install -e .
```

### Via conda-forge

Install from the conda-forge channel:

```bash
conda install -c conda-forge dragon-ml-toolbox
```
**Note:** This version is outdated or broken due to dependency incompatibilities. Use PyPi instead.

## Optional dependencies

### FreeSimpleGUI

Wrapper library used to build powerful GUIs. Requires the tkinter backend.

```bash
pip install dragon-ml-toolbox[gui]
```

### PyTorch

Different builds available depending on the **platform** and **hardware acceleration** (e.g., CUDA for NVIDIA GPUs on Linux/Windows, or MPS for Apple Silicon on macOS).

Install the default CPU-only version with

```bash
pip install dragon-ml-toolbox[pytorch]
```

To make use of GPU acceleration use the official PyTorch installation instructions:

[PyTorch Instructions](https://pytorch.org/get-started/locally/)

## Usage

After installation, import modules like this:

```python
from ml_tools.utilities import sanitize_filename
from ml_tools.logger import custom_logger
```

## Available modules

```bash
data_exploration
datasetmaster
ensemble_learning
ETL_engineering
GUI_tools
handle_excel
logger
MICE_imputation
ML_callbacks
ML_evaluation
ML_trainer
ML_tutorial
path_manager
PSO_optimization
RNN_forecast
utilities
VIF_factor
```
