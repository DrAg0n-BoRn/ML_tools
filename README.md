# dragon-ml-toolbox

A collection of machine learning pipelines and utilities, structured as modular packages for easy reuse and installation. This package has no base dependencies, allowing for lightweight and customized virtual environments.

### Features:

- Modular scripts for data exploration, logging, machine learning, and more.
- Designed for seamless integration as a Git submodule or installable Python package.

## Installation

**Python 3.12**

### Via PyPI

Install the latest stable release from PyPI:

```bash
pip install dragon-ml-toolbox
```

### Via conda-forge

Install from the conda-forge channel:

```bash
conda install -c conda-forge dragon-ml-toolbox
```

### Via GitHub (Editable)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/DrAg0n-BoRn/ML_tools.git
cd ML_tools
pip install -e .
```

## Modular Installation

This toolbox is designed as a collection of mutually exclusive environments due to conflicting core dependencies, except APP bundlers (PyInstaller/Nuitka).

- Rule: Create a fresh virtual environment for each module to use.

### 📦 Core Machine Learning Toolbox [ML]

Installs a comprehensive set of tools for typical data science workflows, including data manipulation, modeling, and evaluation using PyTorch.

➡️ Prerequisite: PyTorch required. Follow the official instructions: [PyTorch website](https://pytorch.org/get-started/locally/)

```Bash
pip install "dragon-ml-toolbox[ML]"
```

#### Modules:

```Bash
constants
custom_logger
data_exploration
ETL_cleaning
ETL_engineering
math_utilities
ML_callbacks
ML_configuration
ML_datasetmaster
ML_evaluation_captum
ML_evaluation_multi
ML_evaluation
ML_inference
ML_models
ML_models_advanced
ML_optimization_pareto
ML_optimization
ML_scaler
ML_sequence_datasetmaster
ML_sequence_evaluation
ML_sequence_inference
ML_sequence_models
ML_trainer
ML_utilities
ML_vision_datasetmaster
ML_vision_evaluation
ML_vision_inference
ML_vision_models
ML_vision_transformers
optimization_tools
path_manager
serde
SQL
utilities
```

Wrappers for some `pytorch_tabular` models are available:

```Bash
pip install "dragon-ml-toolbox[ML,py-tab]"

# Extra Modules:
ML_models_pytab
ML_configuration_pytab
```

---

### 🌳 Ensemble Learning [ensemble]

Comprehensive set of tools for typical data science workflows focused on **XGBoost** and **LightGBM**.

*Note: Optimizes for NumPy >= 2.0.*

```Bash
pip install "dragon-ml-toolbox[ensemble]"
```

#### Modules:

```Bash
constants
custom_logger
data_exploration
ensemble_evaluation
ensemble_inference
ensemble_learning
ETL_cleaning
ETL_engineering
math_utilities
optimization_tools
path_manager
PSO_optimization
serde
SQL
utilities
```

---

### 🔬 MICE Imputation and Variance Inflation Factor [mice]

Utilities for advanced data cleaning and statistical checks. Features **Multiple Imputation by Chained Equations (MICE)** for handling missing data and **Variance Inflation Factor (VIF)** analysis to detect multicollinearity in features.

```Bash
pip install "dragon-ml-toolbox[mice]"
```

#### Modules:

```Bash
constants
custom_logger
math_utilities
MICE_imputation
serde
VIF_factor
path_manager
utilities
```

---

### 📋 Excel File Handling [excel]

Installs dependencies required to process and handle .xlsx or .xls files.

```Bash
pip install "dragon-ml-toolbox[excel]"
```

#### Modules:

```Bash
custom_logger
handle_excel
path_manager
```

---

### 🎰 GUI for Boosting Algorithms (XGBoost, LightGBM) [gui-boost]

GUI tools compatible with XGBoost and LightGBM models used for inference.

```Bash
pip install "dragon-ml-toolbox[gui-boost]"
```

#### Modules:

```Bash
constants
custom_logger
GUI_tools
ensemble_inference
path_manager
serde
```

---

### 🤖 GUI for PyTorch Models [gui-torch]

GUI tools compatible with PyTorch models used for inference.

```Bash
pip install "dragon-ml-toolbox[gui-torch]"
```

#### Modules:

```Bash
constants
custom_logger
GUI_tools
ML_models
ML_sequence_models
ML_vision_models
ML_inference
ML_sequence_inference
ML_vision_inference
ML_vision_transformers
ML_scaler
path_manager
```

---

### ⚒️ APP bundlers

Dependencies required to compile applications, inference scripts, or GUIs into standalone executables (`.exe` or binary) for distribution. Choose your preferred backend:

```Bash
pip install "dragon-ml-toolbox[pyinstaller]"
```

```Bash
pip install "dragon-ml-toolbox[nuitka]"
```

## Usage

After installation, import modules like this:

```python
from ml_tools.serde import serialize_object, deserialize_object
from ml_tools import custom_logger
```
