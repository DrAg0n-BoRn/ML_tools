# dragon-ml-toolbox

A collection of Python utilities for data science and machine learning, structured as a modular package for easy reuse and installation. This package has no base dependencies, allowing for lightweight and customized virtual environments.

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

### 📦 Core Machine Learning Toolbox [ML]

Installs a comprehensive set of tools for typical data science workflows, including data manipulation, modeling, and evaluation. PyTorch is required.

```Bash
pip install "dragon-ml-toolbox[ML]"
```

⚠️ PyTorch required, follow the official instructions: [PyTorch website](https://pytorch.org/get-started/locally/)

#### Modules:

```bash
constants
custom_logger
data_exploration
ensemble_evaluation
ensemble_inference
ensemble_learning
ETL_cleaning
ETL_engineering
math_utilities
ML_callbacks
ML_datasetmaster
ML_evaluation_multi
ML_evaluation
ML_inference
ML_models
ML_optimization
ML_scaler
ML_trainer
ML_utilities
optimization_tools
path_manager
PSO_optimization
RNN_forecast
serde
SQL
utilities
```

---

### 🔬 MICE Imputation and Variance Inflation Factor [mice]

⚠️ Important: This group has strict version requirements. It is highly recommended to install this group in a separate virtual environment.

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
constants
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
ML_inference
ML_scaler
path_manager
```

---

### ⚒️ APP bundlers

Choose one if needed.

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
