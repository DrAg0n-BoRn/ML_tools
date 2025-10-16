# dragon-ml-toolbox

A collection of Python utilities for data science and machine learning, structured as a modular package for easy reuse and installation. This package has no base dependencies, allowing for lightweight and customized virtual environments.

### Features:

- Modular scripts for data exploration, logging, machine learning, and more.
- Designed for seamless integration as a Git submodule or installable Python package.

## Installation

**Python 3.10+**

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

## Modular Installation

### 📦 Core Machine Learning Toolbox [ML]

Installs a comprehensive set of tools for typical data science workflows, including data manipulation, modeling, and evaluation. PyTorch is required.

```Bash
pip install "dragon-ml-toolbox[ML]"
```

To install the standard CPU-only versions of Torch and Torchvision:

```Bash
pip install "dragon-ml-toolbox[pytorch]"
```

⚠️ To make use of GPU acceleration (highly recommended), follow the official instructions: [PyTorch website](https://pytorch.org/get-started/locally/)

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
ML_callbacks
ML_datasetmaster
ML_evaluation_multi
ML_evaluation
ML_inference
ML_models
ML_optimization
ML_scaler
ML_trainer
optimization_tools
path_manager
PSO_optimization
RNN_forecast
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
MICE_imputation
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

For GUIs that include plotting functionality, you must also install the [plot] extra.

```Bash
pip install "dragon-ml-toolbox[gui-boost]"
```

```Bash
pip install "dragon-ml-toolbox[gui-boost,plot]"
```

#### Modules:

```Bash
constants
custom_logger
GUI_tools
ensemble_inference
path_manager
```

---

### 🤖 GUI for PyTorch Models [gui-torch]

For GUIs that include plotting functionality, you must also install the [plot] extra.

```Bash
pip install "dragon-ml-toolbox[gui-torch]"
```

```Bash
pip install "dragon-ml-toolbox[gui-torch,plot]"
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
from ml_tools.utilities import serialize_object, deserialize_object
from ml_tools import custom_logger
```
