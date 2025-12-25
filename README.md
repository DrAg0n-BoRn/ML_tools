# dragon-ml-toolbox

A collection of machine learning pipelines and utilities, structured as modular packages for easy reuse and installation. This package has no base dependencies, allowing for lightweight and customized virtual environments.

### Features:

- Modular scripts for data science workflows, including data exploration, ETL, model training, evaluation, and inference.
- Support for PyTorch-based models, ensemble learning (XGBoost, LightGBM), and MICE imputation.

## Installation

**Python 3.12**

### Via PyPI

Install the latest stable release from PyPI:

Using pip:

```bash
pip install dragon-ml-toolbox
```

Using UV:

```bash
uv add dragon-ml-toolbox
```

### Via conda-forge

Install from the conda-forge channel:

```bash
conda install -c conda-forge dragon-ml-toolbox
```

## Modular Installation

This toolbox is designed as a collection of mutually exclusive environments due to conflicting core dependencies.

- Rule: Create a fresh virtual environment for each module to use.

### 📦 Core Machine Learning Toolbox [ML]

Installs a comprehensive set of tools for typical data science workflows, including data manipulation, modeling, and evaluation using PyTorch.

➡️ On Windows, the default installation includes the CPU version of PyTorch. Follow the official instructions to install the CUDA version: [PyTorch website](https://pytorch.org/get-started/locally/)

```Bash
pip install "dragon-ml-toolbox[ML]"
```

#### Modules:

```Bash
data_exploration
ETL_cleaning
ETL_engineering
IO_tools
keys
math_utilities
ML_callbacks
ML_chain
ML_configuration
ML_datasetmaster
ML_evaluation
ML_evaluation_captum
ML_finalize_handler
ML_inference
ML_inference_sequence
ML_inference_vision
ML_models
ML_models_sequence
ML_models_vision
ML_optimization
ML_scaler
ML_trainer
ML_utilities
ML_vision_transformers
optimization_tools
path_manager
plot_fonts
schema
serde
SQL
utilities
constants
```

---

### 🌳 Ensemble Learning [ensemble]

Comprehensive set of tools for typical data science workflows focused on **XGBoost** and **LightGBM**.

```Bash
pip install "dragon-ml-toolbox[ensemble]"
```

#### Modules:

```Bash
data_exploration
ensemble_evaluation
ensemble_inference
ensemble_learning
ETL_cleaning
ETL_engineering
IO_tools
math_utilities
optimization_tools
path_manager
plot_fonts
PSO_optimization
schema
serde
SQL
utilities
constants
```

---

### 🔬 MICE Imputation and Variance Inflation Factor [mice]

Utilities for advanced data cleaning and statistical checks. Features **Multiple Imputation by Chained Equations (MICE)** for handling missing data and **Variance Inflation Factor (VIF)** analysis to detect multicollinearity in features.

```Bash
pip install "dragon-ml-toolbox[mice]"
```

#### Modules:

```Bash
IO_tools
math_utilities
MICE
path_manager
plot_fonts
serde
utilities
VIF
```

---

### 📋 Excel File Handling [excel]

Installs dependencies required to process and handle .xlsx or .xls files.

```Bash
pip install "dragon-ml-toolbox[excel]"
```

#### Modules:

```Bash
IO_tools
excel_handler
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
ensemble_inference
GUI_tools
IO_tools
path_manager
schema
serde
constants
```

---

### 🤖 GUI for PyTorch Models [gui-torch]

GUI tools compatible with PyTorch models used for inference.

```Bash
pip install "dragon-ml-toolbox[gui-torch]"
```

#### Modules:

```Bash
GUI_tools
IO_tools
keys
ML_models
ML_models_sequence
ML_models_vision # Requires: torchvision and Pillow
ML_inference
ML_inference_sequence
ML_inference_vision # Requires: torchvision and Pillow
ML_vision_transformers # Requires: torchvision and Pillow
ML_scaler
path_manager
schema
constants
```

---

## Usage

After installation, import modules like this:

```python
from ml_tools.serde import serialize_object, deserialize_object
from ml_tools.IO_tools import train_logger
```
