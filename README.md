# Dragon-ML-Toolbox

A collection of machine learning pipelines and utilities, structured as modular packages for lightweight and customized virtual environments.

Modular scripts for data science workflows, including data exploration, ETL, model training, evaluation, and inference.

Support for PyTorch-based models, ensemble learning (XGBoost, LightGBM), MICE imputation, and basic GUI factory classes for model deployment.

### Base Dependencies

```toml
"numpy>=2.0",
"pandas>=3.0",
"polars>=2.0",
"pyarrow",
"joblib",
"colorlog"
```

### Supported Python Versions

Python 3.12

Python 3.13

Python 3.14


## Installation

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
conda install conda-forge::dragon-ml-toolbox
```

## Modular Installation

This toolbox is designed as a collection of mutually exclusive environments due to conflicting core dependencies.

- Rule: Create a fresh virtual environment for each module to use.

### 📦 Core Machine Learning Toolbox [ML]

Installs a comprehensive set of tools for typical data science workflows, including data manipulation, modeling, and evaluation using PyTorch.

🪟 On Windows, the default installation includes the CPU version of PyTorch. Follow the official instructions to install the CUDA version: [PyTorch website](https://pytorch.org/get-started/locally/)

```Bash
pip install "dragon-ml-toolbox[ML]"
```

#### Modules:

```Bash
data_exploration
ETL_cleaning
ETL_engineering
imputation
IO
keys
math_utilities
callbacks
chain
configuration
datasetmaster
evaluation
evaluation_captum
ML_finalize_handler
finetune
inference
inference_sequence
inference_vision
loss
models
models_diffusion
models_sequence
models_vision
optimization
scaler
trainer
trainer_parallel
ML_utilities
vision_transformers
vision_utilities
optimization_tools
outlier_detection
path_manager
plot_fonts
resampling
schema
serde
SQL
utilities
VIF
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
ensemble_optimization
ETL_cleaning
ETL_engineering
imputation
IO
math_utilities
optimization_tools
outlier_detection
path_manager
plot_fonts
resampling
schema
serde
SQL
utilities
VIF
constants
```

---

### 🔠 Natural Language Processing [NLP]

NLP class using Sentence Transformers for generating dense embeddings from text data. Uses HuggingFace models and includes utilities for text preprocessing, TF-IDF vectorization, and dimensionality reduction.

```Bash
pip install "dragon-ml-toolbox[NLP]"
```

##### Modules:

```Bash
IO
NLP
path_manager
serde
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
imputation
IO
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
IO
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
IO
path_manager
schema
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
GUI_tools
IO
keys
models
models_sequence
models_vision # Requires: torchvision and Pillow
inference
inference_sequence
inference_vision # Requires: torchvision and Pillow
vision_transformers # Requires: torchvision and Pillow
vision_utilities # Requires: torchvision and Pillow
scaler
path_manager
schema
serde
```

---

### Usage

After installation, import modules like this:

```python
from dragon.serde import serialize_object, deserialize_object
from dragon.IO import custom_logger
```
