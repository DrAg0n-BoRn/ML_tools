# dragon-ml-tools

A collection of Python utilities for data science and machine learning, structured as a modular package for easy reuse and installation.

## Features

- Modular scripts for data exploration, logging, machine learning, and more.
- Designed for seamless integration as a Git submodule or installable Python package.

## Installation

**Python 3.9+ recommended.**

### Via PyPI

Install the latest stable release from PyPI:

```bash
pip install dragon-ml-tools
```

### Via conda-forge

Install from the conda-forge channel:

```bash
conda install -c conda-forge dragon-ml-toolbox
```

#### Optional dependencies

```bash
pip install dragon-ml-tools[pytorch]
```

### Via GitHub (Editable)

Clone the repository and install in editable mode with optional dependencies:

```bash
git clone https://github.com/DrAg0n-BoRn/ML_tools.git
cd ML_tools
pip install -e '.[pytorch]'
```

## Usage

After installation, import modules like this:

```python
from ml_tools.utilities import sanitize_filename
from ml_tools.logger import custom_logger
```
