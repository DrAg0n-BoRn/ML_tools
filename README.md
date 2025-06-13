# dragon-ml-tools

A collection of Python utilities and machine learning tools, structured as a modular package for easy reuse and installation.

## Features

- Modular scripts for data exploration, logging, machine learning, and more.
- Optional dependencies grouped by functionality for lightweight installs.
- Designed for seamless integration as a Git submodule or installable Python package.


## Installation

Python 3.9+ recommended.

### Via PyPI (Stable Releases)

Install the latest stable release from PyPI with optional dependencies:

```bash
pip install dragon-ml-tools[logger,trainer]
```

To install dependencies from all modules

```bash
pip install dragon-ml-tools[full]
```

### Via GitHub (Editable)

Clone the repository and install in editable mode with optional dependencies:

```bash
git clone https://github.com/DrAg0n-BoRn/ML_tools.git
cd ML_tools
pip install -e '.[logger]'
```

## Usage

After installation, import modules like this:

```python
from ml_tools.utilities import sanitize_filename
from ml_tools.logger import custom_logger
```
