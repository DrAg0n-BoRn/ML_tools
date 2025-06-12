# dragon-ml-tools

A collection of Python utilities and machine learning tools, structured as a modular package for easy reuse and installation.

## Features

- Modular scripts for data exploration, logging, machine learning, and more.
- Optional dependencies grouped by functionality for lightweight installs.
- Designed for seamless integration as a Git submodule or installable Python package.


## Installation

### Via GitHub (Editable / Development Mode)

Clone the repository and install in editable mode with optional dependencies:

```bash
git clone https://github.com/DrAg0n-BoRn/ML_tools.git
cd ML_tools
pip install -e '.[utilities]'
```

### Via PyPI (Stable Releases)

Install the latest stable release from PyPI with optional dependencies:

```bash
pip install dragon-ml-tools[utilities]
```

## Usage

After installation, import modules like this:

```python
from ml_tools.utilities import sanitize_filename
from ml_tools.logger import custom_logger
```

## Development

Python 3.9+ recommended.

To install all dependencies including development tools:

```python
pip install -e '.[dev]'
```

