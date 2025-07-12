import json
from typing import Literal, Optional, Union
from pathlib import Path
from .logger import _LOGGER
from .utilities import make_fullpath, sanitize_filename


__all__ = [
    "generate_notebook"
]

def _get_notebook_content(kind: str):
    """Helper function to generate the cell content for the notebook."""
    
    # --- Common Cells ---
    imports_cell = {
        "cell_type": "code",
        "source": [
            "import torch\n",
            "from torch import nn\n",
            "from torch.utils.data import TensorDataset, DataLoader\n",
            "import numpy as np\n",
            "from pathlib import Path\n",
            "\n",
            "# Import from dragon_ml_toolbox\n",
            "from ml_tools.ML_trainer import MyTrainer\n",
            "from ml_tools.ML_callbacks import EarlyStopping, ModelCheckpoint"
            "from ml_tools.keys import LogKeys"
        ]
    }
    
    device_cell = {
        "cell_type": "code",
        "source": [
            "import torch\\n",
            "if torch.cuda.is_available():\\n",
            "    device = 'cuda'\\n",
            "elif torch.backends.mps.is_available():\\n",
            "    device = 'mps'\\n",
            "else:\\n",
            "    device = 'cpu'\\n",
            "\\n",
            "print(f'Using device: {device}')"
        ]
    }
    
    model_definition_cell = {
        "cell_type": "markdown",
        "source": [
            "### 3. Define the Model, Criterion, and Optimizer\n",
            "Next, we define a simple neural network for our task. We also need to choose a loss function (`criterion`) and an `optimizer`."
        ]
    }

    callbacks_cell = {
        "cell_type": "code",
        "source": [
            "# Define callbacks for training\n",
            "model_filepath = 'best_model.pth'\n",
            "monitor_metric = LogKeys.VAL_LOSS\n",
            "\n",
            "model_checkpoint = ModelCheckpoint(\n",
            "    filepath=model_filepath, \n",
            "    save_best_only=True, \n",
            "    monitor=monitor_metric, \n",
            "    mode='min'\n",
            ")\n",
            "\n",
            "early_stopping = EarlyStopping(\n",
            "    patience=10, \n",
            "    monitor=monitor_metric, \n",
            "    mode='min'\n",
            ")"
        ]
    }

    trainer_instantiation_cell = {
        "cell_type": "code",
        "source": [
            "trainer = MyTrainer(\n",
            "    model=model,\n",
            "    train_dataset=train_dataset,\n",
            "    test_dataset=test_dataset,\n",
            f"    kind='{kind}',\n",
            "    criterion=criterion,\n",
            "    optimizer=optimizer,\n",
            "    device=device,\\n",
            "    callbacks=[model_checkpoint, early_stopping]\n",
            ")"
        ]
    }

    fit_cell = {
        "cell_type": "code",
        "source": [
            "history = trainer.fit(epochs=100, batch_size=16)"
        ]
    }

    evaluation_cell = {
        "cell_type": "code",
        "source": [
            "save_dir = Path('tutorial_results')\n",
            "\n",
            "# The evaluate method will automatically use the test_loader.\n",
            "# First, we load the best weights saved by ModelCheckpoint.\n",
            "model_path = Path(model_filepath)\n",
            "if model_path.exists():\n",
            "    print(f'Loading best model from {model_path}')\n",
            "    trainer.model.load_state_dict(torch.load(model_path))\n",
            "\n",
            "print('\\n--- Evaluating Model ---')\n",
            "# All evaluation artifacts will be saved in the 'evaluation' subdirectory.\n",
            "trainer.evaluate(save_dir=save_dir / 'evaluation')"
        ]
    }
    
    explanation_cell = {
        "cell_type": "code",
        "source": [
            "print('\\n--- Explaining Model ---')\n",
            "# We can also generate SHAP plots to explain the model's predictions.\n",
            "# All SHAP artifacts will be saved in the 'explanation' subdirectory.\n",
            "trainer.explain(\n",
            "    background_loader=trainer.train_loader,\n",
            "    explain_loader=trainer.test_loader,\n",
            "    save_dir=save_dir / 'explanation'\n",
            ")"
        ]
    }


    # --- Task-Specific Cells ---
    if kind == 'classification':
        title = "Classification Tutorial"
        data_prep_source = [
            "### 2. Prepare the Data\n",
            "For this example, we'll generate some simple, linearly separable mock data for a binary classification task. We'll then wrap it in PyTorch `TensorDataset` objects."
        ]
        data_creation_source = [
            "from sklearn.datasets import make_classification\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=0, random_state=42)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Convert to PyTorch tensors\n",
            "X_train = torch.FloatTensor(X_train)\n",
            "y_train = torch.LongTensor(y_train)\n",
            "X_test = torch.FloatTensor(X_test)\n",
            "y_test = torch.LongTensor(y_test)\n",
            "\n",
            "# Create TensorDatasets\n",
            "train_dataset = TensorDataset(X_train, y_train)\n",
            "test_dataset = TensorDataset(X_test, y_test)"
        ]
        model_creation_source = [
            "class SimpleClassifier(nn.Module):\n",
            "    def __init__(self, input_features, num_classes):\n",
            "        super().__init__()\n",
            "        self.layer_1 = nn.Linear(input_features, 32)\n",
            "        self.layer_2 = nn.Linear(32, num_classes)\n",
            "        self.relu = nn.ReLU()\n",
            "    \n",
            "    def forward(self, x):\n",
            "        return self.layer_2(self.relu(self.layer_1(x)))\n",
            "\n",
            "model = SimpleClassifier(input_features=10, num_classes=2)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)"
        ]

    elif kind == 'regression':
        title = "Regression Tutorial"
        data_prep_source = [
            "### 2. Prepare the Data\n",
            "For this example, we'll generate some simple mock data for a regression task. We'll then wrap it in PyTorch `TensorDataset` objects."
        ]
        data_creation_source = [
            "from sklearn.datasets import make_regression\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)\n",
            "y = y.reshape(-1, 1) # Reshape for compatibility with MSELoss\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Convert to PyTorch tensors\n",
            "X_train = torch.FloatTensor(X_train)\n",
            "y_train = torch.FloatTensor(y_train)\n",
            "X_test = torch.FloatTensor(X_test)\n",
            "y_test = torch.FloatTensor(y_test)\n",
            "\n",
            "# Create TensorDatasets\n",
            "train_dataset = TensorDataset(X_train, y_train)\n",
            "test_dataset = TensorDataset(X_test, y_test)"
        ]
        model_creation_source = [
            "class SimpleRegressor(nn.Module):\n",
            "    def __init__(self, input_features, output_features):\n",
            "        super().__init__()\n",
            "        self.layer_1 = nn.Linear(input_features, 32)\n",
            "        self.layer_2 = nn.Linear(32, output_features)\n",
            "        self.relu = nn.ReLU()\n",
            "    \n",
            "    def forward(self, x):\n",
            "        return self.layer_2(self.relu(self.layer_1(x)))\n",
            "\n",
            "model = SimpleRegressor(input_features=5, output_features=1)\n",
            "criterion = nn.MSELoss()\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)"
        ]
    else:
        raise ValueError("kind must be 'classification' or 'regression'")

    # --- Assemble Notebook ---
    cells = [
        {"cell_type": "markdown", "source": [f"# Dragon ML Toolbox - {title}\n", "This notebook demonstrates how to use the `MyTrainer` class for a complete training and evaluation workflow."]},
        {"cell_type": "markdown", "source": ["### 1. Imports\n", "First, let's import all the necessary components."]},
        imports_cell,
        {"cell_type": "markdown", "source": data_prep_source},
        {"cell_type": "code", "source": data_creation_source},
        model_definition_cell,
        {"cell_type": "code", "source": model_creation_source},
        {"cell_type": "markdown", "source": ["### 4. Configure Callbacks\n", "We'll set up `ModelCheckpoint` to save the best model and `EarlyStopping` to prevent overfitting."]},
        callbacks_cell,
        {"cell_type": "markdown", "source": ["### 5. Initialize the Trainer\\n", "First, we'll determine the best device to run on. Then, we can instantiate `MyTrainer` with all our components."]}, 
        device_cell,
        trainer_instantiation_cell,
        {"cell_type": "markdown", "source": ["### 6. Train the Model\n", "Call the `.fit()` method to start training."]},
        fit_cell,
        {"cell_type": "markdown", "source": ["### 7. Evaluate the Model\n", "Finally, call the `.evaluate()` method to see the performance report and save all plots and metrics."]},
        evaluation_cell,
        {"cell_type": "markdown", "source": ["### 8. Explain the Model\n", "We can also use the `.explain()` method to generate and save SHAP plots for model interpretability."]},
        explanation_cell,
    ]
    
    # Add execution counts to code cells
    for cell in cells:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['metadata'] = {}
            cell['outputs'] = []
            
    return cells


def generate_notebook(kind: Literal['classification', 'regression'] = 'classification', filepath: Optional[Union[str,Path]] = None):
    """
    Generates a tutorial Jupyter Notebook (.ipynb) file.

    This function creates a complete, runnable notebook with mock data,
    a simple model, and a full training/evaluation cycle using MyTrainer.

    Args:
        kind (str): The type of tutorial to generate, either 'classification' or 'regression'.
        filepath (str | Path | None): The path to save the notebook file. 
                                  If None, defaults to 'classification_tutorial.ipynb' or
                                  'regression_tutorial.ipynb' in the current directory.
    """
    if kind not in ["classification", "regression"]:
        raise ValueError("kind must be 'classification' or 'regression'")    
    
    if filepath is None:
        sanitized_filepath = f"{kind}_tutorial.ipynb"
    else:
        sanitized_filepath = sanitize_filename(str(filepath))
    
    # check suffix
    if not sanitized_filepath.endswith(".ipynb"):
        sanitized_filepath = sanitized_filepath + ".ipynb"
        
    new_filepath = make_fullpath(sanitized_filepath, make=True)

    _LOGGER.info(f"Generating {kind} tutorial notebook at: {filepath}")

    cells = _get_notebook_content(kind)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0" 
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    try:
        with open(new_filepath, 'w') as f:
            json.dump(notebook, f, indent=2)
        _LOGGER.info("Notebook generated successfully.")
    except Exception as e:
        _LOGGER.error(f"Error generating notebook: {e}")
