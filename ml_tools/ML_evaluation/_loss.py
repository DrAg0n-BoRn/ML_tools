import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Union

from ..path_manager import make_fullpath
from .._core import get_logger
from ..keys._keys import PyTorchLogKeys, _EvaluationConfig


_LOGGER = get_logger("Loss Plot")


__all__ = [
    "plot_losses", 
]


DPI_value = _EvaluationConfig.DPI


def plot_losses(history: dict, save_dir: Union[str, Path]):
    """
    Plots training & validation loss curves from a history object.
    Also plots the learning rate if available in the history.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss'.
        save_dir (str | Path): Directory to save the plot image.
    """
    train_loss = history.get(PyTorchLogKeys.TRAIN_LOSS, [])
    val_loss = history.get(PyTorchLogKeys.VAL_LOSS, [])
    lr_history = history.get(PyTorchLogKeys.LEARNING_RATE, [])
    
    if not train_loss and not val_loss:
        _LOGGER.warning("Loss history is empty or incomplete. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=_EvaluationConfig.LOSS_PLOT_SIZE, dpi=DPI_value)
    
    # --- Plot Losses (Left Y-axis) ---
    line_handles = [] # To store line objects for the legend
    
    # Plot training loss only if data for it exists
    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        line1, = ax.plot(epochs, train_loss, 'o-', label='Training Loss', color='tab:blue')
        line_handles.append(line1)
    
    # Plot validation loss only if data for it exists
    if val_loss:
        epochs = range(1, len(val_loss) + 1)
        line2, = ax.plot(epochs, val_loss, 'o-', label='Validation Loss', color='tab:orange')
        line_handles.append(line2)
    
    ax.set_title('Training and Validation Loss', fontsize=_EvaluationConfig.LOSS_PLOT_LABEL_SIZE + 2, pad=_EvaluationConfig.LABEL_PADDING)
    ax.set_xlabel('Epochs', fontsize=_EvaluationConfig.LOSS_PLOT_LABEL_SIZE, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax.set_ylabel('Loss', color='tab:blue', fontsize=_EvaluationConfig.LOSS_PLOT_LABEL_SIZE, labelpad=_EvaluationConfig.LABEL_PADDING)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=_EvaluationConfig.LOSS_PLOT_TICK_SIZE)
    ax.tick_params(axis='x', labelsize=_EvaluationConfig.LOSS_PLOT_TICK_SIZE)
    ax.grid(True, linestyle='--')
    
    # --- Plot Learning Rate (Right Y-axis) ---
    if lr_history:
        ax2 = ax.twinx() # Create a second y-axis
        epochs = range(1, len(lr_history) + 1)
        line3, = ax2.plot(epochs, lr_history, 'g--', label='Learning Rate')
        line_handles.append(line3)
        
        ax2.set_ylabel('Learning Rate', color='g', fontsize=_EvaluationConfig.LOSS_PLOT_LABEL_SIZE, labelpad=_EvaluationConfig.LABEL_PADDING)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=_EvaluationConfig.LOSS_PLOT_TICK_SIZE)
        # Use scientific notation if the LR is very small
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # increase the size of the scientific notation
        ax2.yaxis.get_offset_text().set_fontsize(_EvaluationConfig.LOSS_PLOT_TICK_SIZE - 2)
        # remove grid from second y-axis
        ax2.grid(False)
    
    # Combine legends from both axes
    ax.legend(handles=line_handles, loc='best', fontsize=_EvaluationConfig.LOSS_PLOT_LEGEND_SIZE)
    
    # ax.grid(True)
    plt.tight_layout()    
    
    save_dir_path = make_fullpath(save_dir, make=True, enforce="directory")
    save_path = save_dir_path / "loss_plot.svg"
    plt.savefig(save_path)
    _LOGGER.info(f"📉 Loss plot saved as '{save_path.name}'")

    plt.close(fig)

