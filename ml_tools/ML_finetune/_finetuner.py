import torch
import torch.nn as nn
from typing import Union, Iterable

from .._core import get_logger

_LOGGER = get_logger("Finetuner")

__all__ = ["DragonFinetuner"]


class DragonFinetuner:
    """
    An interactive fine-tuning manager for PyTorch and DragonML models.
    
    Designed for Jupyter Notebooks to visualize, freeze, and unfreeze model layers.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes the DragonFinetuner with a given model.
        
        Args:
            model (nn.Module): The PyTorch or DragonML model to be fine-tuned.
        """
        self.model = model
        self.components = self._parse_components()

    def _parse_components(self) -> dict[str, nn.Module]:
        """
        Attempts to parse logical components from DragonML models.
        Falls back to standard PyTorch named_children if the interface is missing.
        """
        if hasattr(self.model, '_get_finetune_components'):
            return self.model._get_finetune_components() # type: ignore
        
        _LOGGER.warning("Model does not have '_get_finetune_components'. Falling back to named_children().")
        
        # Safe fallback for generic PyTorch models
        children = dict(self.model.named_children())
        if not children:
            # If the model has no children (e.g., it's a single layer), map the model itself
            _LOGGER.warning("Model has no named children; returning the entire model as a single component.")
            return {"entire_model": self.model}
        return children

    # ---------------------------------------------------------
    # Jupyter Interactive Visualization
    # ---------------------------------------------------------
    
    def summary(self) -> None:
        """
        Prints an interactive summary of the model's components, parameter counts, and their current trainable state.
        """
        print(f"--- Finetuning Summary for {self.model.__class__.__name__} ---")
        header = f"{'Component':<25} | {'Params':<12} | {'Trainable Params':<20} | {'Status'}"
        print(header)
        print("-" * len(header))
        
        total_params = 0
        total_trainable = 0

        for name, module in self.components.items():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += params
            total_trainable += trainable
            
            if trainable == 0:
                status = "❄️ Frozen"
            elif trainable == params:
                status = "🔥 Unfrozen"
            else:
                status = "🌤️ Partially Unfrozen"
                
            print(f"{name:<25} | {params:<12,d} | {trainable:<20,d} | {status}")
            
        print("-" * len(header))
        print(f"Total Params: {total_params:,d} | Trainable: {total_trainable:,d}")

    # ---------------------------------------------------------
    # Fine-Tuning Actions
    # ---------------------------------------------------------

    def freeze_all(self):
        """Freezes all parameters in the entire model."""
        self._set_requires_grad(self.model.parameters(), False)
        _LOGGER.info("❄️ All model parameters are now frozen.")

    def unfreeze_all(self):
        """Unfreezes all parameters in the entire model."""
        self._set_requires_grad(self.model.parameters(), True)
        _LOGGER.info("🔥 All model parameters are now unfrozen.")

    def unfreeze_components(self, component_names: Union[str, list[str]]):
        """
        Unfreezes specific semantic components (e.g., 'head', 'backbone').
        
        Args:
            component_names (str | list[str]): Name(s) of the components to unfreeze.
        """
        if isinstance(component_names, str):
            component_names = [component_names]
        
        successful_unfrozen = []
        
        for name in component_names:
            if name not in self.components:
                _LOGGER.warning(f"Component '{name}' not found. Available: {list(self.components.keys())}")
                continue
        
            self._set_requires_grad(self.components[name].parameters(), True)
            successful_unfrozen.append(name)
        
        if successful_unfrozen:
            log_msg = "🔥 Unfroze components:\n" + "\n".join(f"    - {name}" for name in successful_unfrozen)
            _LOGGER.info(log_msg)
        
            
    def freeze_components(self, component_names: Union[str, list[str]]):
        """
        Freezes specific semantic components (e.g., 'head', 'backbone').
        
        Args:
            component_names (str | list[str]): Name(s) of the components to freeze.
        """
        if isinstance(component_names, str):
            component_names = [component_names]
            
        success_frozen = []
            
        for name in component_names:
            if name not in self.components:
                _LOGGER.warning(f"Component '{name}' not found. Available: {list(self.components.keys())}")
                continue
            
            self._set_requires_grad(self.components[name].parameters(), False)
            success_frozen.append(name)
            
        if success_frozen:
            log_msg = "❄️ Frozen components:\n" + "\n".join(f"    - {name}" for name in success_frozen)
            _LOGGER.info(log_msg)

    def unfreeze_last_n_parameter_tensors(self, n: int):
        """
        Legacy support: Unfreezes the last N individual parameter tensors (e.g., `weight`, `bias`) while freezing the rest.
        
        For semantic component unfreezing, prefer `unfreeze_components()` or `freeze_components()`.
        """
        if n < 0:
            _LOGGER.error(f"N must be >= 0, but got {n}")
            raise ValueError()
        
        all_params = list(self.model.parameters())
        total_param_tensors = len(all_params)

        if n == 0:
            return self.freeze_all()

        if n >= total_param_tensors:
            _LOGGER.warning(f"Requested to unfreeze {n} tensors, but model only has {total_param_tensors}.")
            return self.unfreeze_all()

        # Freeze all first, then unfreeze the tail end
        self.freeze_all()
        params_to_unfreeze = all_params[-n:]
        unfrozen_count = self._set_requires_grad(params_to_unfreeze, True)
        
        _LOGGER.info(f"Unfroze the last {n} parameter tensors ({unfrozen_count} elements).")

    def get_model(self) -> nn.Module:
        """Returns the modified model."""
        return self.model

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _set_requires_grad(params: Iterable[nn.Parameter], requires_grad: bool) -> int:
        """Helper to toggle gradients and return total elements modified."""
        params_changed = 0
        for param in params:
            if param.requires_grad != requires_grad:
                param.requires_grad = requires_grad
                params_changed += param.numel()
        return params_changed
    
    def __repr__(self) -> str:
        return f"<DragonFinetuner(model={self.model.__class__.__name__})>"
