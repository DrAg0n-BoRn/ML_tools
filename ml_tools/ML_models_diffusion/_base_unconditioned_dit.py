from typing import Any
import torch
from torch import nn
from abc import ABC

from ..ML_models._base_save_load import _ArchitectureHandlerMixin
from ..ML_utilities._artifact_finder import DragonArtifactFinder
from ..ML_finalize_handler import FinalizedFileHandler
from .._core import get_logger

from ._dit_parts import TimeEmbedding


_LOGGER = get_logger("Dragon DiT")


class _BaseDragonDiT(_ArchitectureHandlerMixin, nn.Module, ABC):
    """
    Base Unconditioned DiT model for generating sequences using Flow Matching.
    Child classes must populate `self.blocks` with the desired transformer block version.
    """
    def __init__(self, 
                 embed_dim: int,
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.depth = depth
        
        # Positional Embeddings for the Diffusion Model
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        
        # Map raw time to a feature vector
        self.time_mlp = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # To be populated by subclasses (V1 or V2)
        self.blocks = nn.ModuleList([])
        
        # Final layer to predict velocity v_t
        self.final_layer = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
        
    def forward(self, x_t, t):
        t = t.view(t.size(0), 1)
        c = self.time_mlp(t) 
        
        x = x_t + self.pos_embed
        
        for block in self.blocks:
            x = block(x, c)
            
        v_pred = self.final_layer(x)
        return v_pred
    
    @torch.no_grad()
    def generate_sequence(self, 
                          batch_size: int, 
                          num_steps: int = 25) -> torch.Tensor:
        """
        Generates new discrete token sequences from pure noise using Flow Matching. Must be decoded back to the original feature space using the tokenizer's decoder.
        
        Args:
            batch_size (int): The number of samples to generate in the batch.
            num_steps (int): The number of steps to use in the ODE solver.
        
        Returns:
            torch.Tensor: The generated token sequences in the embedding space, of shape [batch_size, seq_len, embed_dim]. Must be decoded back to the original feature space using the tokenizer's decoder
        """
        self.eval()
        _LOGGER.info(f"Generating a batch of {batch_size} samples with {self.seq_len} features using Flow Matching with {num_steps} steps.")
        
        validated_device = next(self.parameters()).device
        
        x_t = torch.randn(batch_size, self.seq_len, self.embed_dim, device=validated_device)
        t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=validated_device)
        
        for i in range(num_steps):
            t_val = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_val
            
            t_tensor = torch.full((batch_size, 1, 1), t_val.item(), device=validated_device)
            
            v_pred = self(x_t, t_tensor)
            x_euler = x_t + v_pred * dt
            
            if i == num_steps - 1:
                x_t = x_euler
                break
                
            t_next_tensor = torch.full((batch_size, 1, 1), t_next.item(), device=validated_device)
            v_pred_next = self(x_euler, t_next_tensor)
            
            v_heun = 0.5 * (v_pred + v_pred_next)
            x_t = x_t + v_heun * dt
            
        return x_t
    
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int=2) -> nn.Module:
        """ 
        Loads a DragonDiT model from the artifacts found by the provided artifact finder. Ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth
        """
        if not artifact_finder.model_architecture_path:
            _LOGGER.error(f"Model architecture file not found at expected path.")
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            _LOGGER.error(f"Model weights file not found at expected path.")
            raise FileNotFoundError()
        
        model: '_BaseDragonDiT' = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False) # type: ignore
        
        finalized_file = FinalizedFileHandler(artifact_finder.weights_path)
        
        model.load_state_dict(finalized_file.model_state_dict)
        model.eval()

        if verbose >= 2:
            _LOGGER.info("Model architecture and weights successfully loaded.")
        
        return model

    def get_architecture_config(self) -> dict[str, Any]:
        """Returns the configuration necessary to reconstruct the architecture."""
        return {
            "embed_dim": self.embed_dim,
            "seq_len": self.seq_len,
            "num_heads": self.num_heads,
            "depth": self.depth
        }
