from typing import Any
import torch
from torch import nn

from ..ML_models._base_save_load import _ArchitectureHandlerMixin
from ..ML_utilities._artifact_finder import DragonArtifactFinder
from ..ML_finalize_handler import FinalizedFileHandler

from .._core import get_logger

from ._dit_parts import TimeEmbedding, DiTBlockFlash


_LOGGER = get_logger("DragonDiT")


__all__ = [
    "DragonDiT"
]


class DragonDiT(_ArchitectureHandlerMixin, nn.Module):
    """
    Unconditioned DiT model for generating sequences using Flow Matching.
    
    Modality-agnostic architecture that can work with images, text, audio, time series, or tabular data, as long as the input is tokenized into a sequence of embeddings.
    """
    def __init__(self, 
                 embed_dim: int,
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        """
        Initializes the Dragon Diffusion Transformer.
        
        Args:
            embed_dim (int): The dimensionality of the token embeddings. Must be divisible by num_heads. Must match the embedding dimension of the input tokens.
            seq_len (int): The length of the input sequences (number of features). Must match the seq_len dimension of the input tokens.
            num_heads (int): The number of attention heads in the DiT blocks.
            depth (int): The number of DiT blocks to stack.
        """
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
        
        # Uniformly load the SDPA-powered block (modern hardware will automatically use FlashAttention under the hood when available)
        self.blocks = nn.ModuleList([
            DiTBlockFlash(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # Final layer to predict velocity v_t
        self.final_layer = nn.Linear(embed_dim, embed_dim)
        
        # Zero-initialize the final layer to predict zero velocity at the start
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
        
    def forward(self, x_t, t):
        # x_t: [batch_size, seq_len, embed_dim]
        # t: [batch_size, 1, 1]
        
        # Flatten t to [batch_size, 1] for the embedding
        t = t.view(t.size(0), 1)
        c = self.time_mlp(t) # c is the time conditioning vector of shape [batch_size, embed_dim]
        
        # initial input with positional awareness
        x = x_t + self.pos_embed
        
        for block in self.blocks:
            x = block(x, c)
            
        # Predict velocity
        v_pred = self.final_layer(x)
        
        # v_pred shape: [batch_size, seq_len, embed_dim]
        return v_pred
    
    @torch.no_grad()
    def generate_sequence(self, 
                          batch_size: int, 
                          num_steps: int = 20) -> torch.Tensor:
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
        
        # validated_device = validate_torch_device(device)
        # Dynamically infer the device from the model's own parameters
        validated_device = next(self.parameters()).device
        
        
        # 1. Start with pure Gaussian noise at t = 0
        # x_t shape: [batch_size, seq_len, embed_dim]
        x_t = torch.randn(batch_size, self.seq_len, self.embed_dim, device=validated_device)
        
        # Explicitly define the time schedule from exactly 0.0 to 1.0
        t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=validated_device)
        
        # 2. Heun's 2nd-Order ODE Solver Loop
        for i in range(num_steps):
            t_val = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_val
            
            t_tensor = torch.full((batch_size, 1, 1), t_val.item(), device=validated_device)
            
            # Predictor step (Standard Euler)
            v_pred = self(x_t, t_tensor)
            x_euler = x_t + v_pred * dt
            
            # If we are at the final boundary step, an Euler step is sufficient to finish
            if i == num_steps - 1:
                x_t = x_euler
                break
                
            # Corrector step: Evaluate the velocity at the predicted new state
            t_next_tensor = torch.full((batch_size, 1, 1), t_next.item(), device=validated_device)
            v_pred_next = self(x_euler, t_next_tensor)
            
            # Average the velocities for a more accurate 2nd-order trajectory step
            v_heun = 0.5 * (v_pred + v_pred_next)
            x_t = x_t + v_heun * dt
            
        return x_t
    
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int=2) -> "DragonDiT":
        """ 
        Loads a DragonDiT model from the artifacts found by the provided artifact finder. Ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth
        """
        # Validation: Ensure required files exist
        if not artifact_finder.model_architecture_path:
            _LOGGER.error(f"Model architecture file not found at expected path.")
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            _LOGGER.error(f"Model weights file not found at expected path.")
            raise FileNotFoundError()
        
        model: 'DragonDiT' = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False) # type: ignore
        
        finalized_file = FinalizedFileHandler(artifact_finder.weights_path)
        
        model.load_state_dict(finalized_file.model_state_dict)
        model.eval()

        if verbose >= 2:
            base_msg = f"Model architecture and weights successfully loaded."
            _LOGGER.info(base_msg)
        
        return model

    def get_architecture_config(self) -> dict[str, Any]:
        """Returns the configuration necessary to reconstruct the architecture."""
        return {
            "embed_dim": self.embed_dim,
            "seq_len": self.seq_len,
            "num_heads": self.num_heads,
            "depth": self.depth
        }
