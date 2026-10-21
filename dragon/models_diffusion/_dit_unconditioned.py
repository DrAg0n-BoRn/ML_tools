import torch
from torch import nn

from .._core import get_logger

from ._dit_parts import DiTBlockFlash, DiTBlockFlashV2
from ._base_unconditioned_dit import _BaseDragonDiT


_LOGGER = get_logger("Dragon DiT")


__all__ = [
    "DragonDiT",
    "DragonDiTV2"
]


class DragonDiT(_BaseDragonDiT):
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
        super().__init__(embed_dim, seq_len, num_heads, depth)
        
        # Populate the blocks with the V1 implementation
        self.blocks = nn.ModuleList([
            DiTBlockFlash(embed_dim, num_heads) for _ in range(depth)
        ])


class DragonDiTV2(_BaseDragonDiT):
    """
    V2 Unconditioned DiT model.
    
    Upgraded with RMSNorm, SwiGLU Feed-Forward Networks, and QK-Normalization.
    
    References:
    - QK-Normalization: https://ieeexplore.ieee.org/document/9879380
    - QK-Normalization: https://proceedings.mlr.press/v202/dehghani23a.html
    - RMSNorm: https://dl.acm.org/doi/abs/10.5555/3454287.3455397
    - SwiGLU: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, 
                 embed_dim: int,
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        """
        Initializes the Dragon Diffusion Transformer V2.
        
        Args:
            embed_dim (int): The dimensionality of the token embeddings. Must be divisible by num_heads. Must match the embedding dimension of the input tokens.
            seq_len (int): The length of the input sequences (number of features). Must match the seq_len dimension of the input tokens.
            num_heads (int): The number of attention heads in the DiT blocks.
            depth (int): The number of DiT blocks to stack.
        """
        super().__init__(embed_dim, seq_len, num_heads, depth)
        
        # Populate the blocks with the V2 implementation
        self.blocks = nn.ModuleList([
            DiTBlockFlashV2(embed_dim, num_heads) for _ in range(depth)
        ])
