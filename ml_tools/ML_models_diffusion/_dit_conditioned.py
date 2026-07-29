import torch
from torch import nn

from .._core import get_logger

from ._dit_parts import DiTBlockFlash, DiTBlockFlashV2
from ._base_conditioned_dit import _BaseDragonDiTGuided


_LOGGER = get_logger("Dragon DiTGuided")


__all__ = [
    "DragonDiTGuided",
    "DragonDiTGuidedV2"
]


class DragonDiTGuided(_BaseDragonDiTGuided):
    """
    DiT model equipped for Classifier-Free Guidance (CFG) using Flow Matching.
        
    Source Paper: "Classifier-Free Diffusion Guidance" (https://arxiv.org/abs/2207.12598)
    """
    def __init__(self, 
                 embed_dim: int, 
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        """ 
        Initializes the Dragon Diffusion Transformer with Classifier-Free Guidance capabilities.
        
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


class DragonDiTGuidedV2(_BaseDragonDiTGuided):
    """
    V2 Guided DiT model equipped for Classifier-Free Guidance.
    
    Upgraded with RMSNorm, SwiGLU Feed-Forward Networks, QK-Normalization, and CFG rescaling.
    
    References:
    - QK-Normalization: https://ieeexplore.ieee.org/document/9879380
    - QK-Normalization: https://proceedings.mlr.press/v202/dehghani23a.html
    - RMSNorm: https://dl.acm.org/doi/abs/10.5555/3454287.3455397
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - CFG Rescaling: https://ieeexplore.ieee.org/document/10484327
    """
    def __init__(self, 
                 embed_dim: int, 
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        """
        Initializes the Dragon Diffusion Transformer V2 with Classifier-Free Guidance capabilities.
        
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
