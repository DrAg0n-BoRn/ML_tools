import torch
from torch import nn
import torch.nn.functional as F
import math


# Embedding for time using sinusoidal functions, similar to the approach used in transformers
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time shape: [batch_size, 1]
        device = time.device
        # sine function for the first half of the dimensions, cosine for the second half
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        #  Apply the time to the embeddings
        # time shape: [batch_size, 1], embeddings shape: [half_dim]
        embeddings = time * embeddings[None, :] # None adds a new dimension for broadcasting to [batch_size, half_dim]
        # Project to the full dimension by concatenating sine and cosine
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # embeddings shape: [batch_size, dim]
        return embeddings
    


# The standard DiT block uses the multi-head attention, it is no longer used due to PyTorch's native scaled dot-product attention (SDPA) is now available and optimized for modern hardware, 
# including support for FlashAttention when available. 
# The DiTBlockFlash class implements the same functionality as DiTBlock but uses explicit linear layers to generate Q, K, V and calls F.scaled_dot_product_attention directly
class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        
        nn.init.zeros_(self.adaLN_modulation[-1].weight)  # type: ignore
        nn.init.zeros_(self.adaLN_modulation[-1].bias)  # type: ignore

    def forward(self, x, time_conditioning):
        (
            shift_attention, 
            scale_attention, 
            gate_attention, 
            shift_feed_forward, 
            scale_feed_forward, 
            gate_feed_forward
        ) = self.adaLN_modulation(time_conditioning).chunk(6, dim=1)
        
        norm_x = self.norm1(x) * (1 + scale_attention.unsqueeze(1)) + shift_attention.unsqueeze(1)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + gate_attention.unsqueeze(1) * attn_out
        
        norm_x = self.norm2(x) * (1 + scale_feed_forward.unsqueeze(1)) + shift_feed_forward.unsqueeze(1)
        ff_out = self.feed_forward(norm_x)
        x = x + gate_feed_forward.unsqueeze(1) * ff_out
        
        return x


class DiTBlockFlash(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        
        nn.init.zeros_(self.adaLN_modulation[-1].weight)  # type: ignore
        nn.init.zeros_(self.adaLN_modulation[-1].bias) # type: ignore

    def forward(self, x, time_conditioning):
        (
            shift_attention, 
            scale_attention, 
            gate_attention, 
            shift_feed_forward, 
            scale_feed_forward, 
            gate_feed_forward
        ) = self.adaLN_modulation(time_conditioning).chunk(6, dim=1)
        
        norm_x = self.norm1(x) * (1 + scale_attention.unsqueeze(1)) + shift_attention.unsqueeze(1)
        
        B, L, D = norm_x.shape
        
        # Generate Q, K, V and reshape for native SDPA: [Batch, Heads, SeqLen, HeadDim]
        qkv = self.qkv(norm_x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # PyTorch Native Scaled Dot-Product Attention
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back to [Batch, SeqLen, Dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.proj(attn_out)
        
        x = x + gate_attention.unsqueeze(1) * attn_out
        
        norm_x = self.norm2(x) * (1 + scale_feed_forward.unsqueeze(1)) + shift_feed_forward.unsqueeze(1)
        ff_out = self.feed_forward(norm_x)
        x = x + gate_feed_forward.unsqueeze(1) * ff_out
        
        return x

