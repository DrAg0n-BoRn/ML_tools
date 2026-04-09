from typing import Any, Optional
import torch
from torch import nn

from ..ML_utilities import validate_torch_device
from ..ML_models._base_save_load import _ArchitectureHandlerMixin
from ..ML_scaler._ML_scaler import DragonScaler
from ..ML_utilities._artifact_finder import DragonArtifactFinder
from ..ML_finalize_handler import FinalizedFileHandler

from .._core import get_logger
from ..keys._keys import ScalerKeys

from ._dit_parts import TimeEmbedding, DiTBlockFlash


_LOGGER = get_logger("DragonDiTGuided")


__all__ = [
    "DragonDiTGuided"
]


######################################################
# Classifier-free guidance for the DiT model.
class DragonDiTGuided(_ArchitectureHandlerMixin, nn.Module):
    """
    DiT model equipped for Classifier-Free Guidance (CFG) using Flow Matching for **regression tasks**.
    
    Source Paper: "Classifier-Free Diffusion Guidance" (https://arxiv.org/abs/2207.12598)
    """
    def __init__(self, 
                 embed_dim: int, 
                 seq_len: int,
                 num_heads: int=4, 
                 depth: int=2):
        """ 
        Initializes the Dragon Diffusion Transformer with Classifier-Free Guidance capabilities for **regression tasks**.
        
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
        self.target_scaler: Optional[DragonScaler] = None
        
        # Positional Embeddings for the Diffusion Model
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. Target Embedding MLP (Maps a scalar target to embed_dim)
        ### NOTE: This works for scalar regression targets. For classification or multi-dimensional targets, this would need to be modified accordingly.
        self.target_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. The "Null" Embedding for unconditional generation
        self.null_embedding = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlockFlash(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x_t, t, y, drop_mask=None):
        """
        x_t: [batch_size, seq_len, embed_dim]
        t: [batch_size, 1, 1]
        y: [batch_size, 1] (The regression targets)
        drop_mask: [batch_size, 1] Boolean mask where True means drop the target.
        """
        batch_size = x_t.shape[0]
        
        # Process Time
        t_flat = t.view(batch_size, 1)
        c_time = self.time_mlp(t_flat) 
        
        # Process Target
        y_embedded = self.target_mlp(y)
        
        # Apply Classifier-Free Guidance Dropout
        if drop_mask is not None:
            # Expand null embedding to match batch size
            null_emb_expanded = self.null_embedding.expand(batch_size, -1)
            # Where drop_mask is True, use null_embedding. Otherwise, use y_embedded.
            c_target = torch.where(drop_mask, null_emb_expanded, y_embedded)
        else:
            c_target = y_embedded
            
        # Fuse time and target conditions
        c_fused = c_time + c_target
        
        # Initial input with positional awareness
        x = x_t + self.pos_embed
        
        for block in self.blocks:
            x = block(x, c_fused)
            
        v_pred = self.final_layer(x)
        return v_pred
    
    def set_target_scaler(self, scaler: DragonScaler):
        """Sets the target scaler for the model, used for inference."""
        self.target_scaler = scaler

    @torch.no_grad()
    def generate_sequence(self, 
                          batch_size: int, 
                          device: str, 
                          target_value: float, 
                          num_steps: int = 20, 
                          guidance_scale: float = 3.0) -> torch.Tensor:
        """
        Generates sequences using Classifier-Free Guidance. Must be decoded back to the original feature space using the tokenizer's decoder.
        
        Args:
            batch_size (int): The number of samples to generate in the batch.
            device (str): The device to perform generation on. (e.g., "cuda:0", "mps", "cpu"). Will be validated for compatibility.
            target_value (float): The regression target value to condition on during generation.
                - If a `target_scaler` is set, the value provided will be automatically scaled before being fed into the model.
                - Else, it is assumed that the value has been pre-scaled to match the scale used during training.
            num_steps (int): The number of steps to use in the ODE solver.
            guidance_scale (float): The scale of the guidance to use. Higher values result in stronger guidance.
                - `0.0` corresponds to unconditional generation (ignoring the target),
                - `1.0` corresponds to standard conditional generation,
                - `1.5` to `4.0` are common values for stronger guidance.
                - `5.0` to `7.0` strong guidance (can lead to better target adherence but risks sample quality if too high).
                - `>7.0` adversarial collapse (extremely strong guidance that can produce unrealistic samples).
                
        Returns:
            torch.Tensor: The generated token sequences in the embedding space, of shape [batch_size, seq_len, embed_dim]. Must be decoded back to the original feature space using the tokenizer's decoder
        """
        self.eval()
        
        validated_device = validate_torch_device(device)
        
        _LOGGER.info(f"Generating a batch of {batch_size} samples with {self.seq_len} features using Classifier-Free Guidance with guidance scale {guidance_scale} and {num_steps} steps.")
        
        # Scale the target value if a scaler is set
        if self.target_scaler is not None:
            target_value_scaled = self.target_scaler.transform(torch.tensor([[target_value]], device=validated_device)).item()
            _LOGGER.info(f"Target value {target_value} scaled to {target_value_scaled} using the provided target scaler.")
        else:
            target_value_scaled = target_value
            _LOGGER.info(f"No target scaler set. Assuming value {target_value_scaled} is expected by the model.")
            
        x_t = torch.randn(batch_size, self.seq_len, self.embed_dim, device=validated_device)
        t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=validated_device)
        
        # Prepare condition and masks for the batched CFG pass
        # We duplicate inputs to run conditional and unconditional passes simultaneously
        y_cond = torch.full((batch_size, 1), target_value_scaled, device=validated_device, dtype=torch.float32)
        y_uncond = torch.zeros_like(y_cond) # Value doesn't matter, it gets masked
        y_batched = torch.cat([y_cond, y_uncond], dim=0)
        
        mask_cond = torch.zeros((batch_size, 1), device=validated_device, dtype=torch.bool)
        mask_uncond = torch.ones((batch_size, 1), device=validated_device, dtype=torch.bool)
        mask_batched = torch.cat([mask_cond, mask_uncond], dim=0)
        
        for i in range(num_steps):
            t_val = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_val
            
            t_tensor = torch.full((batch_size * 2, 1, 1), t_val.item(), device=validated_device)
            x_t_batched = torch.cat([x_t, x_t], dim=0)
            
            # Predictor step
            v_pred_batched = self(x_t_batched, t_tensor, y_batched, mask_batched)
            v_cond, v_uncond = v_pred_batched.chunk(2, dim=0)
            
            # CFG Extrapolation Formula
            v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
            x_euler = x_t + v_guided * dt
            
            if i == num_steps - 1:
                x_t = x_euler
                break
            
            # Corrector step
            t_next_tensor = torch.full((batch_size * 2, 1, 1), t_next.item(), device=validated_device)
            x_euler_batched = torch.cat([x_euler, x_euler], dim=0)
            
            v_pred_next_batched = self(x_euler_batched, t_next_tensor, y_batched, mask_batched)
            v_cond_next, v_uncond_next = v_pred_next_batched.chunk(2, dim=0)
            
            v_guided_next = v_uncond_next + guidance_scale * (v_cond_next - v_uncond_next)
            v_heun = 0.5 * (v_guided + v_guided_next)
            
            x_t = x_t + v_heun * dt
        
        return x_t
    
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int=2) -> "DragonDiTGuided":
        """ 
        Loads a DragonDiTGuided model from the artifacts found by the provided artifact finder. Ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth
            - (Optional but recommended) Scaler .pth with target scaler.
        """
        # Validation: Ensure required files exist
        if not artifact_finder.model_architecture_path:
            _LOGGER.error(f"Model architecture file not found at expected path.")
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            _LOGGER.error(f"Model weights file not found at expected path.")
            raise FileNotFoundError()
        
        model: 'DragonDiTGuided' = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False) # type: ignore
        
        finalized_file = FinalizedFileHandler(artifact_finder.weights_path)
        
        model.load_state_dict(finalized_file.model_state_dict)
        model.eval()
        
        if artifact_finder.scaler_path is not None:
            scaler_dict = torch.load(artifact_finder.scaler_path, map_location="cpu")
            if ScalerKeys.TARGET_SCALER in scaler_dict:
                model.target_scaler = DragonScaler.load(scaler_dict[ScalerKeys.TARGET_SCALER], verbose=False)
            else:
                if verbose >= 1:
                    _LOGGER.warning(f"'{ScalerKeys.TARGET_SCALER}' key not found in the loaded scaler dictionary.")
        else:
            if verbose >= 1:
                _LOGGER.warning(f"No scaler artifact found.")
                
        if verbose >= 2:
            base_msg = f"Model architecture and weights successfully loaded."
            if model.target_scaler is not None:
                base_msg += f" Target scaler successfully loaded and set."
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
