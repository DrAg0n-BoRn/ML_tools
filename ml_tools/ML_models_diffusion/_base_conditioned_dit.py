from typing import Any, Optional
import torch
from torch import nn
from abc import ABC

from ..ML_models._base_save_load import _ArchitectureHandlerMixin
from ..ML_scaler._ML_scaler import DragonScaler
from ..ML_utilities._artifact_finder import DragonArtifactFinder
from ..ML_finalize_handler import FinalizedFileHandler

from ..keys._keys import ScalerKeys
from .._core import get_logger

from ._dit_parts import TimeEmbedding

_LOGGER = get_logger("Dragon DiT Guided")


class _BaseDragonDiTGuided(_ArchitectureHandlerMixin, nn.Module, ABC):
    """
    Base DiT model equipped for Classifier-Free Guidance (CFG).
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
        self.target_scaler: Optional[DragonScaler] = None
        
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        self.time_mlp = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.target_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.null_embedding = nn.Parameter(torch.zeros(1, embed_dim))
        
        # To be populated by subclasses (V1 or V2)
        self.blocks = nn.ModuleList([])
        
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
        
        t_flat = t.view(batch_size, 1)
        c_time = self.time_mlp(t_flat) 
        
        y_embedded = self.target_mlp(y)
        
        if drop_mask is not None:
            null_emb_expanded = self.null_embedding.expand(batch_size, -1)
            c_target = torch.where(drop_mask, null_emb_expanded, y_embedded)
        else:
            c_target = y_embedded
            
        c_fused = c_time + c_target
        x = x_t + self.pos_embed
        
        for block in self.blocks:
            x = block(x, c_fused)
            
        v_pred = self.final_layer(x)
        return v_pred
    
    def set_target_scaler(self, scaler: DragonScaler):
        self.target_scaler = scaler

    def _apply_cfg_rescale(self, v_cond, v_uncond, guidance_scale, cfg_rescale):
        """Helper to apply standard CFG and optional Rescaling."""
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        # UPGRADE: CFG Rescaling to prevent feature variance collapse
        if cfg_rescale > 0.0:
            std_cond = v_cond.std(dim=(1, 2), keepdim=True)
            std_guided = v_guided.std(dim=(1, 2), keepdim=True)
            
            # Rescale guided velocity to match conditional variance
            v_rescaled = v_guided * (std_cond / (std_guided + 1e-5))
            
            # Interpolate based on the rescale factor
            v_guided = cfg_rescale * v_rescaled + (1.0 - cfg_rescale) * v_guided
            
        return v_guided

    @torch.no_grad()
    def generate_sequence(self, 
                          batch_size: int, 
                          target_value: float, 
                          num_steps: int = 25, 
                          guidance_scale: float = 3.0,
                          cfg_rescale: float = 0.0) -> torch.Tensor:
        """
        Generates sequences using Classifier-Free Guidance with optional CFG Rescaling.
        
        Must be decoded back to the original feature space using the tokenizer's decoder.
        
        Args:
            batch_size (int): The number of samples to generate in the batch.
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
            cfg_rescale (float): Optional rescaling factor for the guided velocity to prevent variance collapse.
                - Min: `0.0` Rescaling is completely turned off. This yields standard CFG behavior.
                - Max: `1.0` The guided prediction's variance is strictly forced to match the original conditional variance. 
                - Recommended 0.5 to 0.7 if guidance_scale is high (5.0 or more).
                
        Returns:
            torch.Tensor: The generated token sequences in the embedding space, of shape [batch_size, seq_len, embed_dim].
        """
        self.eval()
        validated_device = next(self.parameters()).device
        
        _LOGGER.info(f"Generating batch: {batch_size} samples. Scale: {guidance_scale}. Rescale: {cfg_rescale}.")
        
        if self.target_scaler is not None:
            target_value_scaled = self.target_scaler.transform(torch.tensor([[target_value]], device=validated_device, dtype=torch.float32)).item()
        else:
            target_value_scaled = target_value

        x_t = torch.randn(batch_size, self.seq_len, self.embed_dim, device=validated_device)
        t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=validated_device)
        
        y_cond = torch.full((batch_size, 1), target_value_scaled, device=validated_device, dtype=torch.float32)
        y_uncond = torch.zeros_like(y_cond) 
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
            
            # Apply CFG Extrapolation & Rescaling
            v_guided = self._apply_cfg_rescale(v_cond, v_uncond, guidance_scale, cfg_rescale)
            x_euler = x_t + v_guided * dt
            
            if i == num_steps - 1:
                x_t = x_euler
                break
            
            # Corrector step
            t_next_tensor = torch.full((batch_size * 2, 1, 1), t_next.item(), device=validated_device)
            x_euler_batched = torch.cat([x_euler, x_euler], dim=0)
            
            v_pred_next_batched = self(x_euler_batched, t_next_tensor, y_batched, mask_batched)
            v_cond_next, v_uncond_next = v_pred_next_batched.chunk(2, dim=0)
            
            v_guided_next = self._apply_cfg_rescale(v_cond_next, v_uncond_next, guidance_scale, cfg_rescale)
            
            v_heun = 0.5 * (v_guided + v_guided_next)
            x_t = x_t + v_heun * dt
        
        return x_t
    
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int=2) -> nn.Module:
        """ 
        Loads a DragonDiTGuided model from the artifacts found by the provided artifact finder. Ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth
            - (Optional but recommended) Scaler .pth with target scaler.
        """
        if not artifact_finder.model_architecture_path:
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            raise FileNotFoundError()
        
        model: '_BaseDragonDiTGuided' = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False) # type: ignore
        
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
    
    def _get_non_decaying_parameters(self) -> set[str]:
        """
        Excludes positional embeddings and the CFG null token from weight decay.
        """
        return {
            "pos_embed", 
            "null_embedding"
        }
        
    def extra_repr(self) -> str:
        """Provides high-level architecture details for print() and PyTorch inspection."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"seq_len={self.seq_len}, "
            f"num_heads={self.num_heads}, "
            f"depth={self.depth}"
        )
        
    def _get_finetune_components(self) -> dict[str, nn.Module]:
        """Maps Guided DiT layers for the DragonFinetuner."""
        return {
            "time_mlp": self.time_mlp,
            "target_mlp": self.target_mlp,
            "blocks": self.blocks,
            "head": self.final_layer,
            "embeddings": nn.ParameterList([self.pos_embed, self.null_embedding])
        }
