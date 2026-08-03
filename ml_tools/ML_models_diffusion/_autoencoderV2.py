from typing import Any, Union, Literal
import torch
from torch import nn
import math

from ..schema import FeatureSchema
from ..keys._keys import SchemaKeys
from ._base_autoencoder import _BaseAutoencoder


__all__ = ["DragonAutoencoderV2"]


class DragonAutoencoderV2(_BaseAutoencoder):
    """
    V2 Autoencoder for Tabular Data incorporating Flow Matching best practices.
    
    Core Upgrades from V1:
    1. Cross-Feature Transformer Encoder/Decoder mixing for modeling inter-feature dependencies.
    2. Variational Autoencoder (VAE) latent regularization for bounded, Gaussian-like latent spaces.
    3. Pluggable numerical embeddings (Fourier or Piecewise Linear/Binning).
    
    Sources:
    - FT-Transformer: https://arxiv.org/abs/2106.11959
    - TabTransformer: https://arxiv.org/abs/2012.06678
    - PLE for Tabular Data: https://arxiv.org/abs/2203.05556
    - Standard VAE: https://arxiv.org/abs/1312.6114
    - Beta-VAE: ICLR 2017: https://openreview.net/forum?id=Sy2fzU9gl
    - Multi-task uncertainty weighting: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, 
                 schema: FeatureSchema, 
                 embedding_dim: int,
                 numerical_embedding_type: Literal['fourier', 'ple'] = 'ple', 
                 fourier_sigma: float = 1.0,
                 ple_bins: int = 100,
                 transformer_depth: int = 2,
                 transformer_heads: int = 4):
        """
        Initializes the DragonAutoencoderV2.
        
        Args:
            schema (FeatureSchema): The schema describing the features and types.
            embedding_dim (int): Dimensionality of tokens.
            numerical_embedding_type (Literal['fourier', 'ple']): The type of numerical embedding to use, Fourier or Piecewise Linear Embeddings.
            fourier_sigma (float): Bandwidth for Fourier features.
            ple_bins (int): Number of bins to use if using PLE.
            transformer_depth (int): Depth of the cross-feature transformer layers.
            transformer_heads (int): Number of attention heads.
        """
        
        # 1. Initialize Base (Handles schema, indices, cardinalities)
        super().__init__(schema, embedding_dim)
        
        self.model_hparams = {
            "schema": schema,
            "embedding_dim": embedding_dim,
            "numerical_embedding_type": numerical_embedding_type,
            "fourier_sigma": fourier_sigma,
            "ple_bins": ple_bins,
            "transformer_depth": transformer_depth,
            "transformer_heads": transformer_heads
        }
        
        self.numerical_embedding_type = numerical_embedding_type.lower()
        self.fourier_sigma = fourier_sigma
        self.ple_bins = ple_bins
        self.transformer_depth = transformer_depth
        self.transformer_heads = transformer_heads
        
        # 2. Initial Encoders (Raw -> Token Embeddings)
        if self.numerical_indices:
            if self.numerical_embedding_type == 'fourier':
                half_dim = embedding_dim // 2
                self.numerical_frequencies = nn.Parameter(torch.randn(len(self.numerical_indices), half_dim) * self.fourier_sigma)
                self.numerical_projection = nn.Linear(half_dim * 2, embedding_dim)
            elif self.numerical_embedding_type == 'ple':
                self.ple_embeddings = nn.Parameter(torch.randn(len(self.numerical_indices), ple_bins, embedding_dim))
            else:
                raise ValueError(f"Unknown numerical_embedding_type: {numerical_embedding_type}")
                
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=embedding_dim) 
            for c in self.cardinalities
        ])
        
        self.feature_identity_embeddings = nn.Parameter(torch.randn(1, self.num_features, embedding_dim))
        
        # 3. Cross-Feature Transformer (Encoder)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=transformer_heads, 
            dim_feedforward=embedding_dim * 4, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_depth)
        
        # 4. VAE Projections
        self.to_mu = nn.Linear(embedding_dim, embedding_dim)
        self.to_logvar = nn.Linear(embedding_dim, embedding_dim)
            
        # 5. Cross-Feature Transformer (Decoder)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=transformer_heads, 
            dim_feedforward=embedding_dim * 4, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(dec_layer, num_layers=transformer_depth)

        # 6. Final Decoders (Tokens -> Raw)
        self.numerical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in self.numerical_indices
        ])
        self.categorical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, c) for c in self.cardinalities
        ])

        # 7. Uncertainty Weighting
        if self.numerical_indices:
            self.log_var_num = nn.Parameter(torch.zeros(1))
        if self.categorical_indices:
            self.log_var_cat = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size = x.shape[0]
        tokens = torch.zeros(batch_size, self.num_features, self.embedding_dim, device=x.device, dtype=torch.float32)
        
        if self.numerical_indices:
            x_num = x[:, self.numerical_indices].float()
            if self.numerical_embedding_type == 'fourier':
                angles = x_num.unsqueeze(-1) * self.numerical_frequencies * 2 * math.pi
                fourier_feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
                tokens[:, self.numerical_indices, :] = self.numerical_projection(fourier_feats)
            elif self.numerical_embedding_type == 'ple':
                scaled = ((x_num + 3.0) / 6.0) * (self.ple_bins - 1)
                bin_indices = torch.clamp(scaled.long(), min=0, max=self.ple_bins - 1)
                for i, num_idx in enumerate(self.numerical_indices):
                    tokens[:, num_idx, :] = self.ple_embeddings[i, bin_indices[:, i], :]

        if self.categorical_indices:
            x_cat = x[:, self.categorical_indices].long()
            for i, embed_layer in enumerate(self.categorical_embeddings):
                tokens[:, self.categorical_indices[i], :] = embed_layer(x_cat[:, i])
                
        tokens = tokens + self.feature_identity_embeddings
        tokens = self.transformer_encoder(tokens)
            
        mu = self.to_mu(tokens)
        logvar = self.to_logvar(tokens)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return mu 

    def _decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = self.transformer_decoder(tokens)
        base_tokens = tokens - self.feature_identity_embeddings
        
        num_reconstructed = []
        if self.numerical_indices:
            for i, num_idx in enumerate(self.numerical_indices):
                val = self.numerical_decoders[i](base_tokens[:, num_idx, :])
                num_reconstructed.append(val)
            num_reconstructed = torch.cat(num_reconstructed, dim=1)
        else:
            num_reconstructed = torch.empty((tokens.shape[0], 0), device=tokens.device)

        cat_logits = []
        if self.categorical_indices:
            for i, cat_idx in enumerate(self.categorical_indices):
                logits = self.categorical_decoders[i](base_tokens[:, cat_idx, :])
                cat_logits.append(logits)
                
        return num_reconstructed, cat_logits

    def get_architecture_config(self) -> dict[str, Any]:
        """Returns a dictionary containing the architecture configuration of the autoencoder."""
        return {
            SchemaKeys.SCHEMA_DICT: self.schema.to_dict(), 
            "embedding_dim": self.embedding_dim,
            "numerical_embedding_type": self.numerical_embedding_type,
            "fourier_sigma": self.fourier_sigma,
            "ple_bins": self.ple_bins,
            "transformer_depth": self.transformer_depth,
            "transformer_heads": self.transformer_heads
        }
        
    def _get_non_decaying_parameters(self) -> set[str]:
        """
        Excludes Fourier frequencies and 3D feature identity (positional) embeddings from weight decay.
        """
        return {
            "numerical_frequencies",
            "feature_identity_embeddings"
        }
