from typing import Any
import torch
from torch import nn
import math

from ..schema import FeatureSchema
from ..keys._keys import SchemaKeys
from ._base_autoencoder import _BaseAutoencoder


__all__ = ["DragonAutoencoder"]


class DragonAutoencoder(_BaseAutoencoder):
    """
    Bidirectional tokenizer and embedder for Tabular Data.
    
    Maps raw features to a continuous latent space, and 
    decodes latent tokens back to continuous values and categorical logits.
    
    Key Features:
    - Uses Gaussian Fourier Features for numerical data to capture complex relationships.
    - Uses learnable embedding layers for categorical features.
    - Adds feature identity (positional) embeddings to preserve feature-specific information.
    - Includes learnable uncertainty weighting parameters (https://arxiv.org/abs/1705.07115).
    """
    def __init__(self, 
                 schema: FeatureSchema, 
                 embedding_dim: int,
                 fourier_sigma: float = 1.0):
        """
        Initializes the DragonAutoencoder for tabular data.
        
        Args:
            schema (FeatureSchema): The schema describing the features and their types.
            embedding_dim (int): The dimensionality of the token embeddings.
                - Recommended to be a multiple of 4 for better performance with Fourier features, but not strictly required.
                - Common choices for less than 100 features: 32, 64, 128. For larger feature sets, consider 256 or 512.
            fourier_sigma (float): The standard deviation for the Gaussian distribution from which Fourier frequencies are drawn.
                - Recommended range: 0.5 to 2.0.
                - Higher bandwidths provide the high-frequency resolution required to map minute continuous differences into distinct latent representations, but can cause the latent space to become chaotic and overly sensitive.
        """
        # 1. Initialize Base (Handles schema, indices, cardinalities)
        super().__init__(schema, embedding_dim)
        
        self.fourier_sigma = fourier_sigma
        
        self.model_hparams = {
            "schema": schema,
            "embedding_dim": embedding_dim,
            "fourier_sigma": fourier_sigma
        }
        
        # 2. Encoding Layers (Raw -> Tokens)
        if self.numerical_indices:
            half_dim = embedding_dim // 2
            self.numerical_frequencies = nn.Parameter(torch.randn(len(self.numerical_indices), half_dim) * self.fourier_sigma)
            self.numerical_projection = nn.Linear(half_dim * 2, embedding_dim)
        
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=embedding_dim) 
            for c in self.cardinalities
        ])
        
        self.feature_identity_embeddings = nn.Parameter(torch.randn(1, self.num_features, embedding_dim))
        
        # 3. Decoding Layers (Tokens -> Raw)
        self.numerical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in self.numerical_indices
        ])
        self.categorical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, c) for c in self.cardinalities
        ])
        
        # 4. Learnable Uncertainty Weighting Parameters
        if self.numerical_indices:
            self.log_var_num = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_num = None
            
        if self.categorical_indices:
            self.log_var_cat = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_cat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = torch.zeros(batch_size, self.num_features, self.embedding_dim, device=x.device, dtype=torch.float32)
        
        if self.numerical_indices:
            x_numerical = x[:, self.numerical_indices].float()
            angles = x_numerical.unsqueeze(-1) * self.numerical_frequencies * 2 * math.pi
            fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            tokens[:, self.numerical_indices, :] = self.numerical_projection(fourier_features)
            
        if self.categorical_indices:
            x_categorical = x[:, self.categorical_indices].long()
            categorical_tokens = []
            for i, embed_layer in enumerate(self.categorical_embeddings):
                token = embed_layer(x_categorical[:, i]).unsqueeze(1)
                categorical_tokens.append(token)
            tokens[:, self.categorical_indices, :] = torch.cat(categorical_tokens, dim=1)
            
        tokens = tokens + self.feature_identity_embeddings
        return tokens

    def _decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
            "fourier_sigma": self.fourier_sigma
        }
        
    def _get_non_decaying_parameters(self) -> set[str]:
        """
        Excludes Fourier frequencies and 3D feature identity (positional) embeddings from weight decay.
        """
        return {
            "numerical_frequencies",
            "feature_identity_embeddings"
        }
