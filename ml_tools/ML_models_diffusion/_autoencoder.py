from typing import Any, Optional
import torch
from torch import nn
import math
import pandas as pd
import json

from ..schema import FeatureSchema
from ..ML_scaler._ML_scaler import DragonScaler
from ..ML_models._base_save_load import _ArchitectureBuilder
from ..ML_utilities._artifact_finder import DragonArtifactFinder

from ..keys._keys import SchemaKeys, ScalerKeys
from .._core import get_logger


_LOGGER = get_logger("DragonAutoencoder")


__all__ = [
    "DragonAutoencoder",
]


# ---- Autoencoder-Style Tokenizer and Embedder for Tabular Data ----
class DragonAutoencoder(_ArchitectureBuilder):
    """
    Bidirectional tokenizer and embedder for Tabular Data.
    Maps raw features to a continuous latent space, and 
    decodes latent tokens back to continuous values and categorical logits.
    
    Requires training.
    
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
        super().__init__()
        self.schema = schema
        self.embedding_dim = embedding_dim
        
        self.scaler: Optional[DragonScaler] = None
        
        self.fourier_sigma = fourier_sigma
        
        self.model_hparams = {
            "schema": schema,
            "embedding_dim": embedding_dim,
            "fourier_sigma": fourier_sigma
        }
        
        # --- 1. Schema Parsing ---
        cat_map = schema.categorical_index_map or {}
        self.categorical_indices = list(cat_map.keys())
        cardinalities = list(cat_map.values())
        
        self.num_features = len(schema.feature_names)
        all_indices = set(range(self.num_features))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        # --- 2. Encoding Layers (Raw -> Tokens) ---
        # Gaussian Fourier Features for numerical data
        if self.numerical_indices:
            half_dim = embedding_dim // 2
            # Frequencies drawn from a Gaussian distribution N(0, fourier_sigma^2)
            self.numerical_frequencies = nn.Parameter(torch.randn(len(self.numerical_indices), half_dim) * self.fourier_sigma)
            # Projection to mix the sine/cosine features and handle any odd embedding dimensions
            self.numerical_projection = nn.Linear(half_dim * 2, embedding_dim)
        
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=embedding_dim) 
            for c in cardinalities
        ])
        
        # Feature Identity Embeddings (Positional)
        self.feature_identity_embeddings = nn.Parameter(torch.randn(1, self.num_features, embedding_dim))
        
        # --- 3. Decoding Layers (Tokens -> Raw) ---
        self.numerical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in self.numerical_indices
        ])
        
        self.categorical_decoders = nn.ModuleList([
            nn.Linear(embedding_dim, c) for c in cardinalities
        ])
        
        # --- 4. Learnable Uncertainty Weighting Parameters --- 
        # source paper: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (https://arxiv.org/abs/1705.07115)
        # Initialize log(\sigma^2) to 0 (variance = 1.0)
        if self.numerical_indices:
            self.log_var_num = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_num = None
            
        if self.categorical_indices:
            self.log_var_cat = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_cat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms raw tabular data into a sequence of embeddings and adds 
        feature identity embeddings.
        """
        # x shape expected to be (batch_size, num_features)
        batch_size = x.shape[0]
        tokens = torch.zeros(batch_size, self.num_features, self.embedding_dim, device=x.device, dtype=torch.float32)
        
        # Encode numerical features
        if self.numerical_indices:
            x_numerical = x[:, self.numerical_indices].float()
            
            # shape: (batch_size, num_features, half_dim)
            angles = x_numerical.unsqueeze(-1) * self.numerical_frequencies * 2 * math.pi
            
            # Concatenate sine and cosine
            # shape: (batch_size, num_features, half_dim * 2)
            fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            
            # Project to embedding_dim
            tokens[:, self.numerical_indices, :] = self.numerical_projection(fourier_features)
            
        # Encode categorical features
        if self.categorical_indices:
            x_categorical = x[:, self.categorical_indices].long()
            categorical_tokens = []
            for i, embed_layer in enumerate(self.categorical_embeddings):
                token = embed_layer(x_categorical[:, i]).unsqueeze(1)
                categorical_tokens.append(token)
            
            tokens[:, self.categorical_indices, :] = torch.cat(categorical_tokens, dim=1)
            
        # Add Feature Identity (Positional) Embeddings
        tokens = tokens + self.feature_identity_embeddings
        
        # shape: (batch_size, num_features, embedding_dim)
        return tokens

    def _decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Differentiable decoding of tokens back to raw numerical values and categorical logits.
        Used for Autoencoder reconstruction training.
        
        tokens shape expected to be (batch_size, num_features, embedding_dim)
        """
        # 1. Remove Feature Identity (Positional) Embeddings
        base_tokens = tokens - self.feature_identity_embeddings
        
        # 2. Decode Numerical Features
        num_reconstructed = []
        if self.numerical_indices:
            for i, num_idx in enumerate(self.numerical_indices):
                token = base_tokens[:, num_idx, :]
                val = self.numerical_decoders[i](token) # shape: (batch_size, 1)
                num_reconstructed.append(val)
            num_reconstructed = torch.cat(num_reconstructed, dim=1)
        else:
            num_reconstructed = torch.empty((tokens.shape[0], 0), device=tokens.device)

        # 3. Decode Categorical Features
        cat_logits = []
        if self.categorical_indices:
            for i, cat_idx in enumerate(self.categorical_indices):
                token = base_tokens[:, cat_idx, :]
                logits = self.categorical_decoders[i](token) # shape: (batch_size, num_classes)
                cat_logits.append(logits)
                
        return num_reconstructed, cat_logits
    
    def set_scaler(self, scaler: DragonScaler) -> None:
        """Sets the internal scaler to be used during approximate decoding."""
        self.scaler = scaler

    def _decode_to_raw_tensors(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Decodes latent tokens back to raw numerical values (inverse scaled if scaler is present) 
        and categorical logits.
        """
        num_reconstructed, cat_logits = self._decode(tokens)
        
        if self.numerical_indices and self.scaler is not None:
            # Create a temporary tensor of the original feature size to use the scaler's inverse_transform safely
            full_features = torch.zeros((tokens.shape[0], self.num_features), device=tokens.device, dtype=torch.float32)
            full_features[:, self.numerical_indices] = num_reconstructed
            
            full_features_reversed = self.scaler.inverse_transform(full_features)
            num_reconstructed = full_features_reversed[:, self.numerical_indices]
            
        return num_reconstructed, cat_logits
    
    @torch.no_grad()
    def approximate_decode(self, tokens: torch.Tensor) -> pd.DataFrame:
        """
        Approximates the continuous tokens from latent space back into raw tabular data.
        Returns a pandas DataFrame where each row represents a human-readable sample.
        
        ⚠️ Set the internal scaler using `set_scaler()` or load it from artifacts before calling this method to ensure numerical features are properly inverse transformed.
        
        Args:
            tokens (torch.Tensor): The latent token representations to decode, shape (batch_size, num_features, embedding_dim).
        
        Returns:
            pd.DataFrame: A DataFrame containing the decoded tabular data.
        """
        self.eval()
        
        _LOGGER.info(f"Decoding a batch of {tokens.shape[0]} samples from latent tokens back to tabular format")
        
        num_reconstructed, cat_logits = self._decode_to_raw_tensors(tokens)
        
        if self.numerical_indices and self.scaler is None:
            _LOGGER.warning("No scaler is set. Numerical features will be returned in the form they were fed into the model.")
        
        decoded_columns = {}
        
        # 1. Map numerical features back
        for i, num_idx in enumerate(self.numerical_indices):
            feat_name = self.schema.feature_names[num_idx]
            decoded_columns[feat_name] = num_reconstructed[:, i].cpu().numpy()

        # 2. Map categorical features back
        for i, cat_idx in enumerate(self.categorical_indices):
            feat_name = self.schema.feature_names[cat_idx]
            logits = cat_logits[i]
            predicted_indices = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Reverse mapping dictionary for this feature
            idx_to_str: dict[int, str] = {}
            if self.schema.categorical_mappings and feat_name in self.schema.categorical_mappings:
                idx_to_str = {v: k for k, v in self.schema.categorical_mappings[feat_name].items()}
            
            decoded_columns[feat_name] = [idx_to_str.get(idx, idx) for idx in predicted_indices]
        
        # Ensure column order matches the original schema
        ordered_data = {feat: decoded_columns[feat] for feat in self.schema.feature_names if feat in decoded_columns}
            
        return pd.DataFrame(ordered_data)
    
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int=2) -> 'DragonAutoencoder':
        """
        Loads a DragonAutoencoder model from the artifacts found by the provided artifact finder. Ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth
            - (Optional but recommended) Scaler .pth with feature scaler.
        """
        # Validation: Ensure required files exist
        if not artifact_finder.model_architecture_path:
            _LOGGER.error(f"Model architecture file not found at expected path.")
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            _LOGGER.error(f"Model weights file not found at expected path.")
            raise FileNotFoundError()
        
        model: 'DragonAutoencoder' = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False) # type: ignore
        
        state_dict = torch.load(artifact_finder.weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        
        if artifact_finder.scaler_path is not None:
            scaler_dict = torch.load(artifact_finder.scaler_path, map_location="cpu")
            if ScalerKeys.FEATURE_SCALER in scaler_dict:
                model.scaler = DragonScaler.load(scaler_dict[ScalerKeys.FEATURE_SCALER], verbose=False)
            else:
                if verbose >= 1:
                    _LOGGER.warning(f"'{ScalerKeys.FEATURE_SCALER}' key not found in the loaded scaler dictionary.")
        else:
            if verbose >= 1:
                _LOGGER.warning(f"No scaler artifact found.")
        
        if verbose >= 2:
            base_msg = f"Model architecture and weights successfully loaded."
            if model.scaler is not None:
                base_msg += f" Feature scaler successfully loaded and set."
            _LOGGER.info(base_msg)
        
        return model

    def get_architecture_config(self) -> dict[str, Any]:
        """Returns the configuration necessary to reconstruct the architecture."""
        return {
            SchemaKeys.SCHEMA_DICT: self.schema.to_dict(),
            "embedding_dim": self.embedding_dim,
            "fourier_sigma": self.fourier_sigma
        }
