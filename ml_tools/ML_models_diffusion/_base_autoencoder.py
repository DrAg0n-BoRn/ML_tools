from abc import ABC, abstractmethod
from typing import Optional
import torch
import pandas as pd
from torch import nn

from ..schema import FeatureSchema
from ..ML_scaler._ML_scaler import DragonScaler
from ..ML_models._base_save_load import _ArchitectureBuilder
from ..ML_utilities._artifact_finder import DragonArtifactFinder
from ..ML_finalize_handler import FinalizedFileHandler
from ..keys._keys import ScalerKeys
from .._core import get_logger


_LOGGER = get_logger("Dragon Autoencoder")


class _BaseAutoencoder(_ArchitectureBuilder, ABC):
    """
    Abstract base class for Dragon Autoencoders.
    Handles schema parsing, decoding utilities, and standardized artifact loading.
    """
    def __init__(self, schema: FeatureSchema, embedding_dim: int):
        super().__init__()
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.scaler: Optional[DragonScaler] = None
        
        # --- 1. Schema Parsing (Centralized) ---
        cat_map = schema.categorical_index_map or {}
        self.categorical_indices = list(cat_map.keys())
        self.cardinalities = list(cat_map.values())
        
        self.num_features = len(schema.feature_names)
        all_indices = set(range(self.num_features))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))

    @abstractmethod
    def _decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Differentiable decoding to be implemented by specific architectures."""
        pass

    # --- 2. Shared Inference Utilities ---
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
            full_features = torch.zeros((tokens.shape[0], self.num_features), device=tokens.device, dtype=torch.float32)
            full_features[:, self.numerical_indices] = num_reconstructed
            
            full_features_reversed = self.scaler.inverse_transform(full_features)
            num_reconstructed = full_features_reversed[:, self.numerical_indices]
            
        return num_reconstructed, cat_logits
    
    @torch.no_grad()
    def approximate_decode(self, tokens: torch.Tensor) -> pd.DataFrame:
        """
        Approximates continuous tokens from latent space back into raw tabular data.
        Returns a pandas DataFrame where each row represents a human-readable sample.
        """
        self.eval()
        _LOGGER.info(f"Decoding a batch of {tokens.shape[0]} samples from latent tokens back to tabular format")
        
        num_reconstructed, cat_logits = self._decode_to_raw_tensors(tokens)
        
        if self.numerical_indices and self.scaler is None:
            _LOGGER.warning("No scaler is set. Numerical features will be returned in the form they were fed into the model.")
        
        decoded_columns = {}
        
        for i, num_idx in enumerate(self.numerical_indices):
            feat_name = self.schema.feature_names[num_idx]
            decoded_columns[feat_name] = num_reconstructed[:, i].cpu().numpy()

        for i, cat_idx in enumerate(self.categorical_indices):
            feat_name = self.schema.feature_names[cat_idx]
            predicted_indices = torch.argmax(cat_logits[i], dim=-1).cpu().numpy()
            
            idx_to_str: dict[int, str] = {}
            if self.schema.categorical_mappings and feat_name in self.schema.categorical_mappings:
                idx_to_str = {v: k for k, v in self.schema.categorical_mappings[feat_name].items()}
            
            decoded_columns[feat_name] = [idx_to_str.get(idx, idx) for idx in predicted_indices]
        
        ordered_data = {feat: decoded_columns[feat] for feat in self.schema.feature_names if feat in decoded_columns}
        return pd.DataFrame(ordered_data)

    # --- 3. Shared Artifact Loading ---
    @classmethod
    def from_artifact_finder(cls, artifact_finder: DragonArtifactFinder, verbose: int = 2) -> nn.Module:
        """
        Loads a DragonAutoencoder model (V1 or V2) from artifacts, ready for inference.
        
        Expects the artifact finder to locate the following files:
            - Model architecture JSON
            - Model weights .pth (Finalized-file)
            - (Optional but recommended) Scaler .pth with feature scaler.
        """
        if not artifact_finder.model_architecture_path:
            _LOGGER.error(f"Model architecture file not found at expected path.")
            raise FileNotFoundError()
        if not artifact_finder.weights_path:
            _LOGGER.error(f"Model weights file not found at expected path.")
            raise FileNotFoundError()
        
        model = cls.load_architecture(artifact_finder.model_architecture_path, verbose=False)
        
        finalized_file = FinalizedFileHandler(artifact_finder.weights_path)
        model.load_state_dict(finalized_file.model_state_dict)
        model.eval()
        
        if artifact_finder.scaler_path is not None:
            scaler_dict = torch.load(artifact_finder.scaler_path, map_location="cpu")
            if ScalerKeys.FEATURE_SCALER in scaler_dict:
                model.scaler = DragonScaler.load(scaler_dict[ScalerKeys.FEATURE_SCALER], verbose=False) # type: ignore
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
