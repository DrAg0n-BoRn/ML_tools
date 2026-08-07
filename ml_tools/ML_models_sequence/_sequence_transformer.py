import math
import torch
from torch import nn
from typing import Literal, Union

from ..schema import FeatureSchema
from ..ML_models._base_save_load import _ArchitectureBuilder

from .._core import get_logger
from ..keys._keys import MLTaskKeys, SchemaKeys


_LOGGER = get_logger("Sequence Transformer")


__all__ = [
    "DragonSequenceTransformer"
]

# References:
# Pre-layer normalization: On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) https://proceedings.mlr.press/v119/xiong20b
# ALiBi: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2022) https://arxiv.org/abs/2108.12409
# Gelu activation: "Gaussian Error Linear Units (GELUs)" (Hendrycks et al., 2020) https://arxiv.org/abs/1606.08415


def get_alibi_mask(seq_len: int, 
                   nhead: int, 
                   device: torch.device, 
                   is_causal: bool = False) -> torch.Tensor:
    """
    Generates an ALiBi (Attention with Linear Biases) mask.
    Replaces standard absolute positional encodings to improve length generalization.
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(nhead))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [base ** i for i in range(1, closest_power_of_2 + 1)]
    
    if closest_power_of_2 < nhead:
        base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        slopes.extend([base ** i for i in range(1, 2 * (nhead - closest_power_of_2) + 1, 2)])
        
    slopes = torch.tensor(slopes, device=device).view(nhead, 1, 1)
    
    seq_positions = torch.arange(seq_len, device=device)
    
    if is_causal:
        distances = seq_positions.unsqueeze(0) - seq_positions.unsqueeze(1)
        alibi = slopes * distances
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        mask = alibi + causal_mask.unsqueeze(0)
    else:
        distances = torch.abs(seq_positions.unsqueeze(0) - seq_positions.unsqueeze(1))
        mask = slopes * -distances
        
    return mask


# Replaced by ALiBi in the Transformer architecture for better length generalization.
class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :] # type: ignore # self.pe is a buffer
        return self.dropout(x)


class DragonSequenceTransformer(_ArchitectureBuilder):
    """
    A Transformer-based network for multivariate sequence prediction tasks.
    
    Dynamically generates separate prediction heads for continuous and categorical 
    targets based on the provided FeatureSchema.
    
    Utilizes ALiBi (Attention with Linear Biases) for positional encoding.
    """
    def __init__(self, 
                 schema: FeatureSchema,
                 targets: list[str],
                 prediction_mode: Union[Literal["autoregressive-sequence-to-sequence", 
                                                "autoregressive-sequence-to-value", 
                                                "exogenous-sequence-to-sequence", 
                                                "exogenous-sequence-to-value"], str],
                 sequence_length: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Initializes the multi-head Transformer for multivariate sequence prediction.
    
        This architecture dynamically constructs distinct output heads for each target based on 
        its data type (continuous vs. categorical) as defined in the provided FeatureSchema. 
        
        Args:
            schema (FeatureSchema): The definitive schema containing feature names, data types, 
                and categorical cardinality configurations.
            targets (list[str]): A list of column names indicating the target variables to predict.
            prediction_mode (str): Determines the temporal structure of the outputs.
                - 'autoregressive': Includes target variables in the input sequence for rolling predictions.
                - 'exogenous': Excludes target variables from the input sequence, relying solely on exogenous features.
                - 'sequence-to-sequence': Predicts a full sequence matching the input time steps.
                - 'sequence-to-value': Predicts single values corresponding to the final time step.
            sequence_length (int): The exact length of the sliding windows generated by the dataset. 
                Used to optimally allocate the pre-computed positional encoding buffer.
            d_model (int): The number of expected features in the encoder inputs (embedding dimension). 
            nhead (int): The number of heads in the multi-head attention models. Should evenly divide `d_model`.
            num_layers (int): The number of sub-encoder layers in the Transformer. 
            dim_feedforward (int): The dimension of the feedforward network model within the encoder. 
            dropout (float): The dropout probability applied to the encoder layers and positional encodings.
        
        ### Note:
        For autoregressive sequence-to-sequence tasks, causal masking is applied to prevent information leakage.
        """
        super().__init__()

        # --- 1. Validation ---
        if prediction_mode not in MLTaskKeys.ALL_SEQUENCE_TASKS:
            _LOGGER.error(f"Unrecognized prediction mode: '{prediction_mode}'.")
            raise ValueError()
            
        if not targets:
            _LOGGER.error("The 'targets' list cannot be empty.")
            raise ValueError()

        # --- 2. Save configuration for _ArchitectureBuilder ---
        self.model_hparams = {
            SchemaKeys.SCHEMA_DICT: schema.to_dict(),
            'targets': targets,
            'prediction_mode': prediction_mode,
            'sequence_length': sequence_length,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
        }
        
        self.prediction_mode = prediction_mode
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Adjust input features for exogenous modes
        if prediction_mode in [MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE, MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE]:
            self.in_features = schema.number_of_features - len(targets)
        else: # autoregressive modes
            self.in_features = schema.number_of_features

        # --- 3. Transformer Backbone ---
        self.input_projection = nn.Linear(self.in_features, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,  # Crucial for (batch, seq, feature) inputs
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # --- 4. Multi-Head Outputs ---
        self.target_heads = nn.ModuleDict()
        
        for target in targets:
            if target in schema.categorical_feature_names:
                target_idx = schema.feature_names.index(target)
                if schema.categorical_index_map and target_idx in schema.categorical_index_map:
                    cardinality = schema.categorical_index_map[target_idx]
                else:
                    _LOGGER.warning(f"Cardinality for categorical target '{target}' not found. Defaulting to 2.")
                    cardinality = 2
                
                self.target_heads[target] = nn.Linear(d_model, cardinality)
            else:
                self.target_heads[target] = nn.Linear(d_model, 1)
                
        # Causal model
        # Causal masking applies an upper-triangular matrix filled with negative infinity to the attention scores. 
        # This physically blocks the model to "look ahead" or "time travel."
        # Only autoregressive-sequence-to-sequence tasks require causal masking. Exogenous tasks do not.
        self._is_causal = (prediction_mode == MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE)

    @property
    def _is_seq_to_val(self) -> bool:
        return self.prediction_mode in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE, MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE]

    @property
    def _is_seq_to_seq(self) -> bool:
        return self.prediction_mode in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping each target name to its prediction tensor.
        """
        # --- Handle Input Shape ---
        if x.ndim == 2:
            x = x.unsqueeze(-1)
            
        batch_size, seq_len, _ = x.shape

        # 1. Project input (Positional encoding removed in favor of ALiBi mask)
        x = self.input_projection(x)

        # 2. Generate ALiBi mask (handles both position biases and causal masking)
        alibi_mask = get_alibi_mask(seq_len, 
                                    self.model_hparams['nhead'], 
                                    x.device, 
                                    self._is_causal)
        # PyTorch 3D mask shape requirement: (batch_size * nhead, seq_len, seq_len)
        alibi_mask = alibi_mask.repeat(batch_size, 1, 1)

        # 3. Transformer Encode
        # shape: (batch_size, seq_len, d_model)
        memory = self.transformer_encoder(x, mask=alibi_mask)
        
        # 4. Route through Multi-Heads
        predictions = {}
        
        if self._is_seq_to_val:
            last_step = memory[:, -1, :]
            
            for target_name, head in self.target_heads.items():
                out = head(last_step)
                predictions[target_name] = out.squeeze(-1) if out.shape[-1] == 1 else out
                
        elif self._is_seq_to_seq:
            for target_name, head in self.target_heads.items():
                out = head(memory)
                predictions[target_name] = out.squeeze(-1) if out.shape[-1] == 1 else out

        return predictions
    
    def get_architecture_config(self) -> dict:
        """Returns the configuration of the model for serialization."""
        return self.model_hparams
    
    def extra_repr(self) -> str:
        """Provides high-level architecture details for print() and PyTorch inspection."""
        return (
            f"prediction_mode='{self.prediction_mode}', "
            f"targets={len(self.targets)}, "
            f"sequence_length={self.sequence_length}, "
            f"in_features={self.in_features}, "
            f"d_model={self.model_hparams['d_model']}, "
            f"nhead={self.model_hparams['nhead']}, "
            f"num_layers={self.model_hparams['num_layers']}, "
            f"dim_feedforward={self.model_hparams['dim_feedforward']}, "
            f"dropout={self.model_hparams['dropout']}"
        )

    def _get_finetune_components(self) -> dict[str, nn.Module]:
        """Maps Transformer sequence model layers for the DragonFinetuner."""
        return {
            "embeddings": self.input_projection,
            "encoder": self.transformer_encoder,
            "heads": self.target_heads
        }
