import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Literal

from ..schema import FeatureSchema

from .._core import get_logger
from ..keys._keys import MLTaskKeys, DatasetKeys

from ._base_sequence_inference import _BaseSequenceInferenceHandler


_LOGGER = get_logger("Exogenous Inference")


__all__ = [
    "DragonSequenceExogenousHandler"
]


class DragonSequenceExogenousHandler(_BaseSequenceInferenceHandler):
    """
    Handles inference and exogenous forecasting for sequence models.
    Targets are excluded from inputs; the rolling window steps forward 
    utilizing strictly future exogenous data.
    
    This handler automatically scales inputs and de-scales outputs.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 schema: Union[str, Path, FeatureSchema],
                 scaler: Optional[Union[str, Path]],
                 target_types: Optional[dict[str, str]] = None,
                 task: Optional[Literal["exogenous-sequence-to-sequence", "exogenous-sequence-to-value"]]=None,
                 device: str = 'cpu'):
        """
        Initializes the handler for exogenous sequence tasks.

        Args:
            model (nn.Module): A trained model architecture that must output a dictionary mapping target names to their respective prediction tensors.
            state_dict (str | Path): Path to the saved .pth model state_dict file or a FinalizedFile format.
            scaler (str | Path): File path to a saved DragonScaler state. This is required to correctly scale continuous inputs and de-scale predictions.
            schema (str | Path | FeatureSchema): The feature schema or a path to a directory containing the schema.
            target_types (dict[str, str], optional): A dictionary mapping target names to their types ('categorical', 'continuous').
            task (str, optional): The type of sequence task. If None, detected from file if a FinalizedFile is provided.
            device (str): The device to run inference on ('cpu', 'cuda', 'mps'). 
        """
        super().__init__(model=model,
                         state_dict=state_dict,
                         schema=schema,
                         scaler=scaler,
                         target_types=target_types,
                         task=task,
                         device=device)
    
        if self.task not in [MLTaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE, MLTaskKeys.EXOGENOUS_SEQUENCE_VALUE]:
            _LOGGER.error(f"Invalid task '{self.task}' for exogenous forecasting.")
            raise ValueError()
        
        if self._num_exog == 0:
            _LOGGER.error("No exogenous features found in the dataset schema. Cannot perform exogenous forecasting.")
            raise ValueError()
    
    def _preprocess_input(self, features: torch.Tensor) -> torch.Tensor:
        """
        Overrides base preprocessing to ensure the exogenous model only 
        receives exogenous features after full-width scaling.
        """
        scaled_features = super()._preprocess_input(features)
        return scaled_features[..., self._exog_indices]
    
    def forecast(self, 
                 n_steps: int,
                 future_exogenous: pd.DataFrame,
                 initial_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 ) -> pd.DataFrame:
        """
        Forecast using sliding exogenous windows for 'n_steps' into the future.
        
        Exogenous features must be provided as a pandas DataFrame containing at least 'n_steps' of future data.
        
        Args:
            n_steps (int): The number of future time steps to predict.
            future_exogenous (pd.DataFrame): Future known exogenous variables (un-scaled).
            initial_sequence (np.ndarray | torch.Tensor, optional): The historical sequence (un-scaled).

        Returns:
            pd.DataFrame: A DataFrame containing the forecasts.
        """
        if n_steps <= 0:
            _LOGGER.error(f"'n_steps' must be a positive integer greater than 0. Received: {n_steps}")
            raise ValueError()
            
        if not isinstance(future_exogenous, pd.DataFrame):
            _LOGGER.error("'future_exogenous' must be a pandas DataFrame to ensure safe column alignment.")
            raise TypeError()   
        
        # --- 1. Resolve Initial Sequence ---
        if initial_sequence is None:
            if self.initial_sequence is None:
                _LOGGER.error("No 'initial_sequence' provided/loaded. Cannot forecast.")
                raise ValueError()
            initial_sequence_tensor = torch.from_numpy(self.initial_sequence).float()
        elif isinstance(initial_sequence, np.ndarray):
            initial_sequence_tensor = torch.from_numpy(initial_sequence).float()
        else:
            initial_sequence_tensor = initial_sequence.float()

        if initial_sequence_tensor.ndim != 2:
             _LOGGER.error(f"initial_sequence must be 2D (seq_len, num_features). Got {initial_sequence_tensor.ndim}D.")
             raise ValueError()
             
        seq_len, num_features = initial_sequence_tensor.shape
        
        if num_features != self._num_features:
             _LOGGER.error(f"initial_sequence feature mismatch. Expected {self._num_features}, got {num_features}.")
             raise ValueError()

        # --- 2. Scale Initial Sequence & Isolate Exogenous Features ---
        if self.feature_scaler is None:
            scaled_initial = initial_sequence_tensor
        else:
            # Scaler expects the full width of the original schema
            scaled_sequence_flat = self.feature_scaler.transform(initial_sequence_tensor)
            scaled_initial = scaled_sequence_flat.reshape(seq_len, num_features)
            
        # The model only takes exogenous inputs
        current_scaled_exog_sequence = scaled_initial[:, self._exog_indices].to(self.device)

        # --- 3. Resolve & Scale Future Exogenous Data ---
        if len(future_exogenous) < n_steps:
            _LOGGER.error(f"'future_exogenous' must have at least {n_steps} rows for a {n_steps}-step forecast.")
            raise ValueError()
            
        missing_cols = [col for col in self._exog_names if col not in future_exogenous.columns]
        if missing_cols:
            _LOGGER.error(f"'future_exogenous' is missing required exogenous columns: {missing_cols}")
            raise ValueError()
            
        exog_np = future_exogenous[self._exog_names].iloc[:n_steps].to_numpy()
        exog_tensor = torch.from_numpy(exog_np).float()
        
        if self.feature_scaler:
            # Pad the missing targets with zeros to satisfy the full-width scaler transformation
            dummy_full = torch.zeros((n_steps, num_features))
            dummy_full[:, self._exog_indices] = exog_tensor
            dummy_scaled = self.feature_scaler.transform(dummy_full)
            scaled_exog_tensor = dummy_scaled[:, self._exog_indices].to(self.device)
        else:
            scaled_exog_tensor = exog_tensor.to(self.device)

        # --- 4. Sliding Exogenous Window Loop ---
        descaled_predictions = []
        with torch.no_grad():
            for t in range(n_steps):
                input_tensor = current_scaled_exog_sequence.unsqueeze(0)
                model_output_dict = self.model(input_tensor) 
                
                next_step_preds = []
                for target in self.target_names:
                    pred = model_output_dict[target].squeeze(0)
                    
                    if self._is_seq_to_val:
                        target_pred = pred
                    else: 
                        target_pred = pred[-1] 
                        
                    if self.target_types.get(target) == DatasetKeys.TARGET_CATEGORICAL:
                        target_pred = torch.argmax(target_pred, dim=-1).float()
                        
                    next_step_preds.append(target_pred)
                
                scaled_prediction = torch.stack(next_step_preds)
                
                if self.target_scaler:
                    descaled_val = self.target_scaler.inverse_transform(scaled_prediction.unsqueeze(0)).squeeze(0).cpu().numpy()
                else:
                    descaled_val = scaled_prediction.cpu().numpy()
                    
                descaled_predictions.append(descaled_val)
                
                # Slide window strictly using future exogenous data (no target feedback)
                next_step_exog = scaled_exog_tensor[t]
                current_scaled_exog_sequence = torch.cat((current_scaled_exog_sequence[1:], next_step_exog.unsqueeze(0)))
                
        # --- 5. Return Results ---
        forecast_df = pd.DataFrame(descaled_predictions, columns=[self.schema.feature_names[i] for i in self.target_indices])
        
        return forecast_df
