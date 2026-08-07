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


_LOGGER = get_logger("Autoregressive Inference")


__all__ = [
    "DragonSequenceAutoregressiveHandler"
]


class DragonSequenceAutoregressiveHandler(_BaseSequenceInferenceHandler):
    """
    Handles inference and autoregressive forecasting for sequence models.
    Targets are actively predicted and fed back into the rolling window for the next time step.
    
    This handler automatically scales inputs and de-scales outputs.
    """
    def __init__(self,
                 model: nn.Module,
                 state_dict: Union[str, Path],
                 schema: Union[str, Path, FeatureSchema],
                 scaler: Optional[Union[str, Path]],
                 target_types: Optional[dict[str, str]] = None,
                 task: Optional[Literal["autoregressive-sequence-to-sequence", "autoregressive-sequence-to-value"]]=None,
                 device: str = 'cpu'):
        """
        Initializes the handler for autoregressive sequence tasks.
    
        Args:
            model (nn.Module): A trained model architecture that must output a dictionary mapping target names to their respective prediction tensors.
            state_dict (str | Path): Path to the saved .pth model state_dict file or a FinalizedFile format.
            schema (str | Path | FeatureSchema): The feature schema or a path to a directory containing the schema.
            scaler (str | Path | None): File path to a saved DragonScaler state. This is required to correctly scale continuous inputs and de-scale predictions.
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
        
        if self.task not in [MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_SEQUENCE, MLTaskKeys.AUTOREGRESSIVE_SEQUENCE_VALUE]:
            _LOGGER.error(f"Invalid task '{self.task}' for autoregressive forecasting.")
            raise ValueError()
    
    def forecast(self, 
                 n_steps: int,
                 future_exogenous: Optional[pd.DataFrame],
                 initial_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None) -> pd.DataFrame:
        """
        Autoregressively forecasts 'n_steps' into the future.
        
        If the model requires exogenous features (features that are not targets) per step prediction,
        'future_exogenous' must be provided as a pandas DataFrame containing at least 'n_steps' of future data.
        
        Args:
            n_steps (int): The number of future time steps to predict.
            future_exogenous (pd.DataFrame | None): Future known variables (un-scaled).
            initial_sequence (np.ndarray | torch.Tensor, optional): The historical sequence (un-scaled).
        
        Returns:
            pd.DataFrame: A DataFrame containing the forecasts.
        """
        if n_steps <= 0:
            _LOGGER.error(f"'n_steps' must be a positive integer greater than 0. Received: {n_steps}")
            raise ValueError()
        
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

        # --- 2. Scale Initial Sequence ---
        if self.feature_scaler is None:
            current_scaled_sequence = initial_sequence_tensor.to(self.device)
        else:
            scaled_sequence_flat = self.feature_scaler.transform(initial_sequence_tensor)
            current_scaled_sequence = scaled_sequence_flat.reshape(seq_len, num_features).to(self.device)

        # --- 3. Resolve & Scale Future Exogenous Data ---
        scaled_exog_tensor = None
        if self._num_exog > 0:
            if future_exogenous is None:
                _LOGGER.error(f"Model requires {self._num_exog} exogenous features, but 'future_exogenous' was not provided.")
                raise ValueError()
            
            if not isinstance(future_exogenous, pd.DataFrame):
                _LOGGER.error("'future_exogenous' must be a pandas DataFrame to ensure safe column alignment.")
                raise TypeError()
                
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
                dummy_full = torch.zeros((n_steps, num_features))
                dummy_full[:, self._exog_indices] = exog_tensor
                dummy_scaled = self.feature_scaler.transform(dummy_full)
                scaled_exog_tensor = dummy_scaled[:, self._exog_indices].to(self.device)
            else:
                scaled_exog_tensor = exog_tensor.to(self.device)

        # --- 4. Autoregressive Loop ---
        descaled_predictions = []
        with torch.no_grad():
            for t in range(n_steps):
                input_tensor = current_scaled_sequence.unsqueeze(0)
                model_output_dict = self.model(input_tensor) 
                
                next_step_preds = []
                for target in self.target_names:
                    pred = model_output_dict[target].squeeze(0)
                    
                    if self._is_seq_to_val:
                        target_pred = pred
                    else: # sequence-to-sequence
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
                
                # Reconstruct the complete feature vector and feedback prediction
                next_step = torch.zeros(num_features, device=self.device)
                next_step[self.target_indices] = scaled_prediction
                
                if self._num_exog > 0 and scaled_exog_tensor is not None:
                    next_step[self._exog_indices] = scaled_exog_tensor[t]
                    
                current_scaled_sequence = torch.cat((current_scaled_sequence[1:], next_step.unsqueeze(0)))
                
        # --- 5. Return Results ---
        forecast_df = pd.DataFrame(descaled_predictions, columns=[self.schema.feature_names[i] for i in self.target_indices])
        
        return forecast_df
