import torch
from torch import nn
from typing import Union, Dict, Any
from pathlib import Path
import json

from ._logger import _LOGGER
from .path_manager import make_fullpath
from ._keys import PytorchModelArchitectureKeys
from ._schema import FeatureSchema
from ._script_info import _script_info
from .ML_models import _ArchitectureHandlerMixin

# Imports from pytorch_tabular
try:
    from omegaconf import DictConfig
    from pytorch_tabular.models import GatedAdditiveTreeEnsembleModel, NODEModel
except ImportError:
    _LOGGER.error(f"GATE and NODE require 'pip install pytorch_tabular omegaconf' dependencies.")
    raise ImportError()


__all__ = [
    "DragonGateModel",
    "DragonNodeModel",
]


class _BasePytabWrapper(nn.Module, _ArchitectureHandlerMixin):
    """
    Internal Base Class: Do not use directly.
    
    This is an adapter to make pytorch_tabular models compatible with the
    dragon-ml-toolbox pipeline.
    
    It handles:
    1.  Schema-based initialization.
    2.  Single-tensor forward pass, which is then split into the
        dict {'continuous': ..., 'categorical': ...} that pytorch_tabular expects.
    3.  Saving/Loading architecture using the pipeline's _ArchitectureHandlerMixin.
    """
    def __init__(self, schema: FeatureSchema):
        super().__init__()
        
        self.schema = schema
        self.model_name = "Base" # To be overridden by child
        self.internal_model: nn.Module = None # type: ignore # To be set by child
        self.model_hparams: Dict = dict() # To be set by child

        # --- Derive indices from schema ---
        categorical_map = schema.categorical_index_map
        
        if categorical_map:
            # The order of keys/values is implicitly linked and must be preserved
            self.categorical_indices = list(categorical_map.keys())
            self.cardinalities = list(categorical_map.values())
        else:
            self.categorical_indices = []
            self.cardinalities = []
        
        # Derive numerical indices by finding what's not categorical
        all_indices = set(range(len(schema.feature_names)))
        categorical_indices_set = set(self.categorical_indices)
        self.numerical_indices = sorted(list(all_indices - categorical_indices_set))

    def _build_pt_config(self, out_targets: int, **kwargs) -> DictConfig:
        """Helper to create the minimal config dict for a pytorch_tabular model."""
        # 'regression' is the most neutral for model architecture. The final output_dim is what truly matters.
        task = "regression"

        config_dict = {
            # --- Data / Schema Params ---
            'task': task,
            'continuous_cols': list(self.schema.continuous_feature_names),
            'categorical_cols': list(self.schema.categorical_feature_names),
            'continuous_dim': len(self.numerical_indices),
            'categorical_dim': len(self.categorical_indices),
            'categorical_cardinality': self.cardinalities,
            'target': ['dummy_target'], # Required, but not used
            
            # --- Model Params ---
            'output_dim': out_targets,
            **kwargs
        }
        
        # Add common params that most models need
        if 'loss' not in config_dict:
            config_dict['loss'] = 'NotUsed'
        if 'metrics' not in config_dict:
            config_dict['metrics'] = []
            
        return DictConfig(config_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts a single tensor and converts it to the dict
        that pytorch_tabular models expect.
        """
        # 1. Split the single tensor input
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()

        # 2. Create the input dict
        input_dict = {
            'continuous': x_cont,
            'categorical': x_cat
        }
        
        # 3. Pass to the internal pytorch_tabular model
        #    The model returns a dict, we extract the logits
        model_output_dict = self.internal_model(input_dict)
        
        # 4. Return the logits tensor
        return model_output_dict['logits']

    def get_architecture_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
        # Deconstruct schema into a JSON-friendly dict
        schema_dict = {
            'feature_names': self.schema.feature_names,
            'continuous_feature_names': self.schema.continuous_feature_names,
            'categorical_feature_names': self.schema.categorical_feature_names,
            'categorical_index_map': self.schema.categorical_index_map,
            'categorical_mappings': self.schema.categorical_mappings
        }
        
        config = {
            'schema_dict': schema_dict,
            'out_targets': self.out_targets,
            **self.model_hparams
        }
        return config

    @classmethod
    def load(cls: type, file_or_dir: Union[str, Path], verbose: bool = True) -> nn.Module:
        """Loads a model architecture from a JSON file."""
        user_path = make_fullpath(file_or_dir)
        
        if user_path.is_dir():
            json_filename = PytorchModelArchitectureKeys.SAVENAME + ".json"
            target_path = make_fullpath(user_path / json_filename, enforce="file")
        elif user_path.is_file():
            target_path = user_path
        else:
            _LOGGER.error(f"Invalid path: '{file_or_dir}'")
            raise IOError()

        with open(target_path, 'r') as f:
            saved_data = json.load(f)

        saved_class_name = saved_data[PytorchModelArchitectureKeys.MODEL]
        config = saved_data[PytorchModelArchitectureKeys.CONFIG]

        if saved_class_name != cls.__name__:
            _LOGGER.error(f"Model class mismatch. File specifies '{saved_class_name}', but '{cls.__name__}' was expected.")
            raise ValueError()

        # --- RECONSTRUCTION LOGIC ---
        if 'schema_dict' not in config:
            _LOGGER.error("Invalid architecture file: missing 'schema_dict'. This file may be from an older version.")
            raise ValueError("Missing 'schema_dict' in config.")
            
        schema_data = config.pop('schema_dict')
        
        # JSON saves all dict keys as strings, convert them back to int.
        raw_index_map = schema_data['categorical_index_map']
        if raw_index_map is not None:
            rehydrated_index_map = {int(k): v for k, v in raw_index_map.items()}
        else:
            rehydrated_index_map = None

        # JSON deserializes tuples as lists, convert them back.
        schema = FeatureSchema(
            feature_names=tuple(schema_data['feature_names']),
            continuous_feature_names=tuple(schema_data['continuous_feature_names']),
            categorical_feature_names=tuple(schema_data['categorical_feature_names']),
            categorical_index_map=rehydrated_index_map,
            categorical_mappings=schema_data['categorical_mappings']
        )
        
        config['schema'] = schema
        # --- End Reconstruction ---

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model
    
    def __repr__(self) -> str:
        internal_model_str = str(self.internal_model)
        # Grab the first line of the internal model's repr
        internal_repr = internal_model_str.split('\n')[0]
        return f"{self.model_name}(internal_model={internal_repr})"


class DragonGateModel(_BasePytabWrapper):
    """
    Adapter for the Gated Additive Tree Ensemble (GATE) model from the 'pytorch_tabular' library.
    
    GATE is a hybrid model that uses Gated Feature Learning Units (GFLUs) to
    learn powerful feature representations. These learned features are then
    fed into an additive ensemble of differentiable decision trees, combining
    the representation learning of deep networks with the structured
    decision-making of tree ensembles.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 gflu_stages: int = 6,
                 num_trees: int = 20,
                 tree_depth: int = 5,
                 dropout: float = 0.1):
        """
        Args:
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            out_targets (int): 
                Number of output targets.
            embedding_dim (int):
                Dimension of the categorical embeddings. (Recommended: 16 to 64)
            gflu_stages (int):
                Number of Gated Feature Learning Units (GFLU) stages. (Recommended: 2 to 6)
            num_trees (int):
                Number of trees in the ensemble. (Recommended: 10 to 50)
            tree_depth (int):
                Depth of each tree. (Recommended: 4 to 8)
            dropout (float):
                Dropout rate for the GFLU.
        """
        super().__init__(schema)
        self.model_name = "DragonGateModel"
        self.out_targets = out_targets
        
        # Store hparams for saving/loading
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'gflu_stages': gflu_stages,
            'num_trees': num_trees,
            'tree_depth': tree_depth,
            'dropout': dropout
        }

        # Build the minimal config for the GateModel
        pt_config = self._build_pt_config(
            out_targets=out_targets,
            embedding_dim=embedding_dim,
            gflu_stages=gflu_stages,
            num_trees=num_trees,
            tree_depth=tree_depth,
            dropout=dropout,
            # GATE-specific params
            gflu_dropout=dropout,
            chain_trees=False,
        )

        # Instantiate the internal pytorch_tabular model
        self.internal_model = GatedAdditiveTreeEnsembleModel(config=pt_config)


class DragonNodeModel(_BasePytabWrapper):
    """
    Adapter for the Neural Oblivious Decision Ensembles (NODE) model from the 'pytorch_tabular' library.
    
    NODE is a model based on an ensemble of differentiable 'oblivious'
    decision trees. An oblivious tree uses the same splitting feature and
    threshold across all nodes at the same depth. This structure, combined
    with a differentiable formulation, allows the model to be trained
    end-to-end with gradient descent, learning feature interactions and
    splitting thresholds simultaneously.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 num_trees: int = 1024,
                 tree_depth: int = 6,
                 dropout: float = 0.1):
        """
        Args:
            schema (FeatureSchema): 
                The definitive schema object from data_exploration.
            out_targets (int): 
                Number of output targets.
            embedding_dim (int):
                Dimension of the categorical embeddings. (Recommended: 16 to 64)
            num_trees (int):
                Total number of trees in the ensemble. (Recommended: 256 to 2048)
            tree_depth (int):
                Depth of each tree. (Recommended: 4 to 8)
            dropout (float):
                Dropout rate.
        """
        super().__init__(schema)
        self.model_name = "DragonNodeModel"
        self.out_targets = out_targets

        # Store hparams for saving/loading
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'num_trees': num_trees,
            'tree_depth': tree_depth,
            'dropout': dropout
        }

        # Build the minimal config for the NodeModel
        pt_config = self._build_pt_config(
            out_targets=out_targets,
            embedding_dim=embedding_dim,
            num_trees=num_trees,
            tree_depth=tree_depth,
            # NODE-specific params
            num_layers=1, # NODE uses num_layers=1 for a single ensemble
            total_trees=num_trees,
            dropout_rate=dropout,
        )

        # Instantiate the internal pytorch_tabular model
        self.internal_model = NODEModel(config=pt_config)


def info():
    _script_info(__all__)
