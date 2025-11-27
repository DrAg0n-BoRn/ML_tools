import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Union, Dict, Any, Literal
from pathlib import Path
import json
import warnings

from ._ML_models import _ArchitectureHandlerMixin
from ._path_manager import make_fullpath
from ._keys import PytorchModelArchitectureKeys
from ._schema import FeatureSchema
from ._script_info import _script_info
from ._logger import _LOGGER

# Imports from pytorch_tabular
try:
    from omegaconf import DictConfig
    from pytorch_tabular.models import (
        GatedAdditiveTreeEnsembleModel as _GATE, 
        NODEModel as _NODE,
        TabNetModel as _TabNet,
        AutoIntModel as _AutoInt
    )
except ImportError:
    _LOGGER.error(f"GATE and NODE require 'pip install pytorch_tabular omegaconf' dependencies.")
    raise ImportError()
else:
    # Silence pytorch_tabular INFO logs up to error level
    import logging
    logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)
    logging.getLogger("pytorch_tabular.models.node.node_model").setLevel(logging.ERROR)


__all__ = [
    "PyTabGateModel",
    "PyTabTabNet",
    "PyTabAutoInt",
    "PyTabNodeModel"
]


class _BasePytabWrapper(nn.Module, _ArchitectureHandlerMixin):
    """
    Internal Base Class: Do not use directly.
    
    This is an adapter to make pytorch_tabular models compatible with the
    dragon-ml-toolbox pipeline.
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
            'target_range': None,
            **kwargs
        }
        
        if 'loss' not in config_dict:
            config_dict['loss'] = 'MSELoss' # Dummy
        if 'metrics' not in config_dict:
            config_dict['metrics'] = []
            
        return DictConfig(config_dict)
    
    def _build_inferred_config(self, out_targets: int, embedding_dim: int = None) -> DictConfig:
        """
        Helper to create the inferred_config required by pytorch_tabular v1.0+.
        Includes explicit embedding_dims calculation to satisfy BaseModel assertions.
        """
        # 1. Calculate embedding_dims list of tuples: [(cardinality, dim), ...]
        if self.categorical_indices:
            if embedding_dim is not None:
                # Use the user-provided fixed dimension for all categorical features
                embedding_dims = [(card, embedding_dim) for card in self.cardinalities]
            else:
                # Default heuristic: min(50, (card + 1) // 2)
                embedding_dims = [(card, min(50, (card + 1) // 2)) for card in self.cardinalities]
        else:
            embedding_dims = []

        # 2. Calculate the total dimension of concatenated embeddings
        # This fixes the 'Missing key embedded_cat_dim' error
        embedded_cat_dim = sum([dim for _, dim in embedding_dims])

        return DictConfig({
            "continuous_dim": len(self.numerical_indices),
            "categorical_dim": len(self.categorical_indices),
            "categorical_cardinality": self.cardinalities,
            "output_dim": out_targets,
            "embedding_dims": embedding_dims,
            "embedded_cat_dim": embedded_cat_dim, 
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts a single tensor and converts it to the dict
        that pytorch_tabular models expect.
        """
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()

        input_dict = {
            'continuous': x_cont,
            'categorical': x_cat
        }
        
        model_output_dict = self.internal_model(input_dict)
        return model_output_dict['logits']

    def get_architecture_config(self) -> Dict[str, Any]:
        """Returns the full configuration of the model."""
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
            _LOGGER.error("Invalid architecture file: missing 'schema_dict'.")
            raise ValueError("Missing 'schema_dict' in config.")
            
        schema_data = config.pop('schema_dict')
        
        raw_index_map = schema_data['categorical_index_map']
        if raw_index_map is not None:
            rehydrated_index_map = {int(k): v for k, v in raw_index_map.items()}
        else:
            rehydrated_index_map = None

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
        internal_repr = internal_model_str.split('\n')[0]
        return f"{self.model_name}(internal_model={internal_repr})"


class PyTabGateModel(_BasePytabWrapper):
    """
    Adapter for the Gated Additive Tree Ensemble (GATE) model.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 gflu_stages: int = 4,
                 num_trees: int = 20,
                 tree_depth: int = 4,
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
                Depth of each tree. (Recommended: 4 to 6)
            dropout (float):
                Dropout rate for the GFLU.
        """
        super().__init__(schema)
        
        warnings.filterwarnings("ignore", message="Implicit dimension choice for softmax")
        warnings.filterwarnings("ignore", message="Ignoring head config")
        
        
        self.model_name = "PyTabGateModel"
        self.out_targets = out_targets
        
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'gflu_stages': gflu_stages,
            'num_trees': num_trees,
            'tree_depth': tree_depth,
            'dropout': dropout
        }

        # Build Hyperparameter Config with defaults
        pt_config = self._build_pt_config(
            out_targets=out_targets,
            embedding_dim=embedding_dim,
            
            # GATE Specific Mappings
            gflu_stages=gflu_stages,
            num_trees=num_trees,
            tree_depth=tree_depth,
            gflu_dropout=dropout,
            tree_dropout=dropout,
            tree_wise_attention=True,
            tree_wise_attention_dropout=dropout,
            
            # GATE Defaults
            chain_trees=False,
            binning_activation="sigmoid",
            feature_mask_function="softmax",
            share_head_weights=True,
            
            # Sparsity
            gflu_feature_init_sparsity=0.3,
            tree_feature_init_sparsity=0.3,
            learnable_sparsity=True,
            
            # Head Configuration
            head="LinearHead",
            head_config={
                "layers": "", 
                "activation": "ReLU", 
                "dropout": 0.0, 
                "use_batch_norm": False, 
                "initialization": "kaiming"
            },
            
            # General Defaults (Required to prevent initialization errors)
            embedding_dropout=0.0,
            batch_norm_continuous_input=False,
            virtual_batch_size=None,
            learning_rate=1e-3, 
            target_range=None,
        )
        
        # Build Data Inference Config (Required by PyTabular v1.0+)
        inferred_config = self._build_inferred_config(
            out_targets=out_targets, 
            embedding_dim=embedding_dim
        )

        # Instantiate the internal pytorch_tabular model
        self.internal_model = _GATE(
            config=pt_config, 
            inferred_config=inferred_config
        )
        
    def __repr__(self) -> str:
        return (f"{self.model_name}(\n"
                f"  out_targets={self.out_targets},\n"
                f"  embedding_dim={self.model_hparams.get('embedding_dim')},\n"
                f"  gflu_stages={self.model_hparams.get('gflu_stages')},\n"
                f"  num_trees={self.model_hparams.get('num_trees')},\n"
                f"  tree_depth={self.model_hparams.get('tree_depth')},\n"
                f"  dropout={self.model_hparams.get('dropout')}\n"
                ")")


class PyTabTabNet(_BasePytabWrapper):
    """
    Adapter for Google's TabNet (Attentive Interpretable Tabular Learning).
    
    TabNet uses sequential attention to choose which features to reason 
    from at each decision step, enabling interpretability.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 virtual_batch_size: int = 128,
                 mask_type: Literal['sparsemax', 'entmax', 'softmax'] = 'sparsemax'):
        """
        Args:
            schema (FeatureSchema): The definitive schema object.
            out_targets (int): Number of output targets.
            n_d (int): Dimension of the prediction layer (usually 8-64).
            n_a (int): Dimension of the attention layer (usually equal to n_d).
            n_steps (int): Number of sequential attention steps (usually 3-10).
            gamma (float): Relaxation parameter for sparsity (usually 1.0-2.0).
            n_independent (int): Number of independent GLU layers in each block.
            n_shared (int): Number of shared GLU layers in each block.
            virtual_batch_size (int): Batch size for Ghost Batch Normalization.
            mask_type (str): Masking function.
                - 'sparsemax' for sparse feature selection.
                - 'entmax' for moderately sparse selection.
                - 'softmax' for dense selection (safest).
        """
        super().__init__(schema)
        self.model_name = "PyTabTabNet"
        self.out_targets = out_targets
        
        self.model_hparams = {
            'n_d': n_d,
            'n_a': n_a,
            'n_steps': n_steps,
            'gamma': gamma,
            'n_independent': n_independent,
            'n_shared': n_shared,
            'virtual_batch_size': virtual_batch_size,
            'mask_type': mask_type
        }

        # TabNet does not use standard embeddings, so we don't pass embedding_dim
        pt_config = self._build_pt_config(
            out_targets=out_targets,
            
            # TabNet Params
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            virtual_batch_size=virtual_batch_size,
            
            # TabNet Defaults
            mask_type=mask_type,
            
            # Head Configuration
            head="LinearHead",
            head_config={
                "layers": "", 
                "activation": "ReLU", 
                "dropout": 0.0, 
                "use_batch_norm": False, 
                "initialization": "kaiming"
            },
            
            # General Defaults
            batch_norm_continuous_input=False,
            learning_rate=1e-3
        )
        
        inferred_config = self._build_inferred_config(out_targets=out_targets)

        self.internal_model = _TabNet(
            config=pt_config,
            inferred_config=inferred_config
        )

    def __repr__(self) -> str:
        return (f"{self.model_name}(\n"
                f"  out_targets={self.out_targets},\n"
                f"  n_d={self.model_hparams.get('n_d')},\n"
                f"  n_a={self.model_hparams.get('n_a')},\n"
                f"  n_steps={self.model_hparams.get('n_steps')},\n"
                f"  gamma={self.model_hparams.get('gamma')},\n"
                f"  virtual_batch_size={self.model_hparams.get('virtual_batch_size')}\n"
                f"  mask_type='{self.model_hparams.get('mask_type')}'\n"
                f")")


class PyTabAutoInt(_BasePytabWrapper):
    """
    Adapter for AutoInt (Automatic Feature Interaction Learning).
    
    Uses Multi-Head Self-Attention to automatically learn high-order 
    feature interactions.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 num_heads: int = 2,
                 num_attn_blocks: int = 3,
                 attn_dropout: float = 0.1,
                 has_residuals: bool = True,
                 deep_layers: bool = True,
                 layers: str = "128-64-32"):
        """
        Args:
            schema (FeatureSchema): The definitive schema object.
            out_targets (int): Number of output targets.
            embedding_dim (int): Dimension of feature embeddings (attn_embed_dim).
            num_heads (int): Number of attention heads.
            num_attn_blocks (int): Number of attention layers.
            attn_dropout (float): Dropout between attention layers.
            has_residuals (bool): If True, adds residual connections.
            deep_layers (bool): If True, adds a standard MLP after attention.
            layers (str): Hyphen-separated layer sizes for the deep MLP part.
        """
        super().__init__(schema)
        self.model_name = "PyTabAutoInt"
        self.out_targets = out_targets
        
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'num_attn_blocks': num_attn_blocks,
            'attn_dropout': attn_dropout,
            'has_residuals': has_residuals,
            'deep_layers': deep_layers,
            'layers': layers
        }

        pt_config = self._build_pt_config(
            out_targets=out_targets,
            
            # AutoInt Params
            attn_embed_dim=embedding_dim,
            num_heads=num_heads,
            num_attn_blocks=num_attn_blocks,
            attn_dropouts=attn_dropout,
            has_residuals=has_residuals,
            
            # Deep MLP part (Optional in AutoInt, but usually good)
            deep_layers=deep_layers,
            layers=layers,
            activation="ReLU",
            
            # Head Configuration
            head="LinearHead",
            head_config={
                "layers": "", 
                "activation": "ReLU", 
                "dropout": 0.0, 
                "use_batch_norm": False, 
                "initialization": "kaiming"
            },
            
            # General Defaults
            embedding_dropout=0.0,
            batch_norm_continuous_input=False,
            learning_rate=1e-3
        )
        
        inferred_config = self._build_inferred_config(
            out_targets=out_targets, 
            embedding_dim=embedding_dim
        )

        self.internal_model = _AutoInt(
            config=pt_config,
            inferred_config=inferred_config
        )

    def __repr__(self) -> str:
        return (f"{self.model_name}(\n"
                f"  out_targets={self.out_targets},\n"
                f"  embedding_dim={self.model_hparams.get('embedding_dim')},\n"
                f"  num_heads={self.model_hparams.get('num_heads')},\n"
                f"  num_attn_blocks={self.model_hparams.get('num_attn_blocks')},\n"
                f"  deep_layers={self.model_hparams.get('deep_layers')}\n"
                f")")


class PyTabNodeModel(_BasePytabWrapper):
    """
    Adapter for the Neural Oblivious Decision Ensembles (NODE) model.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 num_trees: int = 1024,
                 num_layers: int = 2,
                 tree_depth: int = 6,
                 dropout: float = 0.1,
                 backend_function: Literal['softmax', 'entmax15'] = 'softmax'):
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
            num_layers (int):
                Number of NODE layers (stacked ensembles). (Recommended: 2 to 4)
            tree_depth (int):
                Depth of each tree. (Recommended: 4 to 8)
            dropout (float):
                Dropout rate.
            backend_function ('softmax' | 'entmax15'):
                Function for feature selection. 'entmax15' (sparse) or 'softmax' (dense).
                Use 'softmax' if dealing with convergence issues.
        """
        super().__init__(schema)
        self.model_name = "PyTabNodeModel"
        self.out_targets = out_targets
        
        warnings.filterwarnings("ignore", message="Ignoring head config because NODE has a specific head")

        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'num_trees': num_trees,
            'num_layers': num_layers,
            'tree_depth': tree_depth,
            'dropout': dropout,
            'backend_function': backend_function
        }

        # Build Hyperparameter Config with ALL defaults
        pt_config = self._build_pt_config(
            out_targets=out_targets,
            embedding_dim=embedding_dim,
            
            # NODE Specific Mappings
            num_trees=num_trees,
            depth=tree_depth, # Map tree_depth -> depth
            num_layers=num_layers,     # num_layers=1 for a single ensemble
            total_trees=num_trees,
            dropout_rate=dropout,
            
            # NODE Defaults (Manually populated to satisfy backbone requirements)
            additional_tree_output_dim=0,
            input_dropout=0.0,
            choice_function=backend_function,
            bin_function=backend_function,
            initialize_response="normal",
            initialize_selection_logits="uniform",
            threshold_init_beta=1.0,
            threshold_init_cutoff=1.0,
            max_features=None,
            
            # General Defaults (Required to prevent initialization errors)
            embedding_dropout=0.0,
            batch_norm_continuous_input=False,
            virtual_batch_size=None,
            learning_rate=1e-3, 
            
            # NODE schema
            data_aware_init_batch_size=2000, # Required by NodeConfig schema
            augment_dim=0,
        )

        # Build Data Inference Config (Required by PyTabular v1.0+)
        inferred_config = self._build_inferred_config(
            out_targets=out_targets, 
            embedding_dim=embedding_dim
        )

        # Instantiate the internal pytorch_tabular model
        self.internal_model = _NODE(
            config=pt_config,
            inferred_config=inferred_config
        )
    
    def perform_data_aware_initialization(self, train_dataset: Any, batch_size: int = 2000):
        """
        CRITICAL: Initializes NODE decision thresholds using a batch of data.
        
        Call this ONCE before training starts with a large batch (e.g., 2000 samples).
        
        Use the CPU
        
        Args:
            train_dataset: a PyTorch Dataset.
            batch_size: Number of samples to use for initialization.
        """
        # Use a DataLoader to robustly fetch a single batch
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        try:
            batch = next(iter(loader))
        except StopIteration:
            _LOGGER.error("Dataset is empty. Cannot perform data-aware initialization.")
            return

        x_tensor, _ = batch 
        
        # Prepare input dict
        # Prepare input dict matching pytorch_tabular expectations
        # Ensure we are on the same device as the model (CPU here)
        device = next(self.parameters()).device
        x_cont = x_tensor[:, self.numerical_indices].float().to(device)
        x_cat = x_tensor[:, self.categorical_indices].long().to(device)
        
        input_dict = {
            'continuous': x_cont,
            'categorical': x_cat
        }
        
        # --- MOCK DATA MODULE ---
        # datamodule.train_dataloader() -> yields the batch
        class _MockDataModule:
            def train_dataloader(self, batch_size=None):
                # Accepts 'batch_size' argument to satisfy the caller
                # Returns a list containing just the single pre-processed batch dictionary
                return [input_dict]
        
        mock_dm = _MockDataModule()
        
        _LOGGER.info(f"Running NODE Data-Aware Initialization with {batch_size} samples...")
        try:
            with torch.no_grad():
                # Call init on the BACKBONE, not the wrapper
                self.internal_model.data_aware_initialization(mock_dm)
            _LOGGER.info("NODE Initialization Complete. Ready to train.")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize NODE model: {e}")
            raise e
    
    def __repr__(self) -> str:
        return (f"{self.model_name}(\n"
                f"  out_targets={self.out_targets},\n"
                f"  embedding_dim={self.model_hparams.get('embedding_dim')},\n"
                f"  num_trees={self.model_hparams.get('num_trees')},\n"
                f"  num_layers={self.model_hparams.get('num_layers')},\n"
                f"  tree_depth={self.model_hparams.get('tree_depth')},\n"
                f"  dropout={self.model_hparams.get('dropout')}\n"
                f"  backend_function={self.model_hparams.get('backend_function')}\n"
                f")")


def info():
    _script_info(__all__)
