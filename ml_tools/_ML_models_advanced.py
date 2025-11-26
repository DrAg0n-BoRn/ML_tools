import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path
import json

from ._ML_models import _ArchitectureHandlerMixin
from ._path_manager import make_fullpath
from ._schema import FeatureSchema
from ._keys import PytorchModelArchitectureKeys
from ._logger import _LOGGER
from ._script_info import _script_info
from ._model_helpers import (
    Embedding1dLayer,
    GatedFeatureLearningUnit,
    NeuralDecisionTree,
    entmax15,
    entmoid15,
    sparsemax,
    sparsemoid,
    t_softmax,
    SimpleLinearHead,
    DenseODSTBlock,
    Embedding2dLayer,
    FeatTransformer, 
    AttentiveTransformer, 
    initialize_non_glu
)

__all__ = [
    "DragonGateModel",
    "DragonNodeModel",
    "DragonAutoInt",
    "DragonTabNet"
    ]

# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


class _GateHead(nn.Module):
    """
    Custom Prediction Head for GATE.
    Aggregates outputs from multiple trees using learnable weights (eta).
    """
    def __init__(self, input_dim: int, output_dim: int, num_trees: int, share_head_weights: bool = True):
        super().__init__()
        self.num_trees = num_trees
        self.output_dim = output_dim
        self.share_head_weights = share_head_weights

        # Decision Tree Heads
        if share_head_weights:
            self.head = nn.Linear(input_dim, output_dim)
        else:
            self.head = nn.ModuleList(
                [nn.Linear(input_dim, output_dim) for _ in range(num_trees)]
            )

        # Learnable mixing weights (eta) for the trees
        self.eta = nn.Parameter(torch.rand(num_trees, requires_grad=True))
        
        # Global Bias (T0) - initialized to 0, but often updated via data-aware init
        self.T0 = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # backbone_features shape: (Batch, Output_Dim_Tree, Num_Trees) if using attention?
        # Actually GATE backbone returns: (Batch, Output_Dim_Tree, Num_Trees)
        
        # 1. Apply Linear Head to Tree Outputs
        if self.share_head_weights:
            # Transpose to (Batch, Num_Trees, Feature_Dim) to apply the same linear layer
            # y_hat: (Batch, Num_Trees, Output_Dim)
            y_hat = self.head(backbone_features.transpose(2, 1))
        else:
            # Apply separate linear layers
            y_hat = torch.cat(
                [h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)],
                dim=1,
            )

        # 2. Weighted Sum using Eta
        # eta reshape: (1, Num_Trees, 1)
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        
        # Sum across trees -> (Batch, Output_Dim)
        y_hat = y_hat.sum(dim=1)

        # 3. Add Global Bias
        y_hat = y_hat + self.T0
        return y_hat


class DragonGateModel(nn.Module, _ArchitectureHandlerMixin):
    """
    Native implementation of the Gated Additive Tree Ensemble (GATE).
    
    Combines Gated Feature Learning Units (GFLU) for feature interaction learning
    with Differentiable Decision Trees for prediction.
    """
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        # "softmax": nn.functional.softmax,
        "softmax": lambda x: nn.functional.softmax(x, dim=-1),
        "t-softmax": t_softmax,
    }
    
    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": torch.sigmoid,
    }

    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 16,
                 gflu_stages: int = 6,
                 gflu_dropout: float = 0.1,
                 num_trees: int = 20,
                 tree_depth: int = 4,
                 tree_dropout: float = 0.1,
                 chain_trees: bool = False,
                 tree_wise_attention: bool = True,
                 tree_wise_attention_dropout: float = 0.1,
                 binning_activation: Literal['entmoid', 'sparsemoid', 'sigmoid'] = "entmoid",
                 feature_mask_function: Literal['entmax', 'sparsemax', 'softmax', 't-softmax'] = "entmax",
                 share_head_weights: bool = True,
                 batch_norm_continuous: bool = True):
        """
        Args:
            schema: FeatureSchema object.
            out_targets: Number of output targets.
            embedding_dim: Embedding dimension for categorical features.
            gflu_stages: Number of Gated Feature Learning Unit stages.
            num_trees: Number of Neural Decision Trees.
            tree_depth: Depth of the trees.
            chain_trees: If True, feeds output of tree T-1 into tree T.
            tree_wise_attention: If True, applies Self-Attention across the tree outputs.
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # -- Configuration for saving --
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'gflu_stages': gflu_stages,
            'gflu_dropout': gflu_dropout,
            'num_trees': num_trees,
            'tree_depth': tree_depth,
            'tree_dropout': tree_dropout,
            'chain_trees': chain_trees,
            'tree_wise_attention': tree_wise_attention,
            'tree_wise_attention_dropout': tree_wise_attention_dropout,
            'binning_activation': binning_activation,
            'feature_mask_function': feature_mask_function,
            'share_head_weights': share_head_weights,
            'batch_norm_continuous': batch_norm_continuous
        }

        # -- 1. Setup Data Processing --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        embedding_dims = [(c, embedding_dim) for c in self.cardinalities]
        n_continuous = len(self.numerical_indices)
        
        # -- 2. Embedding Layer --
        self.embedding_layer = Embedding1dLayer(
            continuous_dim=n_continuous,
            categorical_embedding_dims=embedding_dims,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        # Calculate total feature dimension after embedding
        total_embedded_cat_dim = sum([d for _, d in embedding_dims])
        self.n_features = n_continuous + total_embedded_cat_dim
        
        # -- 3. GFLU Backbone --
        self.gflu_stages = gflu_stages
        if gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=gflu_stages,
                feature_mask_function=self.ACTIVATION_MAP[feature_mask_function],
                dropout=gflu_dropout,
                feature_sparsity=0.3, # Standard default
                learnable_sparsity=True
            )
            
        # -- 4. Neural Decision Trees --
        self.num_trees = num_trees
        self.chain_trees = chain_trees
        self.tree_depth = tree_depth
        
        if num_trees > 0:
            # Calculate input dim for trees (chaining adds to input)
            tree_input_dim = self.n_features
            
            self.trees = nn.ModuleList()
            for _ in range(num_trees):
                tree = NeuralDecisionTree(
                    depth=tree_depth,
                    n_features=tree_input_dim,
                    dropout=tree_dropout,
                    binning_activation=self.BINARY_ACTIVATION_MAP[binning_activation],
                    feature_mask_function=self.ACTIVATION_MAP[feature_mask_function],
                )
                self.trees.append(tree)
                if chain_trees:
                    # Next tree sees original features + output of this tree (2^depth leaves)
                    tree_input_dim += 2**tree_depth

            self.tree_output_dim = 2**tree_depth
            
            # Optional: Tree-wise Attention
            self.tree_wise_attention = tree_wise_attention
            if tree_wise_attention:
                self.tree_attention = nn.MultiheadAttention(
                    embed_dim=self.tree_output_dim,
                    num_heads=1,
                    batch_first=False, # (Seq, Batch, Feature) standard for PyTorch Attn
                    dropout=tree_wise_attention_dropout
                )
        else:
            self.tree_output_dim = self.n_features

        # -- 5. Prediction Head --
        if num_trees > 0:
            self.head = _GateHead(
                input_dim=self.tree_output_dim,
                output_dim=out_targets,
                num_trees=num_trees,
                share_head_weights=share_head_weights
            )
        else:
            # Fallback if no trees (just GFLU -> Linear)
            self.head = SimpleLinearHead(self.n_features, out_targets)
            # Add T0 manually for consistency if needed, but SimpleLinear covers bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embeddings
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. GFLU
        if self.gflu_stages > 0:
            x = self.gflus(x)
            
        # 3. Trees
        if self.num_trees > 0:
            tree_outputs = []
            tree_input = x
            
            for i in range(self.num_trees):
                # Tree returns (leaf_nodes, feature_masks)
                # leaf_nodes shape: (Batch, 2^depth)
                leaf_nodes, _ = self.trees[i](tree_input)
                
                tree_outputs.append(leaf_nodes.unsqueeze(-1))
                
                if self.chain_trees:
                    tree_input = torch.cat([tree_input, leaf_nodes], dim=1)
            
            # Stack: (Batch, Output_Dim_Tree, Num_Trees)
            tree_outputs = torch.cat(tree_outputs, dim=-1)
            
            # 4. Attention
            if self.tree_wise_attention:
                # Permute for MultiheadAttention: (Num_Trees, Batch, Output_Dim_Tree)
                # Treating 'Trees' as the sequence length
                attn_input = tree_outputs.permute(2, 0, 1)
                
                attn_output, _ = self.tree_attention(attn_input, attn_input, attn_input)
                
                # Permute back: (Batch, Output_Dim_Tree, Num_Trees)
                tree_outputs = attn_output.permute(1, 2, 0)
                
            # 5. Head
            return self.head(tree_outputs)
            
        else:
            # No trees, just linear on top of GFLU
            return self.head(x)

    def data_aware_initialization(self, train_dataset, num_samples: int = 2000):
        """
        Performs data-aware initialization for the global bias T0.
        This often speeds up convergence significantly for regression.
        """
        # 1. Prepare Data
        _LOGGER.info(f"Performing GATE data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device
            
        # 2. Extract Targets
        # Fast path: direct tensor access (Works with DragonDataset/_PytorchDataset)
        if hasattr(train_dataset, "labels") and isinstance(train_dataset.labels, torch.Tensor):
            limit = min(len(train_dataset.labels), num_samples)
            targets = train_dataset.labels[:limit]
        else:
            # Slow path: Iterate
            indices = range(min(len(train_dataset), num_samples))
            y_accum = []
            for i in indices:
                # Expecting (features, targets) tuple
                sample = train_dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    # Standard (X, y) tuple
                    y_val = sample[1]
                elif isinstance(sample, dict):
                    # Try common keys
                    y_val = sample.get('target', sample.get('y', None))
                else:
                    y_val = None
                
                if y_val is not None:
                    # Ensure it's a tensor
                    if not isinstance(y_val, torch.Tensor):
                        y_val = torch.tensor(y_val)
                    y_accum.append(y_val)

            if not y_accum:
                _LOGGER.warning("Could not extract targets for T0 initialization. Skipping.")
                return
            
            targets = torch.stack(y_accum)

        targets = targets.to(device).float()
        
        with torch.no_grad():
            if self.num_trees > 0:
                # Initialize T0 to mean of targets
                mean_target = torch.mean(targets, dim=0)
                
                # Check shapes to avoid broadcasting errors
                if self.head.T0.shape == mean_target.shape:
                    self.head.T0.data = mean_target
                    _LOGGER.info(f"Initialized T0 to {mean_target.cpu().numpy()}")
                elif self.head.T0.numel() == 1 and mean_target.numel() == 1:
                    # scalar case
                    self.head.T0.data = mean_target.view(self.head.T0.shape)
                    _LOGGER.info("GATE Initialization Complete. Ready to train.")
                    # _LOGGER.info(f"Initialized T0 to {mean_target.item()}")
                else:
                    _LOGGER.debug(f"Target shape mismatch for T0 init. Model: {self.head.T0.shape}, Data: {mean_target.shape}")
                    _LOGGER.warning(f"GATE initialization skipped due to shape mismatch:\n    Model: {self.head.T0.shape}\n    Data: {mean_target.shape}")

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
    

class DragonNodeModel(nn.Module, _ArchitectureHandlerMixin):
    """
    Native implementation of Neural Oblivious Decision Ensembles (NODE).
    
    NODE layers multiple layers of Oblivious Differentiable Decision Trees (ODST).
    The 'Dense' architecture concatenates the outputs of previous layers to the 
    features of subsequent layers, allowing for deep feature interaction learning.
    """
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": F.softmax,
    }
    
    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": torch.sigmoid,
    }

    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 24,
                 num_trees: int = 1024,
                 num_layers: int = 2,
                 tree_depth: int = 6,
                 additional_tree_output_dim: int = 3,
                 max_features: Optional[int] = None,
                 input_dropout: float = 0.0,
                 embedding_dropout: float = 0.0,
                 choice_function: Literal['entmax', 'sparsemax', 'softmax'] = 'entmax',
                 bin_function: Literal['entmoid', 'sparsemoid', 'sigmoid'] = 'entmoid',
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema: FeatureSchema object.
            out_targets: Number of output targets.
            embedding_dim: Embedding dimension.
            num_trees: Number of trees per layer.
            num_layers: Number of DenseODST layers.
            tree_depth: Depth of the oblivious trees.
            additional_tree_output_dim: Extra output channels per tree. 
                These are used for internal representation in deeper layers 
                but discarded for the final prediction. (Default: 3)
            max_features: Max features to keep in the dense connection (prevents explosion).
            choice_function: Activation for feature selection ('entmax' is sparse).
            bin_function: Activation for binning ('entmoid' is sparse).
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # -- Configuration for saving --
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'num_trees': num_trees,
            'num_layers': num_layers,
            'tree_depth': tree_depth,
            'additional_tree_output_dim': additional_tree_output_dim,
            'max_features': max_features,
            'input_dropout': input_dropout,
            'embedding_dropout': embedding_dropout,
            'choice_function': choice_function,
            'bin_function': bin_function,
            'batch_norm_continuous': batch_norm_continuous
        }

        # -- 1. Setup Embeddings --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        embedding_dims = [(c, embedding_dim) for c in self.cardinalities]
        n_continuous = len(self.numerical_indices)
        
        self.embedding_layer = Embedding1dLayer(
            continuous_dim=n_continuous,
            categorical_embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        total_embedded_dim = n_continuous + sum([d for _, d in embedding_dims])
        
        # -- 2. Backbone (Dense ODST) --
        # The tree output dim includes the target dim + auxiliary dims for deep learning
        self.tree_dim = out_targets + additional_tree_output_dim
        
        self.backbone = DenseODSTBlock(
            input_dim=total_embedded_dim,
            num_trees=num_trees,
            num_layers=num_layers,
            tree_output_dim=self.tree_dim,
            max_features=max_features,
            input_dropout=input_dropout,
            flatten_output=False, # We want (Batch, Num_Layers * Num_Trees, Tree_Dim)
            depth=tree_depth,
            # Activations
            choice_function=self.ACTIVATION_MAP[choice_function],
            bin_function=self.BINARY_ACTIVATION_MAP[bin_function],
            # Init strategies (defaults)
            initialize_response_=nn.init.normal_,
            initialize_selection_logits_=nn.init.uniform_,
        )
        
        # Note: NODE has a fixed Head (averaging) which is defined in forward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embeddings
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. Backbone
        # Output shape: (Batch, Total_Trees, Tree_Dim)
        x = self.backbone(x)
        
        # 3. Head (Averaging)
        # We take the first 'out_targets' channels and average them across all trees
        # subset: x[..., :out_targets]
        # mean: .mean(dim=-2) -> average over Total_Trees dimension
        return x[..., :self.out_targets].mean(dim=-2)

    def data_aware_initialization(self, train_dataset, num_samples: int = 2000):
        """
        Performs data-aware initialization for the ODST trees using a dataset.
        Crucial for NODE convergence.
        """
        # 1. Prepare Data
        _LOGGER.info(f"Performing NODE data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device
            
        # 2. Extract Features
        # Fast path: If the dataset exposes the full feature tensor (like _PytorchDataset)
        if hasattr(train_dataset, "features") and isinstance(train_dataset.features, torch.Tensor):
             # Slice directly
             limit = min(len(train_dataset.features), num_samples)
             x_input = train_dataset.features[:limit]
        else:
            # Slow path: Iterate and stack (Generic Dataset)
            indices = range(min(len(train_dataset), num_samples))
            x_accum = []
            for i in indices:
                # Expecting (features, targets) tuple from standard datasets
                sample = train_dataset[i]
                if isinstance(sample, (tuple, list)):
                    x_accum.append(sample[0])
                elif isinstance(sample, dict) and 'features' in sample:
                    x_accum.append(sample['features'])
                elif isinstance(sample, dict) and 'x' in sample:
                    x_accum.append(sample['x'])
                else:
                    # Fallback: assume the sample itself is the feature
                    x_accum.append(sample)
            
            if not x_accum:
                _LOGGER.warning("Dataset empty or format unrecognized. Skipping NODE initialization.")
                return
                
            x_input = torch.stack(x_accum)
            
        x_input = x_input.to(device).float()
        
        # 3. Process features (Split -> Embed)
        x_cont = x_input[:, self.numerical_indices].float()
        x_cat = x_input[:, self.categorical_indices].long()
        
        with torch.no_grad():
            x_embedded = self.embedding_layer(x_cont, x_cat)
            
            # 4. Initialize Backbone
            if hasattr(self.backbone, 'initialize'):
                self.backbone.initialize(x_embedded)
                _LOGGER.info("NODE Initialization Complete. Ready to train.")
            else:
                _LOGGER.warning("NODE Backbone does not have an 'initialize' method. Skipping.")
            
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
    

class DragonAutoInt(nn.Module, _ArchitectureHandlerMixin):
    """
    Native implementation of AutoInt (Automatic Feature Interaction Learning).
    
    Maps categorical and continuous features into a shared embedding space,
    then uses Multi-Head Self-Attention to learn high-order feature interactions.
    """
    def __init__(self, *,
                 schema: FeatureSchema,
                 out_targets: int,
                 embedding_dim: int = 32,
                 attn_embed_dim: int = 32,
                 num_heads: int = 2,
                 num_attn_blocks: int = 3,
                 attn_dropout: float = 0.1,
                 has_residuals: bool = True,
                 attention_pooling: bool = True,
                 deep_layers: bool = True,
                 layers: str = "128-64-32",
                 activation: str = "ReLU",
                 embedding_dropout: float = 0.0,
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema: FeatureSchema object.
            out_targets: Number of output targets.
            embedding_dim: Initial embedding dimension for features.
            attn_embed_dim: Projection dimension for the attention mechanism.
            num_heads: Number of attention heads.
            num_attn_blocks: Number of self-attention layers.
            attention_pooling: If True, concatenates outputs of all attention blocks.
            deep_layers: If True, adds an MLP before the attention mechanism.
            layers: Hyphen-separated string for MLP layer sizes (e.g., "128-64").
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        self.model_hparams = {
            'embedding_dim': embedding_dim,
            'attn_embed_dim': attn_embed_dim,
            'num_heads': num_heads,
            'num_attn_blocks': num_attn_blocks,
            'attn_dropout': attn_dropout,
            'has_residuals': has_residuals,
            'attention_pooling': attention_pooling,
            'deep_layers': deep_layers,
            'layers': layers,
            'activation': activation,
            'embedding_dropout': embedding_dropout,
            'batch_norm_continuous': batch_norm_continuous
        }
        
        # -- 1. Setup Embeddings --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        n_continuous = len(self.numerical_indices)
        
        self.embedding_layer = Embedding2dLayer(
            continuous_dim=n_continuous,
            categorical_cardinality=self.cardinalities,
            embedding_dim=embedding_dim,
            embedding_dropout=embedding_dropout,
            batch_norm_continuous_input=batch_norm_continuous
        )
        
        # -- 2. Deep Layers (Optional MLP) --
        curr_units = embedding_dim
        self.deep_layers_mod = None
        
        if deep_layers:
            layers_list = []
            layer_sizes = [int(x) for x in layers.split("-")]
            activation_fn = getattr(nn, activation, nn.ReLU)
            
            for units in layer_sizes:
                layers_list.append(nn.Linear(curr_units, units))
                
                # Changed BatchNorm1d to LayerNorm to handle (Batch, Tokens, Embed) shape correctly
                layers_list.append(nn.LayerNorm(units)) 

                layers_list.append(activation_fn())
                layers_list.append(nn.Dropout(embedding_dropout))
                curr_units = units
            
            self.deep_layers_mod = nn.Sequential(*layers_list)
            
        # -- 3. Attention Backbone --
        self.attn_proj = nn.Linear(curr_units, attn_embed_dim)
        
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=attn_embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout
            )
            for _ in range(num_attn_blocks)
        ])
        
        # Residuals
        self.has_residuals = has_residuals
        self.attention_pooling = attention_pooling
        
        if has_residuals:
            # If pooling, we project input to match the concatenated output size
            # If not pooling, we project input to match the single block output size
            res_dim = attn_embed_dim * num_attn_blocks if attention_pooling else attn_embed_dim
            self.V_res_embedding = nn.Linear(curr_units, res_dim)
            
        # -- 4. Output Dimension Calculation --
        num_features = n_continuous + len(self.cardinalities)
        
        # Output is flattened: (Num_Features * Attn_Dim)
        final_dim = num_features * attn_embed_dim
        if attention_pooling:
            final_dim = final_dim * num_attn_blocks
            
        self.output_dim = final_dim
        self.head = nn.Linear(final_dim, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        # 1. Embed -> (Batch, Num_Features, Embed_Dim)
        x = self.embedding_layer(x_cont, x_cat)
        
        # 2. Deep Layers
        if self.deep_layers_mod:
            x = self.deep_layers_mod(x)
            
        # 3. Attention Projection -> (Batch, Num_Features, Attn_Dim)
        cross_term = self.attn_proj(x)
        
        # Transpose for MultiheadAttention (Seq, Batch, Embed)
        cross_term = cross_term.transpose(0, 1)
        
        attention_ops = []
        for self_attn in self.self_attns:
            # Self Attention: Query=Key=Value=cross_term
            # Output: (Seq, Batch, Embed)
            out, _ = self_attn(cross_term, cross_term, cross_term)
            cross_term = out # Sequential connection
            if self.attention_pooling:
                attention_ops.append(out)
                
        if self.attention_pooling:
            # Concatenate all attention outputs along the embedding dimension
            cross_term = torch.cat(attention_ops, dim=-1)
            
        # Transpose back -> (Batch, Num_Features, Final_Attn_Dim)
        cross_term = cross_term.transpose(0, 1)
        
        # 4. Residual Connection
        if self.has_residuals:
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
            
        # 5. Flatten and Head
        # ReLU before flattening as per original implementation
        cross_term = F.relu(cross_term)
        cross_term = cross_term.reshape(cross_term.size(0), -1)
        
        return self.head(cross_term)
    
    def data_aware_initialization(self, train_dataset, num_samples: int = 2000):
        """
        Performs data-aware initialization for the final head bias.
        """
        # 1. Prepare Data
        _LOGGER.info(f"Performing AutoInt data-aware initialization on up to {num_samples} samples...")
        device = next(self.parameters()).device

        # 2. Extract Targets
        if hasattr(train_dataset, "labels") and isinstance(train_dataset.labels, torch.Tensor):
             limit = min(len(train_dataset.labels), num_samples)
             targets = train_dataset.labels[:limit]
        else:
            indices = range(min(len(train_dataset), num_samples))
            y_accum = []
            for i in indices:
                sample = train_dataset[i]
                # Handle tuple (X, y) or dict
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    y_val = sample[1]
                elif isinstance(sample, dict):
                    y_val = sample.get('target', sample.get('y', None))
                else:
                    y_val = None
                
                if y_val is not None:
                    if not isinstance(y_val, torch.Tensor):
                        y_val = torch.tensor(y_val)
                    y_accum.append(y_val)

            if not y_accum:
                _LOGGER.warning("Could not extract targets for AutoInt initialization. Skipping.")
                return
            
            targets = torch.stack(y_accum)

        targets = targets.to(device).float()
        
        # 3. Initialize Head Bias
        with torch.no_grad():
            mean_target = torch.mean(targets, dim=0)
            if hasattr(self.head, 'bias') and self.head.bias is not None:
                if self.head.bias.shape == mean_target.shape:
                    self.head.bias.data = mean_target
                    _LOGGER.info("AutoInt Initialization Complete. Ready to train.")
                    _LOGGER.debug(f"Initialized AutoInt head bias to {mean_target.cpu().numpy()}")
                elif self.head.bias.numel() == 1 and mean_target.numel() == 1:
                    self.head.bias.data = mean_target.view(self.head.bias.shape)
                    _LOGGER.info("AutoInt Initialization Complete. Ready to train.")
                    _LOGGER.debug(f"Initialized AutoInt head bias to {mean_target.item()}")
            else:
                _LOGGER.warning("AutoInt Head does not have a bias parameter. Skipping initialization.")
    
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


class DragonTabNet(nn.Module, _ArchitectureHandlerMixin):
    """
    Native implementation of TabNet (Attentive Interpretable Tabular Learning).
    
    This implementation faithfully follows the original architecture,
    including the Initial Splitter, Ghost Batch Norm, and GLU scaling.
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
                 momentum: float = 0.02,
                 mask_type: Literal['sparsemax', 'entmax', 'softmax'] = 'sparsemax',
                 batch_norm_continuous: bool = False):
        """
        Args:
            schema: FeatureSchema object.
            out_targets: Number of output targets.
            n_d: Dimension of the prediction layer.
            n_a: Dimension of the attention layer.
            n_steps: Number of sequential attention steps.
            gamma: Relaxation parameter for sparsity.
            n_independent: Number of independent GLU layers in each block.
            n_shared: Number of shared GLU layers in each block.
            virtual_batch_size: Batch size for Ghost Batch Normalization.
            mask_type: Masking function ('sparsemax' is default).
        """
        super().__init__()
        self.schema = schema
        self.out_targets = out_targets
        
        # Save config
        self.model_hparams = {
            'n_d': n_d,
            'n_a': n_a,
            'n_steps': n_steps,
            'gamma': gamma,
            'n_independent': n_independent,
            'n_shared': n_shared,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum,
            'mask_type': mask_type,
            'batch_norm_continuous': batch_norm_continuous
        }
        
        # -- 1. Setup Input Features --
        self.categorical_indices = []
        self.cardinalities = []
        if schema.categorical_index_map:
            self.categorical_indices = list(schema.categorical_index_map.keys())
            self.cardinalities = list(schema.categorical_index_map.values())
        
        all_indices = set(range(len(schema.feature_names)))
        self.numerical_indices = sorted(list(all_indices - set(self.categorical_indices)))
        
        # Standard TabNet Embeddings:
        # We use a simple embedding for each categorical feature and concat with continuous.
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, 1) for card in self.cardinalities
        ])
        
        self.n_continuous = len(self.numerical_indices)
        self.input_dim = self.n_continuous + len(self.cardinalities)
        
        # -- 2. TabNet Backbone Components --
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = 1e-15
        
        # Initial BN
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        # Shared GLU Layers
        if n_shared > 0:
            self.shared_feat_transform = nn.ModuleList()
            for i in range(n_shared):
                if i == 0:
                    self.shared_feat_transform.append(
                        nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    self.shared_feat_transform.append(
                        nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )
        else:
            self.shared_feat_transform = None

        # Initial Splitter
        # This processes the input BEFORE the first step to generate the initial attention vector 'a'
        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            self.shared_feat_transform,
            n_glu_independent=n_independent,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        # Steps
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                self.shared_feat_transform,
                n_glu_independent=n_independent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.input_dim, # We assume group_dim = input_dim (no grouping)
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
                mask_type=mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

        # -- 3. Final Mapping Head --
        self.final_mapping = nn.Linear(n_d, out_targets, bias=False)
        initialize_non_glu(self.final_mapping, n_d, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -- Preprocessing --
        x_cont = x[:, self.numerical_indices].float()
        x_cat = x[:, self.categorical_indices].long()
        
        cat_list = []
        for i, embed in enumerate(self.cat_embeddings):
            cat_list.append(embed(x_cat[:, i])) # (B, 1)
        
        if cat_list:
            x_in = torch.cat([x_cont, *cat_list], dim=1)
        else:
            x_in = x_cont
            
        
        # -- TabNet Encoder Pass --
        x_bn = self.initial_bn(x_in)
        # Initial Split
        # The splitter produces [d, a]. We only need 'a' to start the loop.
        att = self.initial_splitter(x_bn)[:, self.n_d :]
        priors = torch.ones(x_bn.shape, device=x.device)
        out_accumulated = 0
        self.regularization_loss = 0

        for step in range(self.n_steps):
            # 1. Attention
            mask = self.att_transformers[step](priors, att)
            # 2. Accumulate sparsity loss matching original implementation
            loss = torch.sum(torch.mul(mask, torch.log(mask + self.epsilon)), dim=1)
            self.regularization_loss += torch.mean(loss)
            # 3. Update Prior
            priors = torch.mul(self.gamma - mask, priors)
            # 4. Masking
            masked_x = torch.mul(mask, x_bn)
            # 5. Feature Transformer
            out = self.feat_transformers[step](masked_x)
            # 6. Split Output
            d = nn.ReLU()(out[:, :self.n_d])
            att = out[:, self.n_d:]
            # 7. Accumulate Decision
            out_accumulated = out_accumulated + d

        self.regularization_loss /= self.n_steps
        return self.final_mapping(out_accumulated)

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

        if 'schema_dict' not in config:
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

        model = cls(**config)
        if verbose:
            _LOGGER.info(f"Successfully loaded architecture for '{saved_class_name}'")
        return model
    

def info():
    _script_info(__all__)
