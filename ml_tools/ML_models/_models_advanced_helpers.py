import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import random
import numpy as np
from typing import Optional, Callable

# HELPER module for ML_models_advanced
# SOURCE CODE: Adapted and modified from:
# https://github.com/manujosephv/pytorch_tabular/blob/main/LICENSE
# https://github.com/Qwicen/node/blob/master/LICENSE.md
# https://github.com/jrzaurin/pytorch-widedeep?tab=readme-ov-file#license
# https://github.com/rixwew/pytorch-fm/blob/master/LICENSE
# https://arxiv.org/abs/1705.08741v2


# ==========================================
# 1. UTILITIES & ACTIVATIONS
# ==========================================

def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization == "uniform":
        nn.init.uniform_(x, a=-1, b=1)
    elif initialization == "normal":
        nn.init.normal_(x, std=1)

def check_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim
    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)

def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    if k is None or k >= X.shape[dim]:
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)
        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_
    return tau, support_size

def _entmax_threshold_and_support(X, dim=-1, k=None):
    if k is None or k >= X.shape[dim]:
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)
        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_
    return tau_star, support_size

class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None

class Entmax15Function(Function):
    @classmethod
    def forward(cls, ctx, X, dim=0, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        X = X / 2
        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)
        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None

def sparsemax(X, dim=-1, k=None):
    return SparsemaxFunction.apply(X, dim, k)

def entmax15(X, dim=-1, k=None):
    return Entmax15Function.apply(X, dim, k)

class Entmoid15(Function):
    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input

def entmoid15(input):
    return Entmoid15.apply(input)

def sparsemoid(input):
    return (0.5 * input + 0.5).clamp_(0, 1)

def t_softmax(input: torch.Tensor, t: torch.Tensor = None, dim: int = -1) -> torch.Tensor: # type: ignore
    if t is None:
        t = torch.tensor(0.5, device=input.device)
    maxes = torch.max(input, dim=dim, keepdim=True).values
    input_minus_maxes = input - maxes
    w = torch.relu(input_minus_maxes + t) + 1e-8
    return torch.softmax(input_minus_maxes + torch.log(w), dim=dim)

class RSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    @classmethod
    def calculate_t(cls, input: torch.Tensor, r: torch.Tensor, dim: int = -1, eps: float = 1e-8):
        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes
        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]
        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
             # This quantile logic can be complex in pure PyTorch across dims
             # Simplified fallback or precise implementation
             # For GATE usage, q is usually a scalar or simple tensor
             t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
             t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
             t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: torch.Tensor, r: torch.Tensor):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return t_softmax(input, t, self.dim)


# ==========================================
# 2. LAYERS & BLOCKS
# ==========================================

class GhostBatchNorm1d(nn.Module):
    """
    Simulates Ghost Batch Normalization by processing the batch in chunks.
    """
    def __init__(self, num_features, virtual_batch_size=None, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x):
        if self.virtual_batch_size is None or x.shape[0] <= self.virtual_batch_size:
            return self.bn(x)
        
        # Split into chunks
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, 0)

class Embedding1dLayer(nn.Module):
    """
    Concatenates continuous features (processed by BN) and categorical embeddings.
    """
    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: list[tuple[int, int]],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        self.cat_embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
            
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = GhostBatchNorm1d(continuous_dim, virtual_batch_size)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cont: (B, Cont_Dim), x_cat: (B, Cat_Dim)
        
        embed = None
        # Process Continuous
        if self.continuous_dim > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(x_cont)
            else:
                embed = x_cont
        
        # Process Categorical
        if len(self.categorical_embedding_dims) > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(x_cat[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
                
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed # type: ignore

class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity

        self.W_in = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, 2 * self.n_features_in) for _ in range(self.n_stages)]
        )
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, self.n_features_in) for _ in range(self.n_stages)]
        )
        
        # Create Feature Masks
        # In the original code, they sample random beta. 
        
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        self.feature_masks = nn.Parameter(feature_masks, requires_grad=True)

        if getattr(self.feature_mask_function, "__name__", "") == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_masks, r=torch.tensor([self.feature_sparsity]), dim=-1)
            self.t = nn.Parameter(t, requires_grad=self.learnable_sparsity)
            
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        is_t_softmax = getattr(self.feature_mask_function, "__name__", "") == "t_softmax"
        t = torch.relu(self.t) if is_t_softmax else None
        
        for d in range(self.n_stages):
            if is_t_softmax:
                feature = self.feature_mask_function(self.feature_masks[d], t[d]) * x # type: ignore
            else:
                feature = self.feature_mask_function(self.feature_masks[d]) * x
            
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h

class NeuralDecisionStump(nn.Module):
    def __init__(
        self,
        n_features: int,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.8,
        learnable_sparsity: bool = True,
    ):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        
        alpha = random.uniform(0.5, 10.0)
        beta = random.uniform(0.5, 10.0)
        feature_mask = (
            torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta])).sample((self.n_features,)).squeeze(-1)
        )
        self.feature_mask = nn.Parameter(feature_mask, requires_grad=True)
        
        if getattr(self.feature_mask_function, "__name__", "") == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_mask, r=torch.tensor([feature_sparsity]))
            self.t = nn.Parameter(t, requires_grad=learnable_sparsity)

        W = torch.linspace(1.0, self._num_cutpoints + 1.0, self._num_cutpoints + 1, requires_grad=False).reshape(1, 1, -1)
        self.register_buffer("W", W)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat([torch.zeros([self.n_features, 1], device=cutpoints.device), cutpoints], 1)
        self.cut_points = nn.Parameter(cutpoints, requires_grad=True)
        self.leaf_responses = nn.Parameter(torch.rand(self.n_features, self._num_leaf), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.feature_mask_function, "__name__", "") == "t_softmax":
            t = torch.relu(self.t)
            feature_mask = self.feature_mask_function(self.feature_mask, t)
        else:
            feature_mask = self.feature_mask_function(self.feature_mask)
            
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W # type: ignore
        # Binning features
        # x: (B, Features) -> (B, Features, 1) -> bmm -> (B, Features, Cutpoints+1)
        x_out = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x_out = self.binning_activation(x_out)
        x_out = x_out * self.leaf_responses.unsqueeze(0)
        x_out = (x_out * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x_out, feature_mask # type: ignore

class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        dropout: float = 0,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.8,
        learnable_sparsity: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self._dropout = dropout
        
        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    f"decision_stump_{d}_{n}",
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        binning_activation,
                        feature_mask_function,
                        feature_sparsity,
                        learnable_sparsity,
                    ),
                )
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_input = x
        feature_masks = []
        layer_nodes = None
        for d in range(self.depth):
            layer_nodes_list = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                leaf_nodes, feature_mask = self._modules[f"decision_stump_{d}_{n}"](tree_input) # type: ignore
                layer_nodes_list.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_nodes = torch.cat(layer_nodes_list, dim=1)
            tree_input = torch.cat([x, layer_nodes], dim=1)
            feature_masks.append(layer_feature_masks)
        
        return self.dropout(layer_nodes), feature_masks # type: ignore

class ODST(nn.Module):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_output_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_output_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        self.response = nn.Parameter(torch.zeros([num_trees, tree_output_dim, 2**depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)

    def forward(self, input):
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        
        feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
        feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        bins = self.bin_function(threshold_logits)
        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        response_weights = torch.prod(bin_matches, dim=-2)
        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        
        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # Data-aware initialization
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
            
            # Thresholds
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            
            # Simple python loop to avoid checking numpy/torch specifics heavily
            flat_fv = feature_values.flatten(1, 2).t().cpu().numpy()
            flat_pq = percentiles_q.flatten()
            
            res = []
            for i in range(len(flat_fv)):
                 res.append(np.percentile(flat_fv[i], flat_pq[i]))
            
            self.feature_thresholds.data[...] = torch.as_tensor(
                res, dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # Temperatures
            diff = abs(feature_values - self.feature_thresholds)
            temperatures = np.percentile(
                diff.detach().cpu().numpy(),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures).to(feature_values.device) + eps)

class SimpleLinearHead(nn.Module):
    def __init__(self, in_units: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_units, output_dim)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout)) # type: ignore
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)


class DenseODSTBlock(nn.Sequential):
    def __init__(
        self,
        input_dim,
        num_trees,
        num_layers,
        tree_output_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=False,
        Module=ODST,
        **kwargs,
    ):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, num_trees, tree_output_dim=tree_output_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + num_trees * tree_output_dim, max_features or float("inf"))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = (
            num_layers,
            num_trees,
            tree_output_dim,
        )
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs
    
    def initialize(self, x):
        """
        Data-aware initialization that respects the dense connectivity.
        """
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            # Replicate the feature slicing logic from forward()
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            
            # Initialize the specific ODST layer
            if hasattr(layer, 'initialize'):
                layer.initialize(layer_inp) # type: ignore
            
            # Compute output to feed the next layer in the dense block
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)


class SharedEmbeddings(nn.Module):
    """Enables different values in a categorical feature to share some embeddings across."""
    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
    ):
        super().__init__()
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        
        # Clamp initialization as per pytorch_tabular defaults
        with torch.no_grad():
            self.embed.weight.data.clamp_(-2, 2)
            
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out

class Embedding2dLayer(nn.Module):
    """
    Embeds categorical and continuous features into a 2D tensor (Batch, Num_Features, Embed_Dim).
    """
    def __init__(
        self,
        continuous_dim: int,
        categorical_cardinality: list[int],
        embedding_dim: int,
        shared_embedding_strategy: Optional[str] = None,
        frac_shared_embed: float = 0.25,
        embedding_bias: bool = False,
        batch_norm_continuous_input: bool = False,
        embedding_dropout: float = 0.0,
        initialization: Optional[str] = None,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_cardinality = categorical_cardinality
        self.embedding_dim = embedding_dim
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.shared_embedding_strategy = shared_embedding_strategy
        self.frac_shared_embed = frac_shared_embed
        self.embedding_bias = embedding_bias
        
        # Initialization Helper
        d_sqrt_inv = 1 / math.sqrt(embedding_dim)
        def init_weights(m):
            if initialization == "kaiming_uniform":
                nn.init.uniform_(m, a=-d_sqrt_inv, b=d_sqrt_inv)
            elif initialization == "kaiming_normal":
                nn.init.normal_(m, std=d_sqrt_inv)

        # 1. Categorical Embeddings
        if self.shared_embedding_strategy is not None:
            self.cat_embedding_layers = nn.ModuleList(
                [
                    SharedEmbeddings(
                        c,
                        self.embedding_dim,
                        add_shared_embed=(self.shared_embedding_strategy == "add"),
                        frac_shared_embed=self.frac_shared_embed,
                    )
                    for c in categorical_cardinality
                ]
            )
        else:
            self.cat_embedding_layers = nn.ModuleList(
                [nn.Embedding(c, self.embedding_dim) for c in categorical_cardinality]
            )
        
        # 2. Categorical Bias
        if embedding_bias:
            self.cat_embedding_bias = nn.Parameter(torch.Tensor(len(categorical_cardinality), self.embedding_dim))
            init_weights(self.cat_embedding_bias)

        # 3. Continuous Embedding (Linear projection for scalar -> vector)
        self.cont_embedding_layer = nn.Embedding(self.continuous_dim, self.embedding_dim)
        if embedding_bias:
            self.cont_embedding_bias = nn.Parameter(torch.Tensor(self.continuous_dim, self.embedding_dim))
            init_weights(self.cont_embedding_bias)

        # 4. Batch Norm
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)
            
        # 5. Dropout
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
            
        # Apply initialization to layers
        if initialization:
            for m in self.cat_embedding_layers:
                if isinstance(m, nn.Embedding): init_weights(m.weight)
                elif isinstance(m, SharedEmbeddings): 
                    init_weights(m.embed.weight)
                    init_weights(m.shared_embed)
            init_weights(self.cont_embedding_layer.weight)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cont: (B, N_cont), x_cat: (B, N_cat)
        embed = None
        
        # --- Process Continuous ---
        if self.continuous_dim > 0:
            if self.batch_norm_continuous_input:
                x_cont = self.normalizing_batch_norm(x_cont)
            
            # Continuous embedding usually works by multiplying value * weight_vector
            # Or using an index-based lookup if we treat feature indices as tokens
            cont_idx = torch.arange(self.continuous_dim, device=x_cont.device).expand(x_cont.size(0), -1)
            
            # (B, N_cont, Embed_Dim)
            cont_embed = torch.mul(
                x_cont.unsqueeze(2),
                self.cont_embedding_layer(cont_idx)
            )
            
            if self.embedding_bias:
                cont_embed += self.cont_embedding_bias
            embed = cont_embed

        # --- Process Categorical ---
        if len(self.categorical_cardinality) > 0:
            cat_embed = torch.cat(
                [
                    embedding_layer(x_cat[:, i]).unsqueeze(1)
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1
            )
            if self.embedding_bias:
                cat_embed += self.cat_embedding_bias
            
            if embed is None:
                embed = cat_embed
            else:
                embed = torch.cat([embed, cat_embed], dim=1)
                
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
            
        return embed # type: ignore
    


# ==========================================
# 3. TABNET SPECIFIC BLOCKS (Source Faithful)
# ==========================================

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class GBN(nn.Module):
    """
    Ghost Batch Normalization
    """
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        if self.training:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class GLU_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        super(GLU_Layer, self).__init__()
        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
            initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out

class GLU_Block(nn.Module):
    """
    Independent GLU block, specific to each step
    """
    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None, virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x

class FeatTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size=128, momentum=0.02):
        super(FeatTransformer, self).__init__()
        params = {"n_glu": n_glu_independent, "virtual_batch_size": virtual_batch_size, "momentum": momentum}

        if shared_layers is None:
            self.shared = nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            self.specifics = nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(spec_input_dim, output_dim, first=is_first, **params)

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x

class Sparsemax(nn.Module):
    def __init__(self, dim=-1, k=None):
        super().__init__()
        self.dim = dim
        self.k = k

    def forward(self, input):
        return sparsemax(input, dim=self.dim, k=self.k)

class Entmax15(nn.Module):
    def __init__(self, dim=-1, k=None):
        super().__init__()
        self.dim = dim
        self.k = k

    def forward(self, input):
        return entmax15(input, dim=self.dim, k=self.k)

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, group_dim, virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GBN(group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        if mask_type == "sparsemax":
            self.selector = Sparsemax(dim=-1) # Uses _model_helpers Sparsemax
        elif mask_type == "entmax":
            self.selector = Entmax15(dim=-1) # Uses _model_helpers Entmax15
        else:
            self.selector = nn.Softmax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x

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
                [h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)], # type: ignore
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
