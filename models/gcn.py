import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _scipy_to_torch_sparse(mat: sp.csr_matrix, device=None, dtype=torch.float32):
    mat = mat.tocoo()
    idx = torch.tensor(np.vstack([mat.row, mat.col]), dtype=torch.long, device=device)
    val = torch.tensor(mat.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=mat.shape, device=device, dtype=dtype).coalesce()


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, edge_weight: Optional[torch.Tensor] = None) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    if edge_weight is None:
        val = np.ones(edge_index.size(1), dtype=np.float32)
    else:
        val = edge_weight.detach().cpu().numpy().astype(np.float32)
    adj = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.tocsr()
    # Symmetrize: use max to preserve higher trust weights when one direction is missing.
    adj = adj.maximum(adj.T)
    # Always add self-loops with weight=1.
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="csr")
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    d_inv_sqrt = sp.diags(deg_inv_sqrt.astype(np.float32))
    return (d_inv_sqrt @ adj @ d_inv_sqrt).tocsr()


class GCN(nn.Module):
    """
    Pure-PyTorch 2-layer GCN (Kipf & Welling) that does NOT require torch_geometric.
    """

    def __init__(self, num_features, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(num_features, num_hidden, bias=True)
        self.lin2 = nn.Linear(num_hidden, num_classes, bias=True)
        self.dropout = float(dropout)

    def _get_S(self, data, layer: int = 1):
        """
        Get normalized adjacency for a specific layer.

        This repo supports defenses that produce *layer-specific* edge weights:
          - data.edge_weight_l1 for layer-1 message passing
          - data.edge_weight_l2 for layer-2 message passing
        Fallbacks:
          - data.edge_weight (shared for both layers)
          - unit weights (None)
        """
        if layer not in (1, 2):
            raise ValueError("layer must be 1 or 2")

        # Separate caches for each layer (edge weights can differ).
        cache_sp_name = "_S1_sp" if layer == 1 else "_S2_sp"
        cache_hash_name = "_S1_edge_hash" if layer == 1 else "_S2_edge_hash"

        if not hasattr(data, cache_sp_name) or not hasattr(data, cache_hash_name):
            setattr(data, cache_sp_name, None)
            setattr(data, cache_hash_name, None)

        edge_hash = int(torch.sum(data.edge_index).item()) + int(data.edge_index.numel()) + int(layer * 1000003)
        ew = None
        if layer == 1 and getattr(data, "edge_weight_l1", None) is not None:
            ew = data.edge_weight_l1
        elif layer == 2 and getattr(data, "edge_weight_l2", None) is not None:
            ew = data.edge_weight_l2
        else:
            ew = getattr(data, "edge_weight", None)

        if ew is not None:
            edge_hash += int(torch.sum(ew).item() * 1000) + int(ew.numel())

        cached = getattr(data, cache_sp_name)
        cached_hash = getattr(data, cache_hash_name)
        if cached is None or cached_hash != edge_hash:
            S = normalized_adjacency(data.edge_index, data.num_nodes, edge_weight=ew)
            setattr(data, cache_sp_name, S)
            setattr(data, cache_hash_name, edge_hash)
        return getattr(data, cache_sp_name)

    def forward(self, data):
        x = data.x
        S1_sp = self._get_S(data, layer=1)
        S2_sp = self._get_S(data, layer=2)
        S1 = _scipy_to_torch_sparse(S1_sp, device=x.device, dtype=x.dtype)
        S2 = _scipy_to_torch_sparse(S2_sp, device=x.device, dtype=x.dtype)

        h0 = self.lin1(x)
        h1 = torch.sparse.mm(S1, h0)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.lin2(h1)
        out = torch.sparse.mm(S2, h2)
        return F.log_softmax(out, dim=1)

    def forward_with_debug(self, data):
        x = data.x
        device = x.device
        S1_sp = self._get_S(data, layer=1)
        S2_sp = self._get_S(data, layer=2)
        S1 = _scipy_to_torch_sparse(S1_sp, device=device, dtype=x.dtype)
        S2 = _scipy_to_torch_sparse(S2_sp, device=device, dtype=x.dtype)

        def mem_mb(tensor):
            return float((tensor.numel() * tensor.element_size()) / (1024**2))

        pre_agg_1 = x @ self.lin1.weight.t()
        post_norm_1 = torch.sparse.mm(S1, pre_agg_1) + self.lin1.bias
        post_act_1 = F.relu(post_norm_1)
        pre_agg_2 = post_act_1 @ self.lin2.weight.t()
        post_norm_2 = torch.sparse.mm(S2, pre_agg_2) + self.lin2.bias
        probs = torch.softmax(post_norm_2, dim=1)
        preds = probs.argmax(dim=1)

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_memory = {
            "input_x_mb": mem_mb(x),
            "pre_aggregation_l1_mb": mem_mb(pre_agg_1),
            "post_normalization_l1_mb": mem_mb(post_norm_1),
            "post_activation_l1_mb": mem_mb(post_act_1),
            "pre_aggregation_l2_mb": mem_mb(pre_agg_2),
            "post_normalization_l2_mb": mem_mb(post_norm_2),
            "softmax_probabilities_mb": mem_mb(probs),
        }

        return {
            "pre_aggregation_l1": pre_agg_1.detach(),
            "post_normalization_l1": post_norm_1.detach(),
            "post_activation_l1": post_act_1.detach(),
            "pre_aggregation_l2": pre_agg_2.detach(),
            "post_normalization_l2": post_norm_2.detach(),
            "softmax_probabilities": probs.detach(),
            "predictions": preds.detach(),
            "memory": {
                "parameter_memory_mb": float(param_bytes / (1024**2)),
                "activation_memory_mb": activation_memory,
                "total_estimated_mb": float(param_bytes / (1024**2)) + sum(activation_memory.values()),
            },
        }

    def get_embeddings(self, data):
        x = data.x
        S1_sp = self._get_S(data, layer=1)
        S1 = _scipy_to_torch_sparse(S1_sp, device=x.device, dtype=x.dtype)
        h0 = self.lin1(x)
        h1 = torch.sparse.mm(S1, h0)
        return F.relu(h1)
