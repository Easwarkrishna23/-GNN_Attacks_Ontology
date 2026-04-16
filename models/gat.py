import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_self_loops(edge_index, num_nodes):
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    loops = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loops], dim=1)


class GAT(nn.Module):
    """
    Pure-PyTorch 2-layer GAT (Velickovic et al.) without torch_geometric.
    Notes:
    - Implemented for small/medium graphs like Cora (E ~ 5k) on CPU.
    - Uses scatter_reduce-based softmax over incoming edges.
    """

    def __init__(self, num_features, num_hidden, num_classes, heads=4, dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

        # Layer 1: multi-head (concat)
        self.W1 = nn.Linear(num_features, num_hidden * self.heads, bias=False)
        self.a1 = nn.Parameter(torch.empty(self.heads, 2 * num_hidden))

        # Layer 2: single-head output
        self.W2 = nn.Linear(num_hidden * self.heads, num_classes, bias=False)
        self.a2 = nn.Parameter(torch.empty(1, 2 * num_classes))

        self.bias1 = nn.Parameter(torch.zeros(num_hidden * self.heads))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a1)
        nn.init.xavier_uniform_(self.a2)

    def _edge_softmax(self, e, dst, num_nodes):
        # e: (E,) scores per edge, normalize over incoming edges per dst
        max_per_dst = torch.full((num_nodes,), float("-inf"), device=e.device, dtype=e.dtype)
        max_per_dst = max_per_dst.scatter_reduce(0, dst, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[dst])
        sum_per_dst = torch.zeros((num_nodes,), device=e.device, dtype=e.dtype)
        sum_per_dst.scatter_add_(0, dst, exp_e)
        return exp_e / (sum_per_dst[dst] + 1e-12)

    def _attn_layer(self, x, edge_index, W, a, bias, heads, out_dim, concat=True, edge_weight=None):
        # x: (N, Fin), edge_index: (2, E), edge_weight: (E,) optional trust weights
        N = x.size(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = W(x)  # (N, heads*out_dim)
        h = h.view(N, heads, out_dim)  # (N, heads, out_dim)

        # Ensure self-loops; extend weights accordingly.
        if edge_weight is not None:
            ew = edge_weight
            if ew.dim() != 1 or ew.numel() != edge_index.size(1):
                raise ValueError("edge_weight must be shape (E,) aligned with edge_index.")
            loop_w = torch.ones((N,), device=ew.device, dtype=ew.dtype)
            edge_weight = torch.cat([ew, loop_w], dim=0)
        edge_index = _ensure_self_loops(edge_index, N)
        src = edge_index[0]
        dst = edge_index[1]

        hs = h[src]  # (E, heads, out_dim)
        hd = h[dst]  # (E, heads, out_dim)
        cat = torch.cat([hs, hd], dim=-1)  # (E, heads, 2*out_dim)
        # a: (heads, 2*out_dim)
        e = (cat * a.unsqueeze(0)).sum(dim=-1)  # (E, heads)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Semantic edge trust: incorporate as an additive log-prior before softmax.
        # This preserves the attention normalization while down-weighting untrusted edges.
        if edge_weight is not None:
            e = e + torch.log(edge_weight.clamp_min(1e-12)).unsqueeze(1)

        # softmax per dst node for each head (loop heads; small graph)
        alphas = []
        for k in range(heads):
            alpha_k = self._edge_softmax(e[:, k], dst, N)
            alphas.append(alpha_k)
        alpha = torch.stack(alphas, dim=1)  # (E, heads)

        # Aggregate: sum over incoming edges per dst
        out = torch.zeros((N, heads, out_dim), device=x.device, dtype=x.dtype)
        msg = alpha.unsqueeze(-1) * hs  # (E, heads, out_dim)
        for k in range(heads):
            out[:, k, :].index_add_(0, dst, msg[:, k, :])

        out = out.reshape(N, heads * out_dim) if concat else out.mean(dim=1)
        out = out + bias
        return out, (edge_index, alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Support layer-wise edge trust weights (used by base-paper defense).
        ew1 = getattr(data, "edge_weight_l1", None)
        ew2 = getattr(data, "edge_weight_l2", None)
        edge_weight = getattr(data, "edge_weight", None)
        h1, _ = self._attn_layer(
            x,
            edge_index,
            self.W1,
            self.a1,
            self.bias1,
            heads=self.heads,
            out_dim=self.W1.out_features // self.heads,
            concat=True,
            edge_weight=ew1 if ew1 is not None else edge_weight,
        )
        h1 = F.elu(h1)
        logits, _ = self._attn_layer(
            h1,
            edge_index,
            self.W2,
            self.a2,
            self.bias2,
            heads=1,
            out_dim=self.W2.out_features,
            concat=False,
            edge_weight=ew2 if ew2 is not None else edge_weight,
        )
        return F.log_softmax(logits, dim=1)

    def forward_with_debug(self, data):
        x, edge_index = data.x, data.edge_index
        ew1 = getattr(data, "edge_weight_l1", None)
        ew2 = getattr(data, "edge_weight_l2", None)
        edge_weight = getattr(data, "edge_weight", None)

        def mem_mb(tensor):
            return float((tensor.numel() * tensor.element_size()) / (1024**2))

        h1, (ei1, alpha1) = self._attn_layer(
            x,
            edge_index,
            self.W1,
            self.a1,
            self.bias1,
            heads=self.heads,
            out_dim=self.W1.out_features // self.heads,
            concat=True,
            edge_weight=ew1 if ew1 is not None else edge_weight,
        )
        h1_act = F.elu(h1)
        logits, (ei2, alpha2) = self._attn_layer(
            h1_act,
            edge_index,
            self.W2,
            self.a2,
            self.bias2,
            heads=1,
            out_dim=self.W2.out_features,
            concat=False,
            edge_weight=ew2 if ew2 is not None else edge_weight,
        )
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_memory = {
            "input_x_mb": mem_mb(x),
            "hidden_after_attention_mb": mem_mb(h1_act),
            "attention_alpha_mb": mem_mb(alpha1),
            "logits_mb": mem_mb(logits),
            "softmax_probabilities_mb": mem_mb(probs),
        }
        return {
            "attention_edge_index": ei1.detach(),
            "attention_weights": alpha1.detach(),
            "node_aggregation_hidden": h1_act.detach(),
            "logits": logits.detach(),
            "softmax_probabilities": probs.detach(),
            "predictions": preds.detach(),
            "memory": {
                "parameter_memory_mb": float(param_bytes / (1024**2)),
                "activation_memory_mb": activation_memory,
                "total_estimated_mb": float(param_bytes / (1024**2)) + sum(activation_memory.values()),
            },
        }

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        ew1 = getattr(data, "edge_weight_l1", None)
        edge_weight = getattr(data, "edge_weight", None)
        h1, _ = self._attn_layer(
            x,
            edge_index,
            self.W1,
            self.a1,
            self.bias1,
            heads=self.heads,
            out_dim=self.W1.out_features // self.heads,
            concat=True,
            edge_weight=ew1 if ew1 is not None else edge_weight,
        )
        return F.elu(h1)

    def get_attention_weights(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        _, (ei, alpha) = self._attn_layer(
            x,
            edge_index,
            self.W1,
            self.a1,
            self.bias1,
            heads=self.heads,
            out_dim=self.W1.out_features // self.heads,
            concat=True,
            edge_weight=edge_weight,
        )
        return ei, alpha
