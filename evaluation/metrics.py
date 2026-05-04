from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import networkx as nx


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            out["roc_auc_ovr"] = 0.0
    else:
        out["roc_auc_ovr"] = 0.0
    return out


def robustness_score(clean_acc: float, scenario_acc: float, eps: float = 1e-9) -> float:
    return float(max(0.0, scenario_acc) / max(eps, clean_acc))


def attack_success_rate(y_true: np.ndarray, pred_clean: np.ndarray, pred_attack: np.ndarray, targets: np.ndarray) -> float:
    if targets.size == 0:
        return 0.0
    clean_correct = pred_clean[targets] == y_true[targets]
    if clean_correct.sum() == 0:
        return 0.0
    success = (pred_attack[targets] != y_true[targets]) & clean_correct
    return float(success.sum() / clean_correct.sum())


def edge_index_to_csr(edge_index, num_nodes: int) -> sp.csr_matrix:
    import torch

    if isinstance(edge_index, torch.Tensor):
        row = edge_index[0].detach().cpu().numpy()
        col = edge_index[1].detach().cpu().numpy()
    else:
        row = edge_index[0]
        col = edge_index[1]

    vals = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def homophily_ratio(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    r, c = adj.nonzero()
    if r.size == 0:
        return 0.0
    return float(np.mean(labels[r] == labels[c]))


def homophily_recovery_ratio(clean_h: float, attacked_h: float, defended_h: float, eps: float = 1e-9) -> float:
    if abs(clean_h - attacked_h) < eps:
        return 0.0
    return float((defended_h - attacked_h) / max(eps, clean_h - attacked_h))


def graph_density(adj: sp.csr_matrix) -> float:
    n = int(adj.shape[0])
    if n <= 1:
        return 0.0
    edges_undirected = float(adj.nnz / 2.0)
    return float((2.0 * edges_undirected) / (n * (n - 1)))


def graph_modularity(adj: sp.csr_matrix, labels: np.ndarray | None = None) -> float:
    g = nx.from_scipy_sparse_array(adj)
    if g.number_of_edges() == 0 or g.number_of_nodes() <= 2:
        return 0.0
    try:
        if labels is not None and len(labels) == g.number_of_nodes():
            communities = []
            for c in np.unique(labels):
                idx = np.where(labels == c)[0]
                if idx.size > 0:
                    communities.append(set(int(i) for i in idx))
        else:
            communities = nx.algorithms.community.greedy_modularity_communities(g)
        return float(nx.algorithms.community.modularity(g, communities))
    except Exception:
        return 0.0


def graph_conductance_by_labels(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """
    Average class conductance:
    phi(S) = cut(S, ~S) / min(vol(S), vol(~S))
    """
    n = adj.shape[0]
    if n == 0:
        return 0.0
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    total_vol = float(deg.sum())
    if total_vol <= 0:
        return 0.0

    phis = []
    for c in np.unique(labels):
        mask = labels == c
        if not np.any(mask) or np.all(mask):
            continue
        s_idx = np.where(mask)[0]
        t_idx = np.where(~mask)[0]
        cut = float(adj[s_idx][:, t_idx].sum())
        vol_s = float(deg[s_idx].sum())
        vol_t = total_vol - vol_s
        denom = min(vol_s, vol_t)
        if denom > 0:
            phis.append(cut / denom)
    if not phis:
        return 0.0
    return float(np.mean(phis))
