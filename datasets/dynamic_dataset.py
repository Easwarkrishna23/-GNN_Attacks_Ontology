from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    vals = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def csr_to_edge_index(adj: sp.csr_matrix) -> torch.Tensor:
    coo = adj.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def simulate_dynamic_snapshots(base_data: Data, num_snapshots: int = 4, edge_change_rate: float = 0.01, seed: int = 42):
    """
    Create evolving snapshots from a base static graph by edge rewiring.
    Features and labels are preserved; only topology evolves.
    """
    rng = np.random.default_rng(seed)
    adj0 = edge_index_to_csr(base_data.edge_index, base_data.num_nodes)
    snapshots = []

    current = adj0.copy().tolil()
    n = current.shape[0]
    n_changes = max(1, int(edge_change_rate * current.nnz / 2))

    for t in range(num_snapshots):
        # Rewire a small number of edges each snapshot.
        for _ in range(n_changes):
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            if current[u, v] != 0:
                current[u, v] = 0
                current[v, u] = 0
            else:
                current[u, v] = 1
                current[v, u] = 1

        adj_t = current.tocsr()
        adj_t = adj_t.maximum(adj_t.T)
        adj_t.setdiag(0)
        adj_t.eliminate_zeros()

        data_t = Data(
            x=base_data.x.clone(),
            y=base_data.y.clone(),
            edge_index=csr_to_edge_index(adj_t),
            train_mask=base_data.train_mask.clone(),
            val_mask=base_data.val_mask.clone(),
            test_mask=base_data.test_mask.clone(),
        )
        snapshots.append(data_t)

    return snapshots
