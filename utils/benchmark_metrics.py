import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score


def edge_index_to_csr(edge_index, num_nodes: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def csr_to_edge_index(adj: sp.csr_matrix):
    coo = adj.tocoo()
    import torch

    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data.astype(np.float32), dtype=torch.float32)
    return edge_index, edge_weight


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def nettack_asr(y_true, pred_clean, pred_attack, target_nodes) -> float:
    targets = np.asarray(target_nodes, dtype=np.int64)
    if targets.size == 0:
        return 0.0
    clean_ok = pred_clean[targets] == y_true[targets]
    if clean_ok.sum() == 0:
        return 0.0
    success = (pred_attack[targets] != y_true[targets]) & clean_ok
    return float(success.sum() / clean_ok.sum())


def homophily_ratio(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    row, col = adj.nonzero()
    if row.size == 0:
        return 0.0
    return float(np.mean(labels[row] == labels[col]))


def homophily_recovery_ratio(h_clean: float, h_attack: float, h_defended: float, eps: float = 1e-9) -> float:
    if abs(h_clean - h_attack) < eps:
        return 0.0
    denom = max(eps, h_clean - h_attack)
    return float((h_defended - h_attack) / denom)
