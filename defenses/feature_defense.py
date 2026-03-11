import numpy as np
import scipy.sparse as sp
import torch


def normalized_adj_with_self_loops(adj):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_hat = adj + sp.eye(adj.shape[0], dtype=np.float32)
    degree = np.array(adj_hat.sum(axis=1)).flatten()
    d_inv = np.power(degree, -1.0, where=degree > 0)
    d_inv[~np.isfinite(d_inv)] = 0.0
    d_inv = sp.diags(d_inv)
    return d_inv @ adj_hat


def laplacian_feature_smoothing(x, adj, alpha=0.5):
    """
    X_smooth = alpha X + (1-alpha) A_hat X
    """
    a_hat = normalized_adj_with_self_loops(adj)
    x_np = x.detach().cpu().numpy()
    x_smooth = alpha * x_np + (1.0 - alpha) * (a_hat @ x_np)
    return torch.tensor(x_smooth, dtype=x.dtype, device=x.device)


def feature_consistency_regularization(x, adj):
    """
    ||X - A_hat X||^2
    """
    a_hat = normalized_adj_with_self_loops(adj)
    x_np = x.detach().cpu().numpy()
    residual = x_np - (a_hat @ x_np)
    return float(np.mean(residual ** 2))

