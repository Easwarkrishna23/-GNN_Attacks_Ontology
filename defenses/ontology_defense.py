import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity


def build_ontology_matrix(features, labels=None, semantic_weight=0.7):
    """
    Ontology matrix O from semantic similarity, optionally label-guided.
    """
    f = np.asarray(features)
    sim = cosine_similarity(f)
    sim = np.clip(sim, 0.0, 1.0)
    if labels is not None:
        label_mask = (labels[:, None] == labels[None, :]).astype(float)
        sim = semantic_weight * sim + (1.0 - semantic_weight) * label_mask
    row_sum = sim.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return sim / row_sum


def ontology_reweight_adjacency(adj, ontology_matrix, lam=0.3):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    base = adj.toarray().astype(np.float32)
    combined = np.clip(base + lam * ontology_matrix, 0.0, 1.0)
    combined = 0.5 * (combined + combined.T)
    np.fill_diagonal(combined, 1.0)
    return sp.csr_matrix(combined)


def ontology_feature_projection(x, ontology_matrix, lam=0.3):
    """
    H = A_hat XW + lam OX approximation via feature projection term OX.
    """
    x_np = x.detach().cpu().numpy()
    proj = x_np + lam * (ontology_matrix @ x_np)
    proj = np.clip(proj, 0.0, 1.0)
    return torch.tensor(proj, dtype=x.dtype, device=x.device)

