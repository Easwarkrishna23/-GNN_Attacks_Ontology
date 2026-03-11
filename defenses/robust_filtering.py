import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity(features):
    """
    Compute Jaccard similarity for binary/discrete features.
    """
    features = sp.csr_matrix(features)
    intersection = features.dot(features.T)
    row_sums = np.array(features.sum(axis=1)).flatten()
    union = row_sums[:, None] + row_sums[None, :] - intersection
    # Avoid division by zero
    union[union == 0] = 1.0
    return intersection / union

def svd_defense(adj, k=50):
    """
    Apply low-rank approximation using Truncated SVD to filter structural noise.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # Standard SVD for sparse matrices
    adj = adj.asfptype()
    u, s, vt = sp.linalg.svds(adj, k=k)
    s_mat = np.diag(s)
    adj_approx = u @ s_mat @ vt
    
    # Clip and sparsify (keep structure, refine weights)
    adj_approx = np.clip(adj_approx, 0, 1)
    # Threshold very low values to maintain semi-sparsity without killingconnectivity
    adj_approx[adj_approx < 0.01] = 0 
    return sp.csr_matrix(adj_approx)

def embedding_similarity_weighting(adj, embeddings, threshold=0.1):
    """
    Use latent embeddings from a surrogate model to weight edges.
    Latent space often captures semantic relationships better than raw features.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # Normalize embeddings
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings_norm = embeddings / norm
    
    # Dot product for cosine similarity
    sim = embeddings_norm @ embeddings_norm.T
    sim = np.power(np.clip(sim, 0, 1), 3) # Sharpening

    adj_dense = adj.toarray()
    weighted_adj = adj_dense * sim
    weighted_adj[weighted_adj < threshold] = 0
    
    return sp.csr_matrix(weighted_adj)

def similarity_weighted_adj(adj, features, metric='cosine', threshold=0.001):
    """
    Re-weight the adjacency matrix using feature similarity.
    Provides soft scores to help the GCN distinguish between clean and noisy edges.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # Normalize features for cosine similarity
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    features_norm = features / norm
    sim = features_norm @ features_norm.T
    
    # Square the similarity to sharpen the difference between good and bad edges
    sim = np.power(np.clip(sim, 0, 1), 2)

    adj_dense = adj.toarray()
    weighted_adj = adj_dense * sim
    
    # Very low threshold to keep connectivity but demote noise
    weighted_adj[weighted_adj < threshold] = 0
    return sp.csr_matrix(weighted_adj)

def label_guided_pruning(adj, labels, threshold=0.5):
    """
    Prune edges between nodes that have different labels.
    If labels are ground truth (poisoning check) or predicted (self-defense).
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    adj = adj.tocsr().tolil()
    rows, cols = adj.nonzero()
    for u, v in zip(rows, cols):
        if labels[u] != labels[v]:
            adj[u, v] *= threshold # Soft prune
            
    return adj.tocsr()

def top_k_pruning(adj, features, k=10):
    """
    For each node, keep only the top-k most similar neighbors.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # Compute similarity efficiently
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    features_norm = features / norm
    
    adj = adj.tocsr().tolil()
    new_adj = sp.lil_matrix(adj.shape)
    
    rows, cols = adj.nonzero()
    # Process by node to find top-k
    for i in range(adj.shape[0]):
        neighbors = adj.rows[i]
        if not neighbors: continue
        
        # Get similarities for neighbors only
        neighbor_features = features_norm[neighbors]
        node_feature = features_norm[i]
        sims = neighbor_features @ node_feature
        
        # Take top-k
        top_k_idx = np.argsort(sims)[-k:]
        for idx in top_k_idx:
            neighbor = neighbors[idx]
            new_adj[i, neighbor] = adj[i, neighbor]
            
    return new_adj.tocsr()

def robust_edge_filtering(adj, features, labels_guidance=None, probs_guidance=None, embeddings_guidance=None, threshold=0.01):
    """
    High-Fidelity Reliability Defense (Normalized):
    1. Confidence-Aware Trust Scoring
    2. Latent Semantic Alignment (Power 5)
    3. Spectral Signal Recovery (SVD k=140)
    4. Weight Normalization for GCN Stability
    """
    print(f"Applying High-Fidelity Reliability Defense (Target 80%)...")
    
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    # 1. Latent Similarity Weights
    if embeddings_guidance is not None:
        norm = np.linalg.norm(embeddings_guidance, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        emb_norm = embeddings_guidance / norm
        weights = np.power(np.clip(emb_norm @ emb_norm.T, 0, 1), 5) 
    else:
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        feat_norm = features / norm
        weights = np.power(np.clip(feat_norm @ feat_norm.T, 0, 1), 3)

    # 2. Confidence Guidance
    if labels_guidance is not None and probs_guidance is not None:
        max_probs = np.max(probs_guidance, axis=1)
        # Scale confidence to [0.5, 1.0] range to avoid killing signal
        conf_score = 0.5 + 0.5 * max_probs
        conf_mat = conf_score[:, None] * conf_score[None, :]
        label_match = (labels_guidance[:, None] == labels_guidance[None, :]).astype(float)
        # Penalize inter-class edges heavily
        weights = weights * (0.1 + 0.9 * label_match) * conf_mat

    # 3. Spectral Recovery
    adj_denoised = svd_defense(adj, k=140)
    
    # 4. Integrate
    adj_final_dense = adj_denoised.toarray() * weights
    # Re-weight original edges
    adj_final_dense = 0.8 * adj_final_dense + 0.2 * (adj.toarray() * weights)
    
    # 5. Normalization: Ensure max weight is 1.0 to keep GCN gradient stable
    max_w = np.max(adj_final_dense)
    if max_w > 0:
        adj_final_dense = adj_final_dense / max_w
    
    # Prune noise
    adj_final_dense[adj_final_dense < 0.05] = 0
    
    # 6. Final Connectivity check
    for i in range(adj_final_dense.shape[0]):
        if np.sum(adj_final_dense[i]) == 0:
            adj_final_dense[i, i] = 1.0
            
    return sp.csr_matrix(adj_final_dense)

def feature_smoothing(features, adj, alpha=0.5):
    """
    Apply graph-based feature smoothing.
    X_smooth = (1-alpha)X + alpha * D^-1 A X
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    
    adj_normalized = preprocess_adj(adj)
    
    if torch.is_tensor(features):
        feat_np = features.cpu().numpy()
        feat_smooth = (1-alpha) * feat_np + alpha * adj_normalized.dot(feat_np)
        return torch.tensor(feat_smooth, dtype=torch.float, device=features.device)
    else:
        return (1-alpha) * features + alpha * adj_normalized.dot(features)

def preprocess_adj(adj):
    """
    Row-normalize adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj)

if __name__ == "__main__":
    pass
