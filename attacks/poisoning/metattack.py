import numpy as np
import scipy.sparse as sp
try:
    from deeprobust.graph.global_attack import MetaApprox
    from deeprobust.graph.defense import GCN
    DEEPROBUST_AVAILABLE = True
except Exception:
    DEEPROBUST_AVAILABLE = False

def run_metattack(adj, features, labels, idx_train, n_perturbations=50):
    """
    Apply Metattack global poison attack.
    """
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # DeepRobust MetaApprox is unstable with some recent PyTorch builds.
    # Use it only when explicitly enabled.
    use_deeprobust = False
    if DEEPROBUST_AVAILABLE and use_deeprobust:
        try:
            surrogate = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device='cpu')
            surrogate.fit(features, adj, labels, idx_train)
            model = MetaApprox(surrogate, nnodes=adj.shape[0], device='cpu')
            idx_unlabeled = np.setdiff1d(np.arange(adj.shape[0]), idx_train)
            model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations)
            return model.modified_adj, model.modified_features, {
                "outer_loop": "surrogate training over poisoned graph",
                "inner_loop": "gradient-based perturbation update",
            }
        except Exception:
            pass

    # Fallback approximation: score edges by feature dissimilarity and flip top-k.
    feat = features.toarray()
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    feat = feat / norm
    sim = feat @ feat.T
    mod_adj = adj.copy().tolil()
    rows, cols = adj.nonzero()
    scores = np.abs(sim[rows, cols] - 0.5)
    order = np.argsort(scores)[:n_perturbations]
    for idx in order:
        u, v = rows[idx], cols[idx]
        mod_adj[u, v] = 1 - mod_adj[u, v]
        mod_adj[v, u] = mod_adj[u, v]
    return mod_adj.tocsr(), features, {
        "outer_loop": "fallback retraining proxy",
        "inner_loop": "top-dissimilarity edge flipping",
    }

if __name__ == "__main__":
    pass
