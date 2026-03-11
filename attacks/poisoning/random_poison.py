import numpy as np
import scipy.sparse as sp

def run_random_attack(adj, features=None, n_edge_perturbations=50, feature_corruption_rate=0.0, seed=42):
    """
    Random poisoning attack:
    1) random edge rewiring/insertion
    2) optional random feature corruption
    """
    rng = np.random.default_rng(seed)
    modified_adj = adj.copy().tolil()
    n_nodes = adj.shape[0]
    
    count = 0
    while count < n_edge_perturbations:
        u, v = rng.integers(0, n_nodes, size=2)
        if u != v and modified_adj[u, v] == 0:
            modified_adj[u, v] = 1
            modified_adj[v, u] = 1
            count += 1

    modified_features = None
    if features is not None:
        modified_features = np.array(features, copy=True)
        if feature_corruption_rate > 0:
            n_total = modified_features.size
            n_flip = int(feature_corruption_rate * n_total)
            if n_flip > 0:
                flat_idx = rng.choice(n_total, size=n_flip, replace=False)
                flat = modified_features.reshape(-1)
                # Flip binary-like values, otherwise inject bounded noise.
                binary_mask = np.isin(flat[flat_idx], [0.0, 1.0])
                flat_vals = flat[flat_idx]
                flat_vals[binary_mask] = 1.0 - flat_vals[binary_mask]
                flat_vals[~binary_mask] = np.clip(
                    flat_vals[~binary_mask] + rng.normal(0, 0.1, size=np.count_nonzero(~binary_mask)),
                    0.0,
                    1.0,
                )
                flat[flat_idx] = flat_vals
                modified_features = flat.reshape(modified_features.shape)

    return modified_adj.tocsr(), modified_features

if __name__ == "__main__":
    pass
