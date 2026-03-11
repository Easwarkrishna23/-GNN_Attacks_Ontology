import numpy as np
import scipy.sparse as sp
try:
    from deeprobust.graph.targeted_attack import Nettack
    from deeprobust.graph.defense import GCN
    DEEPROBUST_AVAILABLE = True
except Exception:
    DEEPROBUST_AVAILABLE = False

def get_surrogate(adj, features, labels, idx_train):
    """
    Train a surrogate GCN for Nettack.
    """
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    if not DEEPROBUST_AVAILABLE:
        return None
    surrogate = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device='cpu')
    surrogate.fit(features, adj, labels, idx_train)
    return surrogate

def run_nettack(surrogate, adj, features, labels, target_node, n_perturbations=5):
    """
    Apply Nettack poison attack on a target node using a pre-trained surrogate.
    """
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    if DEEPROBUST_AVAILABLE and surrogate is not None:
        model = Nettack(surrogate, nnodes=adj.shape[0], device='cpu')
        model.attack(features, adj, labels, target_node, n_perturbations)
        return model.modified_adj, model.modified_features, {
            "margin_reduction_proxy": float(n_perturbations),
            "perturbation_score_proxy": float(n_perturbations / max(1, adj.shape[0])),
        }

    # Fallback: targeted edge perturbation around target node.
    mod_adj = adj.copy().tolil()
    rng = np.random.default_rng(42 + int(target_node))
    n_nodes = adj.shape[0]
    changes = 0
    while changes < n_perturbations:
        v = int(rng.integers(0, n_nodes))
        if v == target_node:
            continue
        mod_adj[target_node, v] = 1 - mod_adj[target_node, v]
        mod_adj[v, target_node] = mod_adj[target_node, v]
        changes += 1
    return mod_adj.tocsr(), features, {
        "margin_reduction_proxy": float(changes),
        "perturbation_score_proxy": float(changes / max(1, n_nodes)),
    }

if __name__ == "__main__":
    pass
