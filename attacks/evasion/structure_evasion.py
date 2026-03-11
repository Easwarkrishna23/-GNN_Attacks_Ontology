import numpy as np
import scipy.sparse as sp


def run_structure_evasion(adj, target_node, n_perturbations=10, seed=42):
    """
    Degree-preserving test-time structure perturbation around a target node.
    Keeps the target node degree approximately unchanged by removing one
    existing edge and adding one non-edge per perturbation.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    mod_adj = adj.copy().tolil()
    rng = np.random.default_rng(seed + int(target_node))
    n_nodes = adj.shape[0]

    for _ in range(n_perturbations):
        neighbors = np.array(mod_adj.rows[target_node], dtype=int)
        if neighbors.size == 0:
            break
        to_remove = int(rng.choice(neighbors))

        non_neighbors = np.setdiff1d(np.arange(n_nodes), np.append(neighbors, target_node))
        if non_neighbors.size == 0:
            break
        to_add = int(rng.choice(non_neighbors))

        mod_adj[target_node, to_remove] = 0
        mod_adj[to_remove, target_node] = 0
        mod_adj[target_node, to_add] = 1
        mod_adj[to_add, target_node] = 1

    return mod_adj.tocsr()

