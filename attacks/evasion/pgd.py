import torch
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
import numpy as np

def run_pgd_attack(adj, features, labels, idx_train, n_perturbations=50, steps=20):
    """
    Custom PGD evasion attack (iterative gradient-based structural flipping).
    """
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
        
    # Surrogate GCN
    surrogate = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device='cpu')
    surrogate.fit(features, adj, labels, idx_train)
    surrogate.eval()
    
    # Adjacency as continuous tensor for gradients
    adj_dense = torch.tensor(adj.todense(), dtype=torch.float, requires_grad=True)
    features_tensor = torch.tensor(features.todense(), dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Target some nodes (e.g., test set) for gradient calculation
    idx_target = np.random.choice(np.arange(adj.shape[0]), size=10, replace=False)
    
    for i in range(steps):
        output = surrogate.predict(features_tensor, adj_dense)
        loss = F.nll_loss(output[idx_target], labels_tensor[idx_target])
        loss.backward()
        
        with torch.no_grad():
            # Update adjacency using gradient
            adj_dense.data += 0.01 * adj_dense.grad.data.sign()
            # Projection: stay within [0, 1]
            adj_dense.data.clamp_(0, 1)
            adj_dense.grad.zero_()
            
    # Final discretize: flip top-k
    modified_adj = adj.copy().tolil()
    diff = (adj_dense.detach().cpu().numpy() - adj.todense())
    flat_diff = np.abs(diff).flatten()
    indices = np.argsort(flat_diff)[-n_perturbations:]
    
    for idx in indices:
        u, v = divmod(idx, adj.shape[0])
        # Manually flip the edge
        # Check if edge exists (works for lil, csr, etc.)
        if (modified_adj[u, v] != 0).nnz > 0:
            modified_adj[u, v] = 0
            modified_adj[v, u] = 0
        else:
            modified_adj[u, v] = 1
            modified_adj[v, u] = 1
        
    return modified_adj.tocsr(), features

if __name__ == "__main__":
    pass
