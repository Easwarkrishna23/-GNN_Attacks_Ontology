from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.defense import GCN
import scipy.sparse as sp

def run_fgsm_structure_attack(adj, features, labels, idx_train, target_node, n_perturbations=5):
    """
    Apply Fast Gradient Attack (FGA) - the graph version of FGSM.
    """
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # Surrogate GCN
    surrogate = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device='cpu')
    surrogate.fit(features, adj, labels, idx_train)
    
    # Initialize FGA
    model = FGA(surrogate, nnodes=adj.shape[0], device='cpu')
    
    # Run attack
    model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
    
    return model.modified_adj, model.modified_features

if __name__ == "__main__":
    pass
