from deeprobust.graph.global_attack import DICE
import scipy.sparse as sp

def run_dice_attack(adj, labels, n_perturbations=200):
    """
    Apply DICE global poisoning attack.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
        
    model = DICE()
    model.attack(adj, labels, n_perturbations=n_perturbations)
    
    return model.modified_adj, None # Features remain unchanged
