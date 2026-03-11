import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import modularity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Compute standard classification metrics + uncertainty metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
    }
    
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except:
            metrics['roc_auc'] = 0.0
        try:
            metrics['log_loss'] = log_loss(y_true, y_probs, labels=np.arange(y_probs.shape[1]))
        except:
            metrics['log_loss'] = 0.0
            
        # 1. Classification Margin (Robustness)
        sorted_probs = np.sort(y_probs, axis=1)
        true_class_probs = y_probs[np.arange(len(y_true)), y_true]
        # Margin = prob(correct) - prob(best_wrong)
        # For simplicity, average top1 - top2
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]
        metrics['avg_margin'] = np.mean(top1 - top2)
        
        # 2. Entropy (Uncertainty)
        # -sum(p * log(p))
        entropy = -np.sum(y_probs * np.log(y_probs + 1e-10), axis=1)
        metrics['avg_entropy'] = np.mean(entropy)
        
        # 3. Confidence
        metrics['avg_confidence'] = np.mean(np.max(y_probs, axis=1))
        metrics['classification_margin'] = metrics['avg_margin']
            
    return metrics

def compute_robustness_metrics(clean_metrics, attack_metrics, clean_probs, attack_probs):
    """
    Compute robustness specific metrics.
    """
    asr = (clean_metrics['accuracy'] - attack_metrics['accuracy']) / clean_metrics['accuracy'] if clean_metrics['accuracy'] > 0 else 0
    conf_drop = clean_metrics.get('avg_confidence', 0) - attack_metrics.get('avg_confidence', 0)
    margin_drop = clean_metrics.get('avg_margin', 0) - attack_metrics.get('avg_margin', 0)
    
    robustness_score = max(0.0, 1.0 - asr)
    return {
        'ASR': asr,
        'ConfidenceDrop': conf_drop,
        'MarginDrop': margin_drop,
        'RobustnessScore': robustness_score,
    }

def perturbation_rate(original, modified):
    """
    Fraction of changed entries over total entries.
    Supports dense arrays and scipy sparse matrices.
    """
    if sp.issparse(original) or sp.issparse(modified):
        o = original.tocsr()
        m = modified.tocsr()
        diff = (o != m).nnz
        total = o.shape[0] * o.shape[1]
        return diff / total if total > 0 else 0.0
    o = np.asarray(original)
    m = np.asarray(modified)
    diff = np.count_nonzero(o != m)
    total = o.size
    return diff / total if total > 0 else 0.0

def compute_graph_metrics(adj, labels):
    """
    Compute graph structural and connectivity metrics.
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
        
    G = nx.from_scipy_sparse_array(adj) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(adj)
    
    # 1. Homophily Ratio (fraction of edges connecting nodes of same label)
    edge_index = np.array(list(G.edges()))
    if len(edge_index) > 0:
        src, dst = edge_index[:, 0], edge_index[:, 1]
        same_label = (labels[src] == labels[dst]).sum()
        homophily = same_label / len(edge_index)
    else:
        homophily = 0.0
        
    metrics = {
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
        'homophily': homophily,
    }

    # Community structure based on class labels
    label_sets = []
    labels_np = np.asarray(labels)
    for lab in np.unique(labels_np):
        nodes = set(np.where(labels_np == lab)[0].tolist())
        if 0 < len(nodes) < G.number_of_nodes():
            label_sets.append(nodes)

    # Modularity (using label-based communities)
    try:
        metrics['modularity'] = modularity(G, label_sets) if label_sets else 0.0
    except Exception:
        metrics['modularity'] = 0.0

    # Conductance (average over label-based communities)
    conductances = []
    for s in label_sets:
        try:
            conductances.append(nx.algorithms.cuts.conductance(G, s))
        except Exception:
            continue
    metrics['conductance'] = float(np.mean(conductances)) if conductances else 0.0

    return metrics

if __name__ == "__main__":
    pass
