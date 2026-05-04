import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse, add_self_loops

class NeuroSymbolicDefense(nn.Module):
    """
    Neuro-Symbolic Hybrid Defense optimized for 2-layer GCN/GAT.
    Integrates Adaptive Edge Pruning, Ontology-Driven Semantic Validation,
    Feature Smoothing, Graph Purification, and Self-Correction Logic.
    """
    def __init__(self, rank=50, base_threshold=0.1, alpha=0.5):
        super(NeuroSymbolicDefense, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Parametrized learned thresholding mechanism.
        # Intuitively, we use 'w' and 'b' to adapt the pruning threshold 
        # relative to the node degree, preventing information starvation.
        self.w_degree = nn.Parameter(torch.tensor([-0.05]))
        self.b_thresh = nn.Parameter(torch.tensor([base_threshold]))

    def feature_smoothing(self, x, edge_index, num_nodes):
        """
        Applies a Laplacian-style Feature Smoothing step to neutralize evasion attacks.
        X_smooth = (1 - alpha) X + alpha * (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X)
        """
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop
        
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Compute neighborhood aggregation
        x_neigh = torch.zeros_like(x)
        x_neigh.index_add_(0, row, x[col] * norm.unsqueeze(1))
        
        # Blend original features with smoothed structural features
        return (1 - self.alpha) * x + self.alpha * x_neigh

    def graph_purification(self, edge_index, num_nodes):
        """
        Filters high-frequency adversarial noise using Low-Rank Approximation (SVD).
        """
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        
        # Apply Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(adj)
        k = min(self.rank, S.size(0))
        
        # Reconstruct optimal low-rank representation
        adj_purified = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        
        # Truncate near-zero artifacts created by SVD approximation
        adj_purified[adj_purified < 1e-4] = 0
        
        edge_index_purified, edge_weight_purified = dense_to_sparse(adj_purified)
        return edge_index_purified, edge_weight_purified

    def ontology_validation(self, edge_index, edge_weight, x, labels=None):
        """
        Ontology-Driven Semantic Validation.
        If nodes across an edge violate the Domain-Topic Ontology (e.g. have different 
        domains/labels and 0 shared semantic keywords), weight is set to 0.
        """
        row, col = edge_index
        
        # If labels are provided, nodes with differing labels signify differing ontology topics
        if labels is not None:
            topic_mismatch = labels[row] != labels[col]
        else:
            topic_mismatch = torch.ones(row.size(0), dtype=torch.bool, device=x.device)
            
        # Semantic keyword intersection (Bag-of-Words similarity check)
        shared_keywords = (x[row] * x[col]).sum(dim=-1)
        
        # Violation = Mismatched Topic AND Zero Shared Semantic Features
        violation_mask = topic_mismatch & (shared_keywords == 0)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        else:
            edge_weight = edge_weight.clone()
            
        # Sever violating edges entirely
        edge_weight[violation_mask] = 0.0
        return edge_weight

    def adaptive_edge_pruning(self, edge_index, edge_weight, num_nodes):
        """
        Adaptive Edge Pruning with Learned Thresholding based on node degree.
        """
        row, _ = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        
        # Dynamic threshold computation per-node mapping.
        # Nodes with lower degrees intuitively get lower thresholds, preventing starvation.
        adaptive_thresholds = torch.sigmoid(self.w_degree * deg + self.b_thresh)
        edge_thresholds = adaptive_thresholds[row]
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            
        # Validate weights dynamically against the tailored thresholds
        keep_mask = edge_weight >= edge_thresholds
        return edge_index[:, keep_mask], edge_weight[keep_mask]
        
    def calculate_homophily(self, edge_index, labels):
        """Helper function to calculate empirical homophily ratio."""
        if edge_index.size(1) == 0:
            return 0.0
        row, col = edge_index
        return (labels[row] == labels[col]).sum().item() / row.size(0)

    def forward(self, x, edge_index, labels=None):
        """
        Executes the end-to-end defense pipeline.
        Args:
            x (Tensor): Node feature matrix
            edge_index (Tensor): Graph connectivity 
            labels (Tensor, optional): Known node logic boundaries (pseudo-labels at test time)
        Returns:
            x_defended, edge_index_defended, edge_weight_defended
        """
        num_nodes = x.size(0)
        
        # Snapshot Homophily
        orig_homophily = 0.0
        if labels is not None:
            orig_homophily = self.calculate_homophily(edge_index, labels)
            
        # 1. Feature Smoothing
        x_smooth = self.feature_smoothing(x, edge_index, num_nodes)
        
        # 2. Graph Purification
        edge_index_pur, edge_weight_pur = self.graph_purification(edge_index, num_nodes)
        
        # 3. Ontology Validation
        edge_weight_ont = self.ontology_validation(edge_index_pur, edge_weight_pur, x_smooth, labels)
        
        # 4. Adaptive Edge Pruning
        edge_index_pruned, edge_weight_pruned = self.adaptive_edge_pruning(edge_index_pur, edge_weight_ont, num_nodes)
        
        # 5. Self-Correction Logic
        if labels is not None:
            new_homophily = self.calculate_homophily(edge_index_pruned, labels)
            
            # If the pruned graph damages homophily relative to baseline, 
            # fail-safe fallback reverting the structural layout.
            if new_homophily < orig_homophily:
                return x_smooth, edge_index, None  # Returns smoothed features overlaying unpruned edges
                
        return x_smooth, edge_index_pruned, edge_weight_pruned
