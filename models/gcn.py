import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import numpy as np

class GCN(torch.nn.Module):
    """
    Standard 2-layer GCN architecture.
    """
    def __init__(self, num_features, num_hidden, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)
        self.dropout = dropout

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        
        # If edge_weight is not provided in call, check if data has it
        if edge_weight is None and hasattr(data, 'edge_weight'):
            edge_weight = data.edge_weight

        # Layer 1: Feature aggregation + hidden embedding
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: Class logits
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    @staticmethod
    def _normalized_adjacency(edge_index, num_nodes):
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        val = np.ones(edge_index.size(1), dtype=np.float32)
        adj = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes))
        adj = adj + sp.eye(num_nodes, dtype=np.float32)
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
        d_inv_sqrt = sp.diags(deg_inv_sqrt)
        return (d_inv_sqrt @ adj @ d_inv_sqrt).tocsr()

    def forward_with_debug(self, data):
        """
        Compute logits and expose layer-wise matrices for explanation.
        """
        x = data.x
        device = x.device
        norm_adj = self._normalized_adjacency(data.edge_index, data.num_nodes)
        
        def mem_mb(tensor):
            return float((tensor.numel() * tensor.element_size()) / (1024 ** 2))

        w0 = self.conv1.lin.weight.t()
        b0 = self.conv1.bias if self.conv1.bias is not None else 0.0
        pre_agg_1 = x @ w0
        post_norm_1 = torch.tensor(
            norm_adj @ pre_agg_1.detach().cpu().numpy(),
            dtype=x.dtype,
            device=device,
        ) + b0
        post_act_1 = F.relu(post_norm_1)

        w1 = self.conv2.lin.weight.t()
        b1 = self.conv2.bias if self.conv2.bias is not None else 0.0
        pre_agg_2 = post_act_1 @ w1
        post_norm_2 = torch.tensor(
            norm_adj @ pre_agg_2.detach().cpu().numpy(),
            dtype=x.dtype,
            device=device,
        ) + b1
        probs = torch.softmax(post_norm_2, dim=1)
        preds = probs.argmax(dim=1)
        
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_memory = {
            "input_x_mb": mem_mb(x),
            "pre_aggregation_l1_mb": mem_mb(pre_agg_1),
            "post_normalization_l1_mb": mem_mb(post_norm_1),
            "post_activation_l1_mb": mem_mb(post_act_1),
            "pre_aggregation_l2_mb": mem_mb(pre_agg_2),
            "post_normalization_l2_mb": mem_mb(post_norm_2),
            "softmax_probabilities_mb": mem_mb(probs),
        }

        return {
            "pre_aggregation_l1": pre_agg_1.detach(),
            "post_normalization_l1": post_norm_1.detach(),
            "post_activation_l1": post_act_1.detach(),
            "pre_aggregation_l2": pre_agg_2.detach(),
            "post_normalization_l2": post_norm_2.detach(),
            "softmax_probabilities": probs.detach(),
            "predictions": preds.detach(),
            "memory": {
                "parameter_memory_mb": float(param_bytes / (1024 ** 2)),
                "activation_memory_mb": activation_memory,
                "total_estimated_mb": float(param_bytes / (1024 ** 2)) + sum(activation_memory.values()),
            },
        }

    def get_embeddings(self, data, edge_weight=None):
        """
        Extract hidden layer embeddings H(1).
        """
        x, edge_index = data.x, data.edge_index
        if edge_weight is None and hasattr(data, 'edge_weight'):
            edge_weight = data.edge_weight
            
        x = self.conv1(x, edge_index, edge_weight)
        return F.relu(x)
