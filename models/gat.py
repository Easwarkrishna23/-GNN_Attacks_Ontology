import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    """
    Standard 2-layer GAT architecture.
    """
    def __init__(self, num_features, num_hidden, num_classes, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, num_hidden, heads=heads, dropout=dropout)
        # On the second layer, we concatenate the multi-head outputs by default,
        # but for the final layer we usually average them.
        self.conv2 = GATConv(num_hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def forward_with_debug(self, data):
        """
        Return logits and explainable internals (attention + aggregation output).
        """
        x, edge_index = data.x, data.edge_index
        
        def mem_mb(tensor):
            return float((tensor.numel() * tensor.element_size()) / (1024 ** 2))

        x_dropout = F.dropout(x, p=self.dropout, training=self.training)
        hidden, (attn_edge_index, alpha) = self.conv1(
            x_dropout, edge_index, return_attention_weights=True
        )
        hidden = F.elu(hidden)
        logits = self.conv2(F.dropout(hidden, p=self.dropout, training=self.training), edge_index)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_memory = {
            "input_x_mb": mem_mb(x),
            "dropout_input_mb": mem_mb(x_dropout),
            "hidden_after_attention_mb": mem_mb(hidden),
            "attention_alpha_mb": mem_mb(alpha),
            "logits_mb": mem_mb(logits),
            "softmax_probabilities_mb": mem_mb(probs),
        }
        return {
            "attention_edge_index": attn_edge_index.detach(),
            "attention_weights": alpha.detach(),
            "node_aggregation_hidden": hidden.detach(),
            "logits": logits.detach(),
            "softmax_probabilities": probs.detach(),
            "predictions": preds.detach(),
            "memory": {
                "parameter_memory_mb": float(param_bytes / (1024 ** 2)),
                "activation_memory_mb": activation_memory,
                "total_estimated_mb": float(param_bytes / (1024 ** 2)) + sum(activation_memory.values()),
            },
        }

    def get_embeddings(self, data):
        """
        Extract hidden layer embeddings H(1).
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.elu(x)

    def get_attention_weights(self, data):
        """
        Return attention weights for the first layer.
        """
        x, edge_index = data.x, data.edge_index
        # GATConv returns (edge_index, alpha) if return_attention_weights is True
        _, (edge_index, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return edge_index, alpha
