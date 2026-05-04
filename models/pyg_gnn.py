import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class TwoLayerGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=True, normalize=True)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False, return_details: bool = False):
        layer0_in = x
        h1_linear = self.conv1(layer0_in, edge_index, edge_weight=edge_weight)
        h1_act = F.relu(h1_linear)
        h1_drop = F.dropout(h1_act, p=self.dropout, training=self.training)
        emb = h1_drop
        logits = self.conv2(h1_drop, edge_index, edge_weight=edge_weight)

        if return_details:
            details = {
                "input": layer0_in,
                "layer1_pre_activation": h1_linear,
                "layer1_post_activation": h1_act,
                "layer1_post_dropout": h1_drop,
                "layer2_logits": logits,
                "memory_mb_layer1": float((h1_linear.nelement() * h1_linear.element_size()) / (1024**2)),
                "memory_mb_layer2": float((logits.nelement() * logits.element_size()) / (1024**2)),
            }
            return emb, logits, details

        if return_embedding:
            return emb, logits
        return logits


class TwoLayerGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False, return_details: bool = False):
        layer0_in = x
        h1_linear = self.conv1(layer0_in, edge_index)
        h1_act = F.elu(h1_linear)
        h1_drop = F.dropout(h1_act, p=self.dropout, training=self.training)
        emb = h1_drop
        logits = self.conv2(h1_drop, edge_index)

        if return_details:
            details = {
                "input": layer0_in,
                "layer1_pre_activation": h1_linear,
                "layer1_post_activation": h1_act,
                "layer1_post_dropout": h1_drop,
                "layer2_logits": logits,
                "memory_mb_layer1": float((h1_linear.nelement() * h1_linear.element_size()) / (1024**2)),
                "memory_mb_layer2": float((logits.nelement() * logits.element_size()) / (1024**2)),
            }
            return emb, logits, details

        if return_embedding:
            return emb, logits
        return logits
