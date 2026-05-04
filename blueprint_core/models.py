import torch
from torch_geometric.nn import GCNConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = torch.relu(h1)
        
        # Logits
        h2 = self.conv2(h1, edge_index)
        
        return h2, h1   # logits + embeddings

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        # PyG GATConv expects output dim to be multiplied by heads for the next layer if concat=True
        self.att1 = GATConv(in_channels, hidden, heads=8)
        self.att2 = GATConv(hidden * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        h1 = self.att1(x, edge_index)
        h1 = torch.relu(h1)

        # Logits
        out = self.att2(h1, edge_index)
        
        return out, h1
