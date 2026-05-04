import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def load_dataset(root='data', name='Cora'):
    """Load static dataset."""
    dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    return dataset[0]

def perturb_edges(edge_index):
    """Simulate changes by dropping a few random edges."""
    num_edges = edge_index.size(1)
    drop_num = max(1, int(num_edges * 0.05))
    perm = torch.randperm(num_edges)
    keep_indices = perm[drop_num:]
    return edge_index[:, keep_indices]

def create_dynamic_graph(data, timesteps=5):
    """Simulate a dynamic graph over several timesteps."""
    dynamic_graphs = []

    for t in range(timesteps):
        new_data = data.clone()
        # simulate changes
        new_data.edge_index = perturb_edges(new_data.edge_index)
        dynamic_graphs.append(new_data)

    return dynamic_graphs
