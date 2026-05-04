import torch
from torch_geometric.nn import knn_graph

def rebuild_knn_graph(x, k=5):
    """Rebuild graph connectivity using k-nearest neighbors on features."""
    # PyG knn_graph expects [num_nodes, num_features]
    return knn_graph(x, k=k, loop=False)

def reconstruct_graph(data):
    """Reconstruct module per the blueprint."""
    data = data.clone()
    data.edge_index = rebuild_knn_graph(data.x)
    return data
