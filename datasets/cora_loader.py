import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os

def load_cora(root='data/Cora'):
    """
    Load the Cora dataset using PyTorch Geometric.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        
    Returns:
        data: The Cora dataset object.
    """
    if not os.path.exists(root):
        os.makedirs(root)
        
    dataset = Planetoid(root=root, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    
    print(f'Dataset: Cora')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of features: {data.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')
    
    return dataset, data

if __name__ == "__main__":
    dataset, data = load_cora()
    print(f'Training nodes: {data.train_mask.sum().item()}')
    print(f'Validation nodes: {data.val_mask.sum().item()}')
    print(f'Test nodes: {data.test_mask.sum().item()}')
