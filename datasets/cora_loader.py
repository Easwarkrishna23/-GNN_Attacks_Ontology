import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch

from datasets.simple_data import DatasetInfo, GraphData

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False


def _normalize_features(x):
    # Row-normalize features (same spirit as PyG NormalizeFeatures)
    row_sum = np.array(x.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    inv_mat = sp.diags(inv)
    return (inv_mat @ x).tocsr()


def _planetoid_load(root):
    """
    Load Cora from the classic Planetoid raw files:
    ind.cora.{x,y,tx,ty,allx,ally,graph,test.index}
    This path is used when torch_geometric is unavailable.
    """
    raw_dir = os.path.join(root, "Cora", "raw")
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        path = os.path.join(raw_dir, f"ind.cora.{name}")
        with open(path, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_path = os.path.join(raw_dir, "ind.cora.test.index")
    test_idx = np.loadtxt(test_idx_path, dtype=np.int64)
    test_idx_sorted = np.sort(test_idx)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = _normalize_features(features.tocsr())
    features = np.asarray(features.todense(), dtype=np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = labels.argmax(1).astype(np.int64)

    # Build adjacency from graph dict
    adj = sp.lil_matrix((labels.shape[0], labels.shape[0]), dtype=np.float32)
    for i, nbrs in graph.items():
        adj.rows[i] = list(nbrs)
        adj.data[i] = [1.0] * len(nbrs)
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)  # sym

    rows, cols = adj.nonzero()
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    # Standard Planetoid split: first len(y)=140 are train, next 500 val, 1000 test.
    idx_train = np.arange(y.shape[0])
    idx_val = np.arange(y.shape[0], y.shape[0] + 500)
    idx_test = test_idx_sorted

    n = labels.shape[0]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    data = GraphData(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    dataset = DatasetInfo(num_features=data.num_features, num_classes=int(labels.max() + 1))
    return dataset, data


def load_cora(root="data/Cora"):
    """
    Load the Cora dataset.

    Preferred: torch_geometric Planetoid (when available).
    Fallback: classic Planetoid raw file loader (no torch_geometric dependency).
    """
    os.makedirs(root, exist_ok=True)

    if PYG_AVAILABLE:
        dataset = Planetoid(root=root, name="Cora", transform=T.NormalizeFeatures())
        data = dataset[0]
        # Convert to our lightweight GraphData for consistent downstream use.
        data2 = GraphData(
            x=data.x.detach().clone(),
            y=data.y.detach().clone(),
            edge_index=data.edge_index.detach().clone(),
            train_mask=data.train_mask.detach().clone(),
            val_mask=data.val_mask.detach().clone(),
            test_mask=data.test_mask.detach().clone(),
        )
        ds = DatasetInfo(num_features=int(dataset.num_features), num_classes=int(dataset.num_classes))
        print("Dataset: Cora (torch_geometric)")
        print(f"Number of nodes: {data2.num_nodes}")
        print(f"Number of edges: {data2.num_edges}")
        print(f"Number of features: {data2.num_features}")
        print(f"Number of classes: {ds.num_classes}")
        return ds, data2

    dataset, data = _planetoid_load(root)
    print("Dataset: Cora (Planetoid raw fallback, no torch_geometric)")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    return dataset, data

if __name__ == "__main__":
    dataset, data = load_cora()
    print(f'Training nodes: {data.train_mask.sum().item()}')
    print(f'Validation nodes: {data.val_mask.sum().item()}')
    print(f'Test nodes: {data.test_mask.sum().item()}')
