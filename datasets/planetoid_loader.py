import os
import pickle
from typing import Tuple

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


def _normalize_features(x: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.array(x.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    inv_mat = sp.diags(inv)
    return (inv_mat @ x).tocsr()


def _planetoid_raw_load(root: str, name: str) -> Tuple[DatasetInfo, GraphData]:
    """
    Load Planetoid datasets from the classic raw files:
      ind.<name>.{x,y,tx,ty,allx,ally,graph,test.index}

    Supports: Cora, Citeseer, PubMed.
    """
    name_l = name.lower()
    raw_dir = os.path.join(root, name, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Missing raw Planetoid files at {raw_dir}. "
            f"Run: python3 datasets/download_planetoid_raw.py --dataset {name} --root {root}"
        )
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for n in names:
        path = os.path.join(raw_dir, f"ind.{name_l}.{n}")
        with open(path, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_path = os.path.join(raw_dir, f"ind.{name_l}.test.index")
    test_idx = np.loadtxt(test_idx_path, dtype=np.int64)
    test_idx_sorted = np.sort(test_idx)

    # Citeseer has isolated nodes: tx/ty do not cover full test index range.
    if name_l == "citeseer":
        test_idx_range = np.arange(test_idx_sorted.min(), test_idx_sorted.max() + 1)
        tx_ext = sp.lil_matrix((len(test_idx_range), x.shape[1]), dtype=np.float32)
        ty_ext = np.zeros((len(test_idx_range), y.shape[1]), dtype=np.float32)
        tx_ext[test_idx_sorted - test_idx_range.min(), :] = tx
        ty_ext[test_idx_sorted - test_idx_range.min(), :] = ty
        tx = tx_ext
        ty = ty_ext
        test_idx_sorted = test_idx_range

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = _normalize_features(features.tocsr())
    features = np.asarray(features.todense(), dtype=np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = labels.argmax(1).astype(np.int64)

    # adjacency
    adj = sp.lil_matrix((labels.shape[0], labels.shape[0]), dtype=np.float32)
    for i, nbrs in graph.items():
        adj.rows[i] = list(nbrs)
        adj.data[i] = [1.0] * len(nbrs)
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)

    rows, cols = adj.nonzero()
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    # Standard Planetoid split sizes differ by dataset in original code; we keep the common setting:
    # train = first len(y), val = next 500, test = test_idx_sorted.
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
    ds = DatasetInfo(num_features=data.num_features, num_classes=int(labels.max() + 1))
    return ds, data


def load_planetoid(name: str, root: str = "data") -> Tuple[DatasetInfo, GraphData]:
    """
    Load a Planetoid dataset (Cora, Citeseer, PubMed).

    Preferred: torch_geometric Planetoid when available.
    Fallback: classic Planetoid raw loader (no torch_geometric dependency).
    """
    name = str(name)
    if name.lower() not in {"cora", "citeseer", "pubmed"}:
        raise ValueError("name must be one of: Cora, Citeseer, PubMed")

    os.makedirs(root, exist_ok=True)

    if PYG_AVAILABLE:
        dataset = Planetoid(root=os.path.join(root, name), name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data2 = GraphData(
            x=data.x.detach().clone(),
            y=data.y.detach().clone(),
            edge_index=data.edge_index.detach().clone(),
            train_mask=data.train_mask.detach().clone(),
            val_mask=data.val_mask.detach().clone(),
            test_mask=data.test_mask.detach().clone(),
        )
        ds = DatasetInfo(num_features=int(dataset.num_features), num_classes=int(dataset.num_classes))
        return ds, data2

    return _planetoid_raw_load(root, name)
