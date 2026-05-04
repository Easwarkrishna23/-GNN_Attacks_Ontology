from __future__ import annotations

import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cora(root: str = "data", seed: int = 42):
    set_global_seed(seed)
    dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data
