from dataclasses import dataclass
import torch


@dataclass
class DatasetInfo:
    num_features: int
    num_classes: int


class GraphData:
    """
    Minimal PyG-like data container so the project can run without torch_geometric
    (which may fail on some environments due to binary extension mismatches).
    """

    def __init__(self, x, y, edge_index, train_mask, val_mask, test_mask):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.num_nodes = int(x.size(0))
        self.num_edges = int(edge_index.size(1))
        self.num_features = int(x.size(1))

    def clone(self):
        # Deep copy tensors (share nothing).
        return GraphData(
            x=self.x.clone(),
            y=self.y.clone(),
            edge_index=self.edge_index.clone(),
            train_mask=self.train_mask.clone(),
            val_mask=self.val_mask.clone(),
            test_mask=self.test_mask.clone(),
        )

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self

