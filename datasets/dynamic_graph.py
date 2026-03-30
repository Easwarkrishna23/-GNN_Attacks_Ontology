import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp

from datasets.simple_data import GraphData

class DynamicGraphGenerator:
    """
    Generates a synthetic evolving graph using the Barabási-Albert model.
    """
    def __init__(self, initial_nodes=100, num_features=16, num_classes=3):
        self.num_features = num_features
        self.num_classes = num_classes
        self.G = nx.barabasi_albert_graph(initial_nodes, 2)
        
        # Initialize features and labels
        self.features = np.random.randn(initial_nodes, num_features)
        self.labels = np.random.randint(0, num_classes, initial_nodes)
        
    def evolve(self, new_nodes=10, edges_per_node=2):
        """
        Add new nodes to the graph using preferential attachment.
        """
        current_nodes = self.G.number_of_nodes()
        for i in range(new_nodes):
            new_node_id = current_nodes + i
            # Preferential attachment
            targets = self._get_targets(edges_per_node)
            self.G.add_node(new_node_id)
            for t in targets:
                self.G.add_edge(new_node_id, t)
                
        # Update features and labels
        new_features = np.random.randn(new_nodes, self.num_features)
        new_labels = np.random.randint(0, self.num_classes, new_nodes)
        
        self.features = np.vstack([self.features, new_features])
        self.labels = np.concatenate([self.labels, new_labels])
        
        return self.get_pyg_data()

    def _get_targets(self, m):
        """
        Helper for preferential attachment.
        """
        nodes = list(self.G.nodes())
        degrees = [self.G.degree(n) for n in nodes]
        total_degree = sum(degrees)
        if total_degree == 0:
            probs = [1.0 / len(nodes)] * len(nodes)
        else:
            probs = [d / total_degree for d in degrees]
        return np.random.choice(nodes, size=m, replace=False, p=probs)

    def get_pyg_data(self):
        """
        Convert NetworkX graph to a lightweight GraphData object (PyG-like).
        """
        n = self.G.number_of_nodes()
        edges = np.array(list(self.G.edges()), dtype=np.int64)
        if edges.size == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            # undirected: include both directions
            src = np.concatenate([edges[:, 0], edges[:, 1]])
            dst = np.concatenate([edges[:, 1], edges[:, 0]])
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        x = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.long)
        
        # Add train/val/test masks
        num_nodes = n
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        return GraphData(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

if __name__ == "__main__":
    generator = DynamicGraphGenerator(initial_nodes=50)
    data_t0 = generator.get_pyg_data()
    print(f"Time T=0: Nodes={data_t0.num_nodes}, Edges={data_t0.num_edges}")
    
    data_t1 = generator.evolve(new_nodes=20)
    print(f"Time T=1: Nodes={data_t1.num_nodes}, Edges={data_t1.num_edges}")
