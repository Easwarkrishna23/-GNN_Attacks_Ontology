import torch
import random

# --- Helper Functions for Simplified Attacks ---

def compute_gradient(data, target_node):
    """Mock gradient computation for node features."""
    return torch.randn_like(data.x)

def optimize_graph_structure(data):
    """Mock optimization: drop some edges randomly."""
    num_edges = data.edge_index.size(1)
    keep_edges = max(1, int(num_edges * 0.95))
    perm = torch.randperm(num_edges)
    return data.edge_index[:, perm[:keep_edges]]

def random_node(num_nodes):
    return random.randint(0, num_nodes - 1)

def add_edge(data, u, v):
    new_edge = torch.tensor([[u, v], [v, u]], dtype=torch.long, device=data.edge_index.device)
    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

def flip_edge(data):
    """Randomly drop an existing edge to simulate flip."""
    num_edges = data.edge_index.size(1)
    if num_edges > 0:
        idx_to_drop = random.randint(0, num_edges - 1)
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[idx_to_drop] = False
        data.edge_index = data.edge_index[:, keep_mask]

def compute_gradient_wrt_features(model, data):
    """Mock gradient w.r.t features."""
    return torch.randn_like(data.x)

# --- 🔴 POISONING ATTACKS ---

def nettack(data, target_node=0, steps=10):
    data = data.clone()
    for _ in range(steps):
        grad = compute_gradient(data, target_node)
        data.x += 0.01 * torch.sign(grad)
    return data

def meta_attack(data):
    data = data.clone()
    # optimize adjacency matrix
    perturbed_adj = optimize_graph_structure(data)
    data.edge_index = perturbed_adj
    return data

def random_attack(data, flips=100):
    data = data.clone()
    num_nodes = data.x.size(0)
    for _ in range(flips):
        u = random_node(num_nodes)
        v = random_node(num_nodes)
        add_edge(data, u, v)
    return data

# --- 🔴 EVASION ATTACKS ---

def feature_attack(data, epsilon=0.2):
    data = data.clone()
    noise = torch.randn_like(data.x) * epsilon
    data.x = data.x + noise
    return data

def edge_flip_attack(data, num_flips=50):
    data = data.clone()
    for _ in range(num_flips):
        flip_edge(data)
    return data

def gradient_attack(model, data):
    data = data.clone()
    grad = compute_gradient_wrt_features(model, data)
    data.x += 0.05 * torch.sign(grad)
    return data

def apply_all_attacks(data, model=None):
    """Applies all 6 attacks and returns a dictionary of attacked graphs."""
    return {
        "nettack": nettack(data, target_node=0),
        "meta_attack": meta_attack(data),
        "random_attack": random_attack(data),
        "feature_attack": feature_attack(data),
        "edge_flip_attack": edge_flip_attack(data),
        "gradient_attack": gradient_attack(model, data) if model else feature_attack(data)
    }
