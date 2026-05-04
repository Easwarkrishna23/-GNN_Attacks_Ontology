import torch
import torch.nn.functional as F

# --- Helpers ---

def normalized_adj(data):
    """Mock normalized adjacency matrix for feature smoothing."""
    num_nodes = data.x.size(0)
    adj = torch.zeros((num_nodes, num_nodes), device=data.x.device)
    adj[data.edge_index[0], data.edge_index[1]] = 1.0
    adj += torch.eye(num_nodes, device=data.x.device) # Add self loops
    deg = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def get_neighbors(data, node):
    """Get neighbor indices for a given node."""
    edges = data.edge_index
    neighbors = edges[1, edges[0] == node]
    # Include self
    return torch.cat([neighbors, torch.tensor([node], device=data.x.device)])

# --- 🟢 5.1 STRUCTURAL DEFENSE ---

def prune_edges(data, threshold=0.2):
    data = data.clone()
    new_edges = []
    
    # We transpose to iterate over edge pairs (u, v)
    edges_t = data.edge_index.T
    for (u, v) in edges_t:
        # compute cosine similarity
        sim = F.cosine_similarity(data.x[u].unsqueeze(0), data.x[v].unsqueeze(0))
        if sim.item() > threshold:
            new_edges.append([u, v])
            
    if new_edges:
        data.edge_index = torch.tensor(new_edges, dtype=torch.long, device=data.edge_index.device).T
    else:
        # Fallback if everything is pruned
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)
    return data

def smooth_features(data):
    data = data.clone()
    adj_norm = normalized_adj(data)
    data.x = torch.matmul(adj_norm, data.x)
    return data

# --- 🟢 5.2 ONTOLOGY DEFENSE ---

# Step 1: Build Ontology (Mock)
ontology = {
    0: "AI",
    1: "AI",
    2: "DB",
    3: "Theory",
    4: "ML"
}

def is_consistent(feature_vector, ontology):
    """Mock ontology check. 
    Assume inconsistency if feature vector norm is wildly distorted."""
    return feature_vector.norm().item() < 50.0 

def detect_ontology_violation(data):
    attacked_nodes = []
    for node in range(data.num_nodes):
        if not is_consistent(data.x[node], ontology):
            attacked_nodes.append(node)
    return attacked_nodes

def correct_features(data, attacked_nodes):
    data = data.clone()
    for node in attacked_nodes:
        neighbors = get_neighbors(data, node)
        if len(neighbors) > 0:
            data.x[node] = data.x[neighbors].mean(dim=0)
    return data

def ontology_defense(data):
    """Apply standalone ontology defense."""
    data = data.clone()
    attacked_nodes = detect_ontology_violation(data)
    data = correct_features(data, attacked_nodes)
    return data

# --- 🟣 5.3 HYBRID DEFENSE ---

def hybrid_defense(data):
    data = data.clone()
    data = prune_edges(data)
    data = smooth_features(data)
    
    attacked_nodes = detect_ontology_violation(data)
    data = correct_features(data, attacked_nodes)
    
    return data

def apply_defenses(attacked_data_dict):
    """Apply defenses to each attacked graph. Returns dictionary mapping attack -> defenses applied."""
    defended_dict = {}
    for attack_name, data in attacked_data_dict.items():
        defended_dict[attack_name] = {
            "attacked": data,
            "structural": smooth_features(prune_edges(data)),
            "ontology": ontology_defense(data),
            "hybrid": hybrid_defense(data)
        }
    return defended_dict
