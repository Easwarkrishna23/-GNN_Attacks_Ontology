import torch
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _neighbor_mean(data):
    """Sparse per-node mean of neighbour features (no dense N×N matrix)."""
    src, dst = data.edge_index
    deg = torch.zeros(data.num_nodes, device=data.x.device)
    deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=data.x.device))
    deg = deg.clamp(min=1)
    nb_sum = torch.zeros_like(data.x)
    nb_sum.index_add_(0, dst, data.x[src])
    return nb_sum / deg.unsqueeze(1)


def _class_prototypes(data):
    """
    Per-class mean feature vector from training nodes.

    Even when ALL nodes are attacked, averaging ~20 training nodes per class
    reduces individual noise by √20 ≈ 4.5×, yielding prototypes that are
    substantially cleaner than individual attacked vectors.
    """
    num_classes = int(data.y.max().item()) + 1
    prototypes = torch.zeros(num_classes, data.num_features, device=data.x.device)
    counts = torch.zeros(num_classes, device=data.x.device)
    for idx in data.train_mask.nonzero(as_tuple=True)[0]:
        c = data.y[idx].item()
        prototypes[c] += data.x[idx]
        counts[c] += 1
    counts = counts.clamp(min=1)
    return prototypes / counts.unsqueeze(1)   # [C, F]


# ── 5.1 STRUCTURAL DEFENSE ────────────────────────────────────────────────────

def structural_defense(data, alpha=0.5, steps=2):
    """
    Laplacian smoothing: blend each node's features with the mean of its
    graph neighbours (sparse, no dense N×N matrix).

    Why no edge pruning: Cora's sparse L1-normalised BOW vectors have
    naturally low pairwise cosine similarity (0.1–0.3 even for same-class
    pairs), so a hard threshold prunes most legitimate edges and collapses
    graph structure.

    Why no post-clamp / renorm: the model tolerates mildly out-of-distribution
    inputs better than aggressive renormalisation, which distorts feature
    directions and drops signal below the noise floor.
    """
    data = data.clone()
    for _ in range(steps):
        data.x = alpha * data.x + (1 - alpha) * _neighbor_mean(data)
    return data


# ── 5.2 ONTOLOGY DEFENSE ─────────────────────────────────────────────────────

def ontology_defense(data, blend=0.3):
    """
    Prototype-based semantic projection using Cora's topic ontology.

    Semantic knowledge: each paper belongs to one of seven topic classes
    (Neural Networks, Case-Based, Genetic Algorithms, …).  Training nodes
    (with known labels) define per-class feature prototypes — the "semantic
    centroid" of each topic in feature space.

    Even when all node features are globally attacked, the class prototype is
    the mean of ~20 training vectors, reducing per-feature noise by √20 ≈ 4.5×.
    Each test node is then softly projected 30% toward its nearest prototype,
    pulling its features back toward the correct topic's semantic centre
    without destroying its individual signal.

    This is ontology-guided because the correction is driven by symbolic class
    membership (topic identity), not purely by raw geometric similarity.
    """
    data = data.clone()
    prototypes = _class_prototypes(data)                    # [C, F]

    # Nearest prototype by cosine similarity
    x_norm = F.normalize(data.x, p=2, dim=1)
    p_norm = F.normalize(prototypes, p=2, dim=1)
    nearest = torch.mm(x_norm, p_norm.T).argmax(dim=1)     # [N]

    # Soft blend toward nearest class prototype
    data.x = (1 - blend) * data.x + blend * prototypes[nearest]
    return data


# ── 5.3 HYBRID DEFENSE ───────────────────────────────────────────────────────

def hybrid_defense(data):
    """
    Two-stage pipeline: semantic projection first, then graph smoothing.

    Order matters: smoothing after prototype blending propagates topic-aligned
    features across the graph rather than averaging raw noisy signals.
    """
    data = data.clone()
    data = ontology_defense(data)
    data = structural_defense(data, alpha=0.7, steps=1)
    return data


# ── public API ────────────────────────────────────────────────────────────────

def apply_defenses(attacked_data_dict):
    defended_dict = {}
    for attack_name, attacked_data in attacked_data_dict.items():
        defended_dict[attack_name] = {
            "attacked":   attacked_data,
            "structural": structural_defense(attacked_data),
            "ontology":   ontology_defense(attacked_data),
            "hybrid":     hybrid_defense(attacked_data),
        }
    return defended_dict
