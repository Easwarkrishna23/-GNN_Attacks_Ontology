"""
gnnguard_paper_defense.py

Base_Paper.pdf corresponds to GNNGuard:
  - neighbor importance estimation (edge trust from semantic similarity)
  - pruning suspicious edges
  - layer-wise graph memory:
      omega_uv^k = beta * omega_uv^{k-1} + (1-beta) * alpha_hat_uv^k

This module implements a *pragmatic* version tailored to this repo:
- Works for our 2-layer GCN and GAT implementations (pure PyTorch).
- Produces layer-specific edge weights (edge_weight_l1, edge_weight_l2).
- Uses cosine similarity on node representations to estimate importance.
- Prunes by threshold + optional top-k per node (keeps graph connected).

Important: this is a defense on the *input graph/message passing*, not a change to
the GCN/GAT weights directly. It changes how neighbor messages are aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from datasets.simple_data import GraphData


def _cosine_edge_similarity(x: np.ndarray, edge_index: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine similarity for each edge (u->v) in edge_index.
    Returns sim in [0,1] after clipping negatives to 0.
    """
    src = edge_index[0]
    dst = edge_index[1]
    xs = x[src]
    xd = x[dst]
    num = np.sum(xs * xd, axis=1)
    den = (np.linalg.norm(xs, axis=1) * np.linalg.norm(xd, axis=1)) + eps
    sim = num / den
    sim = np.clip(sim, 0.0, 1.0)
    return sim.astype(np.float32)


def _topk_outgoing_mask(edge_index: np.ndarray, scores: np.ndarray, num_nodes: int, k: int) -> np.ndarray:
    """
    Keep only top-k outgoing edges per source node by score.
    Returns a boolean mask over edges.
    """
    if k <= 0:
        return np.ones((scores.shape[0],), dtype=bool)
    src = edge_index[0]
    keep = np.zeros((scores.shape[0],), dtype=bool)
    # bucket edges by src
    buckets = [[] for _ in range(num_nodes)]
    for ei, u in enumerate(src.tolist()):
        buckets[int(u)].append(ei)
    for u in range(num_nodes):
        idxs = buckets[u]
        if not idxs:
            continue
        if len(idxs) <= k:
            keep[idxs] = True
            continue
        sc = scores[idxs]
        top = np.argpartition(sc, -k)[-k:]
        keep[np.asarray(idxs, dtype=int)[top]] = True
    return keep


def _importance_from_similarity(sim: np.ndarray, power: float = 2.0) -> np.ndarray:
    # Sharpen differences: good edges close to 1 stay high, weak edges get suppressed.
    sim = np.clip(sim, 0.0, 1.0)
    return np.power(sim, float(power)).astype(np.float32)


@dataclass(frozen=True)
class PaperDefenseParams:
    smooth_alpha: float = 0.7
    prune_threshold: float = 0.05
    topk: int = 20
    beta: float = 0.6
    power: float = 2.0


def apply_gnnguard_paper_defense(
    model,
    data: GraphData,
    x_smoothed: Optional[torch.Tensor] = None,
    params: Optional[PaperDefenseParams] = None,
) -> Tuple[GraphData, torch.Tensor, torch.Tensor]:
    """
    Returns:
      data_def: cloned GraphData with x replaced (if x_smoothed provided)
      w1: edge weights for layer-1
      w2: edge weights for layer-2 (with layer-wise memory)
    """
    p = params or PaperDefenseParams()
    data_def = data.clone()
    if x_smoothed is not None:
        data_def.x = x_smoothed

    # Always compute weights from CPU numpy arrays (fast for Cora-scale graphs).
    edge_index = data_def.edge_index.detach().cpu().numpy().astype(np.int64)
    num_nodes = int(data_def.num_nodes)

    X0 = data_def.x.detach().cpu().numpy().astype(np.float32)

    # ----- Layer 1: neighbor importance + pruning -----
    sim1 = _cosine_edge_similarity(X0, edge_index)
    a1 = _importance_from_similarity(sim1, power=p.power)
    # threshold pruning
    a1[a1 < float(p.prune_threshold)] = 0.0
    # top-k pruning (outgoing)
    mask1 = _topk_outgoing_mask(edge_index, a1, num_nodes=num_nodes, k=int(p.topk))
    a1 = a1 * mask1.astype(np.float32)
    omega1 = a1

    # For embedding computation, use omega1 as shared edge_weight (layer-1).
    data_tmp = data_def.clone()
    data_tmp.edge_weight_l1 = torch.tensor(omega1, dtype=torch.float32, device=data_def.edge_index.device)
    # Layer-2 weights aren't used for embedding extraction.

    # Compute first-layer embeddings under the current edge trust.
    with torch.no_grad():
        h1 = model.get_embeddings(data_tmp)
    H1 = h1.detach().cpu().numpy().astype(np.float32)

    # ----- Layer 2: neighbor importance + layer-wise memory -----
    sim2 = _cosine_edge_similarity(H1, edge_index)
    a2 = _importance_from_similarity(sim2, power=p.power)
    a2[a2 < float(p.prune_threshold)] = 0.0
    mask2 = _topk_outgoing_mask(edge_index, a2, num_nodes=num_nodes, k=int(p.topk))
    a2 = a2 * mask2.astype(np.float32)

    beta = float(p.beta)
    omega2 = beta * omega1 + (1.0 - beta) * a2
    omega2 = np.clip(omega2, 0.0, 1.0).astype(np.float32)

    w1 = torch.tensor(omega1, dtype=torch.float32, device=data_def.edge_index.device)
    w2 = torch.tensor(omega2, dtype=torch.float32, device=data_def.edge_index.device)
    return data_def, w1, w2

