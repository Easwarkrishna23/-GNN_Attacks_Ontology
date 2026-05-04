from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from owlready2 import get_ontology
from torch_geometric.data import Data


@dataclass
class DefenseOutcome:
    name: str
    data: Data
    metadata: Dict[str, float]


CORA_TOPICS = [
    "CaseBased",
    "GeneticAlgorithms",
    "NeuralNetworks",
    "ProbabilisticMethods",
    "ReinforcementLearning",
    "RuleLearning",
    "Theory",
]


def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    vals = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def csr_to_edge_index(adj: sp.csr_matrix) -> torch.Tensor:
    coo = adj.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def _symmetrize(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _edge_jaccard(x_bin: np.ndarray, u: int, v: int) -> float:
    fu = x_bin[u]
    fv = x_bin[v]
    inter = float(np.sum((fu > 0) & (fv > 0)))
    union = float(np.sum((fu > 0) | (fv > 0)))
    if union == 0:
        return 0.0
    return inter / union


def adaptive_jaccard_pruning(adj: sp.csr_matrix, features: np.ndarray) -> sp.csr_matrix:
    """
    Defense A: adaptive threshold based on node-local median similarity.
    Keeps sparse-region utility better than fixed-threshold pruning.
    """
    adj = _symmetrize(adj)
    x_bin = (features > 0).astype(np.uint8)
    rows, cols = adj.nonzero()

    sims = np.zeros(rows.shape[0], dtype=np.float32)
    local = [[] for _ in range(adj.shape[0])]

    for i, (u, v) in enumerate(zip(rows, cols)):
        s = _edge_jaccard(x_bin, int(u), int(v))
        sims[i] = s
        local[int(u)].append(float(s))

    tau = np.zeros(adj.shape[0], dtype=np.float32)
    for u in range(adj.shape[0]):
        tau[u] = float(np.median(local[u])) if local[u] else 0.0

    keep = np.zeros(rows.shape[0], dtype=bool)
    for i, (u, v) in enumerate(zip(rows, cols)):
        keep[i] = sims[i] >= min(tau[int(u)], tau[int(v)])

    pruned = sp.coo_matrix(
        (np.ones(int(np.sum(keep)), dtype=np.float32), (rows[keep], cols[keep])),
        shape=adj.shape,
    ).tocsr()
    return _symmetrize(pruned)


def feature_smoothing(adj: sp.csr_matrix, features: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """Graph feature smoothing over trusted adjacency."""
    adj = _symmetrize(adj)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    deg[deg == 0] = 1.0
    d_inv = sp.diags(1.0 / deg)
    agg = d_inv @ adj @ features
    smoothed = alpha * features + (1.0 - alpha) * np.asarray(agg, dtype=np.float32)
    return np.clip(smoothed.astype(np.float32), 0.0, 1.0)


def svd_low_rank_purification(adj: sp.csr_matrix, rank_k: int = 96) -> sp.csr_matrix:
    """Low-rank purification via top-k singular values."""
    adj = _symmetrize(adj).astype(np.float32)
    n = adj.shape[0]
    k = max(2, min(int(rank_k), n - 2))

    try:
        u, s, vt = sp.linalg.svds(adj, k=k)
        order = np.argsort(s)[::-1]
        u, s, vt = u[:, order], s[order], vt[order, :]
        recon = (u * s) @ vt
    except Exception:
        dense = adj.toarray()
        uu, ss, vv = np.linalg.svd(dense, full_matrices=False)
        recon = (uu[:, :k] * ss[:k]) @ vv[:k, :]

    recon = (recon + recon.T) / 2.0
    np.fill_diagonal(recon, 0.0)

    target_edges = int(adj.nnz // 2)
    tri_u, tri_v = np.triu_indices(n, k=1)
    scores = recon[tri_u, tri_v]

    if target_edges <= 0:
        return adj.copy()

    target_edges = min(target_edges, scores.size)
    top_idx = np.argpartition(scores, -target_edges)[-target_edges:]

    keep_u = tri_u[top_idx]
    keep_v = tri_v[top_idx]
    purified = sp.coo_matrix(
        (np.ones(keep_u.shape[0], dtype=np.float32), (keep_u, keep_v)),
        shape=(n, n),
    ).tocsr()
    return _symmetrize(purified)


def knn_graph_reconstruction(features: np.ndarray, k: int = 8) -> sp.csr_matrix:
    """
    Graph reconstruction from semantic feature space using cosine k-NN.
    """
    n = features.shape[0]
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, -1.0)
    knn_idx = np.argpartition(sim, -k, axis=1)[:, -k:]

    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    vals = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return _symmetrize(adj)


class OntologySemanticDefense:
    """
    Defense B: ontology-guided semantic validation + feature repair.
    """

    def __init__(self, ontology_path: str, topic_names: List[str] | None = None):
        self.ontology_path = ontology_path
        self.topic_names = topic_names or CORA_TOPICS
        self.ontology = get_ontology(ontology_path).load()
        self.compat, self.min_support = self._read_semantic_rules()

    def _read_semantic_rules(self) -> Tuple[Dict[str, set], Dict[str, float]]:
        compat = {t: {t} for t in self.topic_names}
        support = {t: 0.5 for t in self.topic_names}

        for ind in self.ontology.individuals():
            cls_names = [c.name for c in ind.is_a if hasattr(c, "name")]
            topic = next((c for c in cls_names if c in self.topic_names), None)
            if topic is None:
                continue

            topic_compat = {topic}
            if hasattr(ind, "compatibleWith"):
                for other in ind.compatibleWith:
                    o_names = [c.name for c in other.is_a if hasattr(c, "name")]
                    o_topic = next((c for c in o_names if c in self.topic_names), None)
                    if o_topic:
                        topic_compat.add(o_topic)
            compat[topic] = topic_compat

            if hasattr(ind, "minSemanticSupport"):
                ms = ind.minSemanticSupport
                if isinstance(ms, (list, tuple)):
                    if len(ms) > 0:
                        support[topic] = float(ms[0])
                elif ms is not None:
                    support[topic] = float(ms)

        return compat, support

    def _feature_topic_affinity(self, x: np.ndarray, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        n_features = x.shape[1]
        n_topics = len(self.topic_names)
        aff = np.zeros((n_features, n_topics), dtype=np.float32)

        for c in range(n_topics):
            idx = np.where(train_mask & (y == c))[0]
            if idx.size > 0:
                aff[:, c] = x[idx].mean(axis=0)

        row_sum = aff.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        return aff / row_sum

    def apply(
        self,
        data: Data,
        support_scale: float = 0.85,
        repair_strength: float = 0.4,
        contradiction_weight: float = 0.5,
    ) -> DefenseOutcome:
        adj = edge_index_to_csr(data.edge_index, data.num_nodes)
        x = data.x.cpu().numpy().astype(np.float32)
        y = data.y.cpu().numpy()
        train_mask = data.train_mask.cpu().numpy().astype(bool)

        affinity = self._feature_topic_affinity(x, y, train_mask)
        topic_scores = x @ affinity
        topic_scores = topic_scores / np.maximum(topic_scores.sum(axis=1, keepdims=True), 1e-9)
        dominant = np.argmax(topic_scores, axis=1)

        rows, cols = adj.nonzero()
        keep = np.zeros(rows.shape[0], dtype=bool)
        contradiction_nodes = np.zeros(data.num_nodes, dtype=np.float32)

        for i, (u, v) in enumerate(zip(rows, cols)):
            tu = self.topic_names[int(dominant[u])]
            tv = self.topic_names[int(dominant[v])]
            compatible = tv in self.compat.get(tu, {tu})
            semantic_support = float(np.dot(topic_scores[u], topic_scores[v]))
            threshold = 0.5 * (self.min_support.get(tu, 0.5) + self.min_support.get(tv, 0.5)) * float(support_scale)
            if not compatible and semantic_support < threshold:
                contradiction_nodes[int(u)] += 1.0
                contradiction_nodes[int(v)] += 1.0
            keep[i] = compatible or (semantic_support >= threshold)

        adj_def = sp.coo_matrix(
            (np.ones(int(np.sum(keep)), dtype=np.float32), (rows[keep], cols[keep])),
            shape=adj.shape,
        ).tocsr()
        adj_def = _symmetrize(adj_def)

        # Semantic rule 2: feature-label domain mismatch
        label_mismatch = (dominant != y).astype(np.float32)

        # Semantic rule 3: local neighborhood deviation
        deg = np.asarray(adj_def.sum(axis=1)).reshape(-1).astype(np.float32)
        deg_safe = deg.copy()
        deg_safe[deg_safe == 0] = 1.0
        x_neighbor = sp.diags(1.0 / deg_safe) @ adj_def @ x
        neighbor_deviation = np.linalg.norm(x - np.asarray(x_neighbor), axis=1)
        neighbor_deviation = neighbor_deviation / (np.max(neighbor_deviation) + 1e-9)

        suspicious_score = (
            contradiction_weight * (contradiction_nodes / (np.max(contradiction_nodes) + 1e-9))
            + 0.3 * label_mismatch
            + 0.2 * neighbor_deviation
        )

        # Use z-score threshold instead of fixed quantile.
        # Quantile always flags 25% of nodes — even on a nearly-clean graph —
        # causing unnecessary feature repair that hurts accuracy.
        # Z-score: if suspicious scores are uniformly low (lightly attacked),
        # σ is small so the threshold is high and almost no nodes are flagged.
        s_mu = float(np.mean(suspicious_score))
        s_sigma = float(np.std(suspicious_score)) + 1e-9
        suspicious_mask = suspicious_score > max(s_mu + 1.5 * s_sigma, 0.40)

        x_sem = topic_scores @ affinity.T
        x_smooth = feature_smoothing(adj_def, x, alpha=0.65)
        x_consensus = 0.6 * x_sem + 0.4 * x_smooth
        x_repair = x.copy()
        x_repair[suspicious_mask] = (1.0 - repair_strength) * x[suspicious_mask] + repair_strength * x_consensus[suspicious_mask]
        x_repair = np.clip(x_repair.astype(np.float32), 0.0, 1.0)

        # kNN reconstruction only for heavily attacked graphs (many suspicious nodes).
        # Old trigger of 0.15 always fired because the quantile mask gave exactly 25%.
        if float(np.mean(suspicious_mask)) > 0.40:
            adj_knn = knn_graph_reconstruction(x_repair, k=8)
            adj_def = _symmetrize((adj_def + adj_knn).sign().tocsr())

        defended = data.clone()
        defended.edge_index = csr_to_edge_index(adj_def).to(data.edge_index.device)
        defended.x = torch.tensor(x_repair, dtype=data.x.dtype, device=data.x.device)

        return DefenseOutcome(
            name="OntologyDefense",
            data=defended,
            metadata={
                "kept_edges": float(adj_def.nnz / 2),
                "repair_strength": float(repair_strength),
                "suspicious_nodes": float(np.sum(suspicious_mask)),
                "avg_suspicious_score": float(np.mean(suspicious_score)),
            },
        )


class TripleDefense:
    """Runs independent structural, ontology, and hybrid defenses."""

    def __init__(self, ontology_path: str):
        self.ontology = OntologySemanticDefense(ontology_path=ontology_path)

    def structural(self, data: Data) -> DefenseOutcome:
        adj = edge_index_to_csr(data.edge_index, data.num_nodes)
        x = data.x.cpu().numpy().astype(np.float32)
        adj_pruned = adaptive_jaccard_pruning(adj, x)
        adj_purified = svd_low_rank_purification(adj_pruned, rank_k=96)
        x_smooth = feature_smoothing(adj_purified, x, alpha=0.8)
        adj_recon = knn_graph_reconstruction(x_smooth, k=8)
        adj_final = _symmetrize((adj_purified + adj_recon).sign().tocsr())
        x_final = feature_smoothing(adj_final, x_smooth, alpha=0.82)

        defended = data.clone()
        defended.edge_index = csr_to_edge_index(adj_final).to(data.edge_index.device)
        defended.x = torch.tensor(x_final, dtype=data.x.dtype, device=data.x.device)
        return DefenseOutcome(
            name="StructuralDefense",
            data=defended,
            metadata={"final_edges": float(adj_final.nnz / 2)},
        )

    def ontology_only(self, data: Data) -> DefenseOutcome:
        # Lower support_scale reduces cross-topic edge pruning aggressiveness.
        # Cora has ~19% legitimate cross-topic citations; pruning them hurts.
        return self.ontology.apply(data, support_scale=0.35, repair_strength=0.30)

    def hybrid(self, data: Data) -> DefenseOutcome:
        structural_out = self.structural(data)
        # After structural cleaning, remaining cross-topic edges are mostly legitimate;
        # use a low support_scale to preserve them.
        ontology_out = self.ontology.apply(structural_out.data, support_scale=0.25, repair_strength=0.18)
        return DefenseOutcome(
            name="HybridDefense",
            data=ontology_out.data,
            metadata={
                "structural_edges": structural_out.metadata.get("final_edges", 0.0),
                "ontology_kept_edges": ontology_out.metadata.get("kept_edges", 0.0),
            },
        )
