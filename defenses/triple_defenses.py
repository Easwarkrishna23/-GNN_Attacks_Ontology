from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from owlready2 import get_ontology

from utils.benchmark_metrics import csr_to_edge_index


CORA_TOPICS = [
    "CaseBased",
    "GeneticAlgorithms",
    "NeuralNetworks",
    "ProbabilisticMethods",
    "ReinforcementLearning",
    "RuleLearning",
    "Theory",
]


def _symmetrize(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _edge_jaccard_similarity(features_bin: np.ndarray, u: int, v: int) -> float:
    fu = features_bin[u]
    fv = features_bin[v]
    inter = float(np.sum((fu > 0) & (fv > 0)))
    union = float(np.sum((fu > 0) | (fv > 0)))
    if union == 0.0:
        return 0.0
    return inter / union


def adaptive_jaccard_pruning(adj: sp.csr_matrix, features: np.ndarray) -> sp.csr_matrix:
    """
    Adaptive Jaccard pruning:
    threshold(node) = median Jaccard among its incident edges.
    Edge (u,v) is kept if sim(u,v) >= min(th(u), th(v)).
    """
    adj = _symmetrize(adj)
    x_bin = (features > 0).astype(np.uint8)
    rows, cols = adj.nonzero()

    sims = np.zeros(rows.shape[0], dtype=np.float32)
    for i, (u, v) in enumerate(zip(rows, cols)):
        sims[i] = _edge_jaccard_similarity(x_bin, int(u), int(v))

    n = adj.shape[0]
    local = [[] for _ in range(n)]
    for i, (u, _v) in enumerate(zip(rows, cols)):
        local[int(u)].append(float(sims[i]))

    thresholds = np.zeros(n, dtype=np.float32)
    for u in range(n):
        if len(local[u]) == 0:
            thresholds[u] = 0.0
        else:
            thresholds[u] = float(np.median(np.asarray(local[u], dtype=np.float32)))

    keep = np.zeros(rows.shape[0], dtype=bool)
    for i, (u, v) in enumerate(zip(rows, cols)):
        tau = min(thresholds[int(u)], thresholds[int(v)])
        keep[i] = sims[i] >= tau

    pruned = sp.coo_matrix((np.ones(int(np.sum(keep)), dtype=np.float32), (rows[keep], cols[keep])), shape=adj.shape).tocsr()
    return _symmetrize(pruned)


def svd_low_rank_purification(adj: sp.csr_matrix, rank_k: int = 64) -> sp.csr_matrix:
    """
    Graph purification via top-k singular values.
    Keeps edge budget near original by selecting top-|E|/2 undirected entries.
    """
    adj = _symmetrize(adj).astype(np.float32)
    n = adj.shape[0]
    k = max(2, min(int(rank_k), n - 2))

    try:
        u, s, vt = sp.linalg.svds(adj, k=k)
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]
        recon = (u * s) @ vt
    except Exception:
        dense = adj.toarray()
        uu, ss, vv = np.linalg.svd(dense, full_matrices=False)
        recon = (uu[:, :k] * ss[:k]) @ vv[:k, :]

    recon = (recon + recon.T) / 2.0
    np.fill_diagonal(recon, 0.0)

    # Preserve original undirected edge count to avoid utility collapse.
    original_edges = int(adj.nnz // 2)
    tri_u, tri_v = np.triu_indices(n, k=1)
    tri_scores = recon[tri_u, tri_v]
    if original_edges <= 0:
        return adj.copy()

    original_edges = min(original_edges, tri_scores.size)
    top_idx = np.argpartition(tri_scores, -original_edges)[-original_edges:]
    keep_u = tri_u[top_idx]
    keep_v = tri_v[top_idx]

    purified = sp.coo_matrix((np.ones(keep_u.shape[0], dtype=np.float32), (keep_u, keep_v)), shape=(n, n)).tocsr()
    return _symmetrize(purified)


@dataclass
class OntologyArtifacts:
    topic_names: List[str]
    topic_compatibility: Dict[str, set]
    topic_min_support: Dict[str, float]


class SemanticValidator:
    """
    Ontology-driven semantic defense:
    1) infer node dominant topic from feature-topic affinity
    2) prune edges that violate ontology compatibility and low semantic support
    3) repair features by suppressing low-affinity tokens for dominant topic
    """

    def __init__(self, ontology_path: str, topic_names: List[str] | None = None):
        self.ontology_path = ontology_path
        self.topic_names = topic_names or CORA_TOPICS
        self.ontology = get_ontology(ontology_path).load()
        self.artifacts = self._read_ontology()

    def _read_ontology(self) -> OntologyArtifacts:
        compat: Dict[str, set] = {t: {t} for t in self.topic_names}
        mins: Dict[str, float] = {t: 0.5 for t in self.topic_names}

        for ind in self.ontology.individuals():
            cls_names = [c.name for c in ind.is_a if hasattr(c, "name")]
            topic_cls = next((c for c in cls_names if c in self.topic_names), None)
            if topic_cls is None:
                continue
            compat_set = set([topic_cls])
            if hasattr(ind, "compatibleWith"):
                for x in ind.compatibleWith:
                    xcls = [c.name for c in x.is_a if hasattr(c, "name")]
                    t = next((c for c in xcls if c in self.topic_names), None)
                    if t is not None:
                        compat_set.add(t)
            compat[topic_cls] = compat_set
            if hasattr(ind, "minSemanticSupport") and len(ind.minSemanticSupport) > 0:
                mins[topic_cls] = float(ind.minSemanticSupport[0])

        return OntologyArtifacts(topic_names=self.topic_names, topic_compatibility=compat, topic_min_support=mins)

    def build_feature_topic_affinity(self, features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        n_features = features.shape[1]
        n_topics = len(self.topic_names)
        affinity = np.zeros((n_features, n_topics), dtype=np.float32)
        for c in range(n_topics):
            idx = np.where(train_mask & (labels == c))[0]
            if idx.size == 0:
                continue
            affinity[:, c] = features[idx].mean(axis=0)
        row_sum = affinity.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return affinity / row_sum

    def defend(
        self,
        adj: sp.csr_matrix,
        features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        strength: float = 0.35,
        support_scale: float = 0.60,
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        adj = _symmetrize(adj)
        affinity = self.build_feature_topic_affinity(features, labels, train_mask)

        topic_scores = features @ affinity
        topic_scores = topic_scores / np.maximum(topic_scores.sum(axis=1, keepdims=True), 1e-9)
        dom = np.argmax(topic_scores, axis=1)

        # Edge pruning by ontology compatibility + semantic support
        rows, cols = adj.nonzero()
        keep = np.zeros(rows.shape[0], dtype=bool)

        for i, (u, v) in enumerate(zip(rows, cols)):
            tu = self.topic_names[int(dom[u])]
            tv = self.topic_names[int(dom[v])]
            compatible = tv in self.artifacts.topic_compatibility.get(tu, {tu})
            support = float(np.dot(topic_scores[u], topic_scores[v]))
            tau = 0.5 * (self.artifacts.topic_min_support.get(tu, 0.5) + self.artifacts.topic_min_support.get(tv, 0.5))
            keep[i] = compatible or (support >= tau * float(support_scale))

        adj_def = sp.coo_matrix((np.ones(int(np.sum(keep)), dtype=np.float32), (rows[keep], cols[keep])), shape=adj.shape).tocsr()
        adj_def = _symmetrize(adj_def)

        # Feature repair by semantic + structural projection:
        # X_sem = node_topic_scores * feature_topic_affinity^T
        # X_graph = D^-1 A X  (local smoothing over defended graph)
        x_sem = topic_scores @ affinity.T
        deg = np.asarray(adj_def.sum(axis=1)).reshape(-1).astype(np.float32)
        deg[deg == 0] = 1.0
        x_graph = sp.diags(1.0 / deg) @ adj_def @ features
        repaired_target = 0.5 * x_sem + 0.5 * np.asarray(x_graph, dtype=np.float32)
        repaired = (1.0 - float(strength)) * features + float(strength) * repaired_target
        repaired = np.clip(repaired.astype(np.float32), 0.0, 1.0)
        return adj_def, repaired


def apply_defense_to_data(data, adj_def: sp.csr_matrix, x_def: np.ndarray | None = None):
    defended = data.clone()
    edge_index, _ = csr_to_edge_index(adj_def)
    defended.edge_index = edge_index.to(data.edge_index.device)
    if x_def is not None:
        defended.x = torch.tensor(x_def, dtype=data.x.dtype, device=data.x.device)
    return defended
