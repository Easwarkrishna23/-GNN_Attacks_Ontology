"""
ontology_builder.py

Builds an OWL-friendly ontology schema and semantic matrices for GNN defense.

This module deliberately separates:
1) Ontology (concepts, properties, constraints, rules) as *symbolic structure*.
2) Semantic matrices (node-topic, feature-topic, contradiction sets) as *numeric
   artifacts* to integrate into GNN training/inference.

Why ontology != similarity pruning
---------------------------------
Similarity pruning: edge trust is derived directly from local similarity of two
feature vectors (cosine/Jaccard). There is no explicit semantics, constraints,
or inheritance.

Ontology defense (this module): semantics is represented by
- typed concepts (topics, subtopics, features),
- explicit properties (indicatesTopic, contradicts, coOccursWith),
- constraints/rules (impossible combinations, contradiction filtering),
- inheritance scoring (subtopic inherits parent topic),
and these symbolic structures *drive* numeric trust and repair signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class OntologyConfig:
    dataset_name: str = "Cora"
    # How many top features to consider per class when building semantic mappings.
    topk_features_per_class: int = 40
    # How many subtopics to create per class (heuristic clustering over top features).
    subtopics_per_class: int = 3
    # Co-occurrence threshold (as fraction of nodes) used to form co-occurrence graph edges.
    cooccur_min_frac: float = 0.002
    # Threshold for marking a feature pair as contradictory if they are strongly class-specific
    # and rarely co-occur.
    contradiction_max_frac: float = 0.0005
    # Minimum normalized affinity for a feature to strongly indicate a class/topic.
    strong_affinity: float = 0.65
    # Weighting between co-occurrence semantics and label affinity semantics when building topic scores.
    semantic_weight: float = 0.6
    # Numeric stability epsilon.
    eps: float = 1e-12


@dataclass
class OntologyArtifacts:
    """
    Numeric artifacts derived from the ontology schema for GNN integration.
    """

    config: OntologyConfig
    feature_names: List[str]
    class_names: List[str]
    # feature -> class affinity (F, C), row-normalized
    feature_class_affinity: np.ndarray
    # node -> topic confidence (N, C), row-normalized
    node_topic_confidence: np.ndarray
    # feature-feature co-occurrence sparse graph (F, F)
    feature_cooccur: sp.csr_matrix
    # feature-feature contradiction pairs (set of (i,j) with i<j)
    contradiction_pairs: set[Tuple[int, int]]
    # per-node semantic anomaly score in [0,1]
    node_anomaly: np.ndarray
    # class hierarchy (topic -> list[subtopic])
    topic_hierarchy: Dict[str, List[str]]
    # concept inheritance score for subtopics (subtopic -> score)
    inheritance_score: Dict[str, float]

    def dominant_topic(self) -> np.ndarray:
        return self.node_topic_confidence.argmax(axis=1)


def _row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return mat / s


def _safe_topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, vec.size)
    # argpartition for speed
    idx = np.argpartition(-vec, k - 1)[:k]
    idx = idx[np.argsort(-vec[idx])]
    return idx.astype(np.int64)


class OntologyBuilder:
    """
    Builds an OWL-friendly ontology schema and numeric semantic artifacts.

    Inputs:
    - X: (N,F) node features (dense float; Planetoid features are typically row-normalized).
    - y: (N,) node labels (int), may contain unlabeled nodes but train_mask selects labeled nodes.
    - train_mask: (N,) boolean mask selecting training nodes for label affinity estimation.

    Outputs:
    - OntologyArtifacts containing semantic matrices and constraint sets.
    """

    def __init__(self, config: Optional[OntologyConfig] = None):
        self.config = config or OntologyConfig()

    def infer_feature_names(self, num_features: int) -> List[str]:
        # Planetoid datasets in many loaders do not ship the vocabulary mapping.
        # We generate stable feature ids. If you have a real vocab, pass it in directly.
        return [f"feat_{i:04d}" for i in range(int(num_features))]

    def infer_class_names(self, num_classes: int, provided: Optional[Sequence[str]] = None) -> List[str]:
        if provided is not None and len(provided) == num_classes:
            return list(provided)
        return [f"Topic_{i}" for i in range(int(num_classes))]

    def build_label_affinity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """
        Compute a feature->class affinity matrix A_fc (F,C), row-normalized.

        We estimate P(f|c) from training nodes and normalize per-feature across classes.
        """
        eps = self.config.eps
        N, F = X.shape
        A = np.zeros((F, num_classes), dtype=np.float32)
        for c in range(num_classes):
            idx = np.where(train_mask & (y == c))[0]
            if idx.size == 0:
                continue
            # mean feature activation within class c
            A[:, c] = X[idx].mean(axis=0)
        # Normalize per-feature: each row is a distribution over classes.
        A = _row_normalize(A + eps, eps=eps)
        return A

    def build_feature_cooccurrence(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Build a sparse feature co-occurrence graph.

        For binary features, co-occurrence counts can be computed as X^T X.
        For float features, we use (X>0) as presence.
        """
        cfg = self.config
        N, F = X.shape
        present = (X > 0).astype(np.float32)
        # Dense co-occurrence; for Citeseer (F~3703), this is ~13M entries which is feasible.
        C = present.T @ present  # (F,F)
        np.fill_diagonal(C, 0.0)
        # Threshold by absolute count based on fraction of nodes.
        min_count = max(1.0, float(cfg.cooccur_min_frac) * float(N))
        C[C < min_count] = 0.0
        coo = sp.coo_matrix(C)
        # Keep only top neighbors per feature to stay sparse.
        # We cap to 50 edges per feature.
        rows, cols, vals = coo.row, coo.col, coo.data
        if vals.size == 0:
            return sp.csr_matrix((F, F), dtype=np.float32)
        # group by row
        keep = np.zeros(vals.shape[0], dtype=bool)
        for i in np.unique(rows):
            idx = np.where(rows == i)[0]
            if idx.size <= 50:
                keep[idx] = True
                continue
            top = idx[np.argpartition(-vals[idx], 49)[:50]]
            keep[top] = True
        coo2 = sp.coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(F, F), dtype=np.float32)
        # Symmetrize and normalize weights to [0,1]
        mat = coo2.tocsr()
        mat = mat.maximum(mat.T)
        if mat.nnz > 0:
            vmax = float(mat.data.max())
            if vmax > 0:
                mat.data = (mat.data / vmax).astype(np.float32)
        return mat.tocsr()

    def detect_contradictions(
        self,
        feature_class_affinity: np.ndarray,
        feature_cooccur: sp.csr_matrix,
        num_nodes: int,
    ) -> set[Tuple[int, int]]:
        """
        Mark feature pairs as contradictory if:
        - each feature strongly indicates a different class, and
        - the pair rarely co-occurs.
        """
        cfg = self.config
        F, C = feature_class_affinity.shape
        strong = feature_class_affinity.max(axis=1) >= cfg.strong_affinity
        dom = feature_class_affinity.argmax(axis=1)

        # Build a quick lookup for "rare co-occur": if not present in sparse graph.
        # feature_cooccur has edges above cfg.cooccur_min_frac; we want even stricter for contradictions.
        # We'll also use raw threshold via cfg.contradiction_max_frac.
        max_count = max(1.0, float(cfg.contradiction_max_frac) * float(num_nodes))

        # We do not have raw counts anymore (only normalized), so we approximate rarity by absence.
        # Additionally, we mark contradictions across dominant classes if cooccur weight == 0.
        contradictions: set[Tuple[int, int]] = set()
        for i in range(F):
            if not strong[i]:
                continue
            for j in range(i + 1, F):
                if not strong[j]:
                    continue
                if dom[i] == dom[j]:
                    continue
                # if they do not co-occur frequently -> contradiction
                if feature_cooccur[i, j] == 0:
                    contradictions.add((i, j))
        return contradictions

    def build_topic_hierarchy(
        self,
        class_names: List[str],
        feature_class_affinity: np.ndarray,
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        Create a lightweight topic->subtopic hierarchy based on splitting top features into groups.

        This is not an external domain ontology; it is a dataset-driven "topic ontology" suitable
        for semantic constraint enforcement in the defense.
        """
        cfg = self.config
        F, C = feature_class_affinity.shape
        hierarchy: Dict[str, List[str]] = {}
        inheritance: Dict[str, float] = {}

        for c, cname in enumerate(class_names):
            aff = feature_class_affinity[:, c]
            top = _safe_topk_indices(aff, cfg.topk_features_per_class)
            # Split into cfg.subtopics_per_class groups (by rank chunks).
            subs: List[str] = []
            if top.size == 0:
                hierarchy[cname] = subs
                continue
            chunks = np.array_split(top, max(1, cfg.subtopics_per_class))
            for si, chunk in enumerate(chunks):
                sname = f"{cname}_Subtopic_{si+1}"
                subs.append(sname)
                # Inheritance score: average affinity of the chunk.
                inheritance[sname] = float(np.clip(aff[chunk].mean(), 0.0, 1.0))
            hierarchy[cname] = subs
        return hierarchy, inheritance

    def node_topic_scores(
        self,
        X: np.ndarray,
        feature_class_affinity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute node->topic confidence: S = X * A_fc.
        Row-normalize to a distribution over topics.
        """
        eps = self.config.eps
        S = X @ feature_class_affinity  # (N,C)
        return _row_normalize(S + eps, eps=eps)

    def node_anomaly_scores(
        self,
        X: np.ndarray,
        contradiction_pairs: set[Tuple[int, int]],
        max_pairs: int = 200,
    ) -> np.ndarray:
        """
        Compute a simple semantic anomaly score in [0,1]:
        fraction of contradictory feature pairs present in a node's feature set.

        To keep it fast, we sample up to max_pairs contradiction pairs.
        """
        N, F = X.shape
        present = X > 0
        pairs = list(contradiction_pairs)
        if len(pairs) == 0:
            return np.zeros((N,), dtype=np.float32)
        if len(pairs) > max_pairs:
            rng = np.random.default_rng(42)
            pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]

        hits = np.zeros((N,), dtype=np.float32)
        for i, j in pairs:
            hits += (present[:, i] & present[:, j]).astype(np.float32)
        # Normalize
        hits = hits / float(len(pairs))
        return np.clip(hits, 0.0, 1.0).astype(np.float32)

    def build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> OntologyArtifacts:
        """
        Build ontology artifacts from dataset tensors.

        Parameters
        - X: (N,F) float32 features
        - y: (N,) int labels
        - train_mask: (N,) bool
        """
        cfg = self.config
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (N,F).")
        N, F = X.shape
        num_classes = int(np.max(y) + 1)

        f_names = list(feature_names) if feature_names is not None else self.infer_feature_names(F)
        c_names = self.infer_class_names(num_classes, provided=class_names)

        affinity = self.build_label_affinity(X, y, train_mask, num_classes=num_classes)  # (F,C)
        cooccur = self.build_feature_cooccurrence(X)
        contradictions = self.detect_contradictions(affinity, cooccur, num_nodes=N)
        hierarchy, inheritance = self.build_topic_hierarchy(c_names, affinity)
        node_scores = self.node_topic_scores(X, affinity)
        anomaly = self.node_anomaly_scores(X, contradictions)

        return OntologyArtifacts(
            config=cfg,
            feature_names=f_names,
            class_names=c_names,
            feature_class_affinity=affinity.astype(np.float32),
            node_topic_confidence=node_scores.astype(np.float32),
            feature_cooccur=cooccur.astype(np.float32),
            contradiction_pairs=contradictions,
            node_anomaly=anomaly.astype(np.float32),
            topic_hierarchy=hierarchy,
            inheritance_score=inheritance,
        )

    def export_gnn_matrices(
        self,
        artifacts: OntologyArtifacts,
        edge_index: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export matrices for GNN use.

        Returns:
        - edge_trust: (E,) trust score in [0,1] for each edge in edge_index
        - node_confidence: (N,) confidence in [0,1] derived from anomaly (1 - anomaly)
        """
        cfg = artifacts.config
        N = artifacts.node_topic_confidence.shape[0]
        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)
        # topic similarity via cosine between topic distributions
        P = artifacts.node_topic_confidence
        p_src = P[src]
        p_dst = P[dst]
        num = np.sum(p_src * p_dst, axis=1)
        den = np.linalg.norm(p_src, axis=1) * np.linalg.norm(p_dst, axis=1)
        sim = num / (den + cfg.eps)
        # down-weight edges adjacent to anomalous nodes
        conf = 1.0 - artifacts.node_anomaly
        trust = sim * conf[src] * conf[dst]
        trust = np.clip(trust, 0.0, 1.0).astype(np.float32)
        return trust, conf.astype(np.float32)

