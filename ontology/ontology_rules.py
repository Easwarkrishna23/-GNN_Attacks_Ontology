"""
ontology_rules.py

Rule engine for ontology-guided semantic defense.

The purpose of this module is to make the "semantic logic" explicit and auditable.
Rather than a single similarity matrix, we define:
- semantic consistency rules (node-topic and edge-topic coherence),
- membership rules (feature evidence implies topic membership),
- impossibility rules (contradictory feature combinations),
- cross-domain contradiction rules (topics far apart should not strongly co-occur),
- reinforcement rules (consistent evidence increases confidence),
- anomaly detection rules (violation patterns suggest adversarial edits),
- repair rules (remove contradictory features; reinforce coherent ones),
and exportable SWRL-style strings for documentation/Protégé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ontology_builder import OntologyArtifacts, _safe_topk_indices


@dataclass(frozen=True)
class SWRLRule:
    """
    Minimal SWRL rule representation.

    `body` and `head` are SWRL-like strings for export/documentation.
    """

    name: str
    body: str
    head: str

    def to_swrl(self) -> str:
        return f"{self.name}: {self.body} -> {self.head}"


@dataclass
class RuleReport:
    """
    Explains which rules were triggered and how they affected trust/repair.
    """

    # Node anomaly explanations: list of (node_id, score, reason)
    node_anomalies: List[Tuple[int, float, str]]
    # Edge trust explanations: list of (src, dst, trust, reason)
    edge_flags: List[Tuple[int, int, float, str]]
    # Feature repair explanations: list of (node_id, removed_features, added_features, reason)
    repairs: List[Tuple[int, List[int], List[int], str]]


class OntologyRuleEngine:
    """
    Ontology semantic logic as explicit rules.

    The engine consumes OntologyArtifacts (topic distributions, contradictions, co-occurrence)
    and produces:
    - edge trust scores (semantic edge trust)
    - node confidence scores
    - feature repair suggestions
    """

    def __init__(
        self,
        contradiction_penalty: float = 0.6,
        cross_topic_penalty: float = 0.2,
        reinforcement_gain: float = 0.1,
        repair_remove_max: int = 10,
        repair_add_max: int = 10,
    ):
        self.contradiction_penalty = float(contradiction_penalty)
        self.cross_topic_penalty = float(cross_topic_penalty)
        self.reinforcement_gain = float(reinforcement_gain)
        self.repair_remove_max = int(repair_remove_max)
        self.repair_add_max = int(repair_add_max)

    def semantic_consistency_rules(self) -> List[SWRLRule]:
        return [
            SWRLRule(
                name="EdgeTrustByTopicAgreement",
                body="hasNeighbor(?u,?v) ^ topicSimHigh(?u,?v)",
                head="edgeTrusted(?u,?v)",
            ),
            SWRLRule(
                name="ContradictoryFeatures",
                body="hasFeature(?u,?f1) ^ hasFeature(?u,?f2) ^ contradicts(?f1,?f2)",
                head="anomalousNode(?u)",
            ),
            SWRLRule(
                name="CrossDomainContradiction",
                body="dominantTopic(?u,?t1) ^ dominantTopic(?v,?t2) ^ farApart(?t1,?t2)",
                head="edgeUntrusted(?u,?v)",
            ),
            SWRLRule(
                name="ReinforceMembership",
                body="hasFeature(?u,?f) ^ indicatesTopic(?f,?t) ^ strongAffinity(?f,?t)",
                head="boostTopicConfidence(?u,?t)",
            ),
            # Cora-specific rules (defense planning semantics)
            SWRLRule(
                name="TopicMismatchSuspiciousCitation",
                body="CitationEdge(?e) ^ citesFrom(?e,?u) ^ citesTo(?e,?v) ^ topicMismatch(?u,?v) ^ lowSimilarity(?e)",
                head="SuspiciousEdge(?e)",
            ),
            SWRLRule(
                name="HomophilyCollapseTriggersPurification",
                body="Dataset(?d) ^ hasHomophilyRatio(?d,?h) ^ swrlb:lessThan(?h,0.45)",
                head="GraphPurificationDefense(?def)",
            ),
            SWRLRule(
                name="EmbeddingDriftTriggersAdvTraining",
                body="Paper(?p) ^ embeddingDrift(?p,?s) ^ swrlb:greaterThan(?s,0.30)",
                head="AdversarialTrainingDefense(?def)",
            ),
            SWRLRule(
                name="BridgeNodeVulnerabilityIsolation",
                body="Paper(?p) ^ BridgeNodeVulnerability(?v) ^ hasVulnerability(?p,?v)",
                head="SubgraphIsolationDefense(?def)",
            ),
        ]

    def compute_node_confidence(self, artifacts: OntologyArtifacts) -> np.ndarray:
        # Confidence is inverse anomaly score. (1 - anomaly) is already in artifacts export helper.
        return (1.0 - artifacts.node_anomaly).astype(np.float32)

    def _count_contradictions_for_node(
        self,
        present_idx: np.ndarray,
        contradiction_pairs: set[Tuple[int, int]],
        max_checks: int = 2000,
    ) -> int:
        # present_idx: sorted feature indices present in node
        if present_idx.size < 2 or len(contradiction_pairs) == 0:
            return 0
        present_set = set(int(i) for i in present_idx.tolist())
        cnt = 0
        checks = 0
        for (i, j) in contradiction_pairs:
            if i in present_set and j in present_set:
                cnt += 1
            checks += 1
            if checks >= max_checks:
                break
        return cnt

    def semantic_edge_trust(
        self,
        artifacts: OntologyArtifacts,
        edge_index: np.ndarray,
        node_topic: Optional[np.ndarray] = None,
        node_confidence: Optional[np.ndarray] = None,
        report: Optional[RuleReport] = None,
    ) -> np.ndarray:
        """
        Compute semantic trust for each edge using:
        - topic similarity (cosine on topic distributions)
        - penalties for node contradictions/anomalies
        - optional cross-topic penalty if dominant topics differ strongly
        """
        cfg = artifacts.config
        N = artifacts.node_topic_confidence.shape[0]
        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)

        P = node_topic if node_topic is not None else artifacts.node_topic_confidence
        conf = node_confidence if node_confidence is not None else self.compute_node_confidence(artifacts)

        p_src = P[src]
        p_dst = P[dst]
        num = np.sum(p_src * p_dst, axis=1)
        den = np.linalg.norm(p_src, axis=1) * np.linalg.norm(p_dst, axis=1)
        sim = (num / (den + cfg.eps)).astype(np.float32)

        trust = sim * conf[src] * conf[dst]

        # Cora-specific: penalize edges incident to bridge-vulnerable nodes when topics mismatch.
        # This captures "bridge node vulnerability" and "topic mismatch vulnerability".
        vul = artifacts.vulnerability_scores if hasattr(artifacts, "vulnerability_scores") else {}
        bridge = vul.get("BridgeNodeVulnerability", None)
        mismatch = vul.get("TopicMismatchVulnerability", None)
        if bridge is not None and mismatch is not None:
            b = bridge[src] * bridge[dst]
            # if either endpoint has high mismatch tendency, reduce trust further
            mfac = 0.5 * (mismatch[src] + mismatch[dst])
            penalty = np.clip(b * mfac, 0.0, 1.0).astype(np.float32)
            trust = trust * (1.0 - 0.35 * penalty)

        # Cross-topic penalty: if dominant topics differ, reduce trust slightly
        dom_src = p_src.argmax(axis=1)
        dom_dst = p_dst.argmax(axis=1)
        diff = (dom_src != dom_dst).astype(np.float32)
        trust = trust * (1.0 - self.cross_topic_penalty * diff)

        trust = np.clip(trust, 0.0, 1.0).astype(np.float32)

        if report is not None:
            # record a small sample for explainability
            for k in range(min(50, trust.size)):
                reason = "topic-similarity"
                if diff[k] > 0:
                    reason += "+cross-topic-penalty"
                report.edge_flags.append((int(src[k]), int(dst[k]), float(trust[k]), reason))
        return trust

    def detect_semantic_contradictions(
        self,
        artifacts: OntologyArtifacts,
        X: np.ndarray,
        nodes: Optional[Sequence[int]] = None,
        report: Optional[RuleReport] = None,
    ) -> np.ndarray:
        """
        Recompute anomaly score using full contradiction pair checks for selected nodes.
        Returns anomaly score in [0,1] for those nodes (or all nodes if nodes is None).
        """
        N, F = X.shape
        nodes_idx = np.arange(N, dtype=np.int64) if nodes is None else np.array(list(nodes), dtype=np.int64)
        anomaly = np.zeros((nodes_idx.size,), dtype=np.float32)
        denom = max(1, min(len(artifacts.contradiction_pairs), 2000))
        for ii, n in enumerate(nodes_idx):
            present = np.where(X[n] > 0)[0]
            cnt = self._count_contradictions_for_node(present, artifacts.contradiction_pairs)
            anomaly[ii] = float(cnt) / float(denom)
            if report is not None and anomaly[ii] > 0:
                report.node_anomalies.append((int(n), float(anomaly[ii]), "contradictory-feature-pairs"))
        return np.clip(anomaly, 0.0, 1.0).astype(np.float32)

    def repair_node_features(
        self,
        artifacts: OntologyArtifacts,
        X: np.ndarray,
        node_id: int,
        target_topic: Optional[int] = None,
        report: Optional[RuleReport] = None,
    ) -> np.ndarray:
        """
        Rule-based semantic repair for a single node feature vector.

        Logic:
        - infer dominant topic from node_topic_confidence unless target_topic is provided
        - remove a small number of features that strongly indicate other topics and are contradictory
        - add (reinforce) a small number of features strongly indicative of dominant topic
        """
        cfg = artifacts.config
        x = X[node_id].copy()
        present = np.where(x > 0)[0].astype(np.int64)

        topic = int(target_topic) if target_topic is not None else int(artifacts.node_topic_confidence[node_id].argmax())
        aff = artifacts.feature_class_affinity  # (F,C)

        # Features that strongly indicate a non-dominant topic
        dom_feat_topic = aff[present].argmax(axis=1)
        dom_feat_conf = aff[present].max(axis=1)
        off_topic = present[(dom_feat_topic != topic) & (dom_feat_conf >= cfg.strong_affinity)]

        # Remove up to repair_remove_max off-topic features with highest off-topic confidence
        if off_topic.size > 0:
            confs = aff[off_topic, dom_feat_topic[(dom_feat_topic != topic) & (dom_feat_conf >= cfg.strong_affinity)]]
            to_remove = off_topic[_safe_topk_indices(confs, min(self.repair_remove_max, confs.size))]
        else:
            to_remove = np.array([], dtype=np.int64)

        # Add up to repair_add_max features strongly indicative of the dominant topic,
        # preferring those that co-occur with existing features (semantic reinforcement).
        topic_scores = aff[:, topic].copy()
        # Avoid adding already-present features.
        topic_scores[present] = 0.0
        # Reinforce by co-occurrence with present features.
        if artifacts.feature_cooccur.nnz > 0 and present.size > 0:
            # average co-occur with present features
            co = artifacts.feature_cooccur[:, present].mean(axis=1)
            co = np.asarray(co).reshape(-1)
            topic_scores = topic_scores + self.reinforcement_gain * co.astype(np.float32)
        add_candidates = _safe_topk_indices(topic_scores, self.repair_add_max)

        # Apply changes (binary features in Planetoid are usually 0/1 after normalization; we keep as sparse flips).
        removed = []
        for f in to_remove.tolist():
            if x[f] > 0:
                x[f] = 0.0
                removed.append(int(f))
        added = []
        for f in add_candidates.tolist():
            if x[f] == 0:
                x[f] = 1.0
                added.append(int(f))

        if report is not None and (removed or added):
            report.repairs.append((int(node_id), removed, added, f"repair toward topic={topic}"))
        return x.astype(np.float32)
