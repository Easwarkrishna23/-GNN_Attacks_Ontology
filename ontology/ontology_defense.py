"""
ontology_defense.py

Ontology-guided semantic defense implementation.

This file focuses on "knowledge-guided robust message passing":
- build an ontology (concepts + constraints) from dataset features/labels
- compute semantic trust for edges (not just raw similarity)
- project/repair node features using semantic logic
- prune or down-weight adversarial edges based on semantic contradictions
- add semantic consistency regularization during training

Integration contract with this repo
----------------------------------
The repo uses a lightweight GraphData container (datasets/simple_data.py).
This defense returns:
- defended features X'
- defended edge_index (optionally pruned)
- edge weights for message passing (edge_weight)
- an optional semantic regularizer callable to add to loss
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from datasets.simple_data import GraphData

from .ontology_builder import OntologyArtifacts, OntologyBuilder, OntologyConfig, _row_normalize
from .ontology_rules import OntologyRuleEngine, RuleReport
from .ontology_export import OntologyExporter


class DefenseVariant(str, Enum):
    COOCCURRENCE_ONLY = "cooccurrence_only"
    LABEL_AFFINITY_ONLY = "label_affinity_only"
    OWL_RULES_ONLY = "owl_rules_only"
    SEMANTIC_PROJECTION_ONLY = "semantic_projection_only"
    FULL_ONTOLOGY = "full_ontology"


@dataclass
class DefenseOutput:
    variant: DefenseVariant
    data: GraphData
    edge_weight: Optional[torch.Tensor]
    # report contains rule triggers / repairs; can be exported for paper appendix.
    rule_report: Optional[RuleReport]
    # ontology artifacts used for the defense (useful for ablations).
    artifacts: OntologyArtifacts


class OntologyGuidedDefense:
    """
    End-to-end semantic defense.

    Typical usage:
      defense = OntologyGuidedDefense(dataset_name="Cora")
      defense.fit(clean_data)
      out = defense.defend(attacked_data, variant=DefenseVariant.FULL_ONTOLOGY)
      train(model, out.data, edge_weight=out.edge_weight, reg=defense.regularizer(...))
    """

    def __init__(
        self,
        dataset_name: str,
        config: Optional[OntologyConfig] = None,
        rule_engine: Optional[OntologyRuleEngine] = None,
        export_dir: str = "results/ontologies",
        export_owl: bool = True,
    ):
        cfg = config or OntologyConfig(dataset_name=dataset_name)
        self.dataset_name = dataset_name
        self.config = cfg
        self.builder = OntologyBuilder(cfg)
        self.rules = rule_engine or OntologyRuleEngine()
        self.export_dir = export_dir
        self.export_owl = bool(export_owl)

        self.artifacts: Optional[OntologyArtifacts] = None

    def fit(self, data: GraphData, class_names: Optional[list[str]] = None) -> OntologyArtifacts:
        """
        Build ontology artifacts from the clean training graph.
        """
        X = data.x.detach().cpu().numpy().astype(np.float32)
        y = data.y.detach().cpu().numpy().astype(np.int64)
        train_mask = data.train_mask.detach().cpu().numpy().astype(bool)

        artifacts = self.builder.build(X, y, train_mask, class_names=class_names)
        self.artifacts = artifacts

        if self.export_owl:
            out_dir = Path(self.export_dir) / self.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                exporter = OntologyExporter(base_iri=f"http://example.org/{self.dataset_name.lower()}#")
                exporter.export_all(artifacts, out_dir=str(out_dir))
            except Exception as e:
                # Keep training/defense functional even if RDF tooling is not installed.
                print(f"[Ontology] Export skipped (install rdflib for .owl/.rdf/.ttl). Reason: {e}")
        return artifacts

    def _require_fit(self) -> OntologyArtifacts:
        if self.artifacts is None:
            raise RuntimeError("OntologyGuidedDefense.fit() must be called before defend().")
        return self.artifacts

    def _semantic_projection_from_topics(self, X: np.ndarray, artifacts: OntologyArtifacts, lam: float) -> np.ndarray:
        """
        Ontology-based feature projection (semantic reconstruction).

        Let P be node-topic distribution (N,C) and A be feature-class affinity (F,C).
        A semantic reconstruction of features is:
          X_sem = P * A^T  (N,F)
        Then project:
          X' = (1-lam) X + lam X_sem
        """
        lam = float(lam)
        P = artifacts.node_topic_confidence  # (N,C)
        A = artifacts.feature_class_affinity  # (F,C)
        X_sem = P @ A.T  # (N,F)
        # blend with original
        Xp = (1.0 - lam) * X + lam * X_sem
        # keep non-negativity; Planetoid features are non-negative
        Xp = np.clip(Xp, 0.0, None).astype(np.float32)
        return Xp

    def _cooccurrence_projection(self, X: np.ndarray, artifacts: OntologyArtifacts, lam: float) -> np.ndarray:
        """
        Co-occurrence-only feature projection:
          X' = (1-lam) X + lam * X W
        where W is a row-normalized feature co-occurrence matrix.
        """
        lam = float(lam)
        W = artifacts.feature_cooccur
        if W.nnz == 0:
            return X.copy().astype(np.float32)
        # row-normalize W
        row_sum = np.array(W.sum(axis=1)).reshape(-1)
        row_sum[row_sum == 0] = 1.0
        Dinv = sp.diags(1.0 / row_sum)
        Wn = (Dinv @ W).tocsr()
        Xp = (1.0 - lam) * X + lam * (X @ Wn.T)  # (N,F)
        Xp = np.clip(Xp, 0.0, None).astype(np.float32)
        return Xp

    def _rule_based_repair(self, X: np.ndarray, artifacts: OntologyArtifacts, anomaly_thresh: float = 0.15) -> Tuple[np.ndarray, RuleReport]:
        """
        Apply rule-based repair to nodes that violate semantic constraints.
        """
        report = RuleReport(node_anomalies=[], edge_flags=[], repairs=[])
        # recompute anomaly on the attacked features for better detection
        anomaly = self.rules.detect_semantic_contradictions(artifacts, X, nodes=None, report=report)
        Xr = X.copy()
        bad = np.where(anomaly > float(anomaly_thresh))[0]
        for n in bad.tolist():
            Xr[n] = self.rules.repair_node_features(artifacts, Xr, node_id=int(n), report=report)
        return Xr.astype(np.float32), report

    def defend(
        self,
        data: GraphData,
        variant: DefenseVariant = DefenseVariant.FULL_ONTOLOGY,
        lam: float = 0.3,
        prune_threshold: float = 0.15,
        anomaly_repair_threshold: float = 0.15,
    ) -> DefenseOutput:
        """
        Produce defended (X', edge_index', edge_weight) according to the selected variant.
        """
        artifacts = self._require_fit()
        X = data.x.detach().cpu().numpy().astype(np.float32)
        y = data.y.detach().cpu().numpy().astype(np.int64)
        edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)

        report: Optional[RuleReport] = None
        X_def = X.copy()

        if variant == DefenseVariant.COOCCURRENCE_ONLY:
            X_def = self._cooccurrence_projection(X_def, artifacts, lam=lam)
        elif variant == DefenseVariant.LABEL_AFFINITY_ONLY:
            # Only semantic projection using label affinity (no co-occurrence, no rules, no edge trust).
            X_def = self._semantic_projection_from_topics(X_def, artifacts, lam=lam)
        elif variant == DefenseVariant.OWL_RULES_ONLY:
            X_def, report = self._rule_based_repair(X_def, artifacts, anomaly_thresh=anomaly_repair_threshold)
        elif variant == DefenseVariant.SEMANTIC_PROJECTION_ONLY:
            # Combine co-occurrence projection and topic reconstruction, but no edge pruning/trust.
            X_def = self._cooccurrence_projection(X_def, artifacts, lam=lam * 0.5)
            X_def = self._semantic_projection_from_topics(X_def, artifacts, lam=lam)
            X_def, report = self._rule_based_repair(X_def, artifacts, anomaly_thresh=anomaly_repair_threshold)
        elif variant == DefenseVariant.FULL_ONTOLOGY:
            X_def = self._cooccurrence_projection(X_def, artifacts, lam=lam * 0.5)
            X_def = self._semantic_projection_from_topics(X_def, artifacts, lam=lam)
            X_def, report = self._rule_based_repair(X_def, artifacts, anomaly_thresh=anomaly_repair_threshold)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Recompute node-topic confidence from defended features (keeps defense consistent with changes).
        node_topic = artifacts.node_topic_confidence
        if variant in (DefenseVariant.SEMANTIC_PROJECTION_ONLY, DefenseVariant.FULL_ONTOLOGY, DefenseVariant.LABEL_AFFINITY_ONLY, DefenseVariant.COOCCURRENCE_ONLY):
            node_topic = _row_normalize(X_def @ artifacts.feature_class_affinity + artifacts.config.eps, eps=artifacts.config.eps)

        # Edge trust / pruning only for FULL_ONTOLOGY and OWL_RULES_ONLY (rules imply semantics on edges).
        edge_weight: Optional[np.ndarray] = None
        edge_index_def = edge_index
        if variant in (DefenseVariant.FULL_ONTOLOGY, DefenseVariant.OWL_RULES_ONLY):
            # Create/extend report for edge flags
            if report is None:
                report = RuleReport(node_anomalies=[], edge_flags=[], repairs=[])
            node_conf = self.rules.compute_node_confidence(artifacts)
            trust = self.rules.semantic_edge_trust(artifacts, edge_index, node_topic=node_topic, node_confidence=node_conf, report=report)
            # Prune edges below threshold (ontology-aware pruning).
            keep = trust >= float(prune_threshold)
            edge_index_def = edge_index[:, keep]
            edge_weight = trust[keep]
        else:
            # no pruning; if projection-only, keep original edges with unit weights
            edge_weight = None

        # Build defended GraphData (preserve masks and labels)
        data_def = data.clone()
        data_def.x = torch.tensor(X_def, dtype=data.x.dtype, device=data.x.device)
        data_def.edge_index = torch.tensor(edge_index_def, dtype=torch.long, device=data.edge_index.device)

        # edge_weight returned separately; integration code should attach it to data if model supports.
        ew_t = None if edge_weight is None else torch.tensor(edge_weight, dtype=torch.float32, device=data.edge_index.device)
        return DefenseOutput(
            variant=variant,
            data=data_def,
            edge_weight=ew_t,
            rule_report=report,
            artifacts=artifacts,
        )

    def semantic_regularizer(
        self,
        logits: torch.Tensor,
        data: GraphData,
        edge_weight: Optional[torch.Tensor] = None,
        lam_edge: float = 0.3,
        lam_topic: float = 0.3,
    ) -> torch.Tensor:
        """
        Ontology-aware loss regularization.

        Terms:
        1) Edge consistency: trusted edges should have similar predicted distributions.
           sum_{(u,v)} w_uv ||p_u - p_v||^2
        2) Topic alignment: predicted probabilities should not deviate too far from semantic topic distribution.
           sum_u KL(p_u || topic_u)
        """
        artifacts = self._require_fit()
        p = torch.softmax(logits, dim=1)
        edge_index = data.edge_index
        src = edge_index[0]
        dst = edge_index[1]

        if edge_weight is None:
            w = torch.ones((edge_index.size(1),), device=logits.device, dtype=logits.dtype)
        else:
            w = edge_weight.to(device=logits.device, dtype=logits.dtype)

        # Edge consistency
        diff = p[src] - p[dst]
        edge_term = (w.unsqueeze(1) * (diff * diff)).sum() / (w.sum() + 1e-12)

        # Topic alignment (use artifacts computed on clean; this is a regularizer, not a hard constraint)
        topic = torch.tensor(artifacts.node_topic_confidence, device=logits.device, dtype=logits.dtype)
        topic = topic / (topic.sum(dim=1, keepdim=True) + 1e-12)
        kl = (p * (torch.log(p + 1e-12) - torch.log(topic + 1e-12))).sum(dim=1).mean()

        return float(lam_edge) * edge_term + float(lam_topic) * kl
