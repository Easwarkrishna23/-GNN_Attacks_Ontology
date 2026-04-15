"""
Adaptive semantic attack.

Goal: bypass defenses that rely on simple similarity pruning by injecting
semantically plausible perturbations that still violate *ontology constraints*
or exploit ontology-guided projections.

Threat model (evasion):
- training graph is untouched
- only inference-time node features are modified
- perturbation is constrained by a budget (number of feature flips per node)

Strategy implemented here:
1) Choose a target (wrong) topic/class for each node (e.g., the model's second-best class).
2) Add features with high ontology affinity to the target topic while ensuring:
   - low contradiction with existing features (avoid contradiction_pairs),
   - plausible co-occurrence (prefer features that co-occur with existing ones),
3) Optionally remove a few features strongly indicating the true class.

This is intentionally different from plain FGSM/feature-flip attacks:
it crafts perturbations that look semantically consistent at the feature level,
thereby bypassing naive similarity pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.simple_data import GraphData
from ontology.ontology_builder import OntologyArtifacts, _safe_topk_indices


@dataclass(frozen=True)
class AdaptiveSemanticAttackConfig:
    flips_per_node: int = 10
    remove_per_node: int = 3
    cooccur_weight: float = 0.2
    contradiction_block: bool = True
    seed: int = 42


class AdaptiveSemanticAttack:
    def __init__(self, config: Optional[AdaptiveSemanticAttackConfig] = None):
        self.cfg = config or AdaptiveSemanticAttackConfig()

    def _choose_target_class(self, probs: np.ndarray, true_y: int) -> int:
        # pick highest-prob class that is not the true class
        order = np.argsort(-probs)
        for c in order:
            if int(c) != int(true_y):
                return int(c)
        return int(order[0])

    def apply(
        self,
        data: GraphData,
        artifacts: OntologyArtifacts,
        probs: torch.Tensor,
        nodes: Optional[Sequence[int]] = None,
    ) -> Tuple[GraphData, dict]:
        """
        Apply adaptive semantic evasion attack to selected nodes.

        Inputs:
        - data: GraphData (features will be modified for inference only)
        - artifacts: ontology artifacts built on clean training data
        - probs: model probabilities/log-softmax converted to probs for target selection
        - nodes: nodes to attack; if None, attacks all test nodes

        Returns:
        - attacked GraphData clone with modified x
        - debug dict with per-node changes
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        attacked = data.clone()
        X = attacked.x.detach().cpu().numpy().astype(np.float32)
        y = attacked.y.detach().cpu().numpy().astype(np.int64)
        P = probs.detach().cpu().numpy().astype(np.float32)

        if nodes is None:
            mask = attacked.test_mask.detach().cpu().numpy().astype(bool)
            nodes_idx = np.where(mask)[0]
        else:
            nodes_idx = np.array(list(nodes), dtype=np.int64)

        A = artifacts.feature_class_affinity  # (F,C)
        co = artifacts.feature_cooccur  # (F,F) sparse
        contradictions = artifacts.contradiction_pairs

        debug = {"changes": []}

        for n in nodes_idx.tolist():
            true_c = int(y[n])
            tgt_c = self._choose_target_class(P[n], true_c)

            x = X[n].copy()
            present = np.where(x > 0)[0].astype(np.int64)
            present_set = set(int(i) for i in present.tolist())

            # Candidate add features: high affinity to target class, not already present.
            base_scores = A[:, tgt_c].copy()
            base_scores[present] = 0.0

            # Plausibility: co-occur with current features.
            if co.nnz > 0 and present.size > 0:
                co_mean = np.asarray(co[:, present].mean(axis=1)).reshape(-1).astype(np.float32)
                base_scores = base_scores + float(cfg.cooccur_weight) * co_mean

            add_candidates = _safe_topk_indices(base_scores, 10 * cfg.flips_per_node)
            to_add = []
            for f in add_candidates.tolist():
                if len(to_add) >= cfg.flips_per_node:
                    break
                if f in present_set:
                    continue
                if cfg.contradiction_block:
                    ok = True
                    for p in present:
                        i, j = (min(int(f), int(p)), max(int(f), int(p)))
                        if (i, j) in contradictions:
                            ok = False
                            break
                    if not ok:
                        continue
                to_add.append(int(f))

            # Candidate removals: strong affinity to true class (to reduce true evidence)
            rem_scores = A[present, true_c] if present.size > 0 else np.array([], dtype=np.float32)
            to_remove = []
            if rem_scores.size > 0 and cfg.remove_per_node > 0:
                rem_idx = _safe_topk_indices(rem_scores, min(cfg.remove_per_node, rem_scores.size))
                to_remove = [int(present[i]) for i in rem_idx.tolist()]

            x_before = present.tolist()
            for f in to_remove:
                x[f] = 0.0
            for f in to_add:
                x[f] = 1.0

            X[n] = x
            debug["changes"].append(
                {
                    "node": int(n),
                    "true_class": int(true_c),
                    "target_class": int(tgt_c),
                    "removed_features": to_remove,
                    "added_features": to_add,
                    "before_nnz": int(len(x_before)),
                    "after_nnz": int(int((x > 0).sum())),
                }
            )

        attacked.x = torch.tensor(X, dtype=attacked.x.dtype, device=attacked.x.device)
        return attacked, debug

