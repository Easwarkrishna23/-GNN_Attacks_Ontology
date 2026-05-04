from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class AttackOutcome:
    name: str
    attack_type: str
    budget: float
    data: Data
    metadata: Dict[str, float]


def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def csr_to_edge_index(adj: sp.csr_matrix) -> torch.Tensor:
    coo = adj.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def _clone_with_graph(base_data: Data, adj: sp.csr_matrix, features: np.ndarray | None = None) -> Data:
    new_data = base_data.clone()
    new_data.edge_index = csr_to_edge_index(adj).to(base_data.edge_index.device)
    if features is not None:
        new_data.x = torch.tensor(features, dtype=base_data.x.dtype, device=base_data.x.device)
    return new_data


def _feature_cosine_similarity(features: np.ndarray, u: int, v: int, eps: float = 1e-9) -> float:
    fu = features[u]
    fv = features[v]
    num = float(np.dot(fu, fv))
    den = float(np.linalg.norm(fu) * np.linalg.norm(fv) + eps)
    return num / den


class PoisoningAttacks:
    """Poisoning attacks that modify graph prior to training."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def metattack_global(self, data: Data, budget: int) -> AttackOutcome:
        """
        Approximate global poisoning: rewire edges from high-centrality nodes
        toward low-similarity nodes to collapse homophily globally.
        """
        adj = edge_index_to_csr(data.edge_index, data.num_nodes).tolil()
        x = data.x.cpu().numpy().astype(np.float32)

        degrees = np.asarray(adj.sum(axis=1)).reshape(-1)
        high_deg_nodes = np.argsort(degrees)[::-1][: max(24, data.num_nodes // 16)]
        y = data.y.cpu().numpy()

        n = data.num_nodes
        rewired = 0
        for _ in range(int(budget)):
            u = int(self.rng.choice(high_deg_nodes))
            neighbors = np.where(np.asarray(adj[u].todense()).reshape(-1) > 0)[0]
            if neighbors.size == 0:
                continue
            same_label_nbrs = [int(v) for v in neighbors if y[int(v)] == y[u]]
            if same_label_nbrs:
                sim_same = np.array([_feature_cosine_similarity(x, u, v) for v in same_label_nbrs], dtype=np.float32)
                v_old = same_label_nbrs[int(np.argmax(sim_same))]
            else:
                v_old = int(self.rng.choice(neighbors))

            cands = self.rng.integers(0, n, size=64)
            cands = [int(v) for v in cands if v != u and adj[u, int(v)] == 0 and y[int(v)] != y[u]]
            if not cands:
                continue
            sims = np.array([_feature_cosine_similarity(x, u, v) for v in cands], dtype=np.float32)
            v_new = cands[int(np.argmin(sims))]

            adj[u, v_old] = 0
            adj[v_old, u] = 0
            adj[u, v_new] = 1
            adj[v_new, u] = 1
            rewired += 1

        adj = adj.tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        attacked = _clone_with_graph(data, adj)
        return AttackOutcome(
            name="Metattack",
            attack_type="poisoning",
            budget=float(budget),
            data=attacked,
            metadata={"rewired_edges": float(rewired)},
        )

    def nettack_targeted(self, data: Data, budget: int, target_fraction: float = 0.06) -> AttackOutcome:
        """
        Approximate targeted poisoning: choose correctly-labeled-like train nodes
        and attach low-similarity distractor neighbors.
        """
        adj = edge_index_to_csr(data.edge_index, data.num_nodes).tolil()
        x = data.x.cpu().numpy().astype(np.float32)
        y = data.y.cpu().numpy()

        target_pool = np.where(data.test_mask.cpu().numpy())[0]
        target_count = max(12, int(target_fraction * len(target_pool)))
        target_nodes = self.rng.choice(target_pool, size=min(target_count, len(target_pool)), replace=False)

        n = data.num_nodes
        rewired = 0
        for u in target_nodes:
            u = int(u)
            for _ in range(max(1, int(budget))):
                nbrs = np.where(np.asarray(adj[u].todense()).reshape(-1) > 0)[0]
                same_label_nbrs = [int(v) for v in nbrs if y[int(v)] == y[u]]
                if same_label_nbrs:
                    sim_same = np.array([_feature_cosine_similarity(x, u, v) for v in same_label_nbrs], dtype=np.float32)
                    v_remove = same_label_nbrs[int(np.argmax(sim_same))]
                    adj[u, v_remove] = 0
                    adj[v_remove, u] = 0

                cands = self.rng.integers(0, n, size=96)
                cands = [
                    int(v)
                    for v in cands
                    if v != u and adj[u, int(v)] == 0 and y[int(v)] != y[u]
                ]
                if not cands:
                    continue
                sims = np.array([_feature_cosine_similarity(x, u, v) for v in cands], dtype=np.float32)
                v_new = cands[int(np.argmin(sims))]
                adj[u, v_new] = 1
                adj[v_new, u] = 1
                rewired += 1

        adj = adj.tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        attacked = _clone_with_graph(data, adj)
        return AttackOutcome(
            name="Nettack",
            attack_type="poisoning",
            budget=float(budget),
            data=attacked,
            metadata={"rewired_edges": float(rewired), "target_nodes": float(len(target_nodes))},
        )

    def random_structure(self, data: Data, budget: int) -> AttackOutcome:
        """Random structure poisoning by symmetric edge flip."""
        adj = edge_index_to_csr(data.edge_index, data.num_nodes).tolil()
        n = data.num_nodes
        y = data.y.cpu().numpy()
        flips = 0
        for _ in range(int(budget)):
            u = int(self.rng.integers(0, n))
            if self.rng.random() < 0.65:
                # Add cross-class edge.
                cands = np.where(y != y[u])[0]
                if cands.size == 0:
                    continue
                v = int(self.rng.choice(cands))
                if u == v:
                    continue
                adj[u, v] = 1
                adj[v, u] = 1
                flips += 1
            else:
                # Remove in-class supporting edge.
                nbrs = np.where(np.asarray(adj[u].todense()).reshape(-1) > 0)[0]
                in_class = [int(v) for v in nbrs if y[int(v)] == y[u]]
                if not in_class:
                    continue
                v = int(self.rng.choice(in_class))
                adj[u, v] = 0
                adj[v, u] = 0
                flips += 1

        adj = adj.tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        attacked = _clone_with_graph(data, adj)
        return AttackOutcome(
            name="RandomStructure",
            attack_type="poisoning",
            budget=float(budget),
            data=attacked,
            metadata={"edge_flips": float(flips)},
        )


class EvasionAttacks:
    """Evasion attacks that modify test-time graph/features."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def feature_perturbation(self, data: Data, budget: float, target_mask: torch.Tensor | None = None) -> AttackOutcome:
        """
        Feature evasion (important): binary feature flips + small Gaussian noise
        only on inference nodes.
        """
        attacked = data.clone()
        x = attacked.x.detach().cpu().numpy().astype(np.float32)

        if target_mask is None:
            mask = attacked.test_mask.cpu().numpy()
        else:
            mask = target_mask.cpu().numpy()

        idx = np.where(mask)[0]
        n_features = x.shape[1]
        flip_per_node = max(1, int(budget * n_features))
        sigma = float(max(0.005, budget * 0.08))

        changed = 0
        for u in idx:
            cols = self.rng.choice(n_features, size=min(flip_per_node, n_features), replace=False)
            x[u, cols] = 1.0 - x[u, cols]
            noise_cols = self.rng.choice(n_features, size=min(flip_per_node, n_features), replace=False)
            x[u, noise_cols] += self.rng.normal(0.0, sigma, size=len(noise_cols))
            changed += len(cols)

        x = np.clip(x, 0.0, 1.0)
        attacked.x = torch.tensor(x, dtype=data.x.dtype, device=data.x.device)
        return AttackOutcome(
            name="FeaturePerturbation",
            attack_type="evasion",
            budget=float(budget),
            data=attacked,
            metadata={"changed_entries": float(changed), "noise_sigma": sigma},
        )

    def edge_flip(self, data: Data, budget: int, target_mask: torch.Tensor | None = None) -> AttackOutcome:
        """Test-time degree-preserving edge rewiring around target/test nodes."""
        adj = edge_index_to_csr(data.edge_index, data.num_nodes).tolil()
        x = data.x.cpu().numpy().astype(np.float32)
        y = data.y.cpu().numpy()

        if target_mask is None:
            targets = np.where(data.test_mask.cpu().numpy())[0]
        else:
            targets = np.where(target_mask.cpu().numpy())[0]

        rewired = 0
        n = data.num_nodes
        for _ in range(int(budget)):
            if targets.size == 0:
                break
            u = int(self.rng.choice(targets))
            nbrs = np.where(np.asarray(adj[u].todense()).reshape(-1) > 0)[0]
            if nbrs.size == 0:
                continue
            same_label_nbrs = [int(v) for v in nbrs if y[int(v)] == y[u]]
            if same_label_nbrs:
                sim_same = np.array([_feature_cosine_similarity(x, u, v) for v in same_label_nbrs], dtype=np.float32)
                v_old = same_label_nbrs[int(np.argmax(sim_same))]
            else:
                v_old = int(self.rng.choice(nbrs))

            cands = self.rng.integers(0, n, size=64)
            cands = [int(v) for v in cands if v != u and adj[u, int(v)] == 0 and y[int(v)] != y[u]]
            if not cands:
                continue
            sims = np.array([_feature_cosine_similarity(x, u, v) for v in cands], dtype=np.float32)
            v_new = cands[int(np.argmin(sims))]

            adj[u, v_old] = 0
            adj[v_old, u] = 0
            adj[u, v_new] = 1
            adj[v_new, u] = 1
            rewired += 1

        adj = adj.tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        attacked = _clone_with_graph(data, adj)
        return AttackOutcome(
            name="EdgeFlip",
            attack_type="evasion",
            budget=float(budget),
            data=attacked,
            metadata={"rewired_edges": float(rewired)},
        )

    def gradient_based(
        self,
        model: torch.nn.Module,
        data: Data,
        budget: float,
        steps: int = 7,
        target_mask: torch.Tensor | None = None,
    ) -> AttackOutcome:
        """PGD-like gradient feature evasion attack."""
        attacked = data.clone()
        x0 = attacked.x.detach()

        if target_mask is None:
            target_mask = attacked.test_mask

        eps = float(budget)
        alpha = max(eps / max(steps // 2, 1), 0.01)

        node_mask = target_mask.float().unsqueeze(1)
        x_adv = x0 + node_mask * torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)

        model.eval()
        for _ in range(int(steps)):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv, attacked.edge_index)
            loss = F.cross_entropy(logits[target_mask], attacked.y[target_mask])
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + node_mask * alpha * x_adv.grad.sign()
                delta = (x_adv - x0).clamp(-eps, eps)
                x_adv = (x0 + node_mask * delta).clamp(0.0, 1.0)

        attacked.x = x_adv.detach()
        return AttackOutcome(
            name="GradientBased",
            attack_type="evasion",
            budget=float(budget),
            data=attacked,
            metadata={"epsilon": eps, "steps": float(steps)},
        )


def build_attack_suite(seed: int = 42) -> Tuple[PoisoningAttacks, EvasionAttacks]:
    return PoisoningAttacks(seed=seed), EvasionAttacks(seed=seed)
