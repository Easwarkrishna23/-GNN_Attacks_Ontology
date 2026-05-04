from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


import os

try:
    from deeprobust.graph.global_attack import Metattack
    from deeprobust.graph.targeted_attack import Nettack
    from deeprobust.graph.defense import GCN as DRGCN

    DEEPROBUST_OK = True
except Exception:
    DEEPROBUST_OK = False


def _to_binary_features(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def metattack_global_poison(
    adj: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    idx_train: np.ndarray,
    budget: int,
    seed: int = 42,
):
    """Global poisoning attack with budget-controlled edge perturbations."""
    np.random.seed(seed)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    feat_sp = sp.csr_matrix(_to_binary_features(features))

    use_deeprobust = DEEPROBUST_OK and os.environ.get("USE_DEEPROBUST", "0") == "1"
    if use_deeprobust:
        try:
            surrogate = DRGCN(nfeat=feat_sp.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5, with_relu=False, with_bias=True, device="cpu")
            surrogate.fit(feat_sp, adj, labels, idx_train)
            attacker = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=feat_sp.shape, attack_structure=True, attack_features=False, lambda_=0.0, device="cpu")
            idx_unlabeled = np.setdiff1d(np.arange(adj.shape[0]), idx_train)
            attacker.attack(feat_sp, adj, labels, idx_train, idx_unlabeled, n_perturbations=int(budget), ll_constraint=False)
            mod_adj = attacker.modified_adj.tocsr()
            return mod_adj
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    mod = adj.copy().tolil()
    n = mod.shape[0]
    for _ in range(int(budget)):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        mod[u, v] = 1 - mod[u, v]
        mod[v, u] = mod[u, v]
    return mod.tocsr()


def nettack_targeted(
    adj: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    idx_train: np.ndarray,
    target_nodes: np.ndarray,
    budget_per_node: int,
    seed: int = 42,
):
    """Targeted Nettack-style perturbation for selected target nodes."""
    np.random.seed(seed)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    feat_sp = sp.csr_matrix(_to_binary_features(features))

    use_deeprobust = DEEPROBUST_OK and os.environ.get("USE_DEEPROBUST", "0") == "1"
    if use_deeprobust:
        try:
            surrogate = DRGCN(nfeat=feat_sp.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5, with_relu=False, with_bias=True, device="cpu")
            surrogate.fit(feat_sp, adj, labels, idx_train)
            mod_adj = adj.copy()
            mod_feat = feat_sp.copy()
            for node in target_nodes:
                attacker = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device="cpu")
                attacker.attack(mod_feat, mod_adj, labels, int(node), n_perturbations=int(budget_per_node))
                mod_adj = attacker.modified_adj.tocsr()
                mod_feat = attacker.modified_features.tocsr()
            return mod_adj
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    mod = adj.copy().tolil()
    n = mod.shape[0]
    for node in target_nodes:
        node = int(node)
        for _ in range(int(budget_per_node)):
            v = int(rng.integers(0, n))
            if v == node:
                continue
            mod[node, v] = 1 - mod[node, v]
            mod[v, node] = mod[node, v]
    return mod.tocsr()


def pgd_feature_evasion(
    model,
    data,
    budget: float,
    steps: int = 7,
    step_size: float | None = None,
    target_mask: torch.Tensor | None = None,
):
    """Projected gradient feature evasion with epsilon=budget."""
    eps = float(budget)
    alpha = float(step_size if step_size is not None else eps / max(1, steps // 2))
    attacked = data.clone()
    x0 = data.x.detach()
    if target_mask is None:
        target_mask = data.test_mask
    node_mask = target_mask.float().unsqueeze(1)
    x_adv = x0 + node_mask * torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(int(steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv, data.edge_index)
        loss = F.cross_entropy(logits[target_mask], data.y[target_mask])
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + node_mask * alpha * x_adv.grad.sign()
            delta = (x_adv - x0).clamp(-eps, eps)
            x_adv = (x0 + node_mask * delta).clamp(0.0, 1.0)

    attacked.x = x_adv.detach()
    return attacked
