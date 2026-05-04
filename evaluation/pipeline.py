from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from attacks.benchmark_attacks import AttackOutcome, EvasionAttacks, PoisoningAttacks
from defenses.benchmark_defenses import TripleDefense
from evaluation.metrics import (
    attack_success_rate,
    classification_metrics,
    edge_index_to_csr,
    graph_conductance_by_labels,
    graph_density,
    graph_modularity,
    homophily_ratio,
    homophily_recovery_ratio,
    robustness_score,
)
from models.pyg_gnn import TwoLayerGAT, TwoLayerGCN


@dataclass
class ExperimentConfig:
    seed: int = 42
    epochs_clean: int = 180
    epochs_poison: int = 150
    lr: float = 0.01
    weight_decay: float = 5e-4
    attack_budgets: Dict[str, float] | None = None


@dataclass
class EvalPack:
    metrics: Dict[str, float]
    pred: np.ndarray
    emb: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    details: Dict[str, np.ndarray | float]


DEFAULT_ATTACK_BUDGETS = {
    "Metattack": 420,
    "Nettack": 8,
    "RandomStructure": 520,
    "FeaturePerturbation": 0.22,
    "EdgeFlip": 420,
    "GradientBased": 0.16,
}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, in_dim: int, out_dim: int):
    if model_name == "GCN":
        return TwoLayerGCN(in_channels=in_dim, hidden_channels=32, out_channels=out_dim, dropout=0.5)
    if model_name == "GAT":
        return TwoLayerGAT(in_channels=in_dim, hidden_channels=8, out_channels=out_dim, heads=4, dropout=0.6)
    raise ValueError(f"Unknown model: {model_name}")


def train_model(model, data, epochs: int, lr: float, weight_decay: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model


def evaluate_model(model, data) -> EvalPack:
    model.eval()
    with torch.no_grad():
        emb, logits, details = model(data.x, data.edge_index, return_details=True)
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    y_prob = probs[data.test_mask].cpu().numpy()
    m = classification_metrics(y_true, y_pred, y_prob)
    details_np = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in details.items()}
    return EvalPack(
        metrics=m,
        pred=pred.cpu().numpy(),
        emb=emb.cpu().numpy(),
        logits=logits.cpu().numpy(),
        probs=probs.cpu().numpy(),
        details=details_np,
    )


def _pick_target_nodes(clean_pred: np.ndarray, labels: np.ndarray, test_mask: np.ndarray, top_k: int = 40) -> np.ndarray:
    test_nodes = np.where(test_mask)[0]
    correct = test_nodes[clean_pred[test_nodes] == labels[test_nodes]]
    if correct.size == 0:
        return test_nodes[:top_k]
    return correct[:top_k]


def _run_single_attack(
    attack_name: str,
    poisoner: PoisoningAttacks,
    evader: EvasionAttacks,
    clean_model,
    base_data,
    budget: float,
) -> AttackOutcome:
    if attack_name == "Metattack":
        return poisoner.metattack_global(base_data, budget=int(budget))
    if attack_name == "Nettack":
        return poisoner.nettack_targeted(base_data, budget=int(budget))
    if attack_name == "RandomStructure":
        return poisoner.random_structure(base_data, budget=int(budget))
    if attack_name == "FeaturePerturbation":
        return evader.feature_perturbation(base_data, budget=float(budget))
    if attack_name == "EdgeFlip":
        return evader.edge_flip(base_data, budget=int(budget))
    if attack_name == "GradientBased":
        return evader.gradient_based(clean_model, base_data, budget=float(budget), steps=7)
    raise ValueError(f"Unknown attack: {attack_name}")


def run_benchmark_for_data(
    data,
    dataset_name: str,
    ontology_path: str,
    config: ExperimentConfig,
    models: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    set_seed(config.seed)

    if config.attack_budgets is None:
        config.attack_budgets = DEFAULT_ATTACK_BUDGETS.copy()

    models = models or ["GCN", "GAT"]
    labels = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy().astype(bool)
    adj_clean = edge_index_to_csr(data.edge_index, data.num_nodes)
    hom_clean = homophily_ratio(adj_clean, labels)

    poisoner = PoisoningAttacks(seed=config.seed)
    evader = EvasionAttacks(seed=config.seed)
    defenses = TripleDefense(ontology_path=ontology_path)

    rows: List[Dict[str, float | str]] = []
    tsne_state_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    graph_state_by_model: Dict[str, Dict[str, object]] = {}
    layerwise_state_by_model: Dict[str, Dict[str, object]] = {}

    for model_name in models:
        clean_model = build_model(model_name, data.num_features, int(labels.max()) + 1)
        clean_model = train_model(clean_model, data, epochs=config.epochs_clean, lr=config.lr, weight_decay=config.weight_decay)
        clean_eval = evaluate_model(clean_model, data)
        clean_acc = clean_eval.metrics["accuracy"]
        layerwise_state_by_model[model_name] = {
            "clean": {
                "embedding_sample": clean_eval.emb[:10].tolist(),
                "logits_sample": clean_eval.logits[:10].tolist(),
                "prob_sample": clean_eval.probs[:10].tolist(),
                "memory_mb_layer1": float(clean_eval.details["memory_mb_layer1"]),
                "memory_mb_layer2": float(clean_eval.details["memory_mb_layer2"]),
            }
        }

        target_nodes = _pick_target_nodes(clean_eval.pred, labels, test_mask, top_k=50)

        best_drop = -1.0
        tsne_pack = None
        graph_state_by_model[model_name] = {}

        for attack_name, budget in config.attack_budgets.items():
            attacked_out = _run_single_attack(attack_name, poisoner, evader, clean_model, data, budget)
            attacked_data = attacked_out.data
            adj_attacked = edge_index_to_csr(attacked_data.edge_index, attacked_data.num_nodes)
            hom_attacked = homophily_ratio(adj_attacked, labels)
            attacked_density = graph_density(adj_attacked)
            attacked_conductance = graph_conductance_by_labels(adj_attacked, labels)
            attacked_modularity = graph_modularity(adj_attacked, labels)

            if attacked_out.attack_type == "poisoning":
                attacked_model = build_model(model_name, data.num_features, int(labels.max()) + 1)
                attacked_model = train_model(
                    attacked_model,
                    attacked_data,
                    epochs=config.epochs_poison,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                attacked_eval = evaluate_model(attacked_model, attacked_data)
            else:
                attacked_eval = evaluate_model(clean_model, attacked_data)

            asr = attack_success_rate(labels, clean_eval.pred, attacked_eval.pred, target_nodes) if attack_name == "Nettack" else 0.0
            attacked_acc = attacked_eval.metrics["accuracy"]
            drop = clean_acc - attacked_acc

            structural_out = defenses.structural(attacked_data)
            ontology_out = defenses.ontology_only(attacked_data)
            hybrid_out = defenses.hybrid(attacked_data)

            defense_variants = [
                ("StructuralDefense", structural_out.data),
                ("OntologyDefense", ontology_out.data),
                ("HybridDefense", hybrid_out.data),
            ]

            defended_cache = {}
            for defense_name, defended_data in defense_variants:
                if attacked_out.attack_type == "poisoning":
                    def_model = build_model(model_name, data.num_features, int(labels.max()) + 1)
                    def_model = train_model(
                        def_model,
                        defended_data,
                        epochs=config.epochs_poison,
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                    )
                    def_eval = evaluate_model(def_model, defended_data)
                else:
                    # For evasion, retraining on defended graph can recover useful structure.
                    def_model = build_model(model_name, data.num_features, int(labels.max()) + 1)
                    def_model = train_model(
                        def_model,
                        defended_data,
                        epochs=max(80, config.epochs_clean - 40),
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                    )
                    def_eval = evaluate_model(def_model, defended_data)

                defended_cache[defense_name] = (def_eval, defended_data)

                def_adj = edge_index_to_csr(defended_data.edge_index, defended_data.num_nodes)
                hom_def = homophily_ratio(def_adj, labels)
                def_density = graph_density(def_adj)
                def_conductance = graph_conductance_by_labels(def_adj, labels)
                def_modularity = graph_modularity(def_adj, labels)

                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Attack": attack_name,
                        "AttackType": attacked_out.attack_type,
                        "Defense": defense_name,
                        "Budget": float(budget),
                        "CleanAccuracy": clean_acc,
                        "AttackedAccuracy": attacked_acc,
                        "DefendedAccuracy": def_eval.metrics["accuracy"],
                        "AttackedPrecision": attacked_eval.metrics["precision_macro"],
                        "AttackedRecall": attacked_eval.metrics["recall_macro"],
                        "AttackedF1": attacked_eval.metrics["f1_macro"],
                        "AttackedROCAUC": attacked_eval.metrics["roc_auc_ovr"],
                        "DefendedPrecision": def_eval.metrics["precision_macro"],
                        "DefendedRecall": def_eval.metrics["recall_macro"],
                        "DefendedF1": def_eval.metrics["f1_macro"],
                        "DefendedROCAUC": def_eval.metrics["roc_auc_ovr"],
                        "RobustnessAttacked": robustness_score(clean_acc, attacked_acc),
                        "RobustnessDefended": robustness_score(clean_acc, def_eval.metrics["accuracy"]),
                        "AccuracyDrop": drop,
                        "AccuracyRecovery": def_eval.metrics["accuracy"] - attacked_acc,
                        "RecoveryRate": (def_eval.metrics["accuracy"] - attacked_acc) / max(1e-9, clean_acc - attacked_acc),
                        "ASR": asr,
                        "HomophilyClean": hom_clean,
                        "HomophilyAttacked": hom_attacked,
                        "HomophilyDefended": hom_def,
                        "HomophilyRecoveryRatio": homophily_recovery_ratio(hom_clean, hom_attacked, hom_def),
                        "DensityAttacked": attacked_density,
                        "ConductanceAttacked": attacked_conductance,
                        "ModularityAttacked": attacked_modularity,
                        "DensityDefended": def_density,
                        "ConductanceDefended": def_conductance,
                        "ModularityDefended": def_modularity,
                    }
                )

            graph_state_by_model[model_name][attack_name] = {
                "attack": attack_name,
                "clean_data": data,
                "attacked_data": attacked_data,
                "defended_data": defended_cache["HybridDefense"][1],
            }

            if drop > best_drop:
                best_drop = drop
                tsne_pack = {
                    "clean": clean_eval.emb,
                    "attacked": attacked_eval.emb,
                    "defended_structural": defended_cache["StructuralDefense"][0].emb,
                    "defended_ontology": defended_cache["OntologyDefense"][0].emb,
                    "defended_hybrid": defended_cache["HybridDefense"][0].emb,
                }
                layerwise_state_by_model[model_name]["most_harmful_attack"] = attack_name
                layerwise_state_by_model[model_name]["attacked"] = {
                    "embedding_sample": attacked_eval.emb[:10].tolist(),
                    "logits_sample": attacked_eval.logits[:10].tolist(),
                    "prob_sample": attacked_eval.probs[:10].tolist(),
                    "memory_mb_layer1": float(attacked_eval.details["memory_mb_layer1"]),
                    "memory_mb_layer2": float(attacked_eval.details["memory_mb_layer2"]),
                }
                for d_name in ["StructuralDefense", "OntologyDefense", "HybridDefense"]:
                    d_eval = defended_cache[d_name][0]
                    layerwise_state_by_model[model_name][d_name] = {
                        "embedding_sample": d_eval.emb[:10].tolist(),
                        "logits_sample": d_eval.logits[:10].tolist(),
                        "prob_sample": d_eval.probs[:10].tolist(),
                        "memory_mb_layer1": float(d_eval.details["memory_mb_layer1"]),
                        "memory_mb_layer2": float(d_eval.details["memory_mb_layer2"]),
                    }

        if tsne_pack is not None:
            tsne_state_by_model[model_name] = tsne_pack

    return pd.DataFrame(rows), tsne_state_by_model, graph_state_by_model, layerwise_state_by_model
