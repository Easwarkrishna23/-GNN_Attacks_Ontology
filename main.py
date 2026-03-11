import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
import networkx as nx

from datasets.cora_loader import load_cora
from datasets.dynamic_graph import DynamicGraphGenerator
from experiments.baseline_training import train_model, evaluate_model
from models.gcn import GCN
from models.gat import GAT
from attacks.poisoning.random_poison import run_random_attack
from attacks.poisoning.nettack import get_surrogate, run_nettack
from attacks.poisoning.metattack import run_metattack
from attacks.evasion.structure_evasion import run_structure_evasion
from attacks.evasion.feature_evasion import run_feature_evasion
from attacks.evasion.fgsm_like import run_fgsm_like_feature_attack
from defenses.feature_defense import laplacian_feature_smoothing, feature_consistency_regularization
from defenses.ontology_defense import (
    build_ontology_matrix,
    ontology_reweight_adjacency,
    ontology_feature_projection,
)
from utils.metrics import compute_robustness_metrics, perturbation_rate, compute_graph_metrics
from visualization.graph_viz import visualize_graph_mosaic
from visualization.plotting import (
    plot_robustness_curves,
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_layer_output_panel,
)
from report_generator import generate_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def adj_from_edge_index(edge_index, num_nodes):
    edge_index_np = edge_index.cpu().numpy()
    return sp.csr_matrix(
        (np.ones(edge_index_np.shape[1]), (edge_index_np[0], edge_index_np[1])),
        shape=(num_nodes, num_nodes),
    )


def pyg_from_adj_and_x(base_data, adj, x_np=None):
    new_data = base_data.clone()
    rows, cols = adj.nonzero()
    new_data.edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    if x_np is not None:
        new_data.x = torch.tensor(x_np, dtype=base_data.x.dtype)
    return new_data


def save_clean_graph_plot(adj, labels, save_path):
    g = nx.from_scipy_sparse_array(adj) if hasattr(nx, "from_scipy_sparse_array") else nx.from_scipy_sparse_matrix(adj)
    sample_nodes = list(range(min(300, adj.shape[0])))
    sg = g.subgraph(sample_nodes)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(sg, seed=42)
    nx.draw(
        sg,
        pos,
        node_size=20,
        node_color=labels[sample_nodes],
        cmap=plt.cm.Set1,
        with_labels=False,
        edge_color="gray",
        alpha=0.7,
    )
    plt.title("Clean Graph Subgraph")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_memory_dict(memory_dict):
    print(f"  Parameter memory (MB): {memory_dict['parameter_memory_mb']:.4f}")
    print(f"  Total estimated memory (MB): {memory_dict['total_estimated_mb']:.4f}")
    print("  Activation memory breakdown (MB):")
    for k, v in memory_dict["activation_memory_mb"].items():
        print(f"    - {k}: {v:.4f}")


def print_gcn_debug(debug_info, node_id):
    print("\n[GCN Layer-wise Output]")
    print(f"Node {node_id} pre-aggregation L1 (first 8): {debug_info['pre_aggregation_l1'][node_id][:8].cpu().numpy()}")
    print(f"Node {node_id} post-normalization L1 (first 8): {debug_info['post_normalization_l1'][node_id][:8].cpu().numpy()}")
    print(f"Node {node_id} post-activation L1 (first 8): {debug_info['post_activation_l1'][node_id][:8].cpu().numpy()}")
    print(f"Node {node_id} post-normalization L2/logits: {debug_info['post_normalization_l2'][node_id].cpu().numpy()}")
    print(f"Node {node_id} softmax probs: {debug_info['softmax_probabilities'][node_id].cpu().numpy()}")
    print(f"Node {node_id} predicted class: {int(debug_info['predictions'][node_id])}")
    print_memory_dict(debug_info["memory"])


def print_gat_debug(debug_info, node_id):
    print("\n[GAT Layer-wise Output]")
    print(f"Attention weights shape: {tuple(debug_info['attention_weights'].shape)}")
    print(f"First 10 attention weights: {debug_info['attention_weights'][:10].cpu().numpy().flatten()}")
    print(f"Node {node_id} aggregation hidden (first 8): {debug_info['node_aggregation_hidden'][node_id][:8].cpu().numpy()}")
    print(f"Node {node_id} logits: {debug_info['logits'][node_id].cpu().numpy()}")
    print(f"Node {node_id} softmax probs: {debug_info['softmax_probabilities'][node_id].cpu().numpy()}")
    print(f"Node {node_id} predicted class: {int(debug_info['predictions'][node_id])}")
    print_memory_dict(debug_info["memory"])


def write_layerwise_debug_file(gcn_debug, gat_debug, node_id, out_path):
    lines = []
    lines.append("# Layer-wise Outputs and Memory")
    lines.append("")
    lines.append(f"Target node for debug: {node_id}")
    lines.append("")
    lines.append("## GCN")
    lines.append(f"- pre_aggregation_l1[:8]: {gcn_debug['pre_aggregation_l1'][node_id][:8].cpu().numpy().tolist()}")
    lines.append(f"- post_normalization_l1[:8]: {gcn_debug['post_normalization_l1'][node_id][:8].cpu().numpy().tolist()}")
    lines.append(f"- post_activation_l1[:8]: {gcn_debug['post_activation_l1'][node_id][:8].cpu().numpy().tolist()}")
    lines.append(f"- logits_l2: {gcn_debug['post_normalization_l2'][node_id].cpu().numpy().tolist()}")
    lines.append(f"- probs: {gcn_debug['softmax_probabilities'][node_id].cpu().numpy().tolist()}")
    lines.append(f"- prediction: {int(gcn_debug['predictions'][node_id])}")
    lines.append(f"- memory: {gcn_debug['memory']}")
    lines.append("")
    lines.append("## GAT")
    lines.append(f"- hidden[:8]: {gat_debug['node_aggregation_hidden'][node_id][:8].cpu().numpy().tolist()}")
    lines.append(f"- logits: {gat_debug['logits'][node_id].cpu().numpy().tolist()}")
    lines.append(f"- probs: {gat_debug['softmax_probabilities'][node_id].cpu().numpy().tolist()}")
    lines.append(f"- prediction: {int(gat_debug['predictions'][node_id])}")
    lines.append(f"- memory: {gat_debug['memory']}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_metric_table(title, df, columns):
    print(f"\n=== {title} ===")
    print(df[columns].to_string(index=False))


def verify_feature_evasion(clean_data, attacked_data, target_nodes):
    x_clean = clean_data.x.cpu().numpy()
    x_attack = attacked_data.x.cpu().numpy()
    edge_unchanged = np.array_equal(clean_data.edge_index.cpu().numpy(), attacked_data.edge_index.cpu().numpy())
    diff_rows = np.where(np.any(x_clean != x_attack, axis=1))[0]
    target_set = set(int(t) for t in target_nodes)
    touched_set = set(int(t) for t in diff_rows)
    only_targets_changed = touched_set.issubset(target_set)
    changed_target_count = len(touched_set)
    return {
        "edge_unchanged": edge_unchanged,
        "only_targets_changed": only_targets_changed,
        "changed_target_count": changed_target_count,
        "target_count": len(target_set),
        "feature_perturbation_rate": perturbation_rate(x_clean, x_attack),
    }


def make_result_row(attack_name, base_metrics, attack_metrics, clean_probs_test, attack_probs_test, budget, p_rate):
    rob = compute_robustness_metrics(base_metrics, attack_metrics, clean_probs_test, attack_probs_test)
    return {
        "Attack": attack_name,
        "Accuracy": float(attack_metrics["accuracy"]),
        "Precision": float(attack_metrics["precision"]),
        "Recall": float(attack_metrics["recall"]),
        "F1": float(attack_metrics["f1_macro"]),
        "Macro F1": float(attack_metrics["f1_macro"]),
        "Micro F1": float(attack_metrics["f1_micro"]),
        "ROC-AUC": float(attack_metrics.get("roc_auc", 0.0)),
        "Log-loss": float(attack_metrics.get("log_loss", 0.0)),
        "Classification Margin": float(attack_metrics.get("classification_margin", 0.0)),
        "Robustness Score": float(rob["RobustnessScore"]),
        "Attack Success Rate": float(rob["ASR"]),
        "Margin Drop": float(rob["MarginDrop"]),
        "Perturbation Rate": float(p_rate),
        "Perturbation Budget": float(budget),
    }


def evaluate_model_under_attacks(model_name, clean_model, model_builder, clean_data, attack_payloads, poison_epochs=120):
    base_metrics, base_pred, base_probs = evaluate_model(clean_model, clean_data)
    rows = [
        {
            "Attack": "Baseline",
            "Accuracy": float(base_metrics["accuracy"]),
            "Precision": float(base_metrics["precision"]),
            "Recall": float(base_metrics["recall"]),
            "F1": float(base_metrics["f1_macro"]),
            "Macro F1": float(base_metrics["f1_macro"]),
            "Micro F1": float(base_metrics["f1_micro"]),
            "ROC-AUC": float(base_metrics.get("roc_auc", 0.0)),
            "Log-loss": float(base_metrics.get("log_loss", 0.0)),
            "Classification Margin": float(base_metrics.get("classification_margin", 0.0)),
            "Robustness Score": 1.0,
            "Attack Success Rate": 0.0,
            "Margin Drop": 0.0,
            "Perturbation Rate": 0.0,
            "Perturbation Budget": 0.0,
        }
    ]

    predictions = {"Baseline": base_pred}
    probabilities = {"Baseline": base_probs}

    print(f"\n=== {model_name}: ATTACK EVALUATION ===")
    for payload in attack_payloads:
        attack_name = payload["name"]
        if payload["type"] == "poison":
            attacked_model = train_model(model_builder(), payload["data"], epochs=poison_epochs)
            m, pred, probs = evaluate_model(attacked_model, payload["data"])
        else:
            m, pred, probs = evaluate_model(clean_model, payload["data"])

        row = make_result_row(
            attack_name,
            base_metrics,
            m,
            base_probs[clean_data.test_mask].cpu().numpy(),
            probs[payload["data"].test_mask].cpu().numpy(),
            payload["budget"],
            payload["p_rate"],
        )
        rows.append(row)
        predictions[attack_name] = pred
        probabilities[attack_name] = probs
        print(f"{model_name} | {attack_name}: accuracy={m['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    baseline_acc = float(df[df["Attack"] == "Baseline"]["Accuracy"].iloc[0])
    df["Accuracy Drop"] = baseline_acc - df["Accuracy"]
    return df, predictions, probabilities


def apply_feature_defenses(clean_adj, clean_features, labels, attacked_data, gcn_model, model_name="GCN"):
    rows = []

    data_def_smooth = attacked_data.clone()
    data_def_smooth.x = laplacian_feature_smoothing(attacked_data.x, clean_adj, alpha=0.7)
    consistency_value = feature_consistency_regularization(data_def_smooth.x, clean_adj)
    m_s, pred_s, p_s = evaluate_model(gcn_model, data_def_smooth)

    ontology = build_ontology_matrix(clean_features, labels=labels, semantic_weight=0.8)
    adj_ontology = ontology_reweight_adjacency(clean_adj, ontology, lam=0.3)
    data_def_onto = pyg_from_adj_and_x(attacked_data, adj_ontology, None)
    data_def_onto.x = ontology_feature_projection(attacked_data.x, ontology, lam=0.3)
    m_o, pred_o, p_o = evaluate_model(gcn_model, data_def_onto)

    rows.append(("Defense: Feature Smoothing", m_s, pred_s, p_s, data_def_smooth, consistency_value, adj_ontology))
    rows.append(("Defense: Ontology", m_o, pred_o, p_o, data_def_onto, consistency_value, adj_ontology))
    print(f"[{model_name}] Feature consistency regularization ||X-A_hatX||^2: {consistency_value:.6f}")
    return rows


def build_dynamic_attack_payloads(data_dyn, adj_dyn, features_dyn, labels_dyn, idx_train_dyn, idx_test_dyn):
    payloads_dyn = []

    adj_rnd_dyn, feat_rnd_dyn = run_random_attack(
        adj_dyn,
        features_dyn,
        n_edge_perturbations=300,
        feature_corruption_rate=0.01,
        seed=123,
    )
    data_rnd_dyn = pyg_from_adj_and_x(data_dyn, adj_rnd_dyn, feat_rnd_dyn)
    payloads_dyn.append(
        {
            "name": "Poisoning: Random Structure",
            "type": "poison",
            "data": data_rnd_dyn,
            "budget": 300,
            "p_rate": perturbation_rate(adj_dyn, adj_rnd_dyn),
            "adj": adj_rnd_dyn,
        }
    )

    surrogate_dyn = get_surrogate(adj_dyn, features_dyn, labels_dyn, idx_train_dyn)
    adj_net_dyn = adj_dyn.copy()
    for t in idx_test_dyn[:4]:
        adj_net_dyn, _, _ = run_nettack(surrogate_dyn, adj_net_dyn, features_dyn, labels_dyn, int(t), n_perturbations=5)
    data_net_dyn = pyg_from_adj_and_x(data_dyn, adj_net_dyn, features_dyn)
    payloads_dyn.append(
        {
            "name": "Poisoning: Nettack",
            "type": "poison",
            "data": data_net_dyn,
            "budget": 20,
            "p_rate": perturbation_rate(adj_dyn, adj_net_dyn),
            "adj": adj_net_dyn,
        }
    )

    adj_meta_dyn, feat_meta_dyn, _ = run_metattack(adj_dyn, features_dyn, labels_dyn, idx_train_dyn, n_perturbations=300)
    feat_meta_dyn_np = np.asarray(feat_meta_dyn.todense()) if sp.issparse(feat_meta_dyn) else feat_meta_dyn
    data_meta_dyn = pyg_from_adj_and_x(data_dyn, adj_meta_dyn, feat_meta_dyn_np)
    payloads_dyn.append(
        {
            "name": "Poisoning: Meta Attack",
            "type": "poison",
            "data": data_meta_dyn,
            "budget": 300,
            "p_rate": perturbation_rate(adj_dyn, adj_meta_dyn),
            "adj": adj_meta_dyn,
        }
    )

    adj_structure_dyn = adj_dyn.copy()
    for t in idx_test_dyn[:20]:
        adj_structure_dyn = run_structure_evasion(adj_structure_dyn, int(t), n_perturbations=1, seed=123)
    data_structure_dyn = pyg_from_adj_and_x(data_dyn, adj_structure_dyn, features_dyn)
    payloads_dyn.append(
        {
            "name": "Evasion: Edge Flip",
            "type": "evasion",
            "data": data_structure_dyn,
            "budget": 20,
            "p_rate": perturbation_rate(adj_dyn, adj_structure_dyn),
            "adj": adj_structure_dyn,
        }
    )

    data_feature_dyn, _ = run_feature_evasion(
        data_dyn,
        target_nodes=idx_test_dyn[:60],
        binary_flip_budget=8,
        continuous_noise_std=0.06,
        seed=123,
    )
    payloads_dyn.append(
        {
            "name": "Evasion: Feature",
            "type": "evasion",
            "data": data_feature_dyn,
            "budget": 8,
            "p_rate": perturbation_rate(data_dyn.x.cpu().numpy(), data_feature_dyn.x.cpu().numpy()),
        }
    )
    return payloads_dyn


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    dataset, data = load_cora()
    adj_clean = adj_from_edge_index(data.edge_index, data.num_nodes)
    features_clean = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    idx_train = np.where(data.train_mask.cpu().numpy())[0]
    idx_test = np.where(data.test_mask.cpu().numpy())[0]

    print("\n=== DATASET SUMMARY ===")
    print(f"Feature dimension: {data.x.size(1)}")
    class_counts = np.bincount(labels, minlength=dataset.num_classes)
    print(f"Class distribution: {class_counts.tolist()}")
    save_clean_graph_plot(adj_clean, labels, "results/clean_graph.png")

    print("\n=== BASELINE TRAINING ===")
    gcn = train_model(GCN(dataset.num_features, 16, dataset.num_classes), data, epochs=120)
    gat = train_model(GAT(dataset.num_features, 8, dataset.num_classes, heads=4), data, epochs=120)
    gcn_metrics, gcn_pred, gcn_probs = evaluate_model(gcn, data)
    gat_metrics, gat_pred, gat_probs = evaluate_model(gat, data)
    print(f"GCN baseline metrics: {gcn_metrics}")
    print(f"GAT baseline metrics: {gat_metrics}")

    target_node = int(idx_test[0])
    gcn_debug = gcn.forward_with_debug(data)
    gat_debug = gat.forward_with_debug(data)
    print_gcn_debug(gcn_debug, target_node)
    print_gat_debug(gat_debug, target_node)
    write_layerwise_debug_file(gcn_debug, gat_debug, target_node, "results/layerwise_debug_report.md")
    plot_layer_output_panel(
        [
            gcn_debug["pre_aggregation_l1"].cpu().numpy(),
            gcn_debug["post_normalization_l1"].cpu().numpy(),
            gcn_debug["post_activation_l1"].cpu().numpy(),
            gcn_debug["post_normalization_l2"].cpu().numpy(),
        ],
        [
            "GCN L1: Pre-Aggregation",
            "GCN L1: Post-Normalization",
            "GCN L1: Post-Activation",
            "GCN L2: Logits",
        ],
        "results/layer_outputs_gcn.png",
    )
    plot_layer_output_panel(
        [
            gat_debug["node_aggregation_hidden"].cpu().numpy(),
            gat_debug["logits"].cpu().numpy(),
            gat_debug["softmax_probabilities"].cpu().numpy(),
        ],
        [
            "GAT Hidden (After Attention)",
            "GAT Logits",
            "GAT Softmax Probabilities",
        ],
        "results/layer_outputs_gat.png",
    )

    print("\n=== STATIC ATTACK SUITE (CORA) ===")
    payloads_static = []

    poison_budget = 1500
    adj_rnd, feat_rnd = run_random_attack(
        adj_clean,
        features_clean,
        n_edge_perturbations=poison_budget,
        feature_corruption_rate=0.01,
        seed=42,
    )
    data_rnd = pyg_from_adj_and_x(data, adj_rnd, feat_rnd)
    payloads_static.append({"name": "Poisoning: Random Structure", "type": "poison", "data": data_rnd, "budget": poison_budget, "p_rate": perturbation_rate(adj_clean, adj_rnd), "adj": adj_rnd})

    surrogate = get_surrogate(adj_clean, features_clean, labels, idx_train)
    adj_net = adj_clean.copy()
    nettack_score = []
    for t in idx_test[:6]:
        adj_net, _, info = run_nettack(surrogate, adj_net, features_clean, labels, int(t), n_perturbations=8)
        nettack_score.append(info["perturbation_score_proxy"])
    print(f"Nettack perturbation-score proxy (avg): {float(np.mean(nettack_score)):.4f}")
    data_net = pyg_from_adj_and_x(data, adj_net, features_clean)
    payloads_static.append({"name": "Poisoning: Nettack", "type": "poison", "data": data_net, "budget": 48, "p_rate": perturbation_rate(adj_clean, adj_net), "adj": adj_net})

    adj_meta, feat_meta, meta_info = run_metattack(adj_clean, features_clean, labels, idx_train, n_perturbations=1500)
    print(f"Meta attack loops: outer={meta_info['outer_loop']}, inner={meta_info['inner_loop']}")
    feat_meta_np = np.asarray(feat_meta.todense()) if sp.issparse(feat_meta) else feat_meta
    data_meta = pyg_from_adj_and_x(data, adj_meta, feat_meta_np)
    payloads_static.append({"name": "Poisoning: Meta Attack", "type": "poison", "data": data_meta, "budget": 1500, "p_rate": perturbation_rate(adj_clean, adj_meta), "adj": adj_meta})

    adj_structure = adj_clean.copy()
    for t in idx_test[:40]:
        adj_structure = run_structure_evasion(adj_structure, int(t), n_perturbations=1, seed=42)
    data_structure = pyg_from_adj_and_x(data, adj_structure, features_clean)
    payloads_static.append({"name": "Evasion: Edge Flip", "type": "evasion", "data": data_structure, "budget": 40, "p_rate": perturbation_rate(adj_clean, adj_structure), "adj": adj_structure})

    data_feature, x_original = run_feature_evasion(
        data,
        target_nodes=idx_test[:120],
        binary_flip_budget=12,
        continuous_noise_std=0.08,
        seed=42,
    )
    payloads_static.append({"name": "Evasion: Feature", "type": "evasion", "data": data_feature, "budget": 12, "p_rate": perturbation_rate(data.x.cpu().numpy(), data_feature.x.cpu().numpy())})

    feature_check = verify_feature_evasion(data, data_feature, idx_test[:120])
    print("\n[Feature Evasion Verification]")
    print(f"Training graph unchanged: {feature_check['edge_unchanged']}")
    print(f"Only target test nodes changed: {feature_check['only_targets_changed']}")
    print(f"Changed targets: {feature_check['changed_target_count']}/{feature_check['target_count']}")
    print(f"Feature perturbation rate: {feature_check['feature_perturbation_rate']:.6f}")
    print(f"Original feature vector (node {target_node}, first 20): {x_original[target_node][:20].cpu().numpy()}")
    print(f"Modified feature vector (node {target_node}, first 20): {data_feature.x[target_node][:20].cpu().numpy()}")
    print(f"Prediction clean -> attacked (node {target_node}): {int(gcn_pred[target_node])} -> {int(evaluate_model(gcn, data_feature)[1][target_node])}")

    data_fgsm = run_fgsm_like_feature_attack(gcn, data, epsilon=0.05)
    payloads_static.append({"name": "Evasion: Gradient (FGSM-like)", "type": "evasion", "data": data_fgsm, "budget": 0.05, "p_rate": perturbation_rate(data.x.cpu().numpy(), data_fgsm.x.cpu().numpy())})

    gcn_builder = lambda: GCN(dataset.num_features, 16, dataset.num_classes)
    gat_builder = lambda: GAT(dataset.num_features, 8, dataset.num_classes, heads=4)
    gcn_df, gcn_preds, gcn_prob = evaluate_model_under_attacks("GCN", gcn, gcn_builder, data, payloads_static, poison_epochs=120)
    gat_df, gat_preds, gat_prob = evaluate_model_under_attacks("GAT", gat, gat_builder, data, payloads_static, poison_epochs=120)

    attack_only = gcn_df[gcn_df["Attack"] != "Baseline"].copy()
    worst_attack = attack_only.sort_values("Accuracy Drop", ascending=False).iloc[0]
    print("\n=== IMPACT ANALYSIS (GCN, STATIC) ===")
    print(f"Most impactful attack: {worst_attack['Attack']} with drop={worst_attack['Accuracy Drop']:.4f}")
    if worst_attack["Attack"] == "Evasion: Feature":
        print("Confirmed: Evasion: Feature is the most impactful.")
    else:
        print("Evasion: Feature is not top in this run; reported actual worst attack.")

    print("\n=== DEFENSES FOR EVASION: FEATURE (STATIC, GCN) ===")
    ontology_pred = None
    ontology_defended_data = None
    adj_onto = None
    defense_rows = apply_feature_defenses(adj_clean, features_clean, labels, data_feature, gcn, model_name="GCN-Static")
    for dname, dmetrics, dpred, dprobs, ddef_data, _, adj_onto_candidate in defense_rows:
        if dname == "Defense: Ontology":
            ontology_pred = dpred
            ontology_defended_data = ddef_data
            adj_onto = adj_onto_candidate
        gcn_df = pd.concat(
            [
                gcn_df,
                pd.DataFrame(
                    [
                        make_result_row(
                            dname,
                            gcn_metrics,
                            dmetrics,
                            gcn_prob["Baseline"][data.test_mask].cpu().numpy(),
                            dprobs[data.test_mask].cpu().numpy(),
                            budget=0.3 if "Ontology" in dname else 0.7,
                            p_rate=0.0,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    gcn_df.to_csv("results/final_evaluation_table.csv", index=False)
    gat_df.to_csv("results/final_evaluation_table_gat.csv", index=False)

    print("\n=== FINAL TABLE (GCN) ===")
    print(gcn_df[["Attack", "Accuracy", "F1", "Attack Success Rate", "Margin Drop", "Perturbation Budget"]].to_string(index=False))
    print_metric_table(
        "FINAL METRICS TABLE (GCN, STATIC)",
        gcn_df,
        [
            "Attack",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro F1",
            "Micro F1",
            "ROC-AUC",
            "Log-loss",
            "Classification Margin",
            "Robustness Score",
            "Attack Success Rate",
            "Margin Drop",
            "Perturbation Rate",
            "Perturbation Budget",
        ],
    )
    print_metric_table(
        "FINAL METRICS TABLE (GAT, STATIC)",
        gat_df,
        [
            "Attack",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro F1",
            "Micro F1",
            "ROC-AUC",
            "Log-loss",
            "Classification Margin",
            "Robustness Score",
            "Attack Success Rate",
            "Margin Drop",
            "Perturbation Rate",
            "Perturbation Budget",
        ],
    )

    graph_rows = []
    for name, adj in [
        ("Clean", adj_clean),
        ("Attacked (Edge Flip)", adj_structure),
        ("Defended (Ontology)", adj_onto if adj_onto is not None else adj_clean),
    ]:
        gm = compute_graph_metrics(adj, labels)
        graph_rows.append(
            {
                "Graph": name,
                "Density": gm.get("density", 0.0),
                "Modularity": gm.get("modularity", 0.0),
                "Conductance": gm.get("conductance", 0.0),
            }
        )
    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv("results/graph_metrics_static.csv", index=False)
    print_metric_table(
        "GRAPH METRICS (STATIC)",
        graph_df,
        ["Graph", "Density", "Modularity", "Conductance"],
    )

    adj_onto_plot = adj_onto if adj_onto is not None else adj_clean
    visualize_graph_mosaic(
        adj_clean,
        adj_structure,
        adj_onto_plot,
        labels,
        target_node=target_node,
        save_path="results/graph_mosaic.png",
    )

    budgets = [0.01, 0.03, 0.05, 0.08, 0.10]
    fgsm_curve = []
    feature_curve = []
    for eps in budgets:
        data_eps = run_fgsm_like_feature_attack(gcn, data, epsilon=eps)
        m_eps, _, _ = evaluate_model(gcn, data_eps)
        fgsm_curve.append(m_eps["accuracy"])
        data_feat_eps, _ = run_feature_evasion(data, idx_test[:100], binary_flip_budget=int(200 * eps), continuous_noise_std=eps, seed=42)
        m_feat_eps, _, _ = evaluate_model(gcn, data_feat_eps)
        feature_curve.append(m_feat_eps["accuracy"])
    plot_robustness_curves(budgets, [fgsm_curve, feature_curve], ["FGSM-like", "Evasion: Feature"], title="Robustness vs Perturbation Budget")
    plt.savefig("results/robustness_curve.png")
    plt.close()

    class_names = [str(i) for i in range(dataset.num_classes)]
    y_true = data.y[data.test_mask].cpu().numpy()
    plot_confusion_matrix(y_true, gcn_preds["Baseline"][data.test_mask].cpu().numpy(), class_names, "results/confusion_baseline.png", "GCN Baseline")
    plot_confusion_matrix(y_true, gcn_preds["Evasion: Feature"][data.test_mask].cpu().numpy(), class_names, "results/confusion_feature_attack.png", "Evasion: Feature")
    plot_confusion_matrix(y_true, ontology_pred[data.test_mask].cpu().numpy(), class_names, "results/confusion_ontology_defense.png", "Ontology Defense")

    with torch.no_grad():
        emb_clean = gcn.get_embeddings(data).cpu().numpy()
        emb_attacked = gcn.get_embeddings(data_feature).cpu().numpy()
        emb_defended = gcn.get_embeddings(ontology_defended_data).cpu().numpy()
    plot_tsne_embeddings(emb_clean, labels, "results/tsne_clean.png", "Layer-wise Embedding (Clean)")
    plot_tsne_embeddings(emb_attacked, labels, "results/tsne_attacked.png", "Layer-wise Embedding (Attacked)")
    plot_tsne_embeddings(emb_defended, labels, "results/tsne_defended.png", "Layer-wise Embedding (Defended)")

    print("\n=== DYNAMIC ATTACK SUITE ===")
    generator = DynamicGraphGenerator(initial_nodes=200, num_features=dataset.num_features, num_classes=dataset.num_classes)
    data_dyn = generator.get_pyg_data()
    for _ in range(3):
        data_dyn = generator.evolve(new_nodes=30, edges_per_node=2)

    adj_dyn = adj_from_edge_index(data_dyn.edge_index, data_dyn.num_nodes)
    features_dyn = data_dyn.x.cpu().numpy()
    labels_dyn = data_dyn.y.cpu().numpy()
    idx_train_dyn = np.where(data_dyn.train_mask.cpu().numpy())[0]
    idx_test_dyn = np.where(data_dyn.test_mask.cpu().numpy())[0]

    gcn_dyn = train_model(GCN(dataset.num_features, 16, dataset.num_classes), data_dyn, epochs=80)
    gat_dyn = train_model(GAT(dataset.num_features, 8, dataset.num_classes, heads=4), data_dyn, epochs=80)

    payloads_dyn = build_dynamic_attack_payloads(data_dyn, adj_dyn, features_dyn, labels_dyn, idx_train_dyn, idx_test_dyn)
    data_fgsm_dyn = run_fgsm_like_feature_attack(gcn_dyn, data_dyn, epsilon=0.05)
    payloads_dyn.append({"name": "Evasion: Gradient (FGSM-like)", "type": "evasion", "data": data_fgsm_dyn, "budget": 0.05, "p_rate": perturbation_rate(data_dyn.x.cpu().numpy(), data_fgsm_dyn.x.cpu().numpy())})
    adj_edgeflip_dyn = next((p.get("adj") for p in payloads_dyn if p["name"] == "Evasion: Edge Flip"), None)

    gcn_dyn_builder = lambda: GCN(dataset.num_features, 16, dataset.num_classes)
    gat_dyn_builder = lambda: GAT(dataset.num_features, 8, dataset.num_classes, heads=4)

    gcn_dyn_df, gcn_dyn_preds, gcn_dyn_prob = evaluate_model_under_attacks("GCN-Dynamic", gcn_dyn, gcn_dyn_builder, data_dyn, payloads_dyn, poison_epochs=80)
    gat_dyn_df, gat_dyn_preds, gat_dyn_prob = evaluate_model_under_attacks("GAT-Dynamic", gat_dyn, gat_dyn_builder, data_dyn, payloads_dyn, poison_epochs=80)

    dynamic_attack_only = gcn_dyn_df[gcn_dyn_df["Attack"] != "Baseline"].copy()
    dynamic_worst = dynamic_attack_only.sort_values("Accuracy Drop", ascending=False).iloc[0]
    print(f"Dynamic most impactful attack (GCN): {dynamic_worst['Attack']} drop={dynamic_worst['Accuracy Drop']:.4f}")

    data_feature_dyn = [p["data"] for p in payloads_dyn if p["name"] == "Evasion: Feature"][0]
    adj_onto_dyn = None
    defense_rows_dyn = apply_feature_defenses(adj_dyn, features_dyn, labels_dyn, data_feature_dyn, gcn_dyn, model_name="GCN-Dynamic")
    gcn_dyn_metrics, _, _ = evaluate_model(gcn_dyn, data_dyn)
    for dname, dmetrics, dpred_dyn, dprobs_dyn, _, _, adj_onto_candidate in defense_rows_dyn:
        if dname == "Defense: Ontology":
            adj_onto_dyn = adj_onto_candidate
        gcn_dyn_df = pd.concat(
            [
                gcn_dyn_df,
                pd.DataFrame(
                    [
                        make_result_row(
                            dname,
                            gcn_dyn_metrics,
                            dmetrics,
                            gcn_dyn_prob["Baseline"][data_dyn.test_mask].cpu().numpy(),
                            dprobs_dyn[data_dyn.test_mask].cpu().numpy(),
                            budget=0.3 if "Ontology" in dname else 0.7,
                            p_rate=0.0,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    gcn_dyn_df.to_csv("results/dynamic_gcn_evaluation_table.csv", index=False)
    gat_dyn_df.to_csv("results/dynamic_gat_evaluation_table.csv", index=False)

    dynamic_summary = []
    for attack in ["Baseline", "Evasion: Feature", "Defense: Feature Smoothing", "Defense: Ontology"]:
        row = gcn_dyn_df[gcn_dyn_df["Attack"] == attack]
        if not row.empty:
            dynamic_summary.append({"Attack": attack, "Accuracy": float(row["Accuracy"].iloc[0])})
    pd.DataFrame(dynamic_summary).to_csv("results/dynamic_summary.csv", index=False)

    print_metric_table(
        "FINAL METRICS TABLE (GCN, DYNAMIC)",
        gcn_dyn_df,
        [
            "Attack",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro F1",
            "Micro F1",
            "ROC-AUC",
            "Log-loss",
            "Classification Margin",
            "Robustness Score",
            "Attack Success Rate",
            "Margin Drop",
            "Perturbation Rate",
            "Perturbation Budget",
        ],
    )
    print_metric_table(
        "FINAL METRICS TABLE (GAT, DYNAMIC)",
        gat_dyn_df,
        [
            "Attack",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro F1",
            "Micro F1",
            "ROC-AUC",
            "Log-loss",
            "Classification Margin",
            "Robustness Score",
            "Attack Success Rate",
            "Margin Drop",
            "Perturbation Rate",
            "Perturbation Budget",
        ],
    )

    dyn_graph_rows = []
    for name, adj in [
        ("Clean", adj_dyn),
        ("Attacked (Edge Flip)", adj_edgeflip_dyn if adj_edgeflip_dyn is not None else adj_dyn),
        ("Defended (Ontology)", adj_onto_dyn if adj_onto_dyn is not None else adj_dyn),
    ]:
        gm = compute_graph_metrics(adj, labels_dyn)
        dyn_graph_rows.append(
            {
                "Graph": name,
                "Density": gm.get("density", 0.0),
                "Modularity": gm.get("modularity", 0.0),
                "Conductance": gm.get("conductance", 0.0),
            }
        )
    dyn_graph_df = pd.DataFrame(dyn_graph_rows)
    dyn_graph_df.to_csv("results/graph_metrics_dynamic.csv", index=False)
    print_metric_table(
        "GRAPH METRICS (DYNAMIC)",
        dyn_graph_df,
        ["Graph", "Density", "Modularity", "Conductance"],
    )

    generate_report("results/final_evaluation_table.csv", "results/final_report.md")

    print("\nOutputs written under results/:")
    print("- final_evaluation_table.csv (GCN static)")
    print("- final_evaluation_table_gat.csv (GAT static)")
    print("- dynamic_gcn_evaluation_table.csv")
    print("- dynamic_gat_evaluation_table.csv")
    print("- graph_metrics_static.csv / graph_metrics_dynamic.csv")
    print("- layerwise_debug_report.md")
    print("- layer_outputs_gcn.png / layer_outputs_gat.png")
    print("- graph_mosaic.png / clean_graph.png / robustness_curve.png")
    print("- tsne_clean.png / tsne_attacked.png / tsne_defended.png")
    print("- confusion_baseline.png / confusion_feature_attack.png / confusion_ontology_defense.png")
    print("- final_report.md")


if __name__ == "__main__":
    main()
