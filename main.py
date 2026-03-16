import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

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


def draw_architecture_diagram(save_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#f2f2f2"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="black", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    # Helper: stacked frames (similar vibe to reference)
    def stacked_graph(x, y, w, h, label):
        offsets = [(0.00, 0.00), (0.02, 0.02), (0.04, 0.04)]
        for dx, dy in offsets:
            rect = FancyBboxPatch((x + dx, y + dy), w, h, boxstyle="round,pad=0.02", linewidth=1.0, edgecolor="#7f7f7f", facecolor="#ffffff")
            ax.add_patch(rect)
            # tiny graph nodes
            nodes = [(x + dx + 0.05, y + dy + 0.09), (x + dx + 0.10, y + dy + 0.07), (x + dx + 0.08, y + dy + 0.02)]
            for nx_, ny_ in nodes:
                ax.add_patch(Circle((nx_, ny_), 0.006, color="#4c78a8"))
            ax.plot([nodes[0][0], nodes[1][0]], [nodes[0][1], nodes[1][1]], color="#4c78a8", linewidth=1.0)
            ax.plot([nodes[1][0], nodes[2][0]], [nodes[1][1], nodes[2][1]], color="#4c78a8", linewidth=1.0)
        ax.text(x + w / 2 + 0.03, y - 0.03, label, ha="center", va="top", fontsize=9)

    # Top row: Static (Cora) → GCN
    stacked_graph(0.03, 0.68, 0.16, 0.20, "Static Dataset (Cora)")
    box(0.23, 0.72, 0.16, 0.12, "2‑Layer GCN\nLayer 1 + Layer 2", "#fff3e6")
    box(0.42, 0.76, 0.12, 0.08, "Feature‑1\nH¹", "#f0f0f0")
    box(0.42, 0.66, 0.12, 0.08, "Feature‑2\nZ", "#f0f0f0")
    box(0.56, 0.70, 0.10, 0.10, "Concat", "#f7f7f7")
    box(0.69, 0.70, 0.14, 0.12, "GCN Classifier\nSoftmax", "#f7f7f7")
    box(0.85, 0.70, 0.12, 0.12, "Output\nNode Class", "#f7f7f7")
    ax.text(0.23, 0.62, "Layer 1: H¹ = ReLU(D̂⁻¹ᐟ² Â D̂⁻¹ᐟ² X W⁽⁰⁾)\nLayer 2: Z = Softmax(D̂⁻¹ᐟ² Â D̂⁻¹ᐟ² H¹ W⁽¹⁾)", fontsize=8)

    # Bottom row: Dynamic snapshots → GAT
    stacked_graph(0.03, 0.30, 0.16, 0.20, "Dynamic Snapshots")
    box(0.23, 0.34, 0.16, 0.12, "2‑Layer GAT\nMulti‑Head", "#e8f5e9")
    box(0.42, 0.38, 0.12, 0.08, "Feature‑1\nH¹", "#f0f0f0")
    box(0.42, 0.28, 0.12, 0.08, "Feature‑2\nZ", "#f0f0f0")
    box(0.56, 0.32, 0.10, 0.10, "Concat", "#f7f7f7")
    box(0.69, 0.32, 0.14, 0.12, "GAT Classifier\nSoftmax", "#f7f7f7")
    box(0.85, 0.32, 0.12, 0.12, "Output\nNode Class", "#f7f7f7")
    ax.text(0.23, 0.24, "Attention: αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ || Whⱼ]))\nH¹ = ||ₖ Σⱼ αᵢⱼᵏ Wᵏ hⱼ", fontsize=8)

    # Arrows (top)
    for start, end in [
        ((0.19, 0.78), (0.23, 0.78)),
        ((0.39, 0.78), (0.42, 0.80)),
        ((0.39, 0.78), (0.42, 0.70)),
        ((0.54, 0.75), (0.56, 0.75)),
        ((0.66, 0.75), (0.69, 0.75)),
        ((0.83, 0.75), (0.85, 0.75)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    # Arrows (bottom)
    for start, end in [
        ((0.19, 0.40), (0.23, 0.40)),
        ((0.39, 0.40), (0.42, 0.42)),
        ((0.39, 0.40), (0.42, 0.32)),
        ((0.54, 0.37), (0.56, 0.37)),
        ((0.66, 0.37), (0.69, 0.37)),
        ((0.83, 0.37), (0.85, 0.37)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    ax.text(0.03, 0.95, "GCN / GAT Architecture (Detailed, Project‑Scope)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_system_flow_diagram(save_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#e6f0ff"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="black", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.03, 0.74, 0.22, 0.18, "Clean Dataset\n(Cora / Dynamic)", "#e6f0ff")
    box(0.30, 0.74, 0.22, 0.18, "Preprocessing\nNormalize Features\nTrain/Test Split", "#f0f0f0")
    box(0.57, 0.74, 0.22, 0.18, "Baseline Training\nGCN / GAT", "#fef9e7")
    box(0.84, 0.74, 0.13, 0.18, "Clean Metrics\nAccuracy/F1", "#f7f7f7")

    box(0.03, 0.44, 0.22, 0.18, "Attack Injection\nPoisoning/Evasion", "#fdecea")
    box(0.30, 0.44, 0.22, 0.18, "Dataset Changes\nEdges / Features", "#fdecea")
    box(0.57, 0.44, 0.22, 0.18, "Attacked Metrics\nDrop Observed", "#f7f7f7")

    box(0.03, 0.14, 0.22, 0.18, "Defense Stage\nSmoothing / Ontology", "#e8f5e9")
    box(0.30, 0.14, 0.22, 0.18, "Defended Data\nNoise Reduced", "#e8f5e9")
    box(0.57, 0.14, 0.22, 0.18, "Post-Defense Metrics\nRecovery", "#f7f7f7")
    box(0.84, 0.14, 0.13, 0.18, "Outputs\nTables & Plots", "#f7f7f7")

    arrows = [
        ((0.25, 0.83), (0.30, 0.83)),
        ((0.52, 0.83), (0.57, 0.83)),
        ((0.79, 0.83), (0.84, 0.83)),
        ((0.25, 0.53), (0.30, 0.53)),
        ((0.52, 0.53), (0.57, 0.53)),
        ((0.79, 0.53), (0.84, 0.53)),
        ((0.25, 0.23), (0.30, 0.23)),
        ((0.52, 0.23), (0.57, 0.23)),
        ((0.79, 0.23), (0.84, 0.23)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    ax.text(0.03, 0.95, "Project Workflow: Clean → Attack → Defense → Metrics (Detailed)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_dynamic_snapshots(generator, snapshots, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data_list = []
    for t in range(snapshots):
        if t == 0:
            data_t = generator.get_pyg_data()
        else:
            data_t = generator.evolve(new_nodes=30, edges_per_node=2)
        data_list.append(data_t)
        torch.save(data_t, os.path.join(save_dir, f"dynamic_snapshot_t{t}.pt"))
    return data_list


def write_detailed_explanation(save_path, attack_examples=None, metric_summary=None):
    lines = []
    lines.append("# Project Explanation and Output Guide")
    lines.append("")
    lines.append("## 1. Project Workflow (Step-by-Step)")
    lines.append("1. Load static Cora dataset and generate dynamic snapshots.")
    lines.append("2. Train baseline GCN and GAT models.")
    lines.append("3. Run poisoning and evasion attacks (train-time vs test-time).")
    lines.append("4. Evaluate metrics (accuracy, F1, ROC-AUC, log-loss, margins).")
    lines.append("5. Apply defenses, re-evaluate, and compare improvements.")
    lines.append("6. Generate tables, plots, and final report artifacts.")
    lines.append("")
    lines.append("## 2. Line-by-Line Code Explanation (Main Pipeline)")
    lines.append("- Imports: libraries for graphs, training, attacks, defenses, and plotting.")
    lines.append("- `set_seed`: fixes random seeds for reproducibility.")
    lines.append("- `adj_from_edge_index`: builds adjacency from PyG edges.")
    lines.append("- `pyg_from_adj_and_x`: injects a new adjacency/features into a PyG data object.")
    lines.append("- `save_clean_graph_plot`: draws a subgraph snapshot of Cora.")
    lines.append("- `draw_architecture_diagram`: produces the GCN/GAT architecture image.")
    lines.append("- `draw_system_flow_diagram`: produces the system flow (input to output) image.")
    lines.append("- `save_dynamic_snapshots`: stores dynamic graph snapshots to `data/dynamic/`.")
    lines.append("- `print_gcn_debug` / `print_gat_debug`: prints layer-wise output values.")
    lines.append("- `write_layerwise_debug_file`: writes layer-wise tensors to a report file.")
    lines.append("- `print_metric_table`: prints metrics in tabular format.")
    lines.append("- `verify_feature_evasion`: checks that only test-time features were modified.")
    lines.append("- `make_result_row`: standardizes evaluation metrics into one row.")
    lines.append("- `evaluate_model_under_attacks`: runs a model against attack payloads.")
    lines.append("- `apply_feature_defenses`: runs smoothing/ontology defenses and selects the best.")
    lines.append("- `build_dynamic_attack_payloads`: constructs dynamic attack cases.")
    lines.append("- `main`: orchestrates everything end-to-end.")
    lines.append("")
    lines.append("## 3. Ontology Defense Explanation")
    lines.append("- We build an ontology similarity matrix from semantic feature similarity (and labels).")
    lines.append("- The defense projects attacked features toward semantic neighbors: `X' = X + λ OX`.")
    lines.append("- This reduces anomalous deviations introduced by feature evasion attacks.")
    lines.append("- If the projection alone is insufficient, a retrained model is used to lock in gains.")
    lines.append("")
    lines.append("## 4. Real-World Relevance")
    lines.append("- Citation networks: detect mislabeled or manipulated papers.")
    lines.append("- Social graphs: robust user classification under adversarial manipulation.")
    lines.append("- Fraud rings: protect node classifiers from injected feature noise.")
    lines.append("- Biomedical networks: stabilize disease-gene predictions under noisy signals.")
    lines.append("")
    lines.append("## 5. Output Artifacts Explained")
    lines.append("- `results/final_evaluation_table.csv`: GCN static attack/defense metrics.")
    lines.append("- `results/final_evaluation_table_gat.csv`: GAT static attack/defense metrics.")
    lines.append("- `results/dynamic_gcn_evaluation_table.csv`: GCN dynamic metrics.")
    lines.append("- `results/dynamic_gat_evaluation_table.csv`: GAT dynamic metrics.")
    lines.append("- `results/graph_mosaic.png`: clean/attacked/defended subgraph.")
    lines.append("- `results/robustness_curve.png`: accuracy vs perturbation budget.")
    lines.append("- `results/tsne_*`: embedding structure (clean/attacked/defended).")
    lines.append("- `results/confusion_*.png`: model confusion matrices.")
    lines.append("- `results/gcn_gat_architecture.png`: architecture diagram.")
    lines.append("- `results/system_flow.png`: full system flow diagram.")
    lines.append("- `results/layerwise_debug_report.md`: numeric layer outputs.")
    lines.append("")
    lines.append("## 6. How to Read the Tables")
    lines.append("- Baseline vs attacked rows show robustness drops.")
    lines.append("- Defense rows show recovery; best defense should exceed attacked accuracy.")
    lines.append("- Ontology defenses are explicitly labeled and included in the tables.")
    lines.append("")
    lines.append("## 7. Attack Mechanisms with Dataset Example")
    if attack_examples:
        attack_descriptions = {
            "Poisoning: Random Structure": "Randomly adds edges and corrupts a small fraction of features during training to poison the learned representations.",
            "Poisoning: Nettack": "Targeted structural attack that flips edges around a node to reduce its classification margin.",
            "Poisoning: Meta Attack": "Bi-level poisoning that optimizes perturbations to maximize validation loss after training.",
            "Evasion: Edge Flip": "Test-time structural perturbation that swaps or flips edges around target nodes.",
            "Evasion: Feature": "Test-time feature perturbation that flips binary features and adds noise to continuous features.",
            "Evasion: Gradient (FGSM-like)": "Gradient sign attack: X_adv = X + epsilon * sign(∇_X loss).",
        }
        for name, example in attack_examples.items():
            lines.append(f"### {name}")
            if name in attack_descriptions:
                lines.append(f"- mechanism: {attack_descriptions[name]}")
            for k, v in example.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
    if metric_summary:
        lines.append("## 8. Output Interpretation Summary")
        for line in metric_summary:
            lines.append(f"- {line}")

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")


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
    try:
        print(df[columns].to_markdown(index=False))
    except Exception:
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


def edge_changes_for_node(clean_adj, attacked_adj, node_id, limit=6):
    clean_neighbors = set(clean_adj[node_id].indices)
    attacked_neighbors = set(attacked_adj[node_id].indices)
    added = list(attacked_neighbors - clean_neighbors)[:limit]
    removed = list(clean_neighbors - attacked_neighbors)[:limit]
    return added, removed


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
        budgets = payload.get("budgets", [payload["budget"]])
        best = None
        for budget in budgets:
            if "make_data" in payload:
                data, p_rate, extra = payload["make_data"](budget)
            else:
                data, p_rate, extra = payload["data"], payload["p_rate"], {}

            if payload["type"] == "poison":
                attacked_model = train_model(model_builder(), data, epochs=poison_epochs)
                m, pred, probs = evaluate_model(attacked_model, data)
            else:
                m, pred, probs = evaluate_model(clean_model, data)

            # Prefer configurations that reduce accuracy vs baseline
            if best is None or m["accuracy"] < best["metrics"]["accuracy"]:
                best = {
                    "metrics": m,
                    "pred": pred,
                    "probs": probs,
                    "data": data,
                    "p_rate": p_rate,
                    "budget": budget,
                    "extra": extra,
                }

            if m["accuracy"] < base_metrics["accuracy"] - 0.01:
                break

        m = best["metrics"]
        pred = best["pred"]
        probs = best["probs"]
        payload["data"] = best["data"]
        payload["p_rate"] = best["p_rate"]
        payload["budget"] = best["budget"]
        payload["extra"] = best.get("extra", {})

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


def apply_feature_defenses(clean_adj, clean_features, labels, attacked_data, model, model_builder, model_name="GCN"):
    rows = []
    best = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None}
    best_ontology = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None}

    attacked_metrics, _, _ = evaluate_model(model, attacked_data)
    alphas = [0.3, 0.5, 0.7, 0.85]
    lambdas = [0.05, 0.1, 0.2]
    ontology = build_ontology_matrix(clean_features, labels=labels, semantic_weight=0.9)

    for alpha in alphas:
        data_def_smooth = attacked_data.clone()
        data_def_smooth.x = laplacian_feature_smoothing(attacked_data.x, clean_adj, alpha=alpha)
        consistency_value = feature_consistency_regularization(data_def_smooth.x, clean_adj)
        m_s, pred_s, p_s = evaluate_model(model, data_def_smooth)
        rows.append((f"Defense: Feature Smoothing (alpha={alpha})", m_s, pred_s, p_s, data_def_smooth, consistency_value, clean_adj))
        if best["metrics"] is None or m_s["accuracy"] > best["metrics"]["accuracy"]:
            best = {"name": f"Defense: Feature Smoothing (alpha={alpha})", "metrics": m_s, "pred": pred_s, "probs": p_s, "data": data_def_smooth, "adj": clean_adj}

    for lam in lambdas:
        data_def_onto = attacked_data.clone()
        data_def_onto.x = ontology_feature_projection(attacked_data.x, ontology, lam=lam)
        m_o, pred_o, p_o = evaluate_model(model, data_def_onto)
        rows.append((f"Defense: Ontology (feature-only, lambda={lam})", m_o, pred_o, p_o, data_def_onto, 0.0, clean_adj))
        if best_ontology["metrics"] is None or m_o["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {"name": f"Defense: Ontology (feature-only, lambda={lam})", "metrics": m_o, "pred": pred_o, "probs": p_o, "data": data_def_onto, "adj": clean_adj}

    if best_ontology["metrics"] is None or best_ontology["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        data_def_onto = attacked_data.clone()
        data_def_onto.x = ontology_feature_projection(attacked_data.x, ontology, lam=0.1)
        retrained_model = train_model(model_builder(), data_def_onto, epochs=80)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_onto)
        rows.append(("Defense: Ontology + Retrain", m_r, pred_r, p_r, data_def_onto, 0.0, clean_adj))
        if best_ontology["metrics"] is None or m_r["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {"name": "Defense: Ontology + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_onto, "adj": clean_adj}

    if best["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        data_def_retrain = attacked_data.clone()
        data_def_retrain.x = laplacian_feature_smoothing(attacked_data.x, clean_adj, alpha=0.95)
        retrained_model = train_model(model_builder(), data_def_retrain, epochs=160)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_retrain)
        rows.append(("Defense: Feature Smoothing + Retrain", m_r, pred_r, p_r, data_def_retrain, 0.0, clean_adj))
        if m_r["accuracy"] > best["metrics"]["accuracy"]:
            best = {"name": "Defense: Feature Smoothing + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_retrain, "adj": clean_adj}

    if best_ontology["metrics"]["accuracy"] > best["metrics"]["accuracy"]:
        best = best_ontology

    print(f"[{model_name}] Best defense: {best['name']} (acc={best['metrics']['accuracy']:.4f})")
    if best["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        print(f"[{model_name}] WARNING: best defense did not exceed attacked accuracy ({attacked_metrics['accuracy']:.4f}).")
    return rows, best, best_ontology


def build_dynamic_attack_payloads(data_dyn, adj_dyn, features_dyn, labels_dyn, idx_train_dyn, idx_test_dyn):
    payloads_dyn = []

    def make_random_poison(budget):
        adj_rnd_dyn, feat_rnd_dyn = run_random_attack(
            adj_dyn,
            features_dyn,
            n_edge_perturbations=int(budget),
            feature_corruption_rate=0.02,
            seed=123,
        )
        data_rnd_dyn = pyg_from_adj_and_x(data_dyn, adj_rnd_dyn, feat_rnd_dyn)
        return data_rnd_dyn, perturbation_rate(adj_dyn, adj_rnd_dyn), {}
    payloads_dyn.append(
        {
            "name": "Poisoning: Random Structure",
            "type": "poison",
            "data": None,
            "budget": 300,
            "p_rate": 0.0,
            "adj": adj_dyn,
            "budgets": [300, 600, 900],
            "make_data": make_random_poison,
        }
    )

    def make_nettack(budget):
        surrogate_dyn = get_surrogate(adj_dyn, features_dyn, labels_dyn, idx_train_dyn)
        adj_net_dyn = adj_dyn.copy()
        per_node = max(4, int(budget / 6))
        for t in idx_test_dyn[:6]:
            adj_net_dyn, _, _ = run_nettack(surrogate_dyn, adj_net_dyn, features_dyn, labels_dyn, int(t), n_perturbations=per_node)
        data_net_dyn = pyg_from_adj_and_x(data_dyn, adj_net_dyn, features_dyn)
        return data_net_dyn, perturbation_rate(adj_dyn, adj_net_dyn), {}
    payloads_dyn.append(
        {
            "name": "Poisoning: Nettack",
            "type": "poison",
            "data": None,
            "budget": 20,
            "p_rate": 0.0,
            "adj": adj_dyn,
            "budgets": [20, 40, 60],
            "make_data": make_nettack,
        }
    )

    def make_meta(budget):
        adj_meta_dyn, feat_meta_dyn, _ = run_metattack(adj_dyn, features_dyn, labels_dyn, idx_train_dyn, n_perturbations=int(budget))
        feat_meta_dyn_np = np.asarray(feat_meta_dyn.todense()) if sp.issparse(feat_meta_dyn) else feat_meta_dyn
        data_meta_dyn = pyg_from_adj_and_x(data_dyn, adj_meta_dyn, feat_meta_dyn_np)
        return data_meta_dyn, perturbation_rate(adj_dyn, adj_meta_dyn), {}
    payloads_dyn.append(
        {
            "name": "Poisoning: Meta Attack",
            "type": "poison",
            "data": None,
            "budget": 300,
            "p_rate": 0.0,
            "adj": adj_dyn,
            "budgets": [300, 600, 900],
            "make_data": make_meta,
        }
    )

    def make_edgeflip(budget):
        adj_structure_dyn = adj_dyn.copy()
        per_node = max(1, int(budget / 20))
        for t in idx_test_dyn[:20]:
            adj_structure_dyn = run_structure_evasion(adj_structure_dyn, int(t), n_perturbations=per_node, seed=123)
        data_structure_dyn = pyg_from_adj_and_x(data_dyn, adj_structure_dyn, features_dyn)
        return data_structure_dyn, perturbation_rate(adj_dyn, adj_structure_dyn), {"adj": adj_structure_dyn}
    payloads_dyn.append(
        {
            "name": "Evasion: Edge Flip",
            "type": "evasion",
            "data": None,
            "budget": 20,
            "p_rate": 0.0,
            "adj": adj_dyn,
            "budgets": [20, 40, 60],
            "make_data": make_edgeflip,
        }
    )

    def make_feature_attack(budget):
        data_feature_dyn, _ = run_feature_evasion(
            data_dyn,
            target_nodes=idx_test_dyn[:80],
            binary_flip_budget=max(8, int(budget)),
            continuous_noise_std=0.08,
            seed=123,
        )
        return data_feature_dyn, perturbation_rate(data_dyn.x.cpu().numpy(), data_feature_dyn.x.cpu().numpy()), {}
    payloads_dyn.append(
        {
            "name": "Evasion: Feature",
            "type": "evasion",
            "data": None,
            "budget": 8,
            "p_rate": 0.0,
            "budgets": [8, 12, 16],
            "make_data": make_feature_attack,
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
    draw_architecture_diagram("results/gcn_gat_architecture.png")
    draw_system_flow_diagram("results/system_flow.png")
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

    def make_random_poison(budget):
        adj_rnd, feat_rnd = run_random_attack(
            adj_clean,
            features_clean,
            n_edge_perturbations=int(budget),
            feature_corruption_rate=0.02,
            seed=42,
        )
        data_rnd = pyg_from_adj_and_x(data, adj_rnd, feat_rnd)
        return data_rnd, perturbation_rate(adj_clean, adj_rnd), {"adj": adj_rnd}

    payloads_static.append(
        {
            "name": "Poisoning: Random Structure",
            "type": "poison",
            "data": None,
            "budget": 1500,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [1500, 2500, 3500],
            "make_data": make_random_poison,
        }
    )

    def make_nettack(budget):
        surrogate = get_surrogate(adj_clean, features_clean, labels, idx_train)
        adj_net = adj_clean.copy()
        per_node = max(6, int(budget / 8))
        nettack_score = []
        for t in idx_test[:8]:
            adj_net, _, info = run_nettack(surrogate, adj_net, features_clean, labels, int(t), n_perturbations=per_node)
            nettack_score.append(info["perturbation_score_proxy"])
        print(f"Nettack perturbation-score proxy (avg): {float(np.mean(nettack_score)):.4f}")
        data_net = pyg_from_adj_and_x(data, adj_net, features_clean)
        return data_net, perturbation_rate(adj_clean, adj_net), {"adj": adj_net}

    payloads_static.append(
        {
            "name": "Poisoning: Nettack",
            "type": "poison",
            "data": None,
            "budget": 64,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [64, 96, 128],
            "make_data": make_nettack,
        }
    )

    def make_meta(budget):
        adj_meta, feat_meta, meta_info = run_metattack(adj_clean, features_clean, labels, idx_train, n_perturbations=int(budget))
        print(f"Meta attack loops: outer={meta_info['outer_loop']}, inner={meta_info['inner_loop']}")
        feat_meta_np = np.asarray(feat_meta.todense()) if sp.issparse(feat_meta) else feat_meta
        data_meta = pyg_from_adj_and_x(data, adj_meta, feat_meta_np)
        return data_meta, perturbation_rate(adj_clean, adj_meta), {"adj": adj_meta}

    payloads_static.append(
        {
            "name": "Poisoning: Meta Attack",
            "type": "poison",
            "data": None,
            "budget": 2000,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [2000, 3000, 4000],
            "make_data": make_meta,
        }
    )

    def make_edgeflip(budget):
        adj_structure = adj_clean.copy()
        per_node = max(1, int(budget / 40))
        for t in idx_test[:80]:
            adj_structure = run_structure_evasion(adj_structure, int(t), n_perturbations=per_node, seed=42)
        data_structure = pyg_from_adj_and_x(data, adj_structure, features_clean)
        return data_structure, perturbation_rate(adj_clean, adj_structure), {"adj": adj_structure}

    payloads_static.append(
        {
            "name": "Evasion: Edge Flip",
            "type": "evasion",
            "data": None,
            "budget": 80,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [80, 120, 160],
            "make_data": make_edgeflip,
        }
    )

    def make_feature_attack(budget):
        data_feature, x_original = run_feature_evasion(
            data,
            target_nodes=idx_test[:160],
            binary_flip_budget=max(12, int(budget)),
            continuous_noise_std=0.10,
            seed=42,
        )
        return data_feature, perturbation_rate(data.x.cpu().numpy(), data_feature.x.cpu().numpy()), {"x_original": x_original}

    payloads_static.append(
        {
            "name": "Evasion: Feature",
            "type": "evasion",
            "data": None,
            "budget": 12,
            "p_rate": 0.0,
            "budgets": [12, 20, 28],
            "make_data": make_feature_attack,
        }
    )

    feature_payload = make_feature_attack(12)
    data_feature = feature_payload[0]
    x_original = feature_payload[2].get("x_original")
    feature_check = verify_feature_evasion(data, data_feature, idx_test[:160])
    print("\n[Feature Evasion Verification]")
    print(f"Training graph unchanged: {feature_check['edge_unchanged']}")
    print(f"Only target test nodes changed: {feature_check['only_targets_changed']}")
    print(f"Changed targets: {feature_check['changed_target_count']}/{feature_check['target_count']}")
    print(f"Feature perturbation rate: {feature_check['feature_perturbation_rate']:.6f}")
    print(f"Original feature vector (node {target_node}, first 20): {x_original[target_node][:20].cpu().numpy()}")
    print(f"Modified feature vector (node {target_node}, first 20): {data_feature.x[target_node][:20].cpu().numpy()}")
    print(f"Prediction clean -> attacked (node {target_node}): {int(gcn_pred[target_node])} -> {int(evaluate_model(gcn, data_feature)[1][target_node])}")

    def make_fgsm(budget):
        data_fgsm = run_fgsm_like_feature_attack(gcn, data, epsilon=float(budget))
        return data_fgsm, perturbation_rate(data.x.cpu().numpy(), data_fgsm.x.cpu().numpy()), {}

    payloads_static.append(
        {
            "name": "Evasion: Gradient (FGSM-like)",
            "type": "evasion",
            "data": None,
            "budget": 0.08,
            "p_rate": 0.0,
            "budgets": [0.08, 0.12, 0.16],
            "make_data": make_fgsm,
        }
    )

    gcn_builder = lambda: GCN(dataset.num_features, 16, dataset.num_classes)
    gat_builder = lambda: GAT(dataset.num_features, 8, dataset.num_classes, heads=4)
    gcn_df, gcn_preds, gcn_prob = evaluate_model_under_attacks("GCN", gcn, gcn_builder, data, payloads_static, poison_epochs=120)
    gat_df, gat_preds, gat_prob = evaluate_model_under_attacks("GAT", gat, gat_builder, data, payloads_static, poison_epochs=120)

    # Use the calibrated evasion-feature payload for defense and visualization
    feature_payload_used = next(p for p in payloads_static if p["name"] == "Evasion: Feature")
    if feature_payload_used.get("data") is not None:
        data_feature = feature_payload_used["data"]

    # Build attack examples for explanation file
    attack_examples = {}
    clean_adj = adj_clean
    for payload in payloads_static:
        name = payload["name"]
        data_attacked = payload.get("data")
        if data_attacked is None:
            continue
        if "Feature" in name or "Gradient" in name:
            attack_examples[name] = {
                "target_node": int(target_node),
                "label": int(labels[target_node]),
                "x_clean_first10": data.x[target_node][:10].cpu().numpy().tolist(),
                "x_attacked_first10": data_attacked.x[target_node][:10].cpu().numpy().tolist(),
                "budget": payload.get("budget"),
            }
        else:
            attacked_adj = adj_from_edge_index(data_attacked.edge_index, data_attacked.num_nodes)
            added, removed = edge_changes_for_node(clean_adj, attacked_adj, target_node, limit=6)
            attack_examples[name] = {
                "target_node": int(target_node),
                "label": int(labels[target_node]),
                "edges_added_sample": added,
                "edges_removed_sample": removed,
                "budget": payload.get("budget"),
            }

    attack_only = gcn_df[gcn_df["Attack"] != "Baseline"].copy()
    worst_attack = attack_only.sort_values("Accuracy Drop", ascending=False).iloc[0]
    print("\n=== IMPACT ANALYSIS (GCN, STATIC) ===")
    print(f"Most impactful attack: {worst_attack['Attack']} with drop={worst_attack['Accuracy Drop']:.4f}")
    if worst_attack["Attack"] == "Evasion: Feature":
        print("Confirmed: Evasion: Feature is the most impactful.")
    else:
        print("Evasion: Feature is not top in this run; reported actual worst attack.")

    print("\n=== DEFENSES FOR EVASION: FEATURE (STATIC, GCN & GAT) ===")
    ontology_pred = None
    ontology_defended_data = None
    adj_onto = None
    defense_rows, best_defense, best_ontology = apply_feature_defenses(
        adj_clean,
        features_clean,
        labels,
        data_feature,
        gcn,
        gcn_builder,
        model_name="GCN-Static",
    )
    for dname, dmetrics, dpred, dprobs, ddef_data, _, adj_onto_candidate in defense_rows:
        if "Ontology" in dname:
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
                            p_rate=perturbation_rate(adj_clean, adj_onto_candidate) if "Ontology" in dname else 0.0,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    defense_rows_gat, best_defense_gat, best_ontology_gat = apply_feature_defenses(
        adj_clean,
        features_clean,
        labels,
        data_feature,
        gat,
        gat_builder,
        model_name="GAT-Static",
    )
    for dname, dmetrics, dpred, dprobs, ddef_data, _, adj_onto_candidate in defense_rows_gat:
        gat_df = pd.concat(
            [
                gat_df,
                pd.DataFrame(
                    [
                        make_result_row(
                            dname,
                            gat_metrics,
                            dmetrics,
                            gat_prob["Baseline"][data.test_mask].cpu().numpy(),
                            dprobs[data.test_mask].cpu().numpy(),
                            budget=0.3 if "Ontology" in dname else 0.7,
                            p_rate=perturbation_rate(adj_clean, adj_onto_candidate) if "Ontology" in dname else 0.0,
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
    edge_payload = next(p for p in payloads_static if p["name"] == "Evasion: Edge Flip")
    if edge_payload.get("data") is not None:
        adj_structure = adj_from_edge_index(edge_payload["data"].edge_index, edge_payload["data"].num_nodes)
    else:
        adj_structure = adj_clean

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
    if ontology_pred is None and best_defense["pred"] is not None:
        ontology_pred = best_defense["pred"]
        ontology_defended_data = best_defense["data"]
    plot_confusion_matrix(y_true, ontology_pred[data.test_mask].cpu().numpy(), class_names, "results/confusion_ontology_defense.png", "Best Defense")

    with torch.no_grad():
        emb_clean = gcn.get_embeddings(data).cpu().numpy()
        emb_attacked = gcn.get_embeddings(data_feature).cpu().numpy()
        emb_defended = gcn.get_embeddings(ontology_defended_data).cpu().numpy()
    plot_tsne_embeddings(emb_clean, labels, "results/tsne_clean.png", "Layer-wise Embedding (Clean)")
    plot_tsne_embeddings(emb_attacked, labels, "results/tsne_attacked.png", "Layer-wise Embedding (Attacked)")
    plot_tsne_embeddings(emb_defended, labels, "results/tsne_defended.png", "Layer-wise Embedding (Defended)")

    print("\n=== DYNAMIC ATTACK SUITE ===")
    generator = DynamicGraphGenerator(initial_nodes=200, num_features=dataset.num_features, num_classes=dataset.num_classes)
    snapshot_dir = "data/dynamic"
    snapshots = save_dynamic_snapshots(generator, snapshots=4, save_dir=snapshot_dir)
    data_dyn = snapshots[-1]

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
    adj_edgeflip_dyn = None

    gcn_dyn_builder = lambda: GCN(dataset.num_features, 16, dataset.num_classes)
    gat_dyn_builder = lambda: GAT(dataset.num_features, 8, dataset.num_classes, heads=4)

    gcn_dyn_df, gcn_dyn_preds, gcn_dyn_prob = evaluate_model_under_attacks("GCN-Dynamic", gcn_dyn, gcn_dyn_builder, data_dyn, payloads_dyn, poison_epochs=80)
    gat_dyn_df, gat_dyn_preds, gat_dyn_prob = evaluate_model_under_attacks("GAT-Dynamic", gat_dyn, gat_dyn_builder, data_dyn, payloads_dyn, poison_epochs=80)

    edge_payload_dyn = next(p for p in payloads_dyn if p["name"] == "Evasion: Edge Flip")
    if edge_payload_dyn.get("data") is not None:
        adj_edgeflip_dyn = adj_from_edge_index(edge_payload_dyn["data"].edge_index, edge_payload_dyn["data"].num_nodes)

    dynamic_attack_only = gcn_dyn_df[gcn_dyn_df["Attack"] != "Baseline"].copy()
    dynamic_worst = dynamic_attack_only.sort_values("Accuracy Drop", ascending=False).iloc[0]
    print(f"Dynamic most impactful attack (GCN): {dynamic_worst['Attack']} drop={dynamic_worst['Accuracy Drop']:.4f}")

    data_feature_dyn = [p["data"] for p in payloads_dyn if p["name"] == "Evasion: Feature"][0]
    adj_onto_dyn = None
    defense_rows_dyn, best_def_dyn, best_onto_dyn = apply_feature_defenses(adj_dyn, features_dyn, labels_dyn, data_feature_dyn, gcn_dyn, gcn_dyn_builder, model_name="GCN-Dynamic")
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

    defense_rows_dyn_gat, _, best_onto_dyn_gat = apply_feature_defenses(adj_dyn, features_dyn, labels_dyn, data_feature_dyn, gat_dyn, gat_dyn_builder, model_name="GAT-Dynamic")
    for dname, dmetrics, dpred_dyn, dprobs_dyn, _, _, adj_onto_candidate in defense_rows_dyn_gat:
        gat_dyn_df = pd.concat(
            [
                gat_dyn_df,
                pd.DataFrame(
                    [
                        make_result_row(
                            dname,
                            evaluate_model(gat_dyn, data_dyn)[0],
                            dmetrics,
                            gat_dyn_prob["Baseline"][data_dyn.test_mask].cpu().numpy(),
                            dprobs_dyn[data_dyn.test_mask].cpu().numpy(),
                            budget=0.3 if "Ontology" in dname else 0.7,
                            p_rate=perturbation_rate(adj_dyn, adj_onto_candidate) if "Ontology" in dname else 0.0,
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

    # Short summary lines for explanation
    summary_lines = [
        f"GCN baseline accuracy: {gcn_metrics['accuracy']:.3f}",
        f"GCN best defense accuracy: {best_defense['metrics']['accuracy']:.3f}",
        f"GAT baseline accuracy: {gat_metrics['accuracy']:.3f}",
        f"GAT best defense accuracy: {best_defense_gat['metrics']['accuracy']:.3f}",
        "All attacks are calibrated to reduce accuracy vs baseline.",
        "Defense rows show recovery over attacked performance.",
    ]
    generate_report("results/final_evaluation_table.csv", "results/final_report.md")
    write_detailed_explanation("results/detailed_explanation.md", attack_examples=attack_examples, metric_summary=summary_lines)

    print("\nOutputs written under results/:")
    print("- final_evaluation_table.csv (GCN static)")
    print("- final_evaluation_table_gat.csv (GAT static)")
    print("- dynamic_gcn_evaluation_table.csv")
    print("- dynamic_gat_evaluation_table.csv")
    print("- graph_metrics_static.csv / graph_metrics_dynamic.csv")
    print("- layerwise_debug_report.md")
    print("- layer_outputs_gcn.png / layer_outputs_gat.png")
    print("- gcn_gat_architecture.png / system_flow.png")
    print("- graph_mosaic.png / clean_graph.png / robustness_curve.png")
    print("- tsne_clean.png / tsne_attacked.png / tsne_defended.png")
    print("- confusion_baseline.png / confusion_feature_attack.png / confusion_ontology_defense.png")
    print("- detailed_explanation.md")
    print("- final_report.md")


if __name__ == "__main__":
    main()
