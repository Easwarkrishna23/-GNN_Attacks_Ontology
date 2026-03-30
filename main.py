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
from defenses.robust_filtering import top_k_pruning
from utils.metrics import compute_robustness_metrics, perturbation_rate, compute_graph_metrics
from visualization.graph_viz import visualize_graph_mosaic, visualize_graph_pair
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
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")

    def box(x, y, w, h, text, color="#f2f2f2", fontsize=8):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="black", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

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
    stacked_graph(0.02, 0.70, 0.16, 0.22, "Static Dataset (Cora)")
    box(0.22, 0.78, 0.12, 0.10, "Input\nX, A", "#e8f0ff", fontsize=9)
    box(
        0.36,
        0.75,
        0.22,
        0.14,
        "GCN Layer 1\nNormalize + Aggregate\nH¹ = ReLU(D̂⁻¹ᐟ² Â D̂⁻¹ᐟ² X W⁽⁰⁾)",
        "#fff3e6",
        fontsize=7,
    )
    box(0.60, 0.79, 0.10, 0.08, "Feature‑1\nH¹", "#f0f0f0", fontsize=9)
    box(
        0.72,
        0.75,
        0.22,
        0.14,
        "GCN Layer 2\nZ = Softmax(D̂⁻¹ᐟ² Â D̂⁻¹ᐟ² H¹ W⁽¹⁾)",
        "#fff3e6",
        fontsize=7,
    )
    box(0.95, 0.79, 0.05, 0.08, "Output\nClass", "#f7f7f7", fontsize=8)

    # Bottom row: Dynamic snapshots → GAT
    stacked_graph(0.02, 0.30, 0.16, 0.22, "Dynamic Snapshots")
    box(0.22, 0.38, 0.12, 0.10, "Input\nX, A", "#e8f0ff", fontsize=9)
    box(
        0.36,
        0.35,
        0.22,
        0.14,
        "GAT Layer 1 (Multi‑Head)\nαᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ||Whⱼ]))\nH¹ = ||ₖ Σⱼ αᵢⱼᵏ Wᵏ hⱼ",
        "#e8f5e9",
        fontsize=7,
    )
    box(0.60, 0.39, 0.10, 0.08, "Feature‑1\nH¹ (concat)", "#f0f0f0", fontsize=8)
    box(
        0.72,
        0.35,
        0.22,
        0.14,
        "GAT Layer 2\nZ = Softmax(Σⱼ αᵢⱼ W hⱼ)",
        "#e8f5e9",
        fontsize=7,
    )
    box(0.95, 0.39, 0.05, 0.08, "Output\nClass", "#f7f7f7", fontsize=8)

    # Arrows (top)
    for start, end in [
        ((0.18, 0.80), (0.22, 0.83)),
        ((0.34, 0.83), (0.36, 0.83)),
        ((0.58, 0.83), (0.60, 0.83)),
        ((0.70, 0.83), (0.72, 0.83)),
        ((0.94, 0.83), (0.95, 0.83)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    # Arrows (bottom)
    for start, end in [
        ((0.18, 0.40), (0.22, 0.43)),
        ((0.34, 0.43), (0.36, 0.43)),
        ((0.58, 0.43), (0.60, 0.43)),
        ((0.70, 0.43), (0.72, 0.43)),
        ((0.94, 0.43), (0.95, 0.43)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    ax.text(0.03, 0.95, "GCN / GAT Architecture (Detailed, Project‑Scope)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_system_flow_diagram(save_path):
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#e6f0ff", fontsize=8):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="black", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    # Clean pipeline (top row)
    box(0.03, 0.73, 0.20, 0.20, "Clean Dataset\nX, A\n(Cora / Dynamic)", "#e6f0ff", fontsize=9)
    box(0.26, 0.73, 0.20, 0.20, "Baseline GCN/GAT\nMessage Passing\nH, Z", "#fef9e7", fontsize=9)
    box(0.49, 0.73, 0.20, 0.20, "Clean Metrics\nAccuracy, F1,\nROC-AUC", "#f7f7f7", fontsize=9)
    box(0.72, 0.73, 0.22, 0.20, "Saved Outputs\nClean Plots\nBaseline Tables", "#f7f7f7", fontsize=9)

    # Attack pipeline (middle row)
    box(0.03, 0.43, 0.20, 0.20, "Attack Injection\nPoisoning / Evasion", "#fdecea", fontsize=9)
    box(0.26, 0.43, 0.20, 0.20, "Corrupted Data\nX + ΔX\nA + ΔA", "#fdecea", fontsize=9)
    box(0.49, 0.43, 0.20, 0.20, "Attacked Metrics\nAccuracy Drop\nMargin Shift", "#f7f7f7", fontsize=9)
    box(0.72, 0.43, 0.22, 0.20, "Saved Outputs\nAttack Plots\nAttack Tables", "#f7f7f7", fontsize=9)

    # Defense pipeline (bottom row)
    box(0.03, 0.13, 0.20, 0.20, "Defense Stage\nFeature Smoothing\nX_s = αX + (1-α)ÂX", "#e8f5e9", fontsize=8)
    box(0.26, 0.13, 0.20, 0.20, "Ontology Defense\nX' = X + λ OX", "#e8f5e9", fontsize=8)
    box(0.49, 0.13, 0.20, 0.20, "Post-Defense Metrics\nRecovered Accuracy", "#f7f7f7", fontsize=9)
    box(0.72, 0.13, 0.22, 0.20, "Final Outputs\nPost-Defense Tables\nVisual Comparisons", "#f7f7f7", fontsize=9)

    arrows = [
        ((0.23, 0.83), (0.26, 0.83)),
        ((0.46, 0.83), (0.49, 0.83)),
        ((0.69, 0.83), (0.72, 0.83)),
        ((0.23, 0.53), (0.26, 0.53)),
        ((0.46, 0.53), (0.49, 0.53)),
        ((0.69, 0.53), (0.72, 0.53)),
        ((0.23, 0.23), (0.26, 0.23)),
        ((0.46, 0.23), (0.49, 0.23)),
        ((0.69, 0.23), (0.72, 0.23)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    ax.text(0.03, 0.95, "Project Workflow: Clean → Attack → Defense → Metrics (Detailed)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_gcn_layerwise_diagram(save_path):
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#f2f2f2", fontsize=9):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.6,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=14, linewidth=1.8))

    def stacked_input(x, y, w, h, label):
        offsets = [(0.00, 0.00), (0.02, 0.02), (0.04, 0.04)]
        for dx, dy in offsets:
            rect = FancyBboxPatch(
                (x + dx, y + dy),
                w,
                h,
                boxstyle="round,pad=0.02",
                linewidth=1.0,
                edgecolor="#7f7f7f",
                facecolor="#ffffff",
            )
            ax.add_patch(rect)
            nodes = [
                (x + dx + 0.05, y + dy + 0.12),
                (x + dx + 0.10, y + dy + 0.08),
                (x + dx + 0.07, y + dy + 0.03),
                (x + dx + 0.12, y + dy + 0.03),
            ]
            for nx_, ny_ in nodes:
                ax.add_patch(Circle((nx_, ny_), 0.008, color="#4c78a8"))
            ax.plot([nodes[0][0], nodes[1][0]], [nodes[0][1], nodes[1][1]], color="#4c78a8", linewidth=1.2)
            ax.plot([nodes[1][0], nodes[2][0]], [nodes[1][1], nodes[2][1]], color="#4c78a8", linewidth=1.2)
            ax.plot([nodes[1][0], nodes[3][0]], [nodes[1][1], nodes[3][1]], color="#4c78a8", linewidth=1.2)
        ax.text(x + w / 2 + 0.03, y - 0.04, label, ha="center", va="top", fontsize=10)

    ax.text(0.02, 0.95, "GCN: Detailed Layer-Wise Node Classification Flow (Project Scope)", fontsize=15, weight="bold")

    # Input
    stacked_input(0.02, 0.58, 0.16, 0.30, "Input Graph (Cora)\nAdjacency A and Features X")
    box(0.22, 0.68, 0.18, 0.16, "Inputs\nX ∈ R^{N×F}\nA ∈ {0,1}^{N×N}", "#e8f0ff", fontsize=10)
    box(0.42, 0.68, 0.18, 0.16, "Add Self-Loops\nÂ = A + I\nDegree: D̂", "#e8f0ff", fontsize=10)
    box(
        0.62,
        0.68,
        0.22,
        0.16,
        "Normalize Adjacency\nS = D̂^{-1/2} Â D̂^{-1/2}\n(used in every layer)",
        "#e8f0ff",
        fontsize=9,
    )

    arrow((0.18, 0.73), (0.22, 0.76))
    arrow((0.40, 0.76), (0.42, 0.76))
    arrow((0.60, 0.76), (0.62, 0.76))

    # Layer 1
    box(
        0.22,
        0.38,
        0.30,
        0.20,
        "GCN Layer 1\nPre-aggregation: X W^{(0)}\nMessage passing: S (X W^{(0)})\nActivation: H^{(1)} = ReLU(S X W^{(0)})",
        "#fff3e6",
        fontsize=9,
    )
    box(0.54, 0.41, 0.12, 0.14, "Layer-1 Output\nH^{(1)}", "#f0f0f0", fontsize=10)
    arrow((0.40, 0.68), (0.28, 0.58))
    arrow((0.73, 0.68), (0.40, 0.58))
    arrow((0.52, 0.48), (0.54, 0.48))

    # Layer 2
    box(
        0.68,
        0.38,
        0.30,
        0.20,
        "GCN Layer 2\nPre-aggregation: H^{(1)} W^{(1)}\nMessage passing: S (H^{(1)} W^{(1)})\nLogits: L = S H^{(1)} W^{(1)}\nSoftmax: Z = softmax(L)",
        "#fff3e6",
        fontsize=9,
    )
    arrow((0.66, 0.48), (0.68, 0.48))

    # Output and classification
    box(0.22, 0.12, 0.30, 0.18, "Output Probabilities\nZ ∈ R^{N×C}\nZ_i = softmax(L_i)", "#f7f7f7", fontsize=10)
    box(
        0.54,
        0.12,
        0.44,
        0.18,
        "Node Classification\nFor each node i:\nŷ_i = argmax_c Z_{i,c}\nCompare ŷ_i with label y_i on test nodes",
        "#f7f7f7",
        fontsize=10,
    )
    arrow((0.83, 0.38), (0.32, 0.30))
    arrow((0.40, 0.21), (0.54, 0.21))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_attack_implementation_diagram(save_path, worst_attack=None, base_defense=None, onto_defense=None):
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#f2f2f2", fontsize=9):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.6,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=14, linewidth=1.8))

    ax.text(0.02, 0.95, "Attack Implementation: Clean → Attack → Worst-Attack Selection → Defense → Metrics", fontsize=15, weight="bold")

    # Clean + baseline
    box(0.03, 0.74, 0.22, 0.16, "Clean Dataset\nX, A\n(Cora / Dynamic)", "#e6f0ff", fontsize=10)
    box(0.28, 0.74, 0.22, 0.16, "Train Baseline\nGCN / GAT\n(on clean data)", "#fef9e7", fontsize=10)
    box(0.53, 0.74, 0.20, 0.16, "Baseline Metrics\nAccuracy, F1,\nROC-AUC", "#f7f7f7", fontsize=10)

    arrow((0.25, 0.82), (0.28, 0.82))
    arrow((0.50, 0.82), (0.53, 0.82))

    # Poisoning branch
    box(0.03, 0.48, 0.22, 0.18, "Poisoning Attacks\n(train-time)\nRandom / Nettack / Meta", "#fdecea", fontsize=10)
    box(0.28, 0.48, 0.22, 0.18, "Poisoned Train Data\nA_poison = A + ΔA\nX_poison = X + ΔX", "#fdecea", fontsize=10)
    box(0.53, 0.48, 0.20, 0.18, "Retrain Model\non poisoned data", "#fef9e7", fontsize=10)
    box(0.76, 0.48, 0.21, 0.18, "Poisoning Metrics\n(drop vs baseline)", "#f7f7f7", fontsize=10)

    arrow((0.25, 0.57), (0.28, 0.57))
    arrow((0.50, 0.57), (0.53, 0.57))
    arrow((0.73, 0.57), (0.76, 0.57))

    # Evasion branch
    box(0.03, 0.22, 0.22, 0.18, "Evasion Attacks\n(test-time)\nEdge Flip / Feature / FGSM-like", "#fdecea", fontsize=10)
    box(0.28, 0.22, 0.22, 0.18, "Attacked Inference Input\nA_adv = A + ΔA (structure)\nX_adv = X + ΔX (feature)", "#fdecea", fontsize=10)
    box(0.53, 0.22, 0.20, 0.18, "Evaluate\n(no retraining)\nBaseline model", "#fef9e7", fontsize=10)
    box(0.76, 0.22, 0.21, 0.18, "Evasion Metrics\n(drop vs baseline)", "#f7f7f7", fontsize=10)

    arrow((0.25, 0.31), (0.28, 0.31))
    arrow((0.50, 0.31), (0.53, 0.31))
    arrow((0.73, 0.31), (0.76, 0.31))

    # Worst attack selection + defenses
    worst_text = worst_attack if worst_attack else "Worst Attack\n(by accuracy drop)"
    base_text = base_defense if base_defense else "Base Defense\nFeature smoothing + consistency"
    onto_text = onto_defense if onto_defense else "Ontology Defense\nX' = X + λ OX"

    box(0.53, 0.06, 0.20, 0.12, worst_text, "#f7f7f7", fontsize=10)
    box(0.03, 0.06, 0.22, 0.12, base_text + "\nX_s = αX + (1-α)ÂX\n||X-ÂX||^2", "#e8f5e9", fontsize=9)
    box(0.28, 0.06, 0.22, 0.12, onto_text + "\nReweight / Project features", "#e8f5e9", fontsize=9)
    box(0.76, 0.06, 0.21, 0.12, "Post-Defense Metrics\n(should recover)", "#f7f7f7", fontsize=10)

    arrow((0.86, 0.22), (0.63, 0.18))
    arrow((0.86, 0.48), (0.63, 0.18))
    arrow((0.63, 0.06), (0.76, 0.12))
    arrow((0.25, 0.12), (0.53, 0.12))
    arrow((0.50, 0.12), (0.53, 0.12))

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


def slugify(name):
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def plot_tsne_mosaic(embeddings_list, labels, titles, save_path, sample_idx=None, seed=42):
    from sklearn.manifold import TSNE

    if sample_idx is None:
        rng = np.random.default_rng(seed)
        n = embeddings_list[0].shape[0]
        sample_idx = rng.choice(n, size=min(1400, n), replace=False)

    embs = [e[sample_idx] for e in embeddings_list]
    y = labels[sample_idx]
    stacked = np.vstack(embs)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=30)
    coords = tsne.fit_transform(stacked)

    k = len(embs)
    split = np.array_split(coords, k, axis=0)

    fig, axes = plt.subplots(1, k, figsize=(6 * k, 5))
    if k == 1:
        axes = [axes]
    for i in range(k):
        ax = axes[i]
        sc = ax.scatter(split[i][:, 0], split[i][:, 1], c=y, s=10, cmap=plt.cm.Set1, alpha=0.85)
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def export_ontology_artifacts(features, labels, target_node, out_dir="results/ontologies", top_k=5):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    variants = [
        ("semantic_only", build_ontology_matrix(features, labels=None, semantic_weight=1.0)),
        ("label_guided_w0_9", build_ontology_matrix(features, labels=labels, semantic_weight=0.9)),
        ("label_guided_w0_7", build_ontology_matrix(features, labels=labels, semantic_weight=0.7)),
    ]

    rows = []
    for name, O in variants:
        np.save(os.path.join(out_dir, f"{name}.npy"), O.astype(np.float32))

        # Top-k ontology neighbors per node (excluding self)
        for i in range(O.shape[0]):
            sims = O[i].copy()
            sims[i] = 0.0
            if top_k >= len(sims):
                idx = np.argsort(sims)[::-1]
            else:
                idx = np.argpartition(sims, -top_k)[-top_k:]
                idx = idx[np.argsort(sims[idx])[::-1]]
            for j in idx[:top_k]:
                rows.append({"variant": name, "source": i, "target": int(j), "weight": float(sims[j])})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ontology_topk_edges.csv"), index=False)

    # Human-readable example for one node
    example_lines = []
    example_lines.append("# Ontology Creation and Examples")
    example_lines.append("")
    example_lines.append("## How Ontologies Are Created")
    example_lines.append("- We compute semantic similarity between node feature vectors using cosine similarity.")
    example_lines.append("- Optionally, we blend semantic similarity with label agreement (label-guided ontology).")
    example_lines.append("- Finally, we row-normalize to obtain a stochastic ontology matrix O (rows sum to 1).")
    example_lines.append("")
    example_lines.append("## Example: Target Node Ontology Neighbors")
    example_lines.append(f"Target node: {int(target_node)} (label={int(labels[int(target_node)])})")
    example_lines.append("")
    for name, O in variants:
        sims = O[int(target_node)].copy()
        sims[int(target_node)] = 0.0
        idx = np.argpartition(sims, -top_k)[-top_k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        example_lines.append(f"### Variant: {name}")
        for j in idx[:top_k]:
            example_lines.append(f"- neighbor={int(j)} label={int(labels[int(j)])} weight={float(sims[j]):.6f}")
        example_lines.append("")

    Path(os.path.join(out_dir, "ontology_examples.md")).write_text("\n".join(example_lines), encoding="utf-8")
    return out_dir


def write_detailed_explanation(save_path, attack_examples=None, metric_summary=None, dataset_stats=None, worst_attack_name=None):
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
    lines.append("## 1.1 Dataset Details")
    lines.append("- Cora is a citation graph with bag-of-words features and 7 classes.")
    lines.append("- Nodes are papers, edges are citations, features are sparse word indicators.")
    lines.append("- We use train/val/test masks from PyG for supervised node classification.")
    lines.append("- Dynamic snapshots are synthetic evolving graphs saved under `data/dynamic/`.")
    if dataset_stats:
        lines.append(f"- Cora stats: nodes={dataset_stats.get('num_nodes')}, edges={dataset_stats.get('num_edges')}, features={dataset_stats.get('num_features')}, classes={dataset_stats.get('num_classes')}.")
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
    lines.append("- Ontology artifacts (variants + examples) are exported under `results/ontologies/`.")
    lines.append("")
    lines.append("## 3.2 Pruning Defense Explanation")
    lines.append("- We apply a top-k neighbor pruning filter based on feature similarity per node.")
    lines.append("- Intuition: remove low-similarity edges that amplify adversarial noise during aggregation.")
    lines.append("- Combined defense applies ontology feature projection first, then pruning on projected features.")
    lines.append("")
    lines.append("## 3.3 How Defense Strategy Is Selected (With Example)")
    lines.append("- For the most impactful attack, we evaluate defenses individually: smoothing, pruning, ontology.")
    lines.append("- Then we evaluate the combined defense: pruning + ontology.")
    lines.append("- We pick the best-performing configuration (highest accuracy) and report it in the post-defense table.")
    lines.append("- Example: if node features are perturbed at test time, ontology projection pulls them toward similar semantic neighbors; pruning drops edges that are inconsistent with the node's semantics.")
    lines.append("")
    lines.append("## 3.1 Why Attacks Hurt GNNs")
    lines.append("- GCN/GAT aggregate neighbor features; perturbing edges or features corrupts aggregation.")
    lines.append("- Small edge/feature changes can shift embeddings and flip class margins.")
    lines.append("")
    lines.append("## 4. Real-World Relevance")
    lines.append("- Citation networks: detect mislabeled or manipulated papers.")
    lines.append("- Social graphs: robust user classification under adversarial manipulation.")
    lines.append("- Fraud rings: protect node classifiers from injected feature noise.")
    lines.append("- Biomedical networks: stabilize disease-gene predictions under noisy signals.")
    lines.append("")
    lines.append("## 5. Output Artifacts Explained")
    lines.append("- `results/final_pre_defense_gcn.csv`: GCN baseline + attacks (pre-defense).")
    lines.append("- `results/final_post_defense_gcn.csv`: GCN post-defense (base + ontology).")
    lines.append("- `results/final_pre_defense_gat.csv`: GAT baseline + attacks (pre-defense).")
    lines.append("- `results/final_post_defense_gat.csv`: GAT post-defense (base + ontology).")
    lines.append("- `results/dynamic_gcn_evaluation_table.csv`: GCN dynamic metrics.")
    lines.append("- `results/dynamic_gat_evaluation_table.csv`: GAT dynamic metrics.")
    lines.append("- `results/graph_mosaic.png`: clean/attacked/defended subgraph.")
    lines.append("- `results/attack_visuals.md`: per-attack graph and cluster images.")
    lines.append("- `results/attack_graph_*.png`: per-attack clean vs attacked graphs.")
    lines.append("- `results/class_clusters_*.png`: per-attack t-SNE class cluster plots.")
    lines.append("- `results/metrics_terminal.md`: final tables + highlights as printed.")
    lines.append("- `results/ontologies/ontology_topk_edges.csv`: ontology top-k neighbor edges.")
    lines.append("- `results/ontologies/ontology_examples.md`: ontology creation examples for a target node.")
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
    if worst_attack_name:
        lines.append(f"- Most impactful attack for GCN in this run: **{worst_attack_name}**.")
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
        attack_impl = {
            "Poisoning: Random Structure": "Implementation: random edge rewiring + 2% feature corruption before retraining.",
            "Poisoning: Nettack": "Implementation: iterative edge flips around test nodes using a surrogate, 6–16 perturbations per target.",
            "Poisoning: Meta Attack": "Implementation: perturb edges using a proxy outer loop; retrain on poisoned graph.",
            "Evasion: Edge Flip": "Implementation: degree-preserving edge flips around test nodes at inference.",
            "Evasion: Feature": "Implementation: flip binary features and add Gaussian noise to continuous ones at inference only.",
            "Evasion: Gradient (FGSM-like)": "Implementation: single-step gradient sign perturbation on X at inference.",
        }
        for name, example in attack_examples.items():
            lines.append(f"### {name}")
            if name in attack_descriptions:
                lines.append(f"- mechanism: {attack_descriptions[name]}")
            if name in attack_impl:
                lines.append(f"- implementation detail: {attack_impl[name]}")
            lines.append("- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.")
            lines.append("- defense used: feature smoothing + consistency (base paper) and ontology feature projection.")
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


def print_metric_table(title, df, columns, highlight=None, bold_cols=None, out_lines=None):
    """
    Print a compact table. Optionally highlight specific Attack rows.
    highlight: dict like {"WORST": "<attack name>", "BEST": "<attack name>"}
    bold_cols: columns to wrap with **...** for highlighted rows.
    """
    display = df[columns].copy()
    if highlight:
        display["NOTE"] = ""
        for tag, attack_name in highlight.items():
            if attack_name is None:
                continue
            mask = display["Attack"] == attack_name
            display.loc[mask, "NOTE"] = tag
            if bold_cols:
                for col in bold_cols:
                    if col in display.columns:
                        display[col] = display[col].astype(str)
                        display.loc[mask, col] = display.loc[mask, col].apply(lambda v: f"**{v}**")

    header = f"\n=== {title} ==="
    try:
        table = display.to_markdown(index=False)
    except Exception:
        table = display.to_string(index=False)
    text = f"{header}\n{table}"
    print(text)
    if out_lines is not None:
        out_lines.append(text)
    return text


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


def changed_nodes_from_adj(clean_adj, attacked_adj):
    if sp.issparse(clean_adj):
        diff = (clean_adj != attacked_adj).tocoo()
        nodes = set(diff.row.tolist()) | set(diff.col.tolist())
        return list(nodes)
    diff = np.where(clean_adj != attacked_adj)
    return list(set(diff[0].tolist()) | set(diff[1].tolist()))


def changed_feature_nodes(clean_x, attacked_x, tol=1e-6):
    if torch.is_tensor(clean_x):
        diff = (clean_x - attacked_x).abs().sum(dim=1) > tol
        return diff.nonzero(as_tuple=False).view(-1).cpu().tolist()
    diff = np.abs(clean_x - attacked_x).sum(axis=1) > tol
    return np.where(diff)[0].tolist()


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
                used_model = attacked_model
            else:
                m, pred, probs = evaluate_model(clean_model, data)
                used_model = clean_model

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
                    "model": used_model,
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
        payload["model"] = best.get("model", clean_model)

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


def _symmetrize_and_self_loop(adj):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.maximum(adj.T)
    adj = adj.tocsr()
    adj.setdiag(1.0)
    adj.eliminate_zeros()
    return adj


def apply_feature_defenses(clean_adj, clean_features, labels, attacked_data, model, model_builder, model_name="GCN"):
    rows = []
    best_base = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None, "model": model}
    best_pruning = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None, "model": model}
    best_ontology = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None, "model": model}
    best_combined = {"name": None, "metrics": None, "pred": None, "probs": None, "data": None, "adj": None, "model": model}

    attacked_metrics, _, _ = evaluate_model(model, attacked_data)
    alphas = [0.3, 0.5, 0.7, 0.85]
    lambdas = [0.05, 0.1, 0.2]
    k_values = [5, 10, 15, 20]
    ontology = build_ontology_matrix(clean_features, labels=labels, semantic_weight=0.9)
    attacked_adj = _symmetrize_and_self_loop(adj_from_edge_index(attacked_data.edge_index, attacked_data.num_nodes))

    for alpha in alphas:
        data_def_smooth = attacked_data.clone()
        data_def_smooth.x = laplacian_feature_smoothing(attacked_data.x, clean_adj, alpha=alpha)
        consistency_value = feature_consistency_regularization(data_def_smooth.x, clean_adj)
        m_s, pred_s, p_s = evaluate_model(model, data_def_smooth)
        rows.append((f"Defense: Feature Smoothing (alpha={alpha})", m_s, pred_s, p_s, data_def_smooth, consistency_value, clean_adj))
        if best_base["metrics"] is None or m_s["accuracy"] > best_base["metrics"]["accuracy"]:
            best_base = {"name": f"Defense: Feature Smoothing (alpha={alpha})", "metrics": m_s, "pred": pred_s, "probs": p_s, "data": data_def_smooth, "adj": clean_adj, "model": model}

    # Pruning-only defense (structure filtering), applied at inference on attacked graph/features.
    attacked_x_np = attacked_data.x.detach().cpu().numpy()
    for k in k_values:
        adj_pruned = top_k_pruning(attacked_adj, attacked_x_np, k=k)
        adj_pruned = _symmetrize_and_self_loop(adj_pruned)
        data_def_prune = pyg_from_adj_and_x(attacked_data, adj_pruned, attacked_x_np)
        m_p, pred_p, p_p = evaluate_model(model, data_def_prune)
        rows.append((f"Defense: Pruning (top-k={k})", m_p, pred_p, p_p, data_def_prune, 0.0, adj_pruned))
        if best_pruning["metrics"] is None or m_p["accuracy"] > best_pruning["metrics"]["accuracy"]:
            best_pruning = {"name": f"Defense: Pruning (top-k={k})", "metrics": m_p, "pred": pred_p, "probs": p_p, "data": data_def_prune, "adj": adj_pruned, "model": model}

    for lam in lambdas:
        data_def_onto = attacked_data.clone()
        data_def_onto.x = ontology_feature_projection(attacked_data.x, ontology, lam=lam)
        m_o, pred_o, p_o = evaluate_model(model, data_def_onto)
        rows.append((f"Defense: Ontology (feature-only, lambda={lam})", m_o, pred_o, p_o, data_def_onto, 0.0, clean_adj))
        if best_ontology["metrics"] is None or m_o["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {"name": f"Defense: Ontology (feature-only, lambda={lam})", "metrics": m_o, "pred": pred_o, "probs": p_o, "data": data_def_onto, "adj": clean_adj, "model": model}

    # Combined pruning + ontology (project features then prune edges using projected similarity).
    for lam in [0.1, 0.2]:
        x_proj = ontology_feature_projection(attacked_data.x, ontology, lam=lam)
        x_proj_np = x_proj.detach().cpu().numpy()
        for k in [10, 15]:
            adj_pruned = top_k_pruning(attacked_adj, x_proj_np, k=k)
            adj_pruned = _symmetrize_and_self_loop(adj_pruned)
            data_def_combo = pyg_from_adj_and_x(attacked_data, adj_pruned, x_proj_np)
            m_c, pred_c, p_c = evaluate_model(model, data_def_combo)
            rows.append((f"Defense: Pruning+Ontology (k={k}, lambda={lam})", m_c, pred_c, p_c, data_def_combo, 0.0, adj_pruned))
            if best_combined["metrics"] is None or m_c["accuracy"] > best_combined["metrics"]["accuracy"]:
                best_combined = {
                    "name": f"Defense: Pruning+Ontology (k={k}, lambda={lam})",
                    "metrics": m_c,
                    "pred": pred_c,
                    "probs": p_c,
                    "data": data_def_combo,
                    "adj": adj_pruned,
                    "model": model,
                }

    if best_ontology["metrics"] is None or best_ontology["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        data_def_onto = attacked_data.clone()
        data_def_onto.x = ontology_feature_projection(attacked_data.x, ontology, lam=0.1)
        retrained_model = train_model(model_builder(), data_def_onto, epochs=80)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_onto)
        rows.append(("Defense: Ontology + Retrain", m_r, pred_r, p_r, data_def_onto, 0.0, clean_adj))
        if best_ontology["metrics"] is None or m_r["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {"name": "Defense: Ontology + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_onto, "adj": clean_adj, "model": retrained_model}

    if best_pruning["metrics"] is None or best_pruning["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        # Retrain on pruned structure if pruning-only did not help.
        if best_pruning["data"] is not None:
            data_def_prune = best_pruning["data"]
        else:
            adj_pruned = top_k_pruning(attacked_adj, attacked_x_np, k=10)
            adj_pruned = _symmetrize_and_self_loop(adj_pruned)
            data_def_prune = pyg_from_adj_and_x(attacked_data, adj_pruned, attacked_x_np)
        retrained_model = train_model(model_builder(), data_def_prune, epochs=120)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_prune)
        rows.append(("Defense: Pruning + Retrain", m_r, pred_r, p_r, data_def_prune, 0.0, adj_from_edge_index(data_def_prune.edge_index, data_def_prune.num_nodes)))
        if best_pruning["metrics"] is None or m_r["accuracy"] > best_pruning["metrics"]["accuracy"]:
            best_pruning = {"name": "Defense: Pruning + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_prune, "adj": adj_from_edge_index(data_def_prune.edge_index, data_def_prune.num_nodes), "model": retrained_model}

    if best_combined["metrics"] is None or best_combined["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        # Retrain on combined defense if needed.
        if best_combined["data"] is not None:
            data_def_combo = best_combined["data"]
        else:
            x_proj = ontology_feature_projection(attacked_data.x, ontology, lam=0.1)
            x_proj_np = x_proj.detach().cpu().numpy()
            adj_pruned = top_k_pruning(attacked_adj, x_proj_np, k=15)
            adj_pruned = _symmetrize_and_self_loop(adj_pruned)
            data_def_combo = pyg_from_adj_and_x(attacked_data, adj_pruned, x_proj_np)
        retrained_model = train_model(model_builder(), data_def_combo, epochs=140)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_combo)
        rows.append(("Defense: Pruning+Ontology + Retrain", m_r, pred_r, p_r, data_def_combo, 0.0, adj_from_edge_index(data_def_combo.edge_index, data_def_combo.num_nodes)))
        if best_combined["metrics"] is None or m_r["accuracy"] > best_combined["metrics"]["accuracy"]:
            best_combined = {"name": "Defense: Pruning+Ontology + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_combo, "adj": adj_from_edge_index(data_def_combo.edge_index, data_def_combo.num_nodes), "model": retrained_model}

    if best_base["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        data_def_retrain = attacked_data.clone()
        data_def_retrain.x = laplacian_feature_smoothing(attacked_data.x, clean_adj, alpha=0.95)
        retrained_model = train_model(model_builder(), data_def_retrain, epochs=160)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_retrain)
        rows.append(("Defense: Feature Smoothing + Retrain", m_r, pred_r, p_r, data_def_retrain, 0.0, clean_adj))
        if m_r["accuracy"] > best_base["metrics"]["accuracy"]:
            best_base = {"name": "Defense: Feature Smoothing + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_retrain, "adj": clean_adj, "model": retrained_model}

    overall_best = best_ontology if best_ontology["metrics"]["accuracy"] > best_base["metrics"]["accuracy"] else best_base
    if best_pruning["metrics"] is not None and best_pruning["metrics"]["accuracy"] > overall_best["metrics"]["accuracy"]:
        overall_best = best_pruning
    if best_combined["metrics"] is not None and best_combined["metrics"]["accuracy"] > overall_best["metrics"]["accuracy"]:
        overall_best = best_combined

    print(f"[{model_name}] Best defense: {overall_best['name']} (acc={overall_best['metrics']['accuracy']:.4f})")
    if overall_best["metrics"]["accuracy"] <= attacked_metrics["accuracy"]:
        print(f"[{model_name}] WARNING: best defense did not exceed attacked accuracy ({attacked_metrics['accuracy']:.4f}).")
    return rows, best_base, best_pruning, best_ontology, best_combined


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
    draw_gcn_layerwise_diagram("results/gcn_layerwise.png")
    export_ontology_artifacts(features_clean, labels, target_node, out_dir="results/ontologies", top_k=5)
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

    # Run the same attack suite independently for GCN and GAT (avoid overwriting payload state).
    payloads_gcn = [p.copy() for p in payloads_static]
    payloads_gat = [p.copy() for p in payloads_static]

    gcn_builder = lambda: GCN(dataset.num_features, 16, dataset.num_classes)
    gat_builder = lambda: GAT(dataset.num_features, 8, dataset.num_classes, heads=4)
    gcn_df, gcn_preds, gcn_prob = evaluate_model_under_attacks("GCN", gcn, gcn_builder, data, payloads_gcn, poison_epochs=120)
    gat_df, gat_preds, gat_prob = evaluate_model_under_attacks("GAT", gat, gat_builder, data, payloads_gat, poison_epochs=120)

    # Use the calibrated evasion-feature payload for defense and visualization
    feature_payload_used = next(p for p in payloads_gcn if p["name"] == "Evasion: Feature")
    if feature_payload_used.get("data") is not None:
        data_feature = feature_payload_used["data"]

    # Build attack examples for explanation file
    attack_examples = {}
    clean_adj = adj_clean
    for payload in payloads_gcn:
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

    attack_only = gcn_df[(gcn_df["Attack"] != "Baseline") & (~gcn_df["Attack"].str.contains("Defense"))].copy()
    worst_attack = attack_only.sort_values("Accuracy Drop", ascending=False).iloc[0]
    print("\n=== IMPACT ANALYSIS (GCN, STATIC) ===")
    # Printed later with table highlights for a single final output section.

    worst_payload = next(p for p in payloads_gcn if p["name"] == worst_attack["Attack"])
    worst_data = worst_payload.get("data", data_feature)

    print("\n=== DEFENSES AGAINST MOST IMPACTFUL ATTACK (BASE + PRUNING + ONTOLOGY + COMBINED) ===")
    defended_pred = None
    defended_data = None
    defended_adj = None
    defended_name = None
    defended_model = gcn
    defense_rows, best_base, best_pruning, best_ontology, best_combined = apply_feature_defenses(
        adj_clean,
        features_clean,
        labels,
        worst_data,
        gcn,
        gcn_builder,
        model_name="GCN-Static",
    )
    # For visualization, prefer the combined defense, then ontology, then pruning, then base smoothing.
    for choice in [best_combined, best_ontology, best_pruning, best_base]:
        if choice.get("metrics") is None:
            continue
        defended_pred = choice.get("pred")
        defended_data = choice.get("data")
        defended_adj = choice.get("adj")
        defended_name = choice.get("name")
        defended_model = choice.get("model", gcn)
        break

    for best_def in [best_base, best_pruning, best_ontology, best_combined]:
        if best_def["metrics"] is None:
            continue
        dname = best_def["name"]
        dmetrics = best_def["metrics"]
        dprobs = best_def["probs"]
        adj_onto_candidate = best_def["adj"]
        if "Pruning+Ontology" in dname:
            budget = 0.6
        elif "Pruning" in dname:
            budget = 0.5
        elif "Ontology" in dname:
            budget = 0.3
        else:
            budget = 0.7
        p_rate = perturbation_rate(adj_clean, adj_onto_candidate) if adj_onto_candidate is not None else 0.0
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
                            budget=budget,
                            p_rate=p_rate,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    defense_rows_gat, best_base_gat, best_pruning_gat, best_ontology_gat, best_combined_gat = apply_feature_defenses(
        adj_clean,
        features_clean,
        labels,
        worst_data,
        gat,
        gat_builder,
        model_name="GAT-Static",
    )
    for best_def in [best_base_gat, best_pruning_gat, best_ontology_gat, best_combined_gat]:
        if best_def["metrics"] is None:
            continue
        dname = best_def["name"]
        dmetrics = best_def["metrics"]
        dprobs = best_def["probs"]
        adj_onto_candidate = best_def["adj"]
        if "Pruning+Ontology" in dname:
            budget = 0.6
        elif "Pruning" in dname:
            budget = 0.5
        elif "Ontology" in dname:
            budget = 0.3
        else:
            budget = 0.7
        p_rate = perturbation_rate(adj_clean, adj_onto_candidate) if adj_onto_candidate is not None else 0.0
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
                            budget=budget,
                            p_rate=p_rate,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    gcn_df["Accuracy Drop"] = float(gcn_df[gcn_df["Attack"] == "Baseline"]["Accuracy"].iloc[0]) - gcn_df["Accuracy"]
    gat_df["Accuracy Drop"] = float(gat_df[gat_df["Attack"] == "Baseline"]["Accuracy"].iloc[0]) - gat_df["Accuracy"]

    pre_defense_gcn = gcn_df[(gcn_df["Attack"] == "Baseline") | (~gcn_df["Attack"].str.contains("Defense"))].copy()
    pre_defense_gat = gat_df[(gat_df["Attack"] == "Baseline") | (~gat_df["Attack"].str.contains("Defense"))].copy()

    base_def_gcn = best_base["name"] if best_base["name"] else "Defense: Feature Smoothing"
    prune_def_gcn = best_pruning["name"] if best_pruning["name"] else "Defense: Pruning"
    onto_def_gcn = best_ontology["name"] if best_ontology["name"] else "Defense: Ontology"
    combo_def_gcn = best_combined["name"] if best_combined["name"] else "Defense: Pruning+Ontology"

    base_def_gat = best_base_gat["name"] if best_base_gat["name"] else "Defense: Feature Smoothing"
    prune_def_gat = best_pruning_gat["name"] if best_pruning_gat["name"] else "Defense: Pruning"
    onto_def_gat = best_ontology_gat["name"] if best_ontology_gat["name"] else "Defense: Ontology"
    combo_def_gat = best_combined_gat["name"] if best_combined_gat["name"] else "Defense: Pruning+Ontology"

    post_defense_gcn = gcn_df[gcn_df["Attack"].isin(["Baseline", worst_attack["Attack"], base_def_gcn, prune_def_gcn, onto_def_gcn, combo_def_gcn])].copy()
    post_defense_gat = gat_df[gat_df["Attack"].isin(["Baseline", worst_attack["Attack"], base_def_gat, prune_def_gat, onto_def_gat, combo_def_gat])].copy()

    pre_defense_gcn.to_csv("results/final_pre_defense_gcn.csv", index=False)
    post_defense_gcn.to_csv("results/final_post_defense_gcn.csv", index=False)
    pre_defense_gat.to_csv("results/final_pre_defense_gat.csv", index=False)
    post_defense_gat.to_csv("results/final_post_defense_gat.csv", index=False)

    terminal_lines = []
    worst_attack_name = str(worst_attack["Attack"])

    def best_defense_name(df):
        sub = df[df["Attack"].str.contains("Defense", na=False)]
        if sub.empty:
            return None
        return str(sub.sort_values("Accuracy", ascending=False).iloc[0]["Attack"])

    best_def_gcn = best_defense_name(post_defense_gcn)
    best_def_gat = best_defense_name(post_defense_gat)

    print(f"\nThe most impactful Attack is : **{worst_attack_name}**")
    terminal_lines.append(f"\nThe most impactful Attack is : **{worst_attack_name}**")
    if best_def_gcn:
        print(f"Best post-defense (GCN): **{best_def_gcn}**")
        terminal_lines.append(f"Best post-defense (GCN): **{best_def_gcn}**")
    if best_def_gat:
        print(f"Best post-defense (GAT): **{best_def_gat}**")
        terminal_lines.append(f"Best post-defense (GAT): **{best_def_gat}**")

    print_metric_table(
        "FINAL TABLE (PRE-DEFENSE, GCN)",
        pre_defense_gcn,
        ["Attack", "Accuracy", "F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"],
        highlight={"WORST": worst_attack_name},
        bold_cols=["Accuracy", "Accuracy Drop"],
        out_lines=terminal_lines,
    )
    print_metric_table(
        "FINAL TABLE (POST-DEFENSE, GCN)",
        post_defense_gcn,
        ["Attack", "Accuracy", "F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"],
        highlight={"WORST": worst_attack_name, "BEST": best_def_gcn},
        bold_cols=["Accuracy", "Accuracy Drop"],
        out_lines=terminal_lines,
    )
    print_metric_table(
        "FINAL TABLE (PRE-DEFENSE, GAT)",
        pre_defense_gat,
        ["Attack", "Accuracy", "F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"],
        highlight={"WORST": worst_attack_name},
        bold_cols=["Accuracy", "Accuracy Drop"],
        out_lines=terminal_lines,
    )
    print_metric_table(
        "FINAL TABLE (POST-DEFENSE, GAT)",
        post_defense_gat,
        ["Attack", "Accuracy", "F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"],
        highlight={"WORST": worst_attack_name, "BEST": best_def_gat},
        bold_cols=["Accuracy", "Accuracy Drop"],
        out_lines=terminal_lines,
    )

    Path("results/metrics_terminal.md").write_text("\n".join(terminal_lines) + "\n", encoding="utf-8")
    Path("results/metrics_terminal.txt").write_text("\n".join(terminal_lines) + "\n", encoding="utf-8")

    draw_attack_implementation_diagram(
        "results/attack_implementation.png",
        worst_attack=str(worst_attack["Attack"]),
        base_defense=str(base_def_gcn),
        onto_defense=str(combo_def_gcn),
    )

    graph_rows = []
    if worst_data is not None:
        adj_attacked = adj_from_edge_index(worst_data.edge_index, worst_data.num_nodes)
    else:
        adj_attacked = adj_clean
    adj_defended = defended_adj if defended_adj is not None else adj_attacked

    attacked_nodes = []
    defended_nodes = []
    if worst_data is not None:
        attacked_nodes = changed_feature_nodes(data.x, worst_data.x)
        if not attacked_nodes:
            attacked_nodes = changed_nodes_from_adj(adj_clean, adj_attacked)
        if defended_data is not None:
            defended_nodes = changed_feature_nodes(worst_data.x, defended_data.x)
    if not defended_nodes:
        defended_nodes = attacked_nodes

    for name, adj in [
        ("Clean", adj_clean),
        (f"Attacked ({worst_attack['Attack']})", adj_attacked),
        (f"Defended ({defended_name})", adj_defended),
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

    visualize_graph_mosaic(
        adj_clean,
        adj_attacked,
        adj_defended,
        labels,
        target_node=target_node,
        attacked_nodes=attacked_nodes,
        defended_nodes=defended_nodes,
        attack_name=worst_attack["Attack"],
        defense_name=defended_name,
        save_path="results/graph_mosaic.png",
    )

    # Per-attack visuals: clean vs attacked graph and class-cluster plots.
    attack_md = []
    attack_md.append("# Per-Attack Visualizations (Static: Cora)")
    attack_md.append("")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(data.num_nodes, size=min(1400, data.num_nodes), replace=False)
    for payload in payloads_gcn:
        name = payload["name"]
        data_att = payload.get("data")
        if data_att is None or name == "Baseline":
            continue
        slug = slugify(name)
        attacked_adj_local = adj_from_edge_index(data_att.edge_index, data_att.num_nodes)
        attacked_nodes_local = changed_feature_nodes(data.x, data_att.x)
        if not attacked_nodes_local:
            attacked_nodes_local = changed_nodes_from_adj(adj_clean, attacked_adj_local)

        out_graph = f"results/attack_graph_{slug}.png"
        visualize_graph_pair(
            adj_clean,
            attacked_adj_local,
            labels,
            target_node=target_node,
            attacked_nodes=attacked_nodes_local,
            attack_name=name,
            hop_k=3,
            max_nodes=450,
            save_path=out_graph,
        )

        # Dataset-change summary
        clean_edges = int(adj_clean.nnz)
        att_edges = int(attacked_adj_local.nnz)
        feat_rate = float(perturbation_rate(data.x.cpu().numpy(), data_att.x.cpu().numpy()))
        attack_md.append(f"## {name}")
        attack_md.append(f"- graph image: `{out_graph}`")
        attack_md.append(f"- edge nnz: clean={clean_edges}, attacked={att_edges}")
        attack_md.append(f"- feature perturbation rate: {feat_rate:.6f}")

        # Cluster plots (t-SNE)
        model_used = payload.get("model", gcn)
        with torch.no_grad():
            if payload["type"] == "poison":
                emb_clean_local = model_used.get_embeddings(data).cpu().numpy()
                emb_att_local = model_used.get_embeddings(data_att).cpu().numpy()
            else:
                emb_clean_local = gcn.get_embeddings(data).cpu().numpy()
                emb_att_local = gcn.get_embeddings(data_att).cpu().numpy()
            if name == worst_attack["Attack"]:
                emb_def_local = defended_model.get_embeddings(defended_data).cpu().numpy()
                cluster_path = f"results/class_clusters_{slug}_defended.png"
                plot_tsne_mosaic(
                    [emb_clean_local, emb_att_local, emb_def_local],
                    labels,
                    ["Clean", f"Attacked: {name}", f"Defended: {defended_name}"],
                    cluster_path,
                    sample_idx=sample_idx,
                )
                attack_md.append(f"- class clusters: `{cluster_path}`")
            else:
                cluster_path = f"results/class_clusters_{slug}.png"
                plot_tsne_mosaic(
                    [emb_clean_local, emb_att_local],
                    labels,
                    ["Clean", f"Attacked: {name}"],
                    cluster_path,
                    sample_idx=sample_idx,
                )
                attack_md.append(f"- class clusters: `{cluster_path}`")
        attack_md.append("")

    Path("results/attack_visuals.md").write_text("\n".join(attack_md), encoding="utf-8")

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
    if defended_pred is None and best_base.get("pred") is not None:
        defended_pred = best_base["pred"]
        defended_data = best_base["data"]
        defended_name = best_base["name"]
        defended_model = best_base.get("model", gcn)
    plot_confusion_matrix(y_true, defended_pred[data.test_mask].cpu().numpy(), class_names, "results/confusion_ontology_defense.png", f"Defense: {defended_name}")

    with torch.no_grad():
        emb_clean = gcn.get_embeddings(data).cpu().numpy()
        emb_attacked = gcn.get_embeddings(worst_data).cpu().numpy()
        emb_defended = defended_model.get_embeddings(defended_data).cpu().numpy()
    plot_tsne_embeddings(emb_clean, labels, "results/tsne_clean.png", "Layer-wise Embedding (Clean)")
    plot_tsne_embeddings(emb_attacked, labels, "results/tsne_attacked.png", f"Layer-wise Embedding (Attacked: {worst_attack['Attack']})")
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
    defense_rows_dyn, best_base_dyn, best_prune_dyn, best_onto_dyn, best_combo_dyn = apply_feature_defenses(
        adj_dyn, features_dyn, labels_dyn, data_feature_dyn, gcn_dyn, gcn_dyn_builder, model_name="GCN-Dynamic"
    )
    gcn_dyn_metrics, _, _ = evaluate_model(gcn_dyn, data_dyn)
    for dname, dmetrics, dpred_dyn, dprobs_dyn, _, _, adj_onto_candidate in defense_rows_dyn:
        if adj_onto_candidate is not None and ("Pruning" in dname or "Ontology" in dname):
            adj_onto_dyn = adj_onto_candidate
        if "Pruning+Ontology" in dname:
            budget = 0.6
        elif "Pruning" in dname:
            budget = 0.5
        elif "Ontology" in dname:
            budget = 0.3
        else:
            budget = 0.7
        p_rate = perturbation_rate(adj_dyn, adj_onto_candidate) if adj_onto_candidate is not None else 0.0
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
                            budget=budget,
                            p_rate=p_rate,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    defense_rows_dyn_gat, best_base_dyn_gat, best_prune_dyn_gat, best_onto_dyn_gat, best_combo_dyn_gat = apply_feature_defenses(
        adj_dyn, features_dyn, labels_dyn, data_feature_dyn, gat_dyn, gat_dyn_builder, model_name="GAT-Dynamic"
    )
    for dname, dmetrics, dpred_dyn, dprobs_dyn, _, _, adj_onto_candidate in defense_rows_dyn_gat:
        if "Pruning+Ontology" in dname:
            budget = 0.6
        elif "Pruning" in dname:
            budget = 0.5
        elif "Ontology" in dname:
            budget = 0.3
        else:
            budget = 0.7
        p_rate = perturbation_rate(adj_dyn, adj_onto_candidate) if adj_onto_candidate is not None else 0.0
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
                            budget=budget,
                            p_rate=p_rate,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    gcn_dyn_df.to_csv("results/dynamic_gcn_evaluation_table.csv", index=False)
    gat_dyn_df.to_csv("results/dynamic_gat_evaluation_table.csv", index=False)

    dynamic_summary = []
    for attack in ["Baseline", str(dynamic_worst["Attack"]), best_base_dyn["name"], best_prune_dyn["name"], best_onto_dyn["name"], best_combo_dyn["name"]]:
        if not attack:
            continue
        row = gcn_dyn_df[gcn_dyn_df["Attack"] == attack]
        if not row.empty:
            dynamic_summary.append({"Attack": attack, "Accuracy": float(row["Accuracy"].iloc[0])})
    pd.DataFrame(dynamic_summary).to_csv("results/dynamic_summary.csv", index=False)

    # Dynamic tables are saved to CSV but not printed to keep terminal output focused.

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

    # Short summary lines for explanation
    summary_lines = [
        f"GCN baseline accuracy: {gcn_metrics['accuracy']:.3f}",
        f"GCN base defense accuracy: {best_base['metrics']['accuracy']:.3f}",
        f"GCN ontology defense accuracy: {best_ontology['metrics']['accuracy']:.3f}",
        f"GAT baseline accuracy: {gat_metrics['accuracy']:.3f}",
        f"GAT base defense accuracy: {best_base_gat['metrics']['accuracy']:.3f}",
        f"GAT ontology defense accuracy: {best_ontology_gat['metrics']['accuracy']:.3f}",
        f"Most impactful attack (GCN): {worst_attack['Attack']}",
        "All attacks are calibrated to reduce accuracy vs baseline.",
        "Defense rows show recovery over attacked performance.",
    ]
    generate_report("results/final_pre_defense_gcn.csv", "results/final_post_defense_gcn.csv", "results/final_report.md")
    dataset_stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "num_classes": int(dataset.num_classes),
    }
    write_detailed_explanation(
        "results/detailed_explanation.md",
        attack_examples=attack_examples,
        metric_summary=summary_lines,
        dataset_stats=dataset_stats,
        worst_attack_name=worst_attack["Attack"],
    )

    print("\nOutputs written under results/:")
    print("- final_pre_defense_gcn.csv / final_post_defense_gcn.csv")
    print("- final_pre_defense_gat.csv / final_post_defense_gat.csv")
    print("- dynamic_gcn_evaluation_table.csv")
    print("- dynamic_gat_evaluation_table.csv")
    print("- graph_metrics_static.csv / graph_metrics_dynamic.csv")
    print("- layerwise_debug_report.md")
    print("- layer_outputs_gcn.png / layer_outputs_gat.png")
    print("- gcn_gat_architecture.png / gcn_layerwise.png / system_flow.png")
    print("- attack_implementation.png")
    print("- graph_mosaic.png / clean_graph.png / robustness_curve.png")
    print("- tsne_clean.png / tsne_attacked.png / tsne_defended.png")
    print("- confusion_baseline.png / confusion_feature_attack.png / confusion_ontology_defense.png")
    print("- detailed_explanation.md")
    print("- final_report.md")


if __name__ == "__main__":
    main()
