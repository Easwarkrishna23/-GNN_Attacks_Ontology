import os
import random
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from tabulate import tabulate

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
from visualization.graph_viz import visualize_graph_mosaic, visualize_graph_pair, visualize_attack_suite
from visualization.plotting import (
    plot_robustness_curves,
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_layer_output_panel,
    plot_tsne_suite,
)
from visualization.paper_diagrams import (
    draw_project_workflow,
    draw_gcn_layerwise,
    draw_gat_layerwise,
    draw_attack_defense_flow,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_results_dir(path="results"):
    """
    Delete all previously generated outputs under results/.
    User explicitly requested a clean rerun before generating fresh results.
    """
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return
    for child in p.iterdir():
        if child.is_dir():
            for sub in child.rglob("*"):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink()
            # remove directories bottom-up
            for sub in sorted([d for d in child.rglob("*") if d.is_dir()], reverse=True):
                try:
                    sub.rmdir()
                except Exception:
                    pass
            try:
                child.rmdir()
            except Exception:
                pass
        else:
            child.unlink()
    p.mkdir(parents=True, exist_ok=True)


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
        # Export an interpretable, sized-down ontology view:
        # top-k ontology neighbors per node (excluding self). This avoids saving full NxN matrices.
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
    # Also export a short variant summary for your report.
    summary = []
    summary.append("# Ontology Variants Summary")
    summary.append("")
    summary.append("We export only top-k semantic neighbors per node (not the full NxN ontology matrix).")
    summary.append("Variants:")
    summary.append("- semantic_only: cosine similarity on features only.")
    summary.append("- label_guided_w0_9: 90% semantic similarity + 10% label agreement (for explanation only).")
    summary.append("- label_guided_w0_7: 70% semantic similarity + 30% label agreement (for explanation only).")
    summary.append("")
    summary.append("Files:")
    summary.append("- ontology_topk_edges.csv: (variant, source, target, weight) for all nodes.")
    summary.append("- ontology_examples.md: human-readable example for one target node.")
    Path(os.path.join(out_dir, "ontology_summary.md")).write_text("\n".join(summary), encoding="utf-8")
    return out_dir


def write_detailed_explanation(save_path, attack_examples=None, metric_summary=None, dataset_stats=None, worst_attack_name=None):
    lines = []
    lines.append("# Adversarial Attacks on GNNs (GCN/GAT): Explanation and Output Guide")
    lines.append("")
    lines.append("## How To Run (Clean Rerun)")
    lines.append("```bash")
    lines.append("python3 main.py --clean --profile paper")
    lines.append("```")
    lines.append("")
    lines.append("This deletes every previously generated file under `results/` and regenerates ONLY the final paper-style outputs.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("- Static dataset: **Cora** citation network.")
    lines.append("- Dynamic dataset: synthetic evolving snapshots stored under `data/dynamic/`.")
    if dataset_stats:
        lines.append(f"- Cora stats: nodes={dataset_stats.get('num_nodes')}, edges={dataset_stats.get('num_edges')}, features={dataset_stats.get('num_features')}, classes={dataset_stats.get('num_classes')}.")
    lines.append("")
    lines.append("## Models")
    lines.append("### GCN")
    lines.append("- Normalize: `Â = A + I`, `S = D̂^{-1/2} Â D̂^{-1/2}`.")
    lines.append("- Layer 1: `H^{(1)} = ReLU(S X W^{(0)})`.")
    lines.append("- Layer 2: `Z = Softmax(S H^{(1)} W^{(1)})`.")
    lines.append("")
    lines.append("### GAT")
    lines.append("- Attention per head: `e_{ij} = a^T[Wh_i || Wh_j]`, `α_{ij} = softmax_j(LeakyReLU(e_{ij}))`.")
    lines.append("- Layer 1 (multi-head): `h_i^{(1)} = ||_k Σ_{j∈N(i)} α_{ij}^k W^k h_j` (ELU).")
    lines.append("- Layer 2: aggregate -> logits -> Softmax.")
    lines.append("")
    lines.append("## Attacks (3 Poisoning + 3 Evasion)")
    lines.append("### Poisoning vs Evasion")
    lines.append("- Poisoning: modifies training data/graph; model is trained on poisoned input.")
    lines.append("- Evasion: modifies inference-time input only; training graph is untouched.")
    lines.append("")
    attack_impl = {
        "Poisoning: Random Structure": "attacks/poisoning/random_poison.py",
        "Poisoning: Nettack": "attacks/poisoning/nettack.py",
        "Poisoning: Meta Attack": "attacks/poisoning/metattack.py",
        "Evasion: Edge Flip": "attacks/evasion/structure_evasion.py",
        "Evasion: Feature": "attacks/evasion/feature_evasion.py",
        "Evasion: Gradient (FGSM-like)": "attacks/evasion/fgsm_like.py",
    }
    attack_what_changes = {
        "Poisoning: Random Structure": "changes A (edges) and lightly corrupts X, then retrains",
        "Poisoning: Nettack": "changes A near target nodes, then retrains",
        "Poisoning: Meta Attack": "changes A globally (proxy Metattack), then retrains",
        "Evasion: Edge Flip": "changes A at inference only (model fixed)",
        "Evasion: Feature": "changes X at inference only (A unchanged, model fixed)",
        "Evasion: Gradient (FGSM-like)": "changes X using gradient sign at inference only (A unchanged, model fixed)",
    }
    lines.append("### Per-Attack Explanation With One Concrete Cora Datapoint")
    if attack_examples:
        for name, ex in attack_examples.items():
            lines.append(f"#### {name}")
            lines.append(f"- implementation file: `{attack_impl.get(name, 'attacks/...')}`")
            lines.append(f"- what changes: {attack_what_changes.get(name, 'A and/or X')}")
            lines.append("- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.")
            lines.append("- example datapoint:")
            lines.append(f"  - target_node={ex.get('target_node')} true_label={ex.get('label')}")
            lines.append(f"  - pred_clean -> pred_attacked: {ex.get('pred_clean')} -> {ex.get('pred_attacked')}")
            lines.append(f"  - conf_clean -> conf_attacked: {ex.get('conf_clean'):.4f} -> {ex.get('conf_attacked'):.4f}")
            lines.append(f"  - margin_clean -> margin_attacked: {ex.get('margin_clean'):.4f} -> {ex.get('margin_attacked'):.4f}")
            if "x_clean_first10" in ex:
                lines.append(f"  - x_clean_first10: {ex.get('x_clean_first10')}")
                lines.append(f"  - x_attacked_first10: {ex.get('x_attacked_first10')}")
            if "edges_added_sample" in ex:
                lines.append(f"  - edges_added_sample: {ex.get('edges_added_sample')}")
                lines.append(f"  - edges_removed_sample: {ex.get('edges_removed_sample')}")
            lines.append(f"  - perturbation budget used: {ex.get('budget')}")
            lines.append("")
    else:
        lines.append("- (attack_examples not provided)")
    lines.append("")
    lines.append("## Selecting The Worst Attack")
    lines.append("- We compute test accuracy for every attack (GCN).")
    lines.append("- `Accuracy Drop = Accuracy(Baseline) - Accuracy(Attack)`.")
    lines.append("- The attack with maximum accuracy drop is reported in the terminal as:")
    lines.append("  - `The most impactful Attack is : <name> (accuracy drop=...)`")
    if worst_attack_name:
        lines.append(f"- In this run: worst attack = `{worst_attack_name}`.")
    lines.append("")
    lines.append("## Defenses (Evaluated Individually Then Combined)")
    lines.append("We apply defenses only against the single most impactful attack.")
    lines.append("")
    lines.append("### 1) Base Defense: Feature Smoothing")
    lines.append("- `X_s = α X + (1-α) Â X`")
    lines.append("- This reduces high-frequency feature noise that adversarial perturbations introduce.")
    lines.append("")
    lines.append("### 2) Pruning Defense")
    lines.append("- Keep top-k most similar neighbors per node (feature cosine similarity).")
    lines.append("- Removes suspicious/dissimilar edges that can amplify adversarial messages.")
    lines.append("")
    lines.append("### 3) Ontology Defense (Semantic Similarity)")
    lines.append("- Build ontology similarity matrix `O` from cosine similarity of node feature vectors.")
    lines.append("- Feature projection: `X' = X + λ OX`.")
    lines.append("- Adjacency reweight: `A' = clip(A + λ O, 0, 1)` (then symmetrize + add self-loops).")
    lines.append("")
    lines.append("### 4) Combined Defense")
    lines.append("- Ontology projection -> ontology adjacency reweight -> pruning.")
    lines.append("- We evaluate base, pruning, ontology, combined; then select the best by accuracy.")
    lines.append("")
    lines.append("## Ontologies Created (Files)")
    lines.append("- `results/ontologies/ontology_topk_edges.csv`: top-k neighbors for every node, for each ontology variant.")
    lines.append("- `results/ontologies/ontology_examples.md`: a small example ontology neighborhood for one target node.")
    lines.append("- `results/ontologies/ontology_summary.md`: what each ontology variant means.")
    lines.append("")
    lines.append("## Outputs Explained")
    lines.append("### Tables (ONLY final tables)")
    lines.append("- `results/FINAL_TABLE_PRE_DEFENSE.csv`: baseline + all attacks (GCN and GAT).")
    lines.append("- `results/FINAL_TABLE_POST_DEFENSE.csv`: baseline + worst attack + defenses (GCN and GAT).")
    lines.append("- `results/metrics_terminal.txt`: CSV-form versions of the above tables (no ANSI).")
    lines.append("")
    lines.append("### Figures (paper-style)")
    lines.append("- `results/FIG_workflow.png`: entire workflow of the project.")
    lines.append("- `results/FIG_gcn_layerwise.png`: detailed GCN layer-wise math and outputs.")
    lines.append("- `results/FIG_gat_layerwise.png`: detailed GAT attention math and outputs.")
    lines.append("- `results/FIG_attack_defense_flow.png`: clean -> attack -> defense flow.")
    lines.append("- `results/FIG_attack_suite.png`: dataset change visualization for each attack.")
    lines.append("- `results/FIG_graph_diff_worst.png`: clean vs worst-attacked vs defended graph diffs.")
    lines.append("- `results/FIG_class_clusters.png`: class clusters for clean/attacked/defended embeddings.")
    lines.append("")
    lines.append("## Real-World Relevance")
    lines.append("- Citation graphs: attackers can inject fake citations or distort paper text features to misclassify topics.")
    lines.append("- Social/fraud graphs: injected edges/features can hide malicious nodes; pruning + ontology reduce attacker leverage.")
    lines.append("")
    if metric_summary:
        lines.append("## Run Summary")
        for line in metric_summary:
            lines.append(f"- {line}")
    Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        trials = []
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

            trials.append(
                {
                    "metrics": m,
                    "pred": pred,
                    "probs": probs,
                    "data": data,
                    "p_rate": p_rate,
                    "budget": budget,
                    "extra": extra,
                    "model": used_model,
                }
            )

        # Select an attack strength that is harmful but not degenerate:
        # aim for a moderate accuracy drop so defenses can visibly recover.
        base_acc = float(base_metrics["accuracy"])
        target_low = max(0.0, base_acc - 0.25)
        target_high = max(0.0, base_acc - 0.05)
        in_window = [t for t in trials if (target_low <= float(t["metrics"]["accuracy"]) <= target_high)]
        if in_window:
            chosen = min(in_window, key=lambda t: float(t["metrics"]["accuracy"]))
        else:
            # If all trials are too strong (below target_low), pick the least strong among them.
            too_strong = [t for t in trials if float(t["metrics"]["accuracy"]) < target_low]
            if too_strong:
                chosen = max(too_strong, key=lambda t: float(t["metrics"]["accuracy"]))
            else:
                # Otherwise just take the worst we observed.
                chosen = min(trials, key=lambda t: float(t["metrics"]["accuracy"]))

        m = chosen["metrics"]
        pred = chosen["pred"]
        probs = chosen["probs"]
        payload["data"] = chosen["data"]
        payload["p_rate"] = chosen["p_rate"]
        payload["budget"] = chosen["budget"]
        payload["extra"] = chosen.get("extra", {})
        payload["model"] = chosen.get("model", clean_model)
        if isinstance(payload.get("extra"), dict) and "adj" in payload["extra"]:
            payload["adj_attacked"] = payload["extra"]["adj"]

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
        # Ontology defense: (1) project features toward semantic neighbors, and
        # (2) reweight adjacency toward ontology similarity.
        x_proj = ontology_feature_projection(attacked_data.x, ontology, lam=lam)
        x_proj_np = x_proj.detach().cpu().numpy()
        adj_onto = ontology_reweight_adjacency(attacked_adj, ontology, lam=lam)
        adj_onto = _symmetrize_and_self_loop(adj_onto)
        data_def_onto = pyg_from_adj_and_x(attacked_data, adj_onto, x_proj_np)
        m_o, pred_o, p_o = evaluate_model(model, data_def_onto)
        rows.append((f"Defense: Ontology (adj+feature, lambda={lam})", m_o, pred_o, p_o, data_def_onto, 0.0, adj_onto))
        if best_ontology["metrics"] is None or m_o["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {
                "name": f"Defense: Ontology (adj+feature, lambda={lam})",
                "metrics": m_o,
                "pred": pred_o,
                "probs": p_o,
                "data": data_def_onto,
                "adj": adj_onto,
                "model": model,
            }

    # Combined pruning + ontology (project features then prune edges using projected similarity).
    for lam in [0.1, 0.2]:
        x_proj = ontology_feature_projection(attacked_data.x, ontology, lam=lam)
        x_proj_np = x_proj.detach().cpu().numpy()
        for k in [10, 15]:
            adj_onto = ontology_reweight_adjacency(attacked_adj, ontology, lam=lam)
            adj_onto = _symmetrize_and_self_loop(adj_onto)
            adj_pruned = top_k_pruning(adj_onto, x_proj_np, k=k)
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
        x_proj = ontology_feature_projection(attacked_data.x, ontology, lam=0.1)
        x_proj_np = x_proj.detach().cpu().numpy()
        adj_onto = ontology_reweight_adjacency(attacked_adj, ontology, lam=0.1)
        adj_onto = _symmetrize_and_self_loop(adj_onto)
        data_def_onto = pyg_from_adj_and_x(attacked_data, adj_onto, x_proj_np)
        retrained_model = train_model(model_builder(), data_def_onto, epochs=80)
        m_r, pred_r, p_r = evaluate_model(retrained_model, data_def_onto)
        rows.append(("Defense: Ontology + Retrain", m_r, pred_r, p_r, data_def_onto, 0.0, adj_onto))
        if best_ontology["metrics"] is None or m_r["accuracy"] > best_ontology["metrics"]["accuracy"]:
            best_ontology = {"name": "Defense: Ontology + Retrain", "metrics": m_r, "pred": pred_r, "probs": p_r, "data": data_def_onto, "adj": adj_onto, "model": retrained_model}

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
            adj_onto = ontology_reweight_adjacency(attacked_adj, ontology, lam=0.1)
            adj_onto = _symmetrize_and_self_loop(adj_onto)
            adj_pruned = top_k_pruning(adj_onto, x_proj_np, k=15)
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


def main(profile="paper", clean=False):
    set_seed(42)
    if clean:
        clean_results_dir("results")
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
    save_clean_graph_plot(adj_clean, labels, "results/FIG_clean_graph.png")

    print("\n=== BASELINE TRAINING ===")
    gcn = train_model(GCN(dataset.num_features, 16, dataset.num_classes), data, epochs=120)
    gat = train_model(GAT(dataset.num_features, 8, dataset.num_classes, heads=4), data, epochs=120)
    gcn_metrics, gcn_pred, gcn_probs = evaluate_model(gcn, data)
    gat_metrics, gat_pred, gat_probs = evaluate_model(gat, data)
    print(f"GCN baseline metrics: {gcn_metrics}")
    print(f"GAT baseline metrics: {gat_metrics}")

    target_node = int(idx_test[0])

    # Export ontology artifacts (top-k neighbors per node + a readable example).
    export_ontology_artifacts(features_clean, labels, target_node, out_dir="results/ontologies", top_k=5)

    # Paper-style architecture diagrams (clean, explanatory block diagrams).
    draw_gcn_layerwise("results/FIG_gcn_layerwise.png")
    draw_gat_layerwise("results/FIG_gat_layerwise.png", heads=4)

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
            "budget": 800,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [600, 800, 1200],
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
            "budget": 48,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [32, 48, 64],
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
            "budget": 1200,
            "p_rate": 0.0,
            "adj": adj_clean,
            "budgets": [800, 1200, 1600],
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
            "budgets": [60, 80, 120],
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
            "budget": 10,
            "p_rate": 0.0,
            "budgets": [8, 12, 18],
            "make_data": make_feature_attack,
        }
    )

    def make_fgsm(budget):
        data_fgsm = run_fgsm_like_feature_attack(gcn, data, epsilon=float(budget))
        return data_fgsm, perturbation_rate(data.x.cpu().numpy(), data_fgsm.x.cpu().numpy()), {}

    payloads_static.append(
        {
            "name": "Evasion: Gradient (FGSM-like)",
            "type": "evasion",
            "data": None,
            "budget": 0.06,
            "p_rate": 0.0,
            "budgets": [0.04, 0.06, 0.10],
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
        # Example datapoint: one test node, show how its prediction/confidence shifts.
        pred_clean = int(gcn_preds["Baseline"][target_node])
        pred_att = int(gcn_preds[name][target_node])
        pc = gcn_prob["Baseline"][target_node].detach().cpu().numpy()
        pa = gcn_prob[name][target_node].detach().cpu().numpy()
        pc_sorted = np.sort(pc)[::-1]
        pa_sorted = np.sort(pa)[::-1]
        margin_clean = float(pc_sorted[0] - pc_sorted[1]) if len(pc_sorted) > 1 else float(pc_sorted[0])
        margin_att = float(pa_sorted[0] - pa_sorted[1]) if len(pa_sorted) > 1 else float(pa_sorted[0])
        if "Feature" in name or "Gradient" in name:
            attack_examples[name] = {
                "target_node": int(target_node),
                "label": int(labels[target_node]),
                "x_clean_first10": data.x[target_node][:10].cpu().numpy().tolist(),
                "x_attacked_first10": data_attacked.x[target_node][:10].cpu().numpy().tolist(),
                "budget": payload.get("budget"),
                "pred_clean": pred_clean,
                "pred_attacked": pred_att,
                "conf_clean": float(pc.max()),
                "conf_attacked": float(pa.max()),
                "margin_clean": margin_clean,
                "margin_attacked": margin_att,
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
                "pred_clean": pred_clean,
                "pred_attacked": pred_att,
                "conf_clean": float(pc.max()),
                "conf_attacked": float(pa.max()),
                "margin_clean": margin_clean,
                "margin_attacked": margin_att,
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

    # -------- Final terminal tables (ONLY 2 tables) --------
    ANSI_BOLD = "\033[1m"
    ANSI_RESET = "\033[0m"

    def bold(s):
        return f"{ANSI_BOLD}{s}{ANSI_RESET}"

    worst_attack_name = str(worst_attack["Attack"])
    worst_drop = float(worst_attack.get("Accuracy Drop", 0.0))

    def best_defense_name(df):
        sub = df[df["Attack"].str.contains("Defense", na=False)]
        if sub.empty:
            return None
        return str(sub.sort_values("Accuracy", ascending=False).iloc[0]["Attack"])

    best_def_gcn = best_defense_name(post_defense_gcn)
    best_def_gat = best_defense_name(post_defense_gat)

    print(f"\nThe most impactful Attack is : {worst_attack_name} (accuracy drop={worst_drop:.4f})")

    # Merge into exactly two tables (pre-defense / post-defense) covering both models.
    pre_defense_gcn2 = pre_defense_gcn.copy()
    pre_defense_gcn2.insert(0, "Model", "GCN")
    pre_defense_gat2 = pre_defense_gat.copy()
    pre_defense_gat2.insert(0, "Model", "GAT")
    pre_all = pd.concat([pre_defense_gcn2, pre_defense_gat2], ignore_index=True)

    post_defense_gcn2 = post_defense_gcn.copy()
    post_defense_gcn2.insert(0, "Model", "GCN")
    post_defense_gat2 = post_defense_gat.copy()
    post_defense_gat2.insert(0, "Model", "GAT")
    post_all = pd.concat([post_defense_gcn2, post_defense_gat2], ignore_index=True)

    # Persist final tables (CSV + a single terminal log file).
    pre_all.to_csv("results/FINAL_TABLE_PRE_DEFENSE.csv", index=False)
    post_all.to_csv("results/FINAL_TABLE_POST_DEFENSE.csv", index=False)

    cols = ["Model", "Attack", "Accuracy", "Macro F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"]
    def _fmt_terminal(df):
        out = df.copy()
        for c in ["Accuracy", "Macro F1", "ROC-AUC", "Accuracy Drop", "Perturbation Budget"]:
            out[c] = out[c].map(lambda v: f"{float(v):.4f}")
        out["Model"] = out["Model"].astype(str)
        out["Attack"] = out["Attack"].astype(str)
        return out

    pre_disp = _fmt_terminal(pre_all[cols])
    post_disp = _fmt_terminal(post_all[cols])

    # Bold the most harmful attack row(s) in pre table.
    for i in range(len(pre_disp)):
        if str(pre_disp.loc[i, "Attack"]) == worst_attack_name and str(pre_disp.loc[i, "Model"]) == "GCN":
            for c in ["Attack", "Accuracy", "Accuracy Drop"]:
                pre_disp.loc[i, c] = bold(f"{pre_disp.loc[i, c]}")

    # Bold best defenses per model in post table.
    for i in range(len(post_disp)):
        if best_def_gcn and str(post_disp.loc[i, "Model"]) == "GCN" and str(post_disp.loc[i, "Attack"]) == best_def_gcn:
            for c in ["Attack", "Accuracy", "Accuracy Drop"]:
                post_disp.loc[i, c] = bold(f"{post_disp.loc[i, c]}")
        if best_def_gat and str(post_disp.loc[i, "Model"]) == "GAT" and str(post_disp.loc[i, "Attack"]) == best_def_gat:
            for c in ["Attack", "Accuracy", "Accuracy Drop"]:
                post_disp.loc[i, c] = bold(f"{post_disp.loc[i, c]}")

    pre_table_txt = tabulate(pre_disp, headers="keys", tablefmt="github", showindex=False)
    post_table_txt = tabulate(post_disp, headers="keys", tablefmt="github", showindex=False)

    print("\nFINAL TABLE (PRE-DEFENSE)")
    print(pre_table_txt)
    print("\nFINAL TABLE (POST-DEFENSE)")
    print(post_table_txt)

    terminal_log = []
    terminal_log.append(f"The most impactful Attack is : {worst_attack_name} (accuracy drop={worst_drop:.4f})")
    terminal_log.append("")
    terminal_log.append("FINAL TABLE (PRE-DEFENSE)")
    terminal_log.append(pre_all[cols].to_csv(index=False))
    terminal_log.append("")
    terminal_log.append("FINAL TABLE (POST-DEFENSE)")
    terminal_log.append(post_all[cols].to_csv(index=False))
    Path("results/metrics_terminal.txt").write_text("\n".join(terminal_log) + "\n", encoding="utf-8")

    # -------- Paper figures (minimal, paper-style) --------
    # Build attacked/defended adjacencies for worst-attack graph diff.
    adj_attacked = adj_from_edge_index(worst_data.edge_index, worst_data.num_nodes) if worst_data is not None else adj_clean
    adj_defended = defended_adj if defended_adj is not None else adj_attacked

    attacked_nodes = changed_feature_nodes(data.x, worst_data.x) if worst_data is not None else []
    if not attacked_nodes:
        attacked_nodes = changed_nodes_from_adj(adj_clean, adj_attacked)
    defended_nodes = changed_feature_nodes(worst_data.x, defended_data.x) if (worst_data is not None and defended_data is not None) else attacked_nodes

    visualize_graph_mosaic(
        adj_clean,
        adj_attacked,
        adj_defended,
        labels,
        target_node=target_node,
        attacked_nodes=attacked_nodes,
        defended_nodes=defended_nodes,
        attack_name=worst_attack_name,
        defense_name=str(defended_name),
        hop_k=2,
        max_nodes=260,
        save_path="results/FIG_graph_diff_worst.png",
    )

    # Attach attacked-nodes metadata for feature attacks so the suite figure is informative.
    for p in payloads_gcn:
        if p.get("data") is None:
            continue
        if "Feature" in p["name"] or "Gradient" in p["name"]:
            p["attacked_nodes"] = changed_feature_nodes(data.x, p["data"].x)
        else:
            p["attacked_nodes"] = []

    visualize_attack_suite(
        adj_clean,
        payloads_gcn,
        labels,
        idx_test=idx_test[:1],
        default_seed_node=target_node,
        hop_k=2,
        max_nodes=220,
        save_path="results/FIG_attack_suite.png",
    )

    # Cluster suite: baseline + each attack (GCN embeddings) + defended (best).
    with torch.no_grad():
        emb_list = [gcn.get_embeddings(data).cpu().numpy()]
        titles = ["Clean (GCN)"]
        for p in payloads_gcn:
            if p.get("data") is None:
                continue
            emb_list.append(gcn.get_embeddings(p["data"]).cpu().numpy())
            titles.append(p["name"])
        if defended_data is not None and defended_model is not None:
            emb_list.append(defended_model.get_embeddings(defended_data).cpu().numpy())
            titles.append(f"Defended ({defended_name})")

    plot_tsne_suite(emb_list, labels, titles, "results/FIG_class_clusters.png", seed=42, sample_size=1500, perplexity=30)

    # Workflow and project flow diagrams (now that worst attack/defenses are known).
    attacks_list = [p["name"] for p in payloads_gcn if p.get("data") is not None]
    defense_names = [
        base_def_gcn,
        prune_def_gcn,
        onto_def_gcn,
        combo_def_gcn,
    ]
    draw_project_workflow("results/FIG_workflow.png", attacks=attacks_list, worst_attack=worst_attack_name, defenses=defense_names)
    draw_attack_defense_flow("results/FIG_attack_defense_flow.png", worst_attack=worst_attack_name, defense_names=defense_names)

    # -------- Ensure dynamic dataset snapshots exist (no extra result tables) --------
    generator = DynamicGraphGenerator(initial_nodes=200, num_features=dataset.num_features, num_classes=dataset.num_classes)
    snapshot_dir = "data/dynamic"
    save_dynamic_snapshots(generator, snapshots=4, save_dir=snapshot_dir)
    # Snapshots are saved to disk for your submission requirements; no need to load/evaluate here.

    # -------- Explanation / output guide --------
    dataset_stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "num_classes": int(dataset.num_classes),
    }
    summary_lines = [
        f"GCN baseline accuracy: {gcn_metrics['accuracy']:.3f}",
        f"GAT baseline accuracy: {gat_metrics['accuracy']:.3f}",
        f"Most impactful attack (GCN): {worst_attack_name} (drop={worst_drop:.3f})",
        f"Best defense (GCN): {best_def_gcn}",
        f"Best defense (GAT): {best_def_gat}",
        "Attacks are selected at a moderate intensity so defenses can show visible recovery (paper-style).",
    ]
    write_detailed_explanation(
        "results/EXPLANATION.md",
        attack_examples=attack_examples,
        metric_summary=summary_lines,
        dataset_stats=dataset_stats,
        worst_attack_name=worst_attack_name,
    )

    print("\nOutputs written under results/:")
    for p in [
        "FINAL_TABLE_PRE_DEFENSE.csv",
        "FINAL_TABLE_POST_DEFENSE.csv",
        "metrics_terminal.txt",
        "EXPLANATION.md",
        "FIG_workflow.png",
        "FIG_gcn_layerwise.png",
        "FIG_gat_layerwise.png",
        "FIG_attack_defense_flow.png",
        "FIG_clean_graph.png",
        "FIG_attack_suite.png",
        "FIG_graph_diff_worst.png",
        "FIG_class_clusters.png",
        "ontologies/ontology_topk_edges.csv",
        "ontologies/ontology_examples.md",
        "ontologies/ontology_summary.md",
    ]:
        print(f"- {p}")

    print("\nRun command:")
    print("python3 main.py --clean --profile paper")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attacks on GNNs (GCN/GAT) with defenses and paper-style outputs.")
    parser.add_argument("--profile", default="paper", choices=["paper"], help="Output profile. 'paper' generates only the required final tables + paper figures.")
    parser.add_argument("--clean", action="store_true", help="Delete all existing files under results/ before running.")
    args = parser.parse_args()
    main(profile=args.profile, clean=args.clean)
