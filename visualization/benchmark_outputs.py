from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


CORA_CMAP = plt.cm.get_cmap("tab10", 7)


def _edge_index_to_graph(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from([(int(u), int(v)) for u, v in zip(edge_index[0], edge_index[1]) if int(u) != int(v)])
    return g


def _sample_nodes_for_plot(g: nx.Graph, max_nodes: int = 260) -> np.ndarray:
    nodes = np.array(list(g.nodes()), dtype=np.int64)
    if len(nodes) <= max_nodes:
        return nodes
    degrees = np.array([g.degree(n) for n in nodes], dtype=np.float32)
    top = nodes[np.argsort(degrees)[::-1][: max_nodes // 2]]
    rest_pool = np.setdiff1d(nodes, top)
    if rest_pool.size == 0:
        return top
    rng = np.random.default_rng(42)
    rest = rng.choice(rest_pool, size=max_nodes - len(top), replace=False)
    return np.unique(np.concatenate([top, rest]))


def plot_graph_triplet(
    clean_data,
    attacked_data,
    defended_data,
    labels: np.ndarray,
    out_path: str,
    title_prefix: str,
    max_nodes: int = 260,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    g_clean = _edge_index_to_graph(clean_data.edge_index.cpu().numpy(), clean_data.num_nodes)
    g_att = _edge_index_to_graph(attacked_data.edge_index.cpu().numpy(), attacked_data.num_nodes)
    g_def = _edge_index_to_graph(defended_data.edge_index.cpu().numpy(), defended_data.num_nodes)

    keep = _sample_nodes_for_plot(g_clean, max_nodes=max_nodes)
    sg_clean = g_clean.subgraph(keep).copy()
    sg_att = g_att.subgraph(keep).copy()
    sg_def = g_def.subgraph(keep).copy()

    pos = nx.spring_layout(sg_clean, seed=42, k=0.22)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    states = [
        (sg_clean, "Clean"),
        (sg_att, "Attacked"),
        (sg_def, "Defended"),
    ]

    for ax, (graph, subtitle) in zip(axes, states):
        node_colors = [labels[int(n)] for n in graph.nodes()]
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, width=0.7, edge_color="#6e6e6e")
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=CORA_CMAP,
            node_size=34,
            linewidths=0.2,
            edgecolors="#222222",
        )
        ax.set_title(f"{title_prefix}: {subtitle}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_tsne_states(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    out_path: str,
    seed: int = 42,
    perplexity: float = 30.0,
):
    required = ["clean", "attacked", "defended_structural", "defended_ontology", "defended_hybrid"]
    for r in required:
        if r not in embeddings:
            raise ValueError(f"Missing embedding state: {r}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    order = required
    mats = [np.asarray(embeddings[k], dtype=np.float32) for k in order]
    stacked = np.vstack(mats)

    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    z = tsne.fit_transform(stacked)
    splits = np.array_split(z, len(order), axis=0)

    x_min, x_max = float(z[:, 0].min()), float(z[:, 0].max())
    y_min, y_max = float(z[:, 1].min()), float(z[:, 1].max())

    names = {
        "clean": "Clean",
        "attacked": "Attacked",
        "defended_structural": "Defended (Structural)",
        "defended_ontology": "Defended (Ontology)",
        "defended_hybrid": "Defended (Hybrid)",
    }

    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    for ax, key, coords in zip(axes, order, splits):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=CORA_CMAP, s=8, alpha=0.85, linewidths=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(names[key])

    fig.suptitle("t-SNE of Latent Embeddings (Pre-Softmax Hidden Layer)", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_accuracy_comparison(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plot_df = df.copy()
    plot_df["AttackDefense"] = plot_df["Attack"] + "\n" + plot_df["Defense"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax_i, model in enumerate(["GCN", "GAT"]):
        ax = axes[ax_i]
        cur = plot_df[plot_df["Model"] == model].sort_values(["Attack", "Defense"]) 
        x = np.arange(len(cur))
        clean = cur["CleanAccuracy"].values
        attacked = cur["AttackedAccuracy"].values
        defended = cur["DefendedAccuracy"].values

        ax.plot(x, clean, color="#2f4f4f", marker="o", linewidth=1.5, label="Clean")
        ax.plot(x, attacked, color="#c0392b", marker="o", linewidth=1.5, label="Attacked")
        ax.plot(x, defended, color="#1f77b4", marker="o", linewidth=1.5, label="Defended")
        ax.set_title(f"{model} Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(cur["AttackDefense"], rotation=75, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
