import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.lines import Line2D

def visualize_graph_mosaic(
    clean_adj,
    attacked_adj,
    defended_adj,
    labels,
    target_node=None,
    attacked_nodes=None,
    defended_nodes=None,
    attacked_pred_changed=None,
    defended_pred_changed=None,
    attacked_newly_wrong=None,
    defended_still_wrong=None,
    attack_name=None,
    defense_name=None,
    hop_k=2,
    max_nodes=220,
    save_path="results/graph_comparison.png",
):
    """
    Plot Clean, Attacked, and Defended graphs side-by-side using a subgraph.
    Differences are explicitly highlighted:
      - Attack-added edges: red
      - Attack-removed edges: dashed black
      - Defense-added edges: green
      - Defense-removed edges: dashed blue

    Additionally, we can ring nodes to show:
      - Feature-touched nodes (attacked_nodes/defended_nodes): red/green ring
      - Prediction-changed nodes: purple ring
      - Newly misclassified nodes under attack: black ring (optional)
    """
    if target_node is None:
        target_node = 0

    def to_graph(adj):
        if sp.issparse(adj):
            return nx.from_scipy_sparse_array(adj) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(adj)
        return nx.from_numpy_array(adj)

    def edge_set(G):
        return set((min(u, v), max(u, v)) for u, v in G.edges())

    G_clean = to_graph(clean_adj)
    G_att = to_graph(attacked_adj)
    G_def = to_graph(defended_adj)

    clean_edges = edge_set(G_clean)
    att_edges = edge_set(G_att)
    def_edges = edge_set(G_def)

    attack_added = att_edges - clean_edges
    attack_removed = clean_edges - att_edges
    defense_added = def_edges - att_edges
    defense_removed = att_edges - def_edges

    def khop_nodes(G, start, k, cap):
        visited = {start}
        frontier = {start}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if len(visited) >= cap:
                break
        if len(visited) > cap:
            visited = set(list(visited)[:cap])
        return visited

    # Build node set from k-hop neighborhood (better class-cluster context) and changed edges
    nodes = set(khop_nodes(G_clean, target_node, hop_k, max_nodes))
    for u, v in attack_added.union(attack_removed, defense_added, defense_removed):
        nodes.add(u)
        nodes.add(v)
    if len(nodes) > max_nodes:
        nodes = set(list(nodes)[:max_nodes])

    sub_clean = G_clean.subgraph(nodes)
    pos = nx.spring_layout(sub_clean, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    attack_title = f"Attacked Graph ({attack_name})" if attack_name else "Attacked Graph (Changes Highlighted)"
    defense_title = f"Post-Defense Graph ({defense_name})" if defense_name else "Post-Defense Graph (Changes Highlighted)"
    titles = ["Clean Graph", attack_title, defense_title]

    # Panel 1: Clean
    nx.draw(sub_clean, pos, ax=axes[0], node_color=labels[list(sub_clean.nodes())],
            node_size=120, cmap=plt.cm.Set1, with_labels=False, edge_color='gray', alpha=0.7)

    # Panel 2: Attacked
    nx.draw(sub_clean, pos, ax=axes[1], node_color=labels[list(sub_clean.nodes())],
            node_size=120, cmap=plt.cm.Set1, with_labels=False, edge_color='lightgray', alpha=0.6)
    nx.draw_networkx_edges(sub_clean, pos, ax=axes[1],
                           edgelist=[e for e in attack_added if e[0] in nodes and e[1] in nodes],
                           edge_color='red', width=2.0)
    nx.draw_networkx_edges(sub_clean, pos, ax=axes[1],
                           edgelist=[e for e in attack_removed if e[0] in nodes and e[1] in nodes],
                           edge_color='black', style='dashed', width=1.5)
    if attacked_nodes:
        attacked_nodes = [n for n in attacked_nodes if n in nodes]
        if attacked_nodes:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[1],
                nodelist=attacked_nodes,
                node_color='none',
                edgecolors='red',
                linewidths=2.2,
                node_size=240,
            )
    if attacked_pred_changed:
        ring = [n for n in attacked_pred_changed if n in nodes]
        if ring:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[1],
                nodelist=ring,
                node_color="none",
                edgecolors="#8e44ad",
                linewidths=2.2,
                node_size=310,
            )
    if attacked_newly_wrong:
        ring = [n for n in attacked_newly_wrong if n in nodes]
        if ring:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[1],
                nodelist=ring,
                node_color="none",
                edgecolors="black",
                linewidths=2.2,
                node_size=380,
            )

    # Panel 3: Defended
    nx.draw(sub_clean, pos, ax=axes[2], node_color=labels[list(sub_clean.nodes())],
            node_size=120, cmap=plt.cm.Set1, with_labels=False, edge_color='lightgray', alpha=0.6)
    nx.draw_networkx_edges(sub_clean, pos, ax=axes[2],
                           edgelist=[e for e in defense_added if e[0] in nodes and e[1] in nodes],
                           edge_color='green', width=2.0)
    nx.draw_networkx_edges(sub_clean, pos, ax=axes[2],
                           edgelist=[e for e in defense_removed if e[0] in nodes and e[1] in nodes],
                           edge_color='blue', style='dashed', width=1.5)
    if defended_nodes:
        defended_nodes = [n for n in defended_nodes if n in nodes]
        if defended_nodes:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[2],
                nodelist=defended_nodes,
                node_color='none',
                edgecolors='green',
                linewidths=2.2,
                node_size=240,
            )
    if defended_pred_changed:
        ring = [n for n in defended_pred_changed if n in nodes]
        if ring:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[2],
                nodelist=ring,
                node_color="none",
                edgecolors="#8e44ad",
                linewidths=2.2,
                node_size=310,
            )
    if defended_still_wrong:
        ring = [n for n in defended_still_wrong if n in nodes]
        if ring:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[2],
                nodelist=ring,
                node_color="none",
                edgecolors="black",
                linewidths=2.2,
                node_size=380,
            )

    # Highlight target node
    for ax in axes:
        if target_node in nodes:
            nx.draw_networkx_nodes(sub_clean, pos, ax=ax, nodelist=[target_node],
                                   node_color='yellow', node_size=240, edgecolors='black')

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

    # Legend + change counts
    legend_elems = [
        Line2D([0], [0], color="gray", lw=2, label="Base edges (subgraph)"),
        Line2D([0], [0], color="red", lw=2, label="Attack added edges"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Attack removed edges"),
        Line2D([0], [0], color="green", lw=2, label="Defense added edges"),
        Line2D([0], [0], color="blue", lw=2, linestyle="--", label="Defense removed edges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="red", markersize=9, label="Feature changed (attack)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="green", markersize=9, label="Feature repaired/changed (defense)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#8e44ad", markersize=9, label="Prediction changed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="black", markersize=9, label="Misclassified"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Edge diffs: attack +{len(attack_added)} / -{len(attack_removed)} | defense +{len(defense_added)} / -{len(defense_removed)}",
        y=0.98,
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def visualize_triplet_separate(
    clean_adj,
    attacked_adj,
    defended_adj,
    labels,
    target_node,
    attacked_nodes=None,
    defended_nodes=None,
    attacked_pred_changed=None,
    defended_pred_changed=None,
    attacked_newly_wrong=None,
    defended_still_wrong=None,
    attack_name=None,
    defense_name=None,
    hop_k=2,
    max_nodes=260,
    out_dir="results",
    prefix="worst",
):
    """
    Save 3 separate images (clean / attacked / defended) using the SAME node positions.
    This is helpful for reports where you want independent panels but perfect alignment.
    """
    import os

    if target_node is None:
        target_node = 0

    def to_graph(adj):
        if sp.issparse(adj):
            return nx.from_scipy_sparse_array(adj) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(adj)
        return nx.from_numpy_array(adj)

    def edge_set(G):
        return set((min(u, v), max(u, v)) for u, v in G.edges())

    def khop_nodes(G, start, k, cap):
        visited = {start}
        frontier = {start}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if len(visited) >= cap:
                break
        if len(visited) > cap:
            visited = set(list(visited)[:cap])
        return visited

    G_clean = to_graph(clean_adj)
    G_att = to_graph(attacked_adj)
    G_def = to_graph(defended_adj)

    clean_edges = edge_set(G_clean)
    att_edges = edge_set(G_att)
    def_edges = edge_set(G_def)

    attack_added = att_edges - clean_edges
    attack_removed = clean_edges - att_edges
    defense_added = def_edges - att_edges
    defense_removed = att_edges - def_edges

    nodes = set(khop_nodes(G_clean, target_node, hop_k, max_nodes))
    for u, v in attack_added.union(attack_removed, defense_added, defense_removed):
        nodes.add(u)
        nodes.add(v)
    if len(nodes) > max_nodes:
        nodes = set(list(nodes)[:max_nodes])

    sub = G_clean.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=42)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def _draw_panel(title, edge_color, add_edges=None, rm_edges=None, ring_red=None, ring_green=None, ring_purple=None, ring_black=None, path=""):
        plt.figure(figsize=(8, 6))
        nx.draw(sub, pos, node_color=labels[list(sub.nodes())], node_size=120, cmap=plt.cm.Set1, with_labels=False, edge_color=edge_color, alpha=0.7)
        if add_edges:
            nx.draw_networkx_edges(sub, pos, edgelist=[e for e in add_edges if e[0] in nodes and e[1] in nodes], edge_color="red", width=2.0)
        if rm_edges:
            nx.draw_networkx_edges(sub, pos, edgelist=[e for e in rm_edges if e[0] in nodes and e[1] in nodes], edge_color="black", style="dashed", width=1.5)
        if ring_red:
            ring = [n for n in ring_red if n in nodes]
            if ring:
                nx.draw_networkx_nodes(sub, pos, nodelist=ring, node_color="none", edgecolors="red", linewidths=2.2, node_size=240)
        if ring_green:
            ring = [n for n in ring_green if n in nodes]
            if ring:
                nx.draw_networkx_nodes(sub, pos, nodelist=ring, node_color="none", edgecolors="green", linewidths=2.2, node_size=240)
        if ring_purple:
            ring = [n for n in ring_purple if n in nodes]
            if ring:
                nx.draw_networkx_nodes(sub, pos, nodelist=ring, node_color="none", edgecolors="#8e44ad", linewidths=2.2, node_size=310)
        if ring_black:
            ring = [n for n in ring_black if n in nodes]
            if ring:
                nx.draw_networkx_nodes(sub, pos, nodelist=ring, node_color="none", edgecolors="black", linewidths=2.2, node_size=380)
        if target_node in nodes:
            nx.draw_networkx_nodes(sub, pos, nodelist=[target_node], node_color="yellow", node_size=240, edgecolors="black")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    paths["clean"] = os.path.join(out_dir, f"FIG_{prefix}_clean.png")
    _draw_panel("Clean Dataset (Subgraph)", "gray", path=paths["clean"])

    paths["attacked"] = os.path.join(out_dir, f"FIG_{prefix}_attacked.png")
    _draw_panel(
        f"Attacked Dataset ({attack_name})" if attack_name else "Attacked Dataset",
        "lightgray",
        add_edges=attack_added,
        rm_edges=attack_removed,
        ring_red=attacked_nodes,
        ring_purple=attacked_pred_changed,
        ring_black=attacked_newly_wrong,
        path=paths["attacked"],
    )

    paths["defended"] = os.path.join(out_dir, f"FIG_{prefix}_defended.png")
    _draw_panel(
        f"Defended Dataset ({defense_name})" if defense_name else "Defended Dataset",
        "lightgray",
        add_edges=defense_added,
        rm_edges=defense_removed,
        ring_green=defended_nodes,
        ring_purple=defended_pred_changed,
        ring_black=defended_still_wrong,
        path=paths["defended"],
    )
    return paths


def visualize_graph_pair(
    clean_adj,
    attacked_adj,
    labels,
    target_node=None,
    attacked_nodes=None,
    attack_name=None,
    hop_k=2,
    max_nodes=220,
    save_path="results/attack_graph.png",
):
    """
    Plot Clean vs Attacked using a consistent subgraph, highlighting:
      - Attack-added edges: red
      - Attack-removed edges: dashed black
      - Feature-touched nodes (optional): red ring
    """
    if target_node is None:
        target_node = 0

    def to_graph(adj):
        if sp.issparse(adj):
            return nx.from_scipy_sparse_array(adj) if hasattr(nx, "from_scipy_sparse_array") else nx.from_scipy_sparse_matrix(adj)
        return nx.from_numpy_array(adj)

    def edge_set(G):
        return set((min(u, v), max(u, v)) for u, v in G.edges())

    def khop_nodes(G, start, k, cap):
        visited = {start}
        frontier = {start}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if len(visited) >= cap:
                break
        if len(visited) > cap:
            visited = set(list(visited)[:cap])
        return visited

    G_clean = to_graph(clean_adj)
    G_att = to_graph(attacked_adj)
    clean_edges = edge_set(G_clean)
    att_edges = edge_set(G_att)
    attack_added = att_edges - clean_edges
    attack_removed = clean_edges - att_edges

    nodes = set(khop_nodes(G_clean, target_node, hop_k, max_nodes))
    for u, v in attack_added.union(attack_removed):
        nodes.add(u)
        nodes.add(v)
    if len(nodes) > max_nodes:
        nodes = set(list(nodes)[:max_nodes])

    sub_clean = G_clean.subgraph(nodes)
    pos = nx.spring_layout(sub_clean, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = ["Clean Graph", f"Attacked ({attack_name})" if attack_name else "Attacked Graph"]

    nx.draw(
        sub_clean,
        pos,
        ax=axes[0],
        node_color=labels[list(sub_clean.nodes())],
        node_size=120,
        cmap=plt.cm.Set1,
        with_labels=False,
        edge_color="gray",
        alpha=0.7,
    )

    nx.draw(
        sub_clean,
        pos,
        ax=axes[1],
        node_color=labels[list(sub_clean.nodes())],
        node_size=120,
        cmap=plt.cm.Set1,
        with_labels=False,
        edge_color="lightgray",
        alpha=0.6,
    )
    nx.draw_networkx_edges(
        sub_clean,
        pos,
        ax=axes[1],
        edgelist=[e for e in attack_added if e[0] in nodes and e[1] in nodes],
        edge_color="red",
        width=2.0,
    )
    nx.draw_networkx_edges(
        sub_clean,
        pos,
        ax=axes[1],
        edgelist=[e for e in attack_removed if e[0] in nodes and e[1] in nodes],
        edge_color="black",
        style="dashed",
        width=1.5,
    )

    if attacked_nodes:
        attacked_nodes = [n for n in attacked_nodes if n in nodes]
        if attacked_nodes:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=axes[1],
                nodelist=attacked_nodes,
                node_color="none",
                edgecolors="red",
                linewidths=2.2,
                node_size=240,
            )

    for ax in axes:
        if target_node in nodes:
            nx.draw_networkx_nodes(
                sub_clean,
                pos,
                ax=ax,
                nodelist=[target_node],
                node_color="yellow",
                node_size=240,
                edgecolors="black",
            )

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def visualize_attack_suite(
    clean_adj,
    attack_payloads,
    labels,
    idx_test=None,
    default_seed_node=0,
    hop_k=2,
    max_nodes=200,
    save_path="results/attack_suite.png",
):
    """
    One figure summarizing how each attack changes the dataset (paper-style).
    Produces a 2x3 grid (6 attacks). For structure attacks, we highlight edge diffs.
    For feature attacks, edges are unchanged so we ring nodes with changed features.
    """
    if idx_test is None:
        idx_test = [default_seed_node]

    def to_graph(adj):
        if sp.issparse(adj):
            return nx.from_scipy_sparse_array(adj) if hasattr(nx, "from_scipy_sparse_array") else nx.from_scipy_sparse_matrix(adj)
        return nx.from_numpy_array(adj)

    def edge_set(G):
        return set((min(u, v), max(u, v)) for u, v in G.edges())

    def khop_nodes(G, start, k, cap):
        visited = {start}
        frontier = {start}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if len(visited) >= cap:
                break
        if len(visited) > cap:
            visited = set(list(visited)[:cap])
        return visited

    def pick_seed(clean_adj_, attacked_adj_):
        if sp.issparse(clean_adj_):
            diff = (clean_adj_ != attacked_adj_).tocoo()
            if diff.nnz == 0:
                return int(idx_test[0])
            counts = {}
            for u, v in zip(diff.row, diff.col):
                counts[u] = counts.get(u, 0) + 1
                counts[v] = counts.get(v, 0) + 1
            return int(max(counts.items(), key=lambda kv: kv[1])[0])
        diff = np.where(clean_adj_ != attacked_adj_)
        if len(diff[0]) == 0:
            return int(idx_test[0])
        nodes = np.concatenate([diff[0], diff[1]])
        vals, cnt = np.unique(nodes, return_counts=True)
        return int(vals[np.argmax(cnt)])

    G_clean = to_graph(clean_adj)
    clean_edges = edge_set(G_clean)

    # Keep only the first 6 attacks for the figure grid (project scope)
    payloads = [p for p in attack_payloads if p.get("data") is not None][:6]
    n = len(payloads)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(22, 12))
    axes = axes.reshape(-1)

    for i in range(rows * cols):
        if i >= n:
            axes[i].axis("off")
            continue
        p = payloads[i]
        name = p["name"]
        data = p["data"]
        attacked_adj = p.get("adj_attacked")
        if attacked_adj is None:
            # build from edge_index
            ei = data.edge_index.cpu().numpy()
            attacked_adj = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=clean_adj.shape)
        G_att = to_graph(attacked_adj)
        att_edges = edge_set(G_att)
        added = att_edges - clean_edges
        removed = clean_edges - att_edges

        # For feature attacks (no edge diffs), rely on provided attacked_nodes list.
        attacked_nodes = p.get("attacked_nodes", [])
        seed = pick_seed(clean_adj, attacked_adj) if (len(added) + len(removed)) > 0 else int(idx_test[0])
        nodes = khop_nodes(G_clean, seed, hop_k, max_nodes)
        sub = G_clean.subgraph(nodes)
        pos = nx.spring_layout(sub, seed=42)

        ax = axes[i]
        nx.draw(
            sub,
            pos,
            ax=ax,
            node_color=labels[list(sub.nodes())],
            node_size=90,
            cmap=plt.cm.Set1,
            with_labels=False,
            edge_color="lightgray",
            alpha=0.7,
        )
        # Overlay edge diffs within subgraph
        added_in = [e for e in added if e[0] in nodes and e[1] in nodes]
        removed_in = [e for e in removed if e[0] in nodes and e[1] in nodes]
        if added_in:
            nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=added_in, edge_color="red", width=2.0)
        if removed_in:
            nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=removed_in, edge_color="black", style="dashed", width=1.6)

        # Ring nodes with feature edits (if any)
        if attacked_nodes:
            ring = [n for n in attacked_nodes if n in nodes]
            if ring:
                nx.draw_networkx_nodes(
                    sub,
                    pos,
                    ax=ax,
                    nodelist=ring,
                    node_color="none",
                    edgecolors="red",
                    linewidths=2.2,
                    node_size=180,
                )

        # Seed highlight
        if seed in nodes:
            nx.draw_networkx_nodes(sub, pos, ax=ax, nodelist=[seed], node_color="yellow", node_size=200, edgecolors="black")

        ax.set_title(f"{name}\n(+{len(added)} / -{len(removed)} edges, ΔX nodes={len(attacked_nodes)})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elems = [
        Line2D([0], [0], color="lightgray", lw=2, label="Existing edges"),
        Line2D([0], [0], color="red", lw=2, label="Added edges"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Removed edges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", markeredgecolor="black", markersize=9, label="Seed node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="red", markersize=9, label="Feature-changed node"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle("Attack Implementation: how each attack changes the dataset (graph/feature diffs highlighted)", y=0.98, fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path

if __name__ == "__main__":
    pass
