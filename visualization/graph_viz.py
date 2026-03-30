import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def visualize_graph_mosaic(
    clean_adj,
    attacked_adj,
    defended_adj,
    labels,
    target_node=None,
    attacked_nodes=None,
    defended_nodes=None,
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

    # Highlight target node
    for ax in axes:
        if target_node in nodes:
            nx.draw_networkx_nodes(sub_clean, pos, ax=ax, nodelist=[target_node],
                                   node_color='yellow', node_size=240, edgecolors='black')

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


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

if __name__ == "__main__":
    pass
