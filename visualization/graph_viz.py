import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def visualize_graph_mosaic(clean_adj, attacked_adj, defended_adj, labels, target_node=None, save_path='results/graph_comparison.png'):
    """
    Plot Clean, Attacked, and Defended graphs side-by-side using a subgraph.
    """
    adjs = [clean_adj, attacked_adj, defended_adj]
    titles = ["Clean Graph", "Attacked Graph", "Post-Defense Graph"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # If no target node, pick node 0
    if target_node is None:
        target_node = 0
        
    for i, adj in enumerate(adjs):
        if sp.issparse(adj):
            G = nx.from_scipy_sparse_array(adj) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(adj)
        else:
            G = nx.from_numpy_array(adj)
            
        # Get 2-hop neighborhood of target node for better visibility
        nodes = {target_node}
        for n in G.neighbors(target_node):
            nodes.add(n)
            for nn in G.neighbors(n):
                nodes.add(nn)
        
        subG = G.subgraph(nodes)
        pos = nx.spring_layout(subG, seed=42)
        
        nx.draw(subG, pos, ax=axes[i], node_color=labels[list(subG.nodes())], 
                node_size=100, cmap=plt.cm.Set1, with_labels=False, edge_color='gray', alpha=0.6)
        
        # Highlight target node
        if target_node in subG:
            nx.draw_networkx_nodes(subG, pos, ax=axes[i], nodelist=[target_node], 
                                   node_color='yellow', node_size=200, edgecolors='black')
            
        axes[i].set_title(titles[i])
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

if __name__ == "__main__":
    pass
