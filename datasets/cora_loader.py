from datasets.planetoid_loader import load_planetoid


def load_cora(root="data"):
    # Backward-compatible wrapper.
    return load_planetoid("Cora", root=root)


if __name__ == "__main__":
    ds, data = load_cora()
    print(f"num_nodes={data.num_nodes} num_edges={data.num_edges} num_features={data.num_features} num_classes={ds.num_classes}")

