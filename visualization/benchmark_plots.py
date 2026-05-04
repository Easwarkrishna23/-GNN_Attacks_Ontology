from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


CORA_CMAP = plt.cm.get_cmap("tab10", 7)


def plot_tsne_states(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    out_dir: str,
    seed: int = 42,
    perplexity: float = 30.0,
):
    """
    Create 4 t-SNE plots with consistent colormap and consistent 2D frame by fitting on stacked embeddings.
    States expected keys:
      clean, attacked, defended_svd, defended_ontology
    """
    required = ["clean", "attacked", "defended_svd", "defended_ontology"]
    for k in required:
        if k not in embeddings:
            raise ValueError(f"Missing embedding state: {k}")

    os.makedirs(out_dir, exist_ok=True)

    # Fit a joint t-SNE space for visual comparability.
    order = required
    mats = [np.asarray(embeddings[k], dtype=np.float32) for k in order]
    n = mats[0].shape[0]
    stacked = np.vstack(mats)

    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    z = tsne.fit_transform(stacked)
    split = np.array_split(z, len(order), axis=0)

    titles = {
        "clean": "Clean",
        "attacked": "Attacked",
        "defended_svd": "Defended (SVD)",
        "defended_ontology": "Defended (Ontology)",
    }

    # Shared axis bounds for faithful comparison.
    x_min, x_max = float(z[:, 0].min()), float(z[:, 0].max())
    y_min, y_max = float(z[:, 1].min()), float(z[:, 1].max())

    out_paths = {}
    for state, coords in zip(order, split):
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=CORA_CMAP, s=10, alpha=0.85, linewidths=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t-SNE Latent Embeddings: {titles[state]}")
        path = os.path.join(out_dir, f"tsne_{state}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        out_paths[state] = path

    return out_paths
