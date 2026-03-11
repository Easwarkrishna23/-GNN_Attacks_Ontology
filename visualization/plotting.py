import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_robustness_curves(budgets, accuracies, attack_names, title="Robustness Curves"):
    """
    Plot Accuracy vs Perturbation Budget.
    """
    plt.figure(figsize=(10, 6))
    for i, attack in enumerate(attack_names):
        plt.plot(budgets, accuracies[i], marker='o', label=attack)
        
    plt.title(title)
    plt.xlabel("Perturbation Budget")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_attack_comparison(results_df):
    """
    Bar plot comparing metrics across attacks.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Attack', y='Accuracy')
    plt.title("Impact of Different Attacks on Model Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.tight_layout()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tsne_embeddings(embeddings, labels, save_path, title):
    emb = np.asarray(embeddings)
    y = np.asarray(labels)
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    z = tsne.fit_transform(emb)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, s=8, cmap="tab10", alpha=0.8)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_layer_output_panel(layers, titles, save_path, max_nodes=60, max_features=60, cmap="viridis"):
    """
    Visualize layer-wise outputs as heatmaps for quick inspection.
    Each layer is shown as a node x feature matrix (truncated for readability).
    """
    if len(layers) != len(titles):
        raise ValueError("layers and titles must have the same length")

    def to_2d(arr):
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a[:, None]
        elif a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        return a

    n = len(layers)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, (layer, title) in enumerate(zip(layers, titles)):
        arr = to_2d(layer)
        arr = arr[:max_nodes, :max_features]
        im = axes[i].imshow(arr, aspect="auto", cmap=cmap)
        axes[i].set_title(title)
        axes[i].set_xlabel("Feature Index")
        axes[i].set_ylabel("Node Index")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    pass
