import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, text, fc="#ffffff", ec="#1f2937", lw=1.6, fs=10, align="center"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if ha == "center" else x + 0.02
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color="#111827")
    return p


def _arrow(ax, x0, y0, x1, y1, color="#2563eb", lw=1.8, style="-|>"):
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(a)
    return a


def draw_project_workflow(save_path, attacks, worst_attack, defenses):
    """
    Figure: end-to-end workflow (dataset -> train -> attack -> evaluate -> defend -> outputs).
    This is intentionally a clean "paper-style" block diagram (not AI-art).
    """
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        0.03,
        0.78,
        0.20,
        0.16,
        "Dataset\nCora (static)\n+ dynamic snapshots",
        fc="#eef2ff",
        fs=11,
    )
    _box(ax, 0.27, 0.78, 0.20, 0.16, "Train Baseline\n2-layer GCN\n2-layer GAT", fc="#ecfeff", fs=11)
    _box(
        ax,
        0.51,
        0.78,
        0.22,
        0.16,
        "Run Attacks\n(poisoning + evasion)\n" + "\n".join(f"- {a}" for a in attacks),
        fc="#fff7ed",
        fs=10,
        align="left",
    )
    _box(
        ax,
        0.77,
        0.78,
        0.20,
        0.16,
        "Evaluate Metrics\nAccuracy / F1 / ASR\nMargin Drop / Log-loss",
        fc="#f1f5f9",
        fs=11,
    )

    _arrow(ax, 0.23, 0.86, 0.27, 0.86)
    _arrow(ax, 0.47, 0.86, 0.51, 0.86)
    _arrow(ax, 0.73, 0.86, 0.77, 0.86)

    _box(
        ax,
        0.03,
        0.52,
        0.34,
        0.20,
        "Impact Analysis (GCN)\nPick attack with largest accuracy drop\nMost impactful: " + str(worst_attack),
        fc="#f8fafc",
        fs=11,
        align="left",
    )
    _box(
        ax,
        0.41,
        0.52,
        0.34,
        0.20,
        "Defenses Against Most Impactful Attack\n" + "\n".join(f"- {d}" for d in defenses),
        fc="#dcfce7",
        fs=10,
        align="left",
    )
    _box(
        ax,
        0.79,
        0.52,
        0.18,
        0.20,
        "Final Outputs\nPre-defense table\nPost-defense table\nPaper figures",
        fc="#f1f5f9",
        fs=11,
    )
    _arrow(ax, 0.87, 0.78, 0.20, 0.72, color="#0f766e", style="-|>")
    _arrow(ax, 0.37, 0.62, 0.41, 0.62, color="#0f766e")
    _arrow(ax, 0.75, 0.62, 0.79, 0.62, color="#0f766e")

    _box(
        ax,
        0.03,
        0.14,
        0.94,
        0.28,
        "Data Flow (high level)\n"
        "Clean graph: (A, X) -> model -> y_hat\n"
        "Attack: (A', X') shifts neighborhood aggregation -> embedding drift -> accuracy drop\n"
        "Defense: prune suspicious edges + ontology projection pulls features toward semantic neighbors -> accuracy recovers",
        fc="#ffffff",
        fs=11,
        align="left",
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_gcn_layerwise(save_path):
    """
    Figure: detailed layer-wise GCN diagram with formulas and intermediate outputs.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: inputs
    _box(ax, 0.03, 0.62, 0.18, 0.26, "Input Features\nX (N x F)", fc="#eef2ff", fs=12)
    _box(ax, 0.03, 0.28, 0.18, 0.26, "Adjacency\nA (N x N)", fc="#eef2ff", fs=12)

    # Preprocessing
    _box(
        ax,
        0.25,
        0.45,
        0.22,
        0.30,
        "Preprocess\nÂ = A + I\nD̂ = diag(Â 1)\nS = D̂^{-1/2} Â D̂^{-1/2}",
        fc="#f1f5f9",
        fs=12,
        align="left",
    )
    _arrow(ax, 0.21, 0.75, 0.25, 0.66)
    _arrow(ax, 0.21, 0.41, 0.25, 0.54)

    # Layer 1
    _box(
        ax,
        0.52,
        0.62,
        0.20,
        0.26,
        "GCN Layer 1\nLinear: X W^{(0)}\nAggregate: S (X W^{(0)})\nActivate: ReLU",
        fc="#fff7ed",
        fs=11,
        align="left",
    )
    _box(ax, 0.74, 0.66, 0.10, 0.18, "H^{(1)}\n(N x H)", fc="#ffffff", fs=12)
    _arrow(ax, 0.47, 0.60, 0.52, 0.72)
    _arrow(ax, 0.72, 0.75, 0.74, 0.75)

    # Layer 2
    _box(
        ax,
        0.52,
        0.28,
        0.20,
        0.26,
        "GCN Layer 2\nLinear: H^{(1)} W^{(1)}\nAggregate: S (H^{(1)} W^{(1)})\nSoftmax",
        fc="#fff7ed",
        fs=11,
        align="left",
    )
    _box(ax, 0.74, 0.32, 0.10, 0.18, "Z\n(N x C)", fc="#ffffff", fs=12)
    _arrow(ax, 0.79, 0.66, 0.62, 0.54, color="#2563eb")
    _arrow(ax, 0.72, 0.41, 0.74, 0.41)

    # Output
    _box(
        ax,
        0.86,
        0.28,
        0.12,
        0.60,
        "Node Classification\nŷ = argmax(Z)\nPer-node probabilities",
        fc="#ecfeff",
        fs=12,
        align="left",
    )
    _arrow(ax, 0.84, 0.41, 0.86, 0.58)
    _arrow(ax, 0.84, 0.75, 0.86, 0.70)

    ax.text(
        0.03,
        0.95,
        "2-Layer GCN: what goes in, what each layer computes, and what comes out",
        fontsize=14,
        weight="bold",
        color="#111827",
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_gat_layerwise(save_path, heads=4):
    """
    Figure: detailed layer-wise GAT diagram with attention math (multi-head).
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.62, 0.18, 0.26, "Input Features\nX (N x F)", fc="#eef2ff", fs=12)
    _box(ax, 0.03, 0.28, 0.18, 0.26, "Edges\nE (edge_index)", fc="#eef2ff", fs=12)

    _box(
        ax,
        0.25,
        0.45,
        0.22,
        0.30,
        "Attention (per head k)\n"
        "e_ij^k = a^{kT}[W^k h_i || W^k h_j]\n"
        "alpha_ij^k = softmax_j(LeakyReLU(e_ij^k))",
        fc="#f1f5f9",
        fs=11,
        align="left",
    )
    _arrow(ax, 0.21, 0.75, 0.25, 0.66)
    _arrow(ax, 0.21, 0.41, 0.25, 0.54)

    _box(
        ax,
        0.52,
        0.62,
        0.20,
        0.26,
        f"GAT Layer 1 ({heads} heads)\n"
        "h_i^(1) = concat_k sum_{j in N(i)} alpha_ij^k * W^k h_j\n"
        "Activation: ELU",
        fc="#dcfce7",
        fs=10,
        align="left",
    )
    _box(ax, 0.74, 0.66, 0.10, 0.18, "H^{(1)}\n(N x (H*K))", fc="#ffffff", fs=11)
    _arrow(ax, 0.47, 0.60, 0.52, 0.72)
    _arrow(ax, 0.72, 0.75, 0.74, 0.75)

    _box(
        ax,
        0.52,
        0.28,
        0.20,
        0.26,
        "GAT Layer 2 (1 head)\n"
        "z_i = sum_{j in N(i)} alpha_ij * W h_j\n"
        "Output: Softmax",
        fc="#dcfce7",
        fs=10,
        align="left",
    )
    _box(ax, 0.74, 0.32, 0.10, 0.18, "Z\n(N x C)", fc="#ffffff", fs=12)
    _arrow(ax, 0.79, 0.66, 0.62, 0.54, color="#2563eb")
    _arrow(ax, 0.72, 0.41, 0.74, 0.41)

    _box(
        ax,
        0.86,
        0.28,
        0.12,
        0.60,
        "Node Classification\nŷ = argmax(Z)\nPer-node probabilities",
        fc="#ecfeff",
        fs=12,
        align="left",
    )
    _arrow(ax, 0.84, 0.41, 0.86, 0.58)
    _arrow(ax, 0.84, 0.75, 0.86, 0.70)

    ax.text(
        0.03,
        0.95,
        "2-Layer GAT: attention coefficients, multi-head aggregation, and final classification",
        fontsize=14,
        weight="bold",
        color="#111827",
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_attack_defense_flow(save_path, worst_attack, defense_names):
    """
    Figure: clean -> attack -> defended (conceptual), emphasizing what changes (A, X)
    and where evaluation happens (train-time vs test-time).
    """
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.93,
        "Project Working: clean dataset -> attack introduces perturbations -> defenses mitigate -> metrics recover",
        fontsize=14,
        weight="bold",
        color="#111827",
    )

    _box(ax, 0.03, 0.58, 0.22, 0.26, "Clean Graph\n(A, X)\nTrain baseline GCN/GAT", fc="#eef2ff", fs=12)
    _box(
        ax,
        0.30,
        0.58,
        0.26,
        0.26,
        "Attack Module\nPoisoning: changes training graph\n(A', X') then retrain\n\nEvasion: changes inference input only\n(A', X') but baseline model fixed",
        fc="#fff7ed",
        fs=11,
        align="left",
    )
    _box(
        ax,
        0.61,
        0.58,
        0.18,
        0.26,
        "Attacked Evaluation\nAccuracy drop\nMargin drop / ASR",
        fc="#f1f5f9",
        fs=12,
    )
    _box(
        ax,
        0.82,
        0.58,
        0.15,
        0.26,
        "Most Impactful\nAttack\n" + str(worst_attack),
        fc="#fee2e2",
        fs=12,
    )

    _arrow(ax, 0.25, 0.71, 0.30, 0.71)
    _arrow(ax, 0.56, 0.71, 0.61, 0.71)
    _arrow(ax, 0.79, 0.71, 0.82, 0.71)

    _box(
        ax,
        0.03,
        0.15,
        0.30,
        0.32,
        "Defenses (evaluated individually)\n"
        + "\n".join(f"- {n}" for n in defense_names),
        fc="#dcfce7",
        fs=11,
        align="left",
    )
    _box(
        ax,
        0.37,
        0.15,
        0.30,
        0.32,
        "Combined Defense\nOntology projection -> pruning\nRe-evaluate (and retrain if poisoning)",
        fc="#bbf7d0",
        fs=12,
        align="left",
    )
    _box(
        ax,
        0.71,
        0.15,
        0.26,
        0.32,
        "Post-Defense Evaluation\nAccuracy improves vs attacked\nSelect best defense",
        fc="#f1f5f9",
        fs=12,
    )

    _arrow(ax, 0.89, 0.58, 0.18, 0.47, color="#0f766e")
    _arrow(ax, 0.33, 0.31, 0.37, 0.31, color="#0f766e")
    _arrow(ax, 0.67, 0.31, 0.71, 0.31, color="#0f766e")

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
