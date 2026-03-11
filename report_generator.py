from pathlib import Path
import pandas as pd


def generate_report(results_csv, output_path):
    df = pd.read_csv(results_csv)
    baseline = df[df["Attack"] == "Baseline"].iloc[0]
    attacked = df[df["Attack"] != "Baseline"].copy()
    attacked["acc_drop"] = baseline["Accuracy"] - attacked["Accuracy"]
    worst = attacked.sort_values("acc_drop", ascending=False).iloc[0]

    lines = []
    lines.append("# Adversarial Attacks on GNNs: Experimental Report")
    lines.append("")
    lines.append("## Mathematical Summary")
    lines.append("- GCN layer: H^(l+1) = sigma(D^(-1/2) A_hat D^(-1/2) H^(l) W^(l))")
    lines.append("- GAT layer: H^(l+1)_i = ||_k sigma(sum_j alpha_ij^k W^k h_j)")
    lines.append("- FGSM-like feature attack: X_adv = X + epsilon * sign(grad_X L)")
    lines.append("- Feature smoothing defense: X_smooth = alpha X + (1-alpha) A_hat X")
    lines.append("- Ontology defense term: H = A_hat XW + lambda OX")
    lines.append("")
    lines.append("## Code Walkthrough")
    lines.append("- `models/gcn.py`: 2-layer GCN with debug tensors for pre/post aggregation and softmax.")
    lines.append("- `models/gat.py`: 2-layer GAT with multi-head attention and extracted attention weights.")
    lines.append("- `attacks/`: poisoning attacks modify train graph/features; evasion attacks modify inference inputs.")
    lines.append("- `defenses/`: denoising + consistency regularization and ontology-based semantic reweighting.")
    lines.append("- `main.py`: trains baselines, runs 6 attacks, computes metrics, generates plots and CSV tables.")
    lines.append("")
    lines.append("## Impact Analysis")
    lines.append(f"- Baseline accuracy: {baseline['Accuracy']:.4f}")
    lines.append(f"- Most impactful attack: {worst['Attack']} (accuracy drop {worst['acc_drop']:.4f})")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("- Attack and defense performance are stored in `results/final_evaluation_table.csv`.")
    lines.append("- Visual diagnostics include graph comparisons, robustness curves, confusion matrices, and t-SNE.")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = generate_report("results/final_evaluation_table.csv", "results/final_report.md")
    print(f"Report written to {out}")

