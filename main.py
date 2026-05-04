from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_gnn_project")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import torch

from datasets.dynamic_dataset import simulate_dynamic_snapshots
from datasets.static_dataset import load_cora, set_global_seed
from evaluation.pipeline import DEFAULT_ATTACK_BUDGETS, ExperimentConfig, run_benchmark_for_data
from ontology.build_semantic_ontology import build_semantic_ontology
from visualization.benchmark_outputs import plot_accuracy_comparison, plot_graph_triplet, plot_tsne_states


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def _clear_previous_outputs(results_dir: str) -> None:
    if not os.path.isdir(results_dir):
        return
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                os.remove(path)
            except OSError:
                pass


def _format_for_terminal(df: pd.DataFrame, cols: List[str]) -> str:
    if df.empty:
        return "<empty>"
    view = df.copy()
    for col in view.select_dtypes(include=["float", "float64", "float32"]).columns:
        view[col] = view[col].map(lambda x: f"{x:.4f}")
    return view[cols].to_string(index=False)


def _identify_most_harmful_attack(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    attack_drop = df.groupby("Attack", as_index=False)["AccuracyDrop"].mean()
    worst = attack_drop.sort_values("AccuracyDrop", ascending=False).iloc[0]
    return str(worst["Attack"])


def _prepare_attack_budgets(args) -> Dict[str, float]:
    budgets = DEFAULT_ATTACK_BUDGETS.copy()
    budgets["Metattack"] = float(args.budget_metattack)
    budgets["Nettack"] = float(args.budget_nettack)
    budgets["RandomStructure"] = float(args.budget_random)
    budgets["FeaturePerturbation"] = float(args.budget_feature)
    budgets["EdgeFlip"] = float(args.budget_edgeflip)
    budgets["GradientBased"] = float(args.budget_gradient)
    return budgets


def _dynamic_summary(df_dynamic: pd.DataFrame) -> pd.DataFrame:
    if df_dynamic.empty:
        return pd.DataFrame()

    group_cols = ["Model", "Attack", "Defense"]
    agg_cols = [
        "CleanAccuracy",
        "AttackedAccuracy",
        "DefendedAccuracy",
        "AttackedPrecision",
        "AttackedRecall",
        "AttackedF1",
        "AttackedROCAUC",
        "DefendedPrecision",
        "DefendedRecall",
        "DefendedF1",
        "DefendedROCAUC",
        "RobustnessAttacked",
        "RobustnessDefended",
        "AccuracyDrop",
        "AccuracyRecovery",
        "RecoveryRate",
        "ASR",
        "HomophilyRecoveryRatio",
        "DensityAttacked",
        "ConductanceAttacked",
        "ModularityAttacked",
        "DensityDefended",
        "ConductanceDefended",
        "ModularityDefended",
    ]
    return df_dynamic.groupby(group_cols, as_index=False)[agg_cols].mean()


def _render_visual_outputs(
    labels: np.ndarray,
    static_tsne: Dict[str, Dict[str, np.ndarray]],
    static_graph: Dict[str, Dict[str, object]],
    static_df: pd.DataFrame,
):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # t-SNE for worst-attack state per model.
    for model_name, emb_map in static_tsne.items():
        out_path = os.path.join(PLOTS_DIR, f"tsne_static_{model_name.lower()}.png")
        plot_tsne_states(embeddings=emb_map, labels=labels, out_path=out_path)

    # Graph clean/attacked/defended triplets for each attack per model.
    for model_name, attack_map in static_graph.items():
        for attack_name, info in attack_map.items():
            out_path = os.path.join(PLOTS_DIR, f"graph_triplet_static_{model_name.lower()}_{attack_name.lower()}.png")
            plot_graph_triplet(
                clean_data=info["clean_data"],
                attacked_data=info["attacked_data"],
                defended_data=info["defended_data"],
                labels=labels,
                out_path=out_path,
                title_prefix=f"{model_name} - {attack_name}",
            )

    # Accuracy comparison line-chart (clean/attacked/defended).
    plot_accuracy_comparison(static_df, os.path.join(PLOTS_DIR, "accuracy_comparison_static.png"))


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(args.data_root, "dynamic"), exist_ok=True)

    # Requirement: remove existing generated outputs first.
    _clear_previous_outputs(RESULTS_DIR)

    set_global_seed(args.seed)
    dataset, static_data = load_cora(root=args.data_root, seed=args.seed)
    ontology_path = build_semantic_ontology(static_data, args.ontology_path)

    attack_budgets = _prepare_attack_budgets(args)

    static_config = ExperimentConfig(
        seed=args.seed,
        epochs_clean=args.epochs_clean,
        epochs_poison=args.epochs_poison,
        lr=args.lr,
        weight_decay=args.weight_decay,
        attack_budgets=attack_budgets,
    )

    dynamic_config = ExperimentConfig(
        seed=args.seed,
        epochs_clean=max(80, args.epochs_clean - 60),
        epochs_poison=max(70, args.epochs_poison - 60),
        lr=args.lr,
        weight_decay=args.weight_decay,
        attack_budgets=attack_budgets,
    )

    # Static benchmark.
    static_df, static_tsne, static_graph, static_layerwise = run_benchmark_for_data(
        data=static_data,
        dataset_name="Cora_Static",
        ontology_path=ontology_path,
        config=static_config,
        models=["GCN", "GAT"],
    )

    # Dynamic benchmark across snapshots.
    snapshots = simulate_dynamic_snapshots(
        static_data,
        num_snapshots=args.dynamic_snapshots,
        edge_change_rate=args.dynamic_edge_change_rate,
        seed=args.seed,
    )
    dynamic_rows = []
    for i, snap in enumerate(snapshots):
        torch.save(snap, os.path.join(args.data_root, "dynamic", f"cora_snapshot_t{i+1}.pt"))
        snap_df, _tsne, _graph, _layerwise = run_benchmark_for_data(
            data=snap,
            dataset_name=f"Cora_Dynamic_t{i+1}",
            ontology_path=ontology_path,
            config=dynamic_config,
            models=["GCN", "GAT"],
        )
        dynamic_rows.append(snap_df)

    dynamic_df = pd.concat(dynamic_rows, ignore_index=True) if dynamic_rows else pd.DataFrame()
    dynamic_summary_df = _dynamic_summary(dynamic_df)

    # Save all result tables.
    static_path = os.path.join(RESULTS_DIR, "results_static.csv")
    dynamic_path = os.path.join(RESULTS_DIR, "results_dynamic.csv")
    dynamic_summary_path = os.path.join(RESULTS_DIR, "results_dynamic_summary.csv")

    static_df.to_csv(static_path, index=False)
    dynamic_df.to_csv(dynamic_path, index=False)
    dynamic_summary_df.to_csv(dynamic_summary_path, index=False)

    attack_summary = (
        static_df[["Dataset", "Model", "Attack", "CleanAccuracy", "AttackedAccuracy", "AccuracyDrop", "ASR"]]
        .drop_duplicates(subset=["Dataset", "Model", "Attack"])
        .sort_values(["Model", "AccuracyDrop"], ascending=[True, False])
    )
    defense_summary = (
        static_df[
            [
                "Dataset",
                "Model",
                "Attack",
                "Defense",
                "AttackedAccuracy",
                "DefendedAccuracy",
                "AccuracyRecovery",
                "RecoveryRate",
                "DefendedF1",
                "DefendedROCAUC",
            ]
        ]
        .sort_values(["Model", "Attack", "Defense"])
    )
    attack_summary.to_csv(os.path.join(RESULTS_DIR, "table_attack_vs_accuracy.csv"), index=False)
    defense_summary.to_csv(os.path.join(RESULTS_DIR, "table_defense_vs_recovery.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "layerwise_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(static_layerwise, f, indent=2)

    # Render required plots.
    labels = static_data.y.cpu().numpy()
    _render_visual_outputs(labels, static_tsne, static_graph, static_df)

    # Final terminal report.
    pre_def_cols = [
        "Dataset",
        "Model",
        "Attack",
        "CleanAccuracy",
        "AttackedAccuracy",
        "AccuracyDrop",
        "ASR",
    ]
    post_def_cols = [
        "Dataset",
        "Model",
        "Attack",
        "Defense",
        "DefendedAccuracy",
        "AccuracyRecovery",
        "RecoveryRate",
        "DefendedF1",
        "DefendedROCAUC",
        "RobustnessDefended",
        "HomophilyRecoveryRatio",
        "DensityDefended",
        "ConductanceDefended",
        "ModularityDefended",
    ]

    pre_def_static = (
        static_df[["Dataset", "Model", "Attack", "CleanAccuracy", "AttackedAccuracy", "AccuracyDrop", "ASR"]]
        .drop_duplicates(subset=["Dataset", "Model", "Attack"])
        .sort_values(["Model", "AccuracyDrop"], ascending=[True, False])
    )

    post_def_static = static_df[post_def_cols].sort_values(["Model", "Attack", "Defense"])
    most_harmful = _identify_most_harmful_attack(pre_def_static)

    pre_final_path = os.path.join(RESULTS_DIR, "final_pre_defense.csv")
    post_final_path = os.path.join(RESULTS_DIR, "final_post_defense.csv")
    pre_def_static.to_csv(pre_final_path, index=False)
    post_def_static.to_csv(post_final_path, index=False)

    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append("FINAL STATIC RESULTS: PRE-DEFENSE")
    report_lines.append("=" * 120)
    report_lines.append(_format_for_terminal(pre_def_static, pre_def_cols))
    report_lines.append("")
    report_lines.append(f"The most impactful Attack is : {most_harmful}")
    report_lines.append("")
    report_lines.append("=" * 120)
    report_lines.append("FINAL STATIC RESULTS: POST-DEFENSE")
    report_lines.append("=" * 120)
    report_lines.append(_format_for_terminal(post_def_static, post_def_cols))
    report_lines.append("")

    if not dynamic_summary_df.empty:
        report_lines.append("=" * 120)
        report_lines.append("DYNAMIC DATASET SUMMARY (AVG OVER SNAPSHOTS)")
        report_lines.append("=" * 120)
        dyn_cols = [
            "Model",
            "Attack",
            "Defense",
            "AttackedAccuracy",
            "DefendedAccuracy",
            "AccuracyRecovery",
            "DefendedF1",
            "RobustnessDefended",
        ]
        report_lines.append(_format_for_terminal(dynamic_summary_df, dyn_cols))

    report = "\n".join(report_lines)
    print(report)

    terminal_file = os.path.join(RESULTS_DIR, "metrics_terminal.txt")
    with open(terminal_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved files:")
    print(f"- {static_path}")
    print(f"- {dynamic_path}")
    print(f"- {dynamic_summary_path}")
    print(f"- {pre_final_path}")
    print(f"- {post_final_path}")
    print(f"- {os.path.join(RESULTS_DIR, 'table_attack_vs_accuracy.csv')}")
    print(f"- {os.path.join(RESULTS_DIR, 'table_defense_vs_recovery.csv')}")
    print(f"- {os.path.join(RESULTS_DIR, 'layerwise_outputs.json')}")
    print(f"- {terminal_file}")
    print(f"- {os.path.join(PLOTS_DIR, 'accuracy_comparison_static.png')}")
    print("\nRun command:")
    print("python3 main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attacks and Defense Mechanisms in GNNs (Cora static + dynamic benchmark)")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--ontology-path", type=str, default="ontology/gnn_security.owl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-clean", type=int, default=160)
    parser.add_argument("--epochs-poison", type=int, default=140)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--budget-metattack", type=float, default=420)
    parser.add_argument("--budget-nettack", type=float, default=8)
    parser.add_argument("--budget-random", type=float, default=520)
    parser.add_argument("--budget-feature", type=float, default=0.22)
    parser.add_argument("--budget-edgeflip", type=float, default=420)
    parser.add_argument("--budget-gradient", type=float, default=0.16)

    parser.add_argument("--dynamic-snapshots", type=int, default=3)
    parser.add_argument("--dynamic-edge-change-rate", type=float, default=0.02)

    args = parser.parse_args()
    main(args)
