from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import re


def _fmt_float(x: float) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
    except Exception:
        pass
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def df_to_latex(df: pd.DataFrame, *, caption: str, label: str, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    latex = []
    latex.append(f"\\begin{{{env}}}[t]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append(df.to_latex(index=False, escape=True, longtable=False, bold_rows=False))
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\end{{{env}}}")
    return "\n".join(latex)


def write_table(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results = repo_root / "results"
    out_dir = repo_root / "paper_ieee" / "tables"

    # Static: attacks (pre-defense) and defenses (post-defense)
    # Prefer the latest unified outputs (FINAL_TABLE_*.csv). Fall back to legacy per-model CSVs.
    pre_all_path = results / "FINAL_TABLE_PRE_DEFENSE.csv"
    post_all_path = results / "FINAL_TABLE_POST_DEFENSE.csv"

    for model in ["GCN", "GAT"]:
        if pre_all_path.exists():
            pre_all = pd.read_csv(pre_all_path)
            pre = pre_all[pre_all["Model"].astype(str).str.upper() == model].copy()
        else:
            pre = pd.read_csv(results / f"final_pre_defense_{model.lower()}.csv")
        pre_cols = [
            "Attack",
            "Accuracy",
            "Macro F1",
            "ROC-AUC",
            "Perturbation Budget",
            "Accuracy Drop",
        ]
        pre = pre[pre_cols].copy()
        for c in pre_cols[1:]:
            pre[c] = pre[c].map(_fmt_float)
        write_table(
            out_dir / f"static_{model.lower()}_attacks.tex",
            df_to_latex(
                pre,
                caption=f"Static (Planetoid) attack suite results for {model} (before defenses).",
                label=f"tab:static_{model.lower()}_attacks",
                wide=True,
            ),
        )

        if post_all_path.exists():
            post_all = pd.read_csv(post_all_path)
            post = post_all[post_all["Model"].astype(str).str.upper() == model].copy()
        else:
            post = pd.read_csv(results / f"final_post_defense_{model.lower()}.csv")
        post_cols = [
            "Attack",
            "Accuracy",
            "Macro F1",
            "ROC-AUC",
            "Perturbation Budget",
        ]
        post = post[post_cols].copy()
        for c in post_cols[1:]:
            post[c] = post[c].map(_fmt_float)
        write_table(
            out_dir / f"static_{model.lower()}_defenses.tex",
            df_to_latex(
                post,
                caption=f"Static (Planetoid) post-attack defenses for {model} on the most impactful attack.",
                label=f"tab:static_{model.lower()}_defenses",
                wide=True,
            ),
        )

    # Graph metrics (optional depending on run profile)
    gm_static_path = results / "graph_metrics_static.csv"
    if gm_static_path.exists():
        gm_static = pd.read_csv(gm_static_path)
        for c in ["Density", "Modularity", "Conductance"]:
            if c in gm_static.columns:
                gm_static[c] = gm_static[c].map(_fmt_float)
        write_table(
            out_dir / "graph_metrics_static.tex",
            df_to_latex(
                gm_static,
                caption="Static graph structural metrics (computed on label-induced communities).",
                label="tab:graph_metrics_static",
                wide=False,
            ),
        )

    gm_dyn_path = results / "graph_metrics_dynamic.csv"
    if gm_dyn_path.exists():
        gm_dyn = pd.read_csv(gm_dyn_path)
        for c in ["Density", "Modularity", "Conductance"]:
            if c in gm_dyn.columns:
                gm_dyn[c] = gm_dyn[c].map(_fmt_float)
        write_table(
            out_dir / "graph_metrics_dynamic.tex",
            df_to_latex(
                gm_dyn,
                caption="Dynamic graph structural metrics (computed on label-induced communities).",
                label="tab:graph_metrics_dynamic",
                wide=False,
            ),
        )

    # Dynamic summary: optional (not always generated in paper profile)
    dyn_path = results / "dynamic_gcn_evaluation_table.csv"
    if not dyn_path.exists():
        dyn = None
    else:
        dyn = pd.read_csv(dyn_path)
    if dyn is not None:
        keep = [
            "Baseline",
            "Poisoning: Random Structure",
            "Poisoning: Meta Attack",
            "Evasion: Edge Flip",
            "Evasion: Feature",
            "Evasion: Gradient (FGSM-like)",
        ]
        dyn_pick = dyn[dyn["Attack"].isin(keep)].copy()
        dyn_pick = dyn_pick[["Attack", "Accuracy", "F1", "ROC-AUC", "Log-loss", "Perturbation Budget"]].copy()
        for c in ["Accuracy", "F1", "ROC-AUC", "Log-loss", "Perturbation Budget"]:
            dyn_pick[c] = dyn_pick[c].map(_fmt_float)
        write_table(
            out_dir / "dynamic_gcn_summary.tex",
            df_to_latex(
                dyn_pick,
                caption="Dynamic graph results (compact summary) for GCN.",
                label="tab:dynamic_gcn_summary",
                wide=True,
            ),
        )

    # Dynamic defense sweep (optional)
    if dyn is not None:
        sweep_rows = []
        for _, r in dyn.iterrows():
            attack_name = str(r.get("Attack", ""))
            if "Defense: Feature Smoothing" in attack_name:
                m = re.search(r"alpha=([0-9.]+)", attack_name)
                setting = f"$\\alpha={m.group(1)}$" if m else ""
                sweep_rows.append({"Defense": "Feature Smoothing", "Setting": setting, "Accuracy": r.get("Accuracy", ""), "F1": r.get("F1", "")})
            if "Defense: Ontology" in attack_name:
                m = re.search(r"lambda=([0-9.]+)", attack_name)
                setting = f"$\\lambda={m.group(1)}$" if m else ""
                sweep_rows.append({"Defense": "Ontology", "Setting": setting, "Accuracy": r.get("Accuracy", ""), "F1": r.get("F1", "")})
        if sweep_rows:
            sweep = pd.DataFrame(sweep_rows)
            for c in ["Accuracy", "F1"]:
                sweep[c] = sweep[c].map(_fmt_float)
            write_table(
                out_dir / "dynamic_defense_sweep.tex",
                df_to_latex(
                    sweep,
                    caption="Dynamic graph defense sweep for GCN (selected hyperparameter settings).",
                    label="tab:dynamic_defense_sweep",
                    wide=True,
                ),
            )

    print(f"Wrote LaTeX tables to: {out_dir}")

    # Semantic logic ablations + defense benchmark tables
    ab_path = results / "SEMANTIC_ABLATION.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        for c in ["GCN_Accuracy", "GAT_Accuracy"]:
            if c in ab.columns:
                ab[c] = ab[c].map(_fmt_float)
        write_table(
            out_dir / "semantic_ablation.tex",
            df_to_latex(
                ab,
                caption="Semantic logic ablation study (ontology variants).",
                label="tab:semantic_ablation",
                wide=False,
            ),
        )

    bench_path = results / "DEFENSE_BENCHMARK.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        for c in ["GCN", "GAT"]:
            if c in bench.columns:
                bench[c] = bench[c].map(_fmt_float)
        write_table(
            out_dir / "defense_benchmark.tex",
            df_to_latex(
                bench,
                caption="Defense benchmark comparison across GCN and GAT (best variant per defense).",
                label="tab:defense_benchmark",
                wide=False,
            ),
        )


if __name__ == "__main__":
    main()
