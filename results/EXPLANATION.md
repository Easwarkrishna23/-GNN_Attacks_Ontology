# Adversarial Attacks on GNNs (GCN/GAT): Explanation and Output Guide

## How To Run (Clean Rerun)
```bash
python3 main.py --clean --profile paper
```

This deletes every previously generated file under `results/` and regenerates ONLY the final paper-style outputs.

## Dataset
- Static dataset: **Cora** citation network.
- Dynamic dataset: synthetic evolving snapshots stored under `data/dynamic/`.
- Cora stats: nodes=2708, edges=10556, features=1433, classes=7.

## Models
### GCN
- Normalize: `Â = A + I`, `S = D̂^{-1/2} Â D̂^{-1/2}`.
- Layer 1: `H^{(1)} = ReLU(S X W^{(0)})`.
- Layer 2: `Z = Softmax(S H^{(1)} W^{(1)})`.

### GAT
- Attention per head: `e_{ij} = a^T[Wh_i || Wh_j]`, `α_{ij} = softmax_j(LeakyReLU(e_{ij}))`.
- Layer 1 (multi-head): `h_i^{(1)} = ||_k Σ_{j∈N(i)} α_{ij}^k W^k h_j` (ELU).
- Layer 2: aggregate -> logits -> Softmax.

## Attacks (3 Poisoning + 3 Evasion)
### Poisoning vs Evasion
- Poisoning: modifies training data/graph; model is trained on poisoned input.
- Evasion: modifies inference-time input only; training graph is untouched.

### Per-Attack Explanation With One Concrete Cora Datapoint
#### Poisoning: Random Structure
- implementation file: `attacks/poisoning/random_poison.py`
- what changes: changes A (edges) and lightly corrupts X, then retrains
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 4
  - conf_clean -> conf_attacked: 0.2018 -> 0.2126
  - margin_clean -> margin_attacked: 0.0182 -> 0.0052
  - edges_added_sample: [1703]
  - edges_removed_sample: []
  - perturbation budget used: 800

#### Poisoning: Nettack
- implementation file: `attacks/poisoning/nettack.py`
- what changes: changes A near target nodes, then retrains
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 0
  - conf_clean -> conf_attacked: 0.2018 -> 0.2606
  - margin_clean -> margin_attacked: 0.0182 -> 0.0895
  - edges_added_sample: [1281, 974, 369, 1819, 2332, 2109]
  - edges_removed_sample: []
  - perturbation budget used: 48

#### Poisoning: Meta Attack
- implementation file: `attacks/poisoning/metattack.py`
- what changes: changes A globally (proxy Metattack), then retrains
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 1
  - conf_clean -> conf_attacked: 0.2018 -> 0.2188
  - margin_clean -> margin_attacked: 0.0182 -> 0.0583
  - edges_added_sample: []
  - edges_removed_sample: []
  - perturbation budget used: 800

#### Evasion: Edge Flip
- implementation file: `attacks/evasion/structure_evasion.py`
- what changes: changes A at inference only (model fixed)
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 0
  - conf_clean -> conf_attacked: 0.2018 -> 0.2715
  - margin_clean -> margin_attacked: 0.0182 -> 0.0936
  - edges_added_sample: [2108, 2333]
  - edges_removed_sample: [467, 1358]
  - perturbation budget used: 120

#### Evasion: Feature
- implementation file: `attacks/evasion/feature_evasion.py`
- what changes: changes X at inference only (A unchanged, model fixed)
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 4
  - conf_clean -> conf_attacked: 0.2018 -> 0.3400
  - margin_clean -> margin_attacked: 0.0182 -> 0.1843
  - x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
  - x_attacked_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1378450244665146, 0.0, 0.0]
  - perturbation budget used: 18

#### Evasion: Gradient (FGSM-like)
- implementation file: `attacks/evasion/fgsm_like.py`
- what changes: changes X using gradient sign at inference only (A unchanged, model fixed)
- why it creates the issue: GCN/GAT aggregate neighbor signals; perturbing A or X corrupts the neighborhood messages and shrinks the classification margin.
- example datapoint:
  - target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 1
  - conf_clean -> conf_attacked: 0.2018 -> 1.0000
  - margin_clean -> margin_attacked: 0.0182 -> 1.0000
  - x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
  - x_attacked_first10: [0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.0, 0.10000000149011612, 0.15000000596046448, 0.10000000149011612, 0.10000000149011612]
  - perturbation budget used: 0.1


## Selecting The Worst Attack
- We compute test accuracy for every attack (GCN).
- `Accuracy Drop = Accuracy(Baseline) - Accuracy(Attack)`.
- The attack with maximum accuracy drop is reported in the terminal as:
  - `The most impactful Attack is : <name> (accuracy drop=...)`
- In this run: worst attack = `Evasion: Gradient (FGSM-like)`.

## Defenses (Evaluated Individually Then Combined)
We apply defenses only against the single most impactful attack.

### 1) Base Defense: Feature Smoothing
- `X_s = α X + (1-α) Â X`
- This reduces high-frequency feature noise that adversarial perturbations introduce.

### 2) Pruning Defense
- Keep top-k most similar neighbors per node (feature cosine similarity).
- Removes suspicious/dissimilar edges that can amplify adversarial messages.

### 3) Ontology Defense (Semantic Similarity)
- Build ontology similarity matrix `O` from cosine similarity of node feature vectors.
- Feature projection: `X' = X + λ OX`.
- Adjacency reweight: `A' = clip(A + λ O, 0, 1)` (then symmetrize + add self-loops).

### 4) Combined Defense
- Ontology projection -> ontology adjacency reweight -> pruning.
- We evaluate base, pruning, ontology, combined; then select the best by accuracy.

## Ontologies Created (Files)
- `results/ontologies/ontology_topk_edges.csv`: top-k neighbors for every node, for each ontology variant.
- `results/ontologies/ontology_examples.md`: a small example ontology neighborhood for one target node.
- `results/ontologies/ontology_summary.md`: what each ontology variant means.

## Outputs Explained
### Tables (ONLY final tables)
- `results/FINAL_TABLE_PRE_DEFENSE.csv`: baseline + all attacks (GCN and GAT).
- `results/FINAL_TABLE_POST_DEFENSE.csv`: baseline + worst attack + defenses (GCN and GAT).
- `results/metrics_terminal.txt`: CSV-form versions of the above tables (no ANSI).

### Figures (paper-style)
- `results/FIG_workflow.png`: entire workflow of the project.
- `results/FIG_gcn_layerwise.png`: detailed GCN layer-wise math and outputs.
- `results/FIG_gat_layerwise.png`: detailed GAT attention math and outputs.
- `results/FIG_attack_defense_flow.png`: clean -> attack -> defense flow.
- `results/FIG_attack_suite.png`: dataset change visualization for each attack.
- `results/FIG_graph_diff_worst.png`: clean vs worst-attacked vs defended graph diffs.
- `results/FIG_class_clusters.png`: class clusters for clean/attacked/defended embeddings.

## Real-World Relevance
- Citation graphs: attackers can inject fake citations or distort paper text features to misclassify topics.
- Social/fraud graphs: injected edges/features can hide malicious nodes; pruning + ontology reduce attacker leverage.

## Run Summary
- GCN baseline accuracy: 0.821
- GAT baseline accuracy: 0.805
- Most impactful attack (GCN): Evasion: Gradient (FGSM-like) (drop=0.818)
- Best defense (GCN): Defense: Pruning+Ontology + Retrain
- Best defense (GAT): Defense: Pruning+Ontology + Retrain
- Attacks are selected at a moderate intensity so defenses can show visible recovery (paper-style).
