# Adversarial Attacks on GNNs (GCN/GAT): Explanation and Output Guide

## Beginner-Friendly Walkthrough (Read This First)
This project is a simple story:
1. Train GCN and GAT on a clean graph dataset (baseline).
2. Apply attacks that modify edges (graph structure) or features (node words).
3. Apply defenses to reduce the damage and re-test node classification.

Key idea:
- GNNs do neighborhood mixing. If neighbors or features are tampered, the mixed signal becomes wrong.

## How To Run
```bash
python3 main.py --clean --profile paper --dataset Cora
```

## Datasets
- Static datasets: Cora, Citeseer, PubMed (citation networks).
- Dynamic dataset: synthetic evolving snapshots saved under `data/dynamic/`.
- Dataset stats: nodes=2708, edges=10556, features=1433, classes=7.

### What Are X and A?
- `X` (features): each node has a feature vector (in Planetoid datasets: word/term features).
- `A` (adjacency): edges (citations) telling which node is connected to which.

## Models (2-Layer Versions)
### GCN
- Adds self-loops so each node also sees itself.
- Normalizes by degree so high-degree nodes do not dominate.
- Layer 1: mix neighbors + linear weights + ReLU.
- Layer 2: mix again + output a probability for each class.

### GAT
- Same goal as GCN, but it learns attention weights (which neighbors matter more).
- Multi-head attention in layer 1 improves stability.

## Attacks
Two categories:
- Poisoning: attacker changes training data; model is retrained on poisoned graph.
- Evasion: attacker changes only inference-time inputs; model weights are fixed.

### Per-Attack Example (One Real Node)
#### Poisoning: Random Structure
- code: `attacks/poisoning/random_poison.py`
- what changes: changes A (edges) and slightly corrupts X, then retrains
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 4
  - confidence: 0.2018 -> 0.2126
  - margin: 0.0182 -> 0.0052
  - sample edges added: [1703]
  - sample edges removed: []

#### Poisoning: Nettack
- code: `attacks/poisoning/nettack.py`
- what changes: changes A near target nodes, then retrains
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 2
  - confidence: 0.2018 -> 0.2988
  - margin: 0.0182 -> 0.0555
  - sample edges added: [2496, 1281, 643, 326, 10, 974]
  - sample edges removed: []

#### Poisoning: Meta Attack
- code: `attacks/poisoning/metattack.py`
- what changes: changes A globally (proxy Metattack), then retrains
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 1
  - confidence: 0.2018 -> 0.2188
  - margin: 0.0182 -> 0.0583
  - sample edges added: []
  - sample edges removed: []

#### Evasion: Edge Flip
- code: `attacks/evasion/structure_evasion.py`
- what changes: changes A at inference only (model fixed)
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 0
  - confidence: 0.2018 -> 0.2715
  - margin: 0.0182 -> 0.0936
  - sample edges added: [2108, 2333]
  - sample edges removed: [467, 1358]

#### Evasion: Feature
- code: `attacks/evasion/feature_evasion.py`
- what changes: changes X at inference only (A unchanged, model fixed)
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 4
  - confidence: 0.2018 -> 0.3400
  - margin: 0.0182 -> 0.1843
  - first10 features (clean): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
  - first10 features (attacked): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1378450244665146, 0.0, 0.0]

#### Evasion: Gradient (FGSM-like)
- code: `attacks/evasion/fgsm_like.py`
- what changes: changes X using gradient sign (A unchanged, model fixed)
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 1
  - confidence: 0.2018 -> 1.0000
  - margin: 0.0182 -> 1.0000
  - first10 features (clean): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
  - first10 features (attacked): [0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.0, 0.10000000149011612, 0.15000000596046448, 0.10000000149011612, 0.10000000149011612]

#### Evasion: Adaptive Semantic
- code: `attacks/evasion/adaptive_semantic.py`
- what changes: changes X in a semantically plausible way to try to bypass pruning (model fixed)
- why it hurts: the model mixes neighbor information; corrupted neighbors/features distort embeddings.
- example node: target_node=1708 true_label=3
  - pred_clean -> pred_attacked: 1 -> 1
  - confidence: 0.2018 -> 0.9856
  - margin: 0.0182 -> 0.9782
  - sample edges added: []
  - sample edges removed: []


## Selecting the Most Harmful Attack
- We compute accuracy for every attack on the test set (GCN).
- Accuracy Drop = Accuracy(Baseline) - Accuracy(Attack).
- Worst attack (this run): `Evasion: Gradient (FGSM-like)`.

## Defenses
We defend ONLY the worst attack (so results are not cherry-picked).

### Smoothing
- Averages features with neighbors to remove noise.

### Similarity pruning / GNNGuard-like filtering
- Assign edge trust using similarity; remove or down-weight suspicious edges.

### GNN-SVD (low-rank filtering)
- Approximates adjacency with a low-rank matrix to remove structural noise.

### Ontology-guided semantic defense (main contribution)
Ontology defense is NOT the same as similarity pruning.
- It creates explicit semantic concepts (topics and subtopics), relations, and rules.
- It detects contradictions (impossible feature combinations).
- It repairs features and computes semantic edge trust for robust message passing.

Protégé exports:
- `results/ontologies/<Dataset>/ontology.owl`
- `results/ontologies/<Dataset>/ontology.ttl`
- `results/ontologies/<Dataset>/ontology.swrl`

## Outputs (What Proves What)
### Terminal tables
- PRE-DEFENSE table: shows attacks actually reduce node classification metrics.
- POST-DEFENSE table: shows defenses recover accuracy/F1 on the worst attack.

### Visual proof (graphs)
- `results/FIG_graph_diff_worst.png`: clean vs attacked vs defended (3 panels).
- `results/FIG_worst_clean.png`, `results/FIG_worst_attacked.png`, `results/FIG_worst_defended.png`: the same subgraph saved separately with aligned positions.
Legend in the images:
- red edges: edges added by attack
- black dashed edges: edges removed by attack
- green edges: edges added by defense
- blue dashed edges: edges removed by defense
- red rings: nodes whose features were changed by the attack
- green rings: nodes whose features were repaired/changed by the defense
- purple rings: nodes whose predicted class changed (node classification effect)
- black rings: misclassified nodes

### CSV tables saved
- `results/FINAL_TABLE_PRE_DEFENSE.csv`
- `results/FINAL_TABLE_POST_DEFENSE.csv`
- `results/SEMANTIC_ABLATION.csv` (co-occurrence only vs rules only vs full ontology)
- `results/DEFENSE_BENCHMARK.csv` (smoothing / GNNGuard-like / SVD / adversarial training / ontology / combined)

## Real-world relevance
- Citation graphs: fake citations or edited text features can misclassify papers.
- Fraud/social graphs: injected edges/features can hide malicious nodes; semantic constraints reduce attacker leverage.

## Run Summary
- GCN baseline accuracy: 0.821
- GAT baseline accuracy: 0.805
- Most impactful attack (GCN): Evasion: Gradient (FGSM-like) (drop=0.818)
- Best defense (GCN): Defense: Full Ontology + Pruning (k=15) + Retrain
- Best defense (GAT): Defense: Full Ontology + Pruning (k=15) + Retrain
- Attacks are selected at a moderate intensity so defenses can show visible recovery (paper-style).
