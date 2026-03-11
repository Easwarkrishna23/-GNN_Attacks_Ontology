# Adversarial Attacks on GNN Architecture (GCN and GAT) - Full Project Explanation

## 1) What your project is doing end-to-end
This project builds, attacks, evaluates, and defends node-classification GNN pipelines on:
- Static graph dataset: **Cora citation graph**
- Dynamic graph dataset: **synthetic evolving graph** (`DynamicGraphGenerator`)

It implements:
- Baselines: 2-layer **GCN** and 2-layer **GAT**
- Poisoning attacks (3): Random, Nettack, Meta Attack
- Evasion attacks (3): Structure, Feature, FGSM-like
- Defense for feature-focused evasion: Feature smoothing + Ontology-based defense
- Metrics and visuals: CSV tables, robustness curves, confusion matrices, t-SNE, graph mosaics, layer-wise outputs, layer-wise memory

---

## 2) 2-layer baseline GCN architecture (requested detailed understanding)
From `models/gcn.py`:
- Layer 1: `GCNConv(num_features -> hidden_dim=16)`
- Activation: `ReLU`
- Regularization: `Dropout`
- Layer 2: `GCNConv(hidden_dim -> num_classes)`
- Output: `log_softmax` over classes

### Mathematical form
For layer `l`:
\[
H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)}\right)
\]
Where:
- \(\hat{A}=A+I\) (self loops)
- \(\hat{D}\) degree of \(\hat{A}\)
- \(H^{(0)}=X\) (node features)
- \(\sigma\)=ReLU in layer 1, identity in final layer before softmax/log-softmax

### Expected output of each stage
- Pre-aggregation L1 (`XW0`): projected feature space
- Post-normalization L1 (`A_norm XW0`): neighborhood-mixed features
- Post-activation L1: nonlinear hidden representation
- Pre-aggregation L2 (`H1W1`): class-space projection
- Post-normalization L2: logits per class
- Final output: log-probabilities (or probabilities via `exp`)

### Layer-wise output + memory now implemented
`forward_with_debug()` now returns:
- Layer tensors (`pre_aggregation_l1`, `post_normalization_l1`, `post_activation_l1`, etc.)
- Probabilities and predictions
- Memory stats:
  - parameter memory
  - per-layer activation memory
  - total estimated memory

Saved report: `results/layerwise_debug_report.md`

---

## 3) 2-layer GAT architecture
From `models/gat.py`:
- Layer 1: multi-head `GATConv(num_features -> hidden_dim, heads=4)`
- Activation: `ELU`
- Layer 2: `GATConv(hidden_dim*heads -> num_classes, heads=1, concat=False)`
- Output: `log_softmax`

GAT learns edge-wise attention \(\alpha_{ij}\), so each node weighs neighbors differently during aggregation.
`forward_with_debug()` returns attention weights + hidden outputs + memory usage.

---

## 4) How each attack works (code-level, individually)

### Poisoning attacks (training-time corruption)
1. `attacks/poisoning/random_poison.py`
- Randomly adds edges (`n_edge_perturbations`)
- Optionally corrupts features globally
- Effect: changes graph structure before retraining, so model learns from poisoned topology

2. `attacks/poisoning/nettack.py`
- Uses DeepRobust Nettack if available, else targeted edge-flip fallback
- Perturbs around selected target test nodes
- Effect: class-boundary confusion around target neighborhoods

3. `attacks/poisoning/metattack.py`
- DeepRobust path is guarded due runtime incompatibility; fallback is used
- Fallback flips edges based on feature dissimilarity scoring
- Effect: injects structurally misleading relationships globally

### Evasion attacks (inference-time corruption)
1. `attacks/evasion/structure_evasion.py`
- For a target node: remove one existing edge and add one non-edge (degree-preserving style)
- Effect: test-time neighborhood message flow changes without retraining

2. `attacks/evasion/feature_evasion.py`
- Only modifies selected target nodes’ feature vectors at inference
- Binary features: bit flips up to `binary_flip_budget`
- Continuous features: Gaussian noise + clamp [0,1]
- Graph edges unchanged
- Effect: moves target representations toward wrong class in feature space

3. `attacks/evasion/fgsm_like.py`
- Computes gradient wrt features and applies
  \(X_{adv}=X+\epsilon \cdot sign(\nabla_X L)\)
- Strong white-box perturbation; can be very destructive

---

## 5) Double-check of "Evasion: Feature" correctness
Verification now printed in `main.py`:
- `Training graph unchanged: True`
- `Only target test nodes changed: True`
- `Changed targets: 120/120` (static run)
- Perturbation rate reported explicitly

So implementation is accurate for an evasion feature attack: **it perturbs test-time features only, not train graph structure**.

---

## 6) Which attack is most impactful?
From `results/final_evaluation_table.csv` (GCN, static):
- Baseline accuracy: **0.802**
- Largest drop came from **Evasion: FGSM-like** (accuracy 0.004)
- `Evasion: Feature` also degrades performance (accuracy 0.764), but not the largest under current hyperparameters

Interpretation:
- Your statement "Evasion: Feature is most vulnerable" can hold in non-white-box settings.
- In strict white-box settings, FGSM-like often dominates because it directly uses model gradients.

---

## 7) Defense strategy implemented for feature-focused evasion
Implemented in `main.py`, `defenses/feature_defense.py`, `defenses/ontology_defense.py`.

### A) Basic defense (base-paper-style smoothing)
- Laplacian feature smoothing:
\[
X_{smooth} = \alpha X + (1-\alpha)\hat{A}X
\]
- Regularizes local feature noise by neighborhood consistency

### B) Ontology-based defense (novel extension)
- Build semantic ontology matrix `O` from feature similarity + optional label guidance
- Reweight adjacency with ontology (`A + λO`), symmetrize, keep self-loops
- Project features with ontology term (`X + λOX`)

This integrates node relationships at a semantic level, reducing vulnerability to local feature perturbations.

---

## 8) Static and dynamic dataset coverage

### Static (Cora)
Generated files:
- `results/final_evaluation_table.csv` (GCN)
- `results/final_evaluation_table_gat.csv` (GAT)
- Visuals: `clean_graph.png`, `graph_mosaic.png`, confusion matrices, t-SNE, robustness curve

### Dynamic
Generated files:
- `results/dynamic_gcn_evaluation_table.csv`
- `results/dynamic_gat_evaluation_table.csv`
- `results/dynamic_summary.csv`

Both static and dynamic pipelines include poisoning + evasion suites and defense evaluation for feature-focused evasion.

---

## 9) File-by-file explanation (every source file)

### Entry + orchestration
- `main.py`
  - Full pipeline controller: load data, train baselines, run 6 attacks, evaluate metrics, validate feature evasion, run defenses, generate plots and reports, run dynamic suite.

- `report_generator.py`
  - Converts final CSV into a markdown summary report (`results/final_report.md`).

### Models
- `models/gcn.py`
  - 2-layer GCN + debug internals + embeddings extraction.
  - Now includes per-layer memory estimation.

- `models/gat.py`
  - 2-layer GAT with attention heads + debug attention outputs + embeddings.
  - Now includes per-layer memory estimation.

### Datasets
- `datasets/cora_loader.py`
  - Loads Planetoid Cora with feature normalization.

- `datasets/dynamic_graph.py`
  - Creates evolving Barabasi-Albert graph with synthetic features/labels and train/val/test masks.

### Training/Evaluation utilities
- `experiments/baseline_training.py`
  - Generic train/eval loops used by GCN/GAT.

- `experiments/attack_evaluation.py`
  - Legacy/auxiliary evaluation script; main production flow is in `main.py`.

- `utils/metrics.py`
  - Classification metrics: accuracy, precision, recall, F1 macro/micro, ROC-AUC, log-loss, margin, entropy, confidence.
  - Robustness metrics: ASR, confidence drop, margin drop, robustness score.
  - Perturbation-rate and graph-structure metrics.

### Poisoning attacks
- `attacks/poisoning/random_poison.py`: random graph/feature corruption.
- `attacks/poisoning/dice.py`: DICE global attack wrapper.
- `attacks/poisoning/nettack.py`: targeted Nettack wrapper + fallback.
- `attacks/poisoning/metattack.py`: Meta attack wrapper with safe fallback path.

### Evasion attacks
- `attacks/evasion/structure_evasion.py`: test-time edge rewiring around target nodes.
- `attacks/evasion/feature_evasion.py`: target-node feature perturbation (binary flips + noise).
- `attacks/evasion/fgsm_like.py`: gradient sign feature attack.
- `attacks/evasion/gradient_feature.py`: alternate gradient feature perturbation utility.
- `attacks/evasion/fgsm.py`: structural FGSM-style attack wrapper (DeepRobust FGA).
- `attacks/evasion/pgd.py`: iterative PGD-style attack utility.

### Defenses
- `defenses/feature_defense.py`
  - Adjacency normalization, Laplacian smoothing, consistency regularization.

- `defenses/ontology_defense.py`
  - Ontology matrix construction, ontology-guided adjacency reweighting, ontology feature projection.

- `defenses/robust_filtering.py`
  - Additional denoising/pruning/spectral filtering toolkit.

- `defenses/adversarial_training.py`
  - Optional adversarial training and randomized smoothing utilities.

### Visualization
- `visualization/plotting.py`
  - Robustness curves, confusion matrices, t-SNE plotting.

- `visualization/graph_viz.py`
  - Clean/attacked/defended graph mosaic visualization around target neighborhood.

### Project dependencies
- `requirements.txt`
  - Declares required libraries (`torch-geometric`, `deeprobust`, `seaborn`, etc.).

---

## 10) How attacks cause node-classification error (mechanism)
- GCN/GAT predictions rely on neighborhood message passing.
- Poisoning changes training graph/features -> wrong message patterns learned.
- Evasion changes test graph/features -> inference aggregation deviates from clean manifold.
- Result: class posteriors shift, margins collapse, confidence/uncertainty changes, leading to misclassification.

---

## 11) Real-world scenario where your project is useful
**Academic citation recommendation/manuscript topic classification**:
- Nodes = papers, edges = citation links, features = textual/topic embeddings.
- Attackers can inject fake citations or manipulate metadata features.
- Your pipeline detects robustness gaps and evaluates defenses before deployment.

Also applicable to:
- Fraud rings in financial transaction graphs
- Fake-user influence in social graphs
- Cyber threat intelligence knowledge graphs

---

## 12) What more this project can achieve (next scope)
1. Certified robustness bounds for GNN node classification.
2. Temporal/streaming adversarial defense with online adaptation.
3. Heterogeneous graph support (multi-node/multi-edge types).
4. Causal/ontology-constrained training objectives.
5. Explainability overlays per attacked node (counterfactual graph edits).

---

## 13) Important generated outputs to use in your report/demo
- `results/final_evaluation_table.csv`
- `results/final_evaluation_table_gat.csv`
- `results/dynamic_gcn_evaluation_table.csv`
- `results/dynamic_gat_evaluation_table.csv`
- `results/layerwise_debug_report.md`
- `results/graph_mosaic.png`
- `results/clean_graph.png`
- `results/robustness_curve.png`
- `results/confusion_baseline.png`
- `results/confusion_feature_attack.png`
- `results/confusion_ontology_defense.png`
- `results/tsne_clean.png`, `results/tsne_attacked.png`, `results/tsne_defended.png`

