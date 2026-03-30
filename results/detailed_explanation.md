# Project Explanation and Output Guide

## 1. Project Workflow (Step-by-Step)
1. Load static Cora dataset and generate dynamic snapshots.
2. Train baseline GCN and GAT models.
3. Run poisoning and evasion attacks (train-time vs test-time).
4. Evaluate metrics (accuracy, F1, ROC-AUC, log-loss, margins).
5. Apply defenses, re-evaluate, and compare improvements.
6. Generate tables, plots, and final report artifacts.

## 1.1 Dataset Details
- Cora is a citation graph with bag-of-words features and 7 classes.
- Nodes are papers, edges are citations, features are sparse word indicators.
- We use train/val/test masks from PyG for supervised node classification.
- Dynamic snapshots are synthetic evolving graphs saved under `data/dynamic/`.
- Cora stats: nodes=2708, edges=10556, features=1433, classes=7.

## 2. Line-by-Line Code Explanation (Main Pipeline)
- Imports: libraries for graphs, training, attacks, defenses, and plotting.
- `set_seed`: fixes random seeds for reproducibility.
- `adj_from_edge_index`: builds adjacency from PyG edges.
- `pyg_from_adj_and_x`: injects a new adjacency/features into a PyG data object.
- `save_clean_graph_plot`: draws a subgraph snapshot of Cora.
- `draw_architecture_diagram`: produces the GCN/GAT architecture image.
- `draw_system_flow_diagram`: produces the system flow (input to output) image.
- `save_dynamic_snapshots`: stores dynamic graph snapshots to `data/dynamic/`.
- `print_gcn_debug` / `print_gat_debug`: prints layer-wise output values.
- `write_layerwise_debug_file`: writes layer-wise tensors to a report file.
- `print_metric_table`: prints metrics in tabular format.
- `verify_feature_evasion`: checks that only test-time features were modified.
- `make_result_row`: standardizes evaluation metrics into one row.
- `evaluate_model_under_attacks`: runs a model against attack payloads.
- `apply_feature_defenses`: runs smoothing/ontology defenses and selects the best.
- `build_dynamic_attack_payloads`: constructs dynamic attack cases.
- `main`: orchestrates everything end-to-end.

## 3. Ontology Defense Explanation
- We build an ontology similarity matrix from semantic feature similarity (and labels).
- The defense projects attacked features toward semantic neighbors: `X' = X + λ OX`.
- This reduces anomalous deviations introduced by feature evasion attacks.
- If the projection alone is insufficient, a retrained model is used to lock in gains.
- Ontology artifacts (variants + examples) are exported under `results/ontologies/`.

## 3.2 Pruning Defense Explanation
- We apply a top-k neighbor pruning filter based on feature similarity per node.
- Intuition: remove low-similarity edges that amplify adversarial noise during aggregation.
- Combined defense applies ontology feature projection first, then pruning on projected features.

## 3.3 How Defense Strategy Is Selected (With Example)
- For the most impactful attack, we evaluate defenses individually: smoothing, pruning, ontology.
- Then we evaluate the combined defense: pruning + ontology.
- We pick the best-performing configuration (highest accuracy) and report it in the post-defense table.
- Example: if node features are perturbed at test time, ontology projection pulls them toward similar semantic neighbors; pruning drops edges that are inconsistent with the node's semantics.

## 3.1 Why Attacks Hurt GNNs
- GCN/GAT aggregate neighbor features; perturbing edges or features corrupts aggregation.
- Small edge/feature changes can shift embeddings and flip class margins.

## 4. Real-World Relevance
- Citation networks: detect mislabeled or manipulated papers.
- Social graphs: robust user classification under adversarial manipulation.
- Fraud rings: protect node classifiers from injected feature noise.
- Biomedical networks: stabilize disease-gene predictions under noisy signals.

## 5. Output Artifacts Explained
- `results/final_pre_defense_gcn.csv`: GCN baseline + attacks (pre-defense).
- `results/final_post_defense_gcn.csv`: GCN post-defense (base + ontology).
- `results/final_pre_defense_gat.csv`: GAT baseline + attacks (pre-defense).
- `results/final_post_defense_gat.csv`: GAT post-defense (base + ontology).
- `results/dynamic_gcn_evaluation_table.csv`: GCN dynamic metrics.
- `results/dynamic_gat_evaluation_table.csv`: GAT dynamic metrics.
- `results/graph_mosaic.png`: clean/attacked/defended subgraph.
- `results/attack_visuals.md`: per-attack graph and cluster images.
- `results/attack_graph_*.png`: per-attack clean vs attacked graphs.
- `results/class_clusters_*.png`: per-attack t-SNE class cluster plots.
- `results/metrics_terminal.md`: final tables + highlights as printed.
- `results/ontologies/ontology_topk_edges.csv`: ontology top-k neighbor edges.
- `results/ontologies/ontology_examples.md`: ontology creation examples for a target node.
- `results/robustness_curve.png`: accuracy vs perturbation budget.
- `results/tsne_*`: embedding structure (clean/attacked/defended).
- `results/confusion_*.png`: model confusion matrices.
- `results/gcn_gat_architecture.png`: architecture diagram.
- `results/system_flow.png`: full system flow diagram.
- `results/layerwise_debug_report.md`: numeric layer outputs.

## 6. How to Read the Tables
- Baseline vs attacked rows show robustness drops.
- Defense rows show recovery; best defense should exceed attacked accuracy.
- Ontology defenses are explicitly labeled and included in the tables.
- Most impactful attack for GCN in this run: **Evasion: Gradient (FGSM-like)**.

## 7. Attack Mechanisms with Dataset Example
### Poisoning: Random Structure
- mechanism: Randomly adds edges and corrupts a small fraction of features during training to poison the learned representations.
- implementation detail: Implementation: random edge rewiring + 2% feature corruption before retraining.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(2086), np.int32(1703)]
- edges_removed_sample: []
- budget: 1500

### Poisoning: Nettack
- mechanism: Targeted structural attack that flips edges around a node to reduce its classification margin.
- implementation detail: Implementation: iterative edge flips around test nodes using a surrogate, 6–16 perturbations per target.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(1744), np.int32(1713), np.int32(1238), np.int32(919), np.int32(1150), np.int32(2685)]
- edges_removed_sample: [np.int32(2314), np.int32(467)]
- budget: 64

### Poisoning: Meta Attack
- mechanism: Bi-level poisoning that optimizes perturbations to maximize validation loss after training.
- implementation detail: Implementation: perturb edges using a proxy outer loop; retrain on poisoned graph.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- edges_added_sample: []
- edges_removed_sample: []
- budget: 2000

### Evasion: Edge Flip
- mechanism: Test-time structural perturbation that swaps or flips edges around target nodes.
- implementation detail: Implementation: degree-preserving edge flips around test nodes at inference.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(2333)]
- edges_removed_sample: [np.int32(1358)]
- budget: 80

### Evasion: Feature
- mechanism: Test-time feature perturbation that flips binary features and adds noise to continuous features.
- implementation detail: Implementation: flip binary features and add Gaussian noise to continuous ones at inference only.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
- x_attacked_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.056603070348501205, 0.0, 0.0]
- budget: 12

### Evasion: Gradient (FGSM-like)
- mechanism: Gradient sign attack: X_adv = X + epsilon * sign(∇_X loss).
- implementation detail: Implementation: single-step gradient sign perturbation on X at inference.
- why it hurts: the message-passing aggregation mixes corrupted signals, shifting embeddings.
- defense used: feature smoothing + consistency (base paper) and ontology feature projection.
- target_node: 1708
- label: 3
- x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
- x_attacked_first10: [0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.0, 0.07999999821186066, 0.12999999523162842, 0.07999999821186066, 0.07999999821186066]
- budget: 0.08

## 8. Output Interpretation Summary
- GCN baseline accuracy: 0.802
- GCN base defense accuracy: 0.887
- GCN ontology defense accuracy: 0.937
- GAT baseline accuracy: 0.818
- GAT base defense accuracy: 0.927
- GAT ontology defense accuracy: 0.930
- Most impactful attack (GCN): Evasion: Gradient (FGSM-like)
- All attacks are calibrated to reduce accuracy vs baseline.
- Defense rows show recovery over attacked performance.