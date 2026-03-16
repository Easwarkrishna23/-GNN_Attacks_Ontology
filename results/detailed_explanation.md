# Project Explanation and Output Guide

## 1. Project Workflow (Step-by-Step)
1. Load static Cora dataset and generate dynamic snapshots.
2. Train baseline GCN and GAT models.
3. Run poisoning and evasion attacks (train-time vs test-time).
4. Evaluate metrics (accuracy, F1, ROC-AUC, log-loss, margins).
5. Apply defenses, re-evaluate, and compare improvements.
6. Generate tables, plots, and final report artifacts.

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

## 4. Real-World Relevance
- Citation networks: detect mislabeled or manipulated papers.
- Social graphs: robust user classification under adversarial manipulation.
- Fraud rings: protect node classifiers from injected feature noise.
- Biomedical networks: stabilize disease-gene predictions under noisy signals.

## 5. Output Artifacts Explained
- `results/final_evaluation_table.csv`: GCN static attack/defense metrics.
- `results/final_evaluation_table_gat.csv`: GAT static attack/defense metrics.
- `results/dynamic_gcn_evaluation_table.csv`: GCN dynamic metrics.
- `results/dynamic_gat_evaluation_table.csv`: GAT dynamic metrics.
- `results/graph_mosaic.png`: clean/attacked/defended subgraph.
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

## 7. Attack Mechanisms with Dataset Example
### Poisoning: Random Structure
- mechanism: Randomly adds edges and corrupts a small fraction of features during training to poison the learned representations.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(2086), np.int32(1703)]
- edges_removed_sample: []
- budget: 1500

### Poisoning: Nettack
- mechanism: Targeted structural attack that flips edges around a node to reduce its classification margin.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(1601), np.int32(2627), np.int32(1765), np.int32(46), np.int32(1744), np.int32(1489)]
- edges_removed_sample: [np.int32(1857), np.int32(873), np.int32(2314), np.int32(2313), np.int32(467)]
- budget: 128

### Poisoning: Meta Attack
- mechanism: Bi-level poisoning that optimizes perturbations to maximize validation loss after training.
- target_node: 1708
- label: 3
- edges_added_sample: []
- edges_removed_sample: []
- budget: 4000

### Evasion: Edge Flip
- mechanism: Test-time structural perturbation that swaps or flips edges around target nodes.
- target_node: 1708
- label: 3
- edges_added_sample: [np.int32(2108), np.int32(2333)]
- edges_removed_sample: [np.int32(467), np.int32(1358)]
- budget: 120

### Evasion: Feature
- mechanism: Test-time feature perturbation that flips binary features and adds noise to continuous features.
- target_node: 1708
- label: 3
- x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
- x_attacked_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.056603070348501205, 0.0, 0.0]
- budget: 12

### Evasion: Gradient (FGSM-like)
- mechanism: Gradient sign attack: X_adv = X + epsilon * sign(∇_X loss).
- target_node: 1708
- label: 3
- x_clean_first10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0, 0.0]
- x_attacked_first10: [0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.07999999821186066, 0.0, 0.07999999821186066, 0.12999999523162842, 0.07999999821186066, 0.07999999821186066]
- budget: 0.08

## 8. Output Interpretation Summary
- GCN baseline accuracy: 0.802
- GCN best defense accuracy: 0.760
- GAT baseline accuracy: 0.818
- GAT best defense accuracy: 0.771
- All attacks are calibrated to reduce accuracy vs baseline.
- Defense rows show recovery over attacked performance.