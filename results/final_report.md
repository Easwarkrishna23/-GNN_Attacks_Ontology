# Adversarial Attacks on GNNs: Experimental Report

## Mathematical Summary
- GCN layer: H^(l+1) = sigma(D^(-1/2) A_hat D^(-1/2) H^(l) W^(l))
- GAT layer: H^(l+1)_i = ||_k sigma(sum_j alpha_ij^k W^k h_j)
- FGSM-like feature attack: X_adv = X + epsilon * sign(grad_X L)
- Feature smoothing defense: X_smooth = alpha X + (1-alpha) A_hat X
- Ontology defense term: H = A_hat XW + lambda OX

## Code Walkthrough
- `models/gcn.py`: 2-layer GCN with debug tensors for pre/post aggregation and softmax.
- `models/gat.py`: 2-layer GAT with multi-head attention and extracted attention weights.
- `attacks/`: poisoning attacks modify train graph/features; evasion attacks modify inference inputs.
- `defenses/`: denoising + consistency regularization and ontology-based semantic reweighting.
- `main.py`: trains baselines, runs 6 attacks, computes metrics, generates plots and CSV tables.

## Impact Analysis
- Baseline accuracy: 0.8020
- Most impactful attack: Evasion: Gradient (FGSM-like) (accuracy drop 0.7980)

## Conclusion
- Attack and defense performance are stored in `results/final_evaluation_table.csv`.
- Visual diagnostics include graph comparisons, robustness curves, confusion matrices, and t-SNE.