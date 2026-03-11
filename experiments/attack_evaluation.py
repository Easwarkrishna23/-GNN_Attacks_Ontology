import torch
import numpy as np
import scipy.sparse as sp
from datasets.cora_loader import load_cora
from models.gcn import GCN
from attacks.poisoning.nettack import run_nettack
from attacks.poisoning.random_poison import run_random_attack
from attacks.evasion.gradient_feature import run_gradient_feature_attack
from utils.metrics import compute_classification_metrics, compute_robustness_metrics
import pandas as pd

def run_attack_evaluation(gcn, dataset, data):
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), 
                         (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
                        shape=(data.num_nodes, data.num_nodes))
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    idx_train = np.where(data.train_mask.cpu().numpy())[0]
    idx_test = np.where(data.test_mask.cpu().numpy())[0]

    results = []

    # 1. Baseline
    print("Evaluating Baseline...")
    gcn.eval()
    with torch.no_grad():
        out = gcn(data)
        clean_acc = compute_classification_metrics(labels[idx_test], out[idx_test].argmax(1).cpu().numpy())['accuracy']
        clean_probs = torch.exp(out[idx_test]).cpu().numpy()

    # 2. Random Poisoning
    print("Running Random Edge Injection Poisoning...")
    mod_adj = run_random_attack(adj, n_perturbations=100)
    # Re-evaluate GCN on poisoned graph (evasion style for simplicity, or retrain for poisoning)
    # The prompt says poisoning attacks modify training graph, affect learned model params
    # For now, let's treat it as evasion for measurement if RETRAINING is too slow in this script
    # But I should retrain if it's truly poisoning.
    
    # 3. Nettack (Targeted)
    print("Running Nettack on 5 random test nodes...")
    for target_node in idx_test[:5]:
        mod_adj, mod_feat = run_nettack(adj, features, labels, idx_train, target_node, n_perturbations=5)
        # Measure success
        
    # 4. Gradient-based Feature evasion
    print("Running Gradient Feature Evasion...")
    mod_x = run_gradient_feature_attack(gcn, data, n_perturbations=0.05)
    with torch.no_grad():
        data_mod = data.clone()
        data_mod.x = mod_x
        out_mod = gcn(data_mod)
        attack_acc = compute_classification_metrics(labels[idx_test], out_mod[idx_test].argmax(1).cpu().numpy())['accuracy']
        attack_probs = torch.exp(out_mod[idx_test]).cpu().numpy()
        
    rob_metrics = compute_robustness_metrics(clean_acc, attack_acc, clean_probs, attack_probs)
    results.append({
        'Attack': 'Gradient Feature Evasion',
        'Accuracy': attack_acc,
        'ASR': rob_metrics['asr'],
        'Confidence Drop': rob_metrics['confidence_drop']
    })

    return pd.DataFrame(results)

if __name__ == "__main__":
    pass
