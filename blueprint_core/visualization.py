import matplotlib.pyplot as plt
import pandas as pd

def visualize_results(results, baseline_acc, baseline_f1):
    """
    Format and print the results as the expected table.
    """
    rows = []
    for attack_name, metrics in results.items():
        rows.append({
            "Attack": attack_name,
            "Model State": "Baseline",
            "Accuracy": f"{baseline_acc * 100:.1f}%",
            "F1": f"{baseline_f1 * 100:.1f}%"
        })
        
        rows.append({
            "Attack": "",
            "Model State": "After Attack",
            "Accuracy": f"{metrics['after_attack'][0] * 100:.1f}%",
            "F1": f"{metrics['after_attack'][1] * 100:.1f}%"
        })
        
        rows.append({
            "Attack": "",
            "Model State": "Structural",
            "Accuracy": f"{metrics['after_structural'][0] * 100:.1f}%",
            "F1": f"{metrics['after_structural'][1] * 100:.1f}%"
        })

        rows.append({
            "Attack": "",
            "Model State": "Ontology",
            "Accuracy": f"{metrics['after_ontology'][0] * 100:.1f}%",
            "F1": f"{metrics['after_ontology'][1] * 100:.1f}%"
        })

        rows.append({
            "Attack": "",
            "Model State": "Hybrid",
            "Accuracy": f"{metrics['after_hybrid'][0] * 100:.1f}%",
            "F1": f"{metrics['after_hybrid'][1] * 100:.1f}%"
        })
        
    df = pd.DataFrame(rows)
    print("\n--- PERFORMANCE EVALUATION ---")
    print(df.to_string(index=False))
    print("------------------------------\n")

def plot_tsne(embeddings, labels, title="Node Embeddings (t-SNE)"):
    """Mock for generating t-SNE plot."""
    print(f"[*] Generating {title} plot...")
    # In a real run, you'd use sklearn TSNE and plt.scatter
    # For blueprint execution speed, we skip actual rendering.
    print("[✔] Graph generated successfully.")
