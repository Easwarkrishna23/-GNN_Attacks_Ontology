from dataset import load_dataset
from train_eval import train_model, evaluate, evaluate_all
from attacks import apply_all_attacks
from defenses import apply_defenses
from visualization import visualize_results, plot_tsne

def main():
    print("[1/6] Loading Dataset...")
    data = load_dataset()
    print(f"      Loaded Cora: {data.num_nodes} nodes, {data.num_edges} edges.")

    print("\n[2/6] Training Baseline Model...")
    baseline_model = train_model(data, model_type="GCN")
    baseline_acc, baseline_f1, baseline_embeddings = evaluate(baseline_model, data)
    print(f"      Baseline Accuracy: {baseline_acc*100:.1f}% | F1: {baseline_f1*100:.1f}%")

    print("\n[3/6] Applying 6 Adversarial Attacks...")
    attacked_data_dict = apply_all_attacks(data, baseline_model)
    print(f"      Generated attacks: {', '.join(attacked_data_dict.keys())}")

    print("\n[4/6] Applying Defenses (Structural, Ontology, Hybrid)...")
    defended_data_dict = apply_defenses(attacked_data_dict)

    print("\n[5/6] Evaluating Models...")
    results = evaluate_all(baseline_model, attacked_data_dict, defended_data_dict)

    print("\n[6/6] Visualizing Results...")
    visualize_results(results, baseline_acc, baseline_f1)
    plot_tsne(baseline_embeddings, data.y, "Clean Embeddings")
    
    print("\n[✔] End-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
