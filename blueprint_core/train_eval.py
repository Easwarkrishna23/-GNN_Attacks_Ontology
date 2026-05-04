import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as sklearn_f1

from models import GCN, GAT

def accuracy(pred, target):
    return (pred == target).sum().item() / target.size(0)

def f1_score(pred, target):
    return sklearn_f1(target.cpu().numpy(), pred.cpu().numpy(), average='macro')

def train_model(data, model_type="GCN", hidden=16, epochs=200):
    
    num_features = data.num_features
    num_classes = int(data.y.max()) + 1
    
    if model_type == "GCN":
        model = GCN(num_features, hidden, num_classes)
    elif model_type == "GAT":
        model = GAT(num_features, hidden, num_classes)
    else:
        raise ValueError("Invalid model_type")
        
    model = model.to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out, embeddings = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Eval on test mask
        test_pred = pred[data.test_mask]
        test_target = data.y[data.test_mask]
        
        acc = accuracy(test_pred, test_target)
        f1 = f1_score(test_pred, test_target)
        
    return acc, f1, embeddings

def evaluate_all(baseline_model, attacked_data_dict, defended_data_dict):
    """Evaluate clean, attacked, and all defense combinations."""
    results = {}
    
    for attack_name in attacked_data_dict.keys():
        results[attack_name] = {}
        
        # Attack
        attacked_data = attacked_data_dict[attack_name]
        acc_attack, f1_attack, _ = evaluate(baseline_model, attacked_data)
        
        # Defenses
        defs = defended_data_dict[attack_name]
        acc_struct, f1_struct, _ = evaluate(baseline_model, defs["structural"])
        acc_onto, f1_onto, _ = evaluate(baseline_model, defs["ontology"])
        acc_hybrid, f1_hybrid, _ = evaluate(baseline_model, defs["hybrid"])
        
        results[attack_name] = {
            "after_attack": (acc_attack, f1_attack),
            "after_structural": (acc_struct, f1_struct),
            "after_ontology": (acc_onto, f1_onto),
            "after_hybrid": (acc_hybrid, f1_hybrid)
        }
        
    return results
