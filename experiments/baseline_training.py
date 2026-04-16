import torch
import torch.nn.functional as F
from datasets.cora_loader import load_cora
from models.gcn import GCN
from models.gat import GAT
from utils.metrics import compute_classification_metrics
import os

def train_model(
    model,
    data,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    regularizer=None,
    reg_weight=0.0,
    edge_weight=None,
    edge_weight_l1=None,
    edge_weight_l2=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Some defenses use shared edge weights; others use layer-wise trust weights.
        if edge_weight is not None:
            data.edge_weight = edge_weight
        if edge_weight_l1 is not None:
            data.edge_weight_l1 = edge_weight_l1
        if edge_weight_l2 is not None:
            data.edge_weight_l2 = edge_weight_l2
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        if regularizer is not None and float(reg_weight) > 0:
            loss = loss + float(reg_weight) * regularizer(out, data)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
            
    return model

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        probs = torch.exp(out)
        
        mask = data.test_mask
        metrics = compute_classification_metrics(
            data.y[mask].cpu().numpy(),
            pred[mask].cpu().numpy(),
            probs[mask].cpu().numpy()
        )
    return metrics, pred, probs

def run_baseline():
    dataset, data = load_cora()
    
    # Train GCN
    print("\nTraining GCN Baseline...")
    gcn = GCN(dataset.num_features, 16, dataset.num_classes)
    gcn = train_model(gcn, data)
    gcn_metrics, _, _ = evaluate_model(gcn, data)
    print(f"GCN Baseline Metrics: {gcn_metrics}")
    
    # Train GAT
    print("\nTraining GAT Baseline...")
    gat = GAT(dataset.num_features, 8, dataset.num_classes)
    gat = train_model(gat, data)
    gat_metrics, _, _ = evaluate_model(gat, data)
    print(f"GAT Baseline Metrics: {gat_metrics}")
    
    # Save models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(gcn.state_dict(), 'checkpoints/gcn_baseline.pth')
    torch.save(gat.state_dict(), 'checkpoints/gat_baseline.pth')
    
    return gcn, gat, gcn_metrics, gat_metrics

if __name__ == "__main__":
    run_baseline()
