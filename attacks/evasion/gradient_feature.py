import torch
import torch.nn.functional as F

def run_gradient_feature_attack(model, data, n_perturbations=0.1):
    """
    Gradient-based feature perturbation attack.
    n_perturbations here is the epsilon for perturbation.
    """
    model.eval()
    data = data.clone()
    data.x.requires_grad = True
    
    logits = model(data)
    loss = F.nll_loss(logits[data.test_mask], data.y[data.test_mask])
    
    loss.backward()
    
    with torch.no_grad():
        # Add perturbation in direction of gradient
        perturbation = n_perturbations * data.x.grad.sign()
        data.x += perturbation
        
    return data.x

if __name__ == "__main__":
    pass
