import torch
import torch.nn.functional as F


def run_fgsm_like_feature_attack(model, data, epsilon=0.05):
    """
    FGSM-like attack over node features:
    X_adv = X + epsilon * sign(grad_X loss)
    """
    model.eval()
    attacked = data.clone()
    attacked.x = attacked.x.clone().detach().requires_grad_(True)
    out = model(attacked)
    loss = F.nll_loss(out[attacked.test_mask], attacked.y[attacked.test_mask])
    loss.backward()
    with torch.no_grad():
        attacked.x = torch.clamp(attacked.x + epsilon * attacked.x.grad.sign(), 0.0, 1.0)
    return attacked

