import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Feature Denoising (Preprocessing Defense)
# ─────────────────────────────────────────────────────────────────────

def feature_denoising(x, method='median', kernel_size=3):
    """
    Denoise node features to remove adversarial sign-based perturbations.

    The gradient feature attack adds noise in the sign direction of
    the gradient: x_adv = x + eps * sign(grad). This creates structured,
    high-frequency noise across the feature vector. Median filtering
    and Gaussian smoothing are highly effective at reversing this.

    Args:
        x         : Node feature tensor [N, F]
        method    : 'median' | 'gaussian' | 'both'
        kernel_size: Window for median filter
    """
    x_np = x.cpu().detach().numpy()

    if method in ('median', 'both'):
        # Median filter per node's feature vector
        from scipy.signal import medfilt
        x_np = np.apply_along_axis(
            lambda row: medfilt(row, kernel_size=kernel_size),
            axis=1, arr=x_np
        )

    if method in ('gaussian', 'both'):
        # Light Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        x_np = gaussian_filter1d(x_np, sigma=0.5, axis=1)

    # Clip to valid range [0, 1]
    x_np = np.clip(x_np, 0.0, 1.0)
    return torch.tensor(x_np, dtype=x.dtype)


# ─────────────────────────────────────────────────────────────────────
# Randomized Smoothing Inference
# ─────────────────────────────────────────────────────────────────────

def smoothed_predict(model, data, sigma=0.05, n_samples=50):
    """
    Randomized Smoothing: average predictions over N copies of the
    input with i.i.d. Gaussian noise added to features.

    This breaks gradient-based attacks because any single gradient
    computed by the attacker is averaged out by the noise distribution.

    Args:
        model    : Trained GCN
        data     : Input graph data
        sigma    : Noise standard deviation
        n_samples: Number of noisy copies to average
    """
    model.eval()
    logit_sum = None

    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(data.x) * sigma
            data_noisy = data.clone()
            data_noisy.x = torch.clamp(data.x + noise, 0.0, 1.0)
            out = model(data_noisy)
            probs = torch.exp(out)

            if logit_sum is None:
                logit_sum = probs
            else:
                logit_sum += probs

    avg_probs = logit_sum / n_samples
    preds = avg_probs.argmax(dim=1)
    return preds, avg_probs


# ─────────────────────────────────────────────────────────────────────
# Combined Adversarial Training (now with larger epsilon & more epochs)
# ─────────────────────────────────────────────────────────────────────

def adversarial_train_model(model, data, epochs=300, lr=0.01,
                            weight_decay=5e-4, epsilon=0.1,
                            adv_ratio=0.5, warmup_epochs=80):
    """
    PGD Adversarial Training with Warm-up + Cosine LR schedule.

    Phase 1 (warmup): Clean training so model builds a proper classifier.
    Phase 2 (hardening): Interleave clean and PGD-perturbed training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if epoch < warmup_epochs:
            # Phase 1: Clean
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        else:
            # Phase 2: Clean + Adversarial
            # --- Clean loss ---
            out_clean = model(data)
            loss_clean = F.nll_loss(out_clean[data.train_mask],
                                    data.y[data.train_mask])

            # --- Generate PGD perturbation (detached) ---
            x_adv = _pgd(model, data, epsilon=epsilon, alpha=epsilon/5, steps=3)
            data_adv = data.clone()
            data_adv.x = x_adv

            model.train()
            optimizer.zero_grad()
            out_adv = model(data_adv)
            loss_adv = F.nll_loss(out_adv[data.train_mask],
                                  data.y[data.train_mask])

            loss = (1 - adv_ratio) * loss_clean + adv_ratio * loss_adv

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 40 == 0:
            phase = "Warmup" if epoch < warmup_epochs else "Hardening"
            print(f'[{phase}] Adv-Epoch {epoch:03d} | Loss={loss.item():.4f}')

    return model


def _pgd(model, data, epsilon, alpha, steps):
    """Inner PGD loop (helper, not exported)."""
    x_orig = data.x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        data_tmp = data.clone()
        data_tmp.x = x_adv
        model.eval()
        out = model(data_tmp)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    return x_adv.detach()
