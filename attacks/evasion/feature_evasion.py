import numpy as np
import torch


def run_feature_evasion(data, target_nodes, binary_flip_budget=20, continuous_noise_std=0.1, seed=42):
    """
    Inference-time feature attack only; leaves training graph untouched.
    """
    rng = np.random.default_rng(seed)
    attacked = data.clone()
    original_x = attacked.x.clone()
    x = attacked.x.clone()

    for node in target_nodes:
        vec = x[node]
        binary_idx = torch.where((vec == 0) | (vec == 1))[0]
        if len(binary_idx) > 0:
            k = min(binary_flip_budget, len(binary_idx))
            choose = rng.choice(binary_idx.cpu().numpy(), size=k, replace=False)
            vec[choose] = 1.0 - vec[choose]
        non_binary_idx = torch.where((vec != 0) & (vec != 1))[0]
        if len(non_binary_idx) > 0:
            noise = torch.tensor(
                rng.normal(0.0, continuous_noise_std, size=len(non_binary_idx)),
                dtype=vec.dtype,
                device=vec.device,
            )
            vec[non_binary_idx] = torch.clamp(vec[non_binary_idx] + noise, 0.0, 1.0)
        x[node] = vec

    attacked.x = x
    return attacked, original_x

