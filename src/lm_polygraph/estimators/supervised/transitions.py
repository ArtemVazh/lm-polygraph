import torch
import torch.nn.functional as F
import math

def layerwise_angles(x: torch.Tensor, degrees=False, eps=1e-8) -> torch.Tensor:
    normed = F.normalize(x, dim=-1, eps=eps)          # [B, L, D]
    h_i, h_ip1 = normed[:, :-1, :], normed[:, 1:, :]
    cos_sim = (h_i * h_ip1).sum(dim=-1).clamp(-1 + eps, 1 - eps)  # [B, L-1]
    angles = torch.acos(cos_sim)
    if degrees:
        angles = angles * (180.0 / math.pi)
    return angles  # [B, L-1]

def layerwise_sigmas(x: torch.Tensor,  eps=1e-8) -> torch.Tensor:
    norms = x.norm(p=2, dim=-1) + eps                # [B, L]
    sigmas = norms[:, 1:] / norms[:, :-1]            # [B, L-1]
    return sigmas     # [B, L-1]