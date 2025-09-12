import torch
import torch.nn.functional as F


# Layerwise Hidden Norm:
# Computes ℓ₂ norm ‖h_i‖ across the hidden dimension for each layer.
# Captures the magnitude of layer activations, often correlated with information density or saturation.
# Simple and fast way to summarize per-layer activation strength.
def layerwise_norms(embeddings: torch.Tensor, norm=2) -> torch.Tensor:
    """
    Compute ℓ₂ norm of hidden activations at each layer.

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L]
    """
    return embeddings.norm(p=norm, dim=-1)


# PCA Projection onto First Principal Direction:
# Projects each layer's embedding onto the first principal axis across the hidden dimension.
# Highlights the dominant direction of semantic variance.
# Captures major shared variation in hidden representation.
def layerwise_pca_projection(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Project each layer onto the first PCA component (shared across batch).

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L]
    """
    # Flatten across batch and layer to fit PCA on [B*L, D]
    B, L, D = embeddings.shape
    flat = embeddings.float().reshape(B * L, D)

    # Centered PCA using SVD (top-1 direction)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(flat_centered, full_matrices=False)
    pc1 = V[0]  # [D]

    # Project each layer onto pc1 → [B, L]
    return (embeddings.float() @ pc1).squeeze(-1)

def per_sample_pca_projection(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Project each layer onto the first PCA component for each sample independently.

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L] — scalar per layer
    """
    B, L, D = embeddings.shape
    projections = []

    for i in range(B):
        x = embeddings[i].float()  # [L, D]
        x_centered = x - x.mean(dim=0, keepdim=True)
        U, S, V = torch.linalg.svd(x_centered, full_matrices=False)
        pc1 = V[0]  # [D]
        proj = (x @ pc1)  # [L]
        projections.append(proj)

    return torch.stack(projections, dim=0)  # [B, L]

def layerwise_softmax_entropy(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply softmax over hidden dim and compute Shannon entropy.

    Args:
        embeddings (Tensor): shape [B, L, D]
        eps (float): stability term

    Returns:
        Tensor: shape [B, L]
    """
    probs = F.softmax(embeddings, dim=-1) + eps
    return -torch.sum(probs * torch.log(probs), dim=-1)

