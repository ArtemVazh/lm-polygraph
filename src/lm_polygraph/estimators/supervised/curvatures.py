import torch
import torch.nn.functional as F
import math

# Angle-Based Curvature (Vector Turning Angle):
# Measures the angle between two consecutive difference vectors:
#     v1 = h_{i+1} - h_i, v2 = h_{i+2} - h_{i+1}
# Curvature is computed as:
#     κ_i = arccos(⟨v1, v2⟩ / (‖v1‖‖v2‖))
# Interprets how sharply the trajectory bends from one step to the next.
# Zero means smooth/straight movement; higher values indicate directional change.
# Invariant to scaling and translation; reflects pure directional deviation.
def angle_curvature(
    embeddings: torch.Tensor,
    degrees: bool = False,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute discrete curvature (angle between consecutive differences) for layerwise embeddings.

    Args:
        embeddings (Tensor): shape [B, L, D]
        degrees (bool): if True, return angle in degrees instead of radians
        eps (float): numerical stability term for norm and acos

    Returns:
        Tensor: shape [B, L-2] containing curvature at interior points
    """
    # Compute layerwise deltas
    v1 = embeddings[:, 1:-1, :] - embeddings[:, :-2, :]   # [B, L-2, D]
    v2 = embeddings[:, 2:, :]   - embeddings[:, 1:-1, :]  # [B, L-2, D]

    # Normalize vectors
    v1_norm = F.normalize(v1, dim=-1, eps=eps)
    v2_norm = F.normalize(v2, dim=-1, eps=eps)

    # Cosine similarity
    cos_sim = (v1_norm * v2_norm).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)  # [B, L-2]

    # Compute angle
    curvature = torch.acos(cos_sim)  # radians

    if degrees:
        curvature = curvature * (180.0 / math.pi)

    return curvature  # [B, L-2]

# Menger Curvature:
# Measures how much three consecutive points deviate from forming a straight line.
# Computed as (4 × area of triangle) / (product of side lengths).
# High when triangle is "sharp", zero when points are collinear.
# Interprets curvature as inverse of the radius of the circle passing through 3 points.
def curvature_menger(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Menger curvature for each triple of consecutive layer embeddings.

    Args:
        embeddings (Tensor): shape [B, L, D]
        eps (float): numerical stability

    Returns:
        Tensor: shape [B, L-2], Menger curvature values
    """
    A, B, C = embeddings[:, :-2], embeddings[:, 1:-1], embeddings[:, 2:]

    AB = B - A
    BC = C - B
    AC = C - A

    # Norms
    a = AB.norm(dim=-1)  # [B, L-2]
    b = BC.norm(dim=-1)
    c = AC.norm(dim=-1)

    # Heron's formula for area
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)).clamp(min=0.0).sqrt()  # [B, L-2]

    # Menger curvature
    curvature = (4 * area) / (a * b * c + eps)

    return curvature

# Second Derivative Norm:
# Discrete analogue of acceleration — ‖h_{i+2} - 2h_{i+1} + h_i‖.
# Measures how much the path “bends” at each interior point.
# Sensitive to sharp changes in direction or sudden non-linearity.
# Simple and computationally cheap to interpret "trajectory sharpness".
def curvature_second_derivative(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute norm of second discrete derivative for each triple of embeddings.

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L-2]
    """
    A, B, C = embeddings[:, :-2], embeddings[:, 1:-1], embeddings[:, 2:]
    second_diff = C - 2 * B + A
    return second_diff.norm(dim=-1)

# Arc-to-Chord Curvature:
# Computes the ratio between path length (arc) and direct distance (chord).
# curvature = (arc length / chord length) - 1
# Zero if path is straight; increases as path deviates from linearity.
# Intuitively reflects how inefficient (curved) the movement between h_i and h_{i+2} is.
def curvature_arc_chord(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute arc-length to chord-length ratio curvature.

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L-2]
    """
    A, B, C = embeddings[:, :-2], embeddings[:, 1:-1], embeddings[:, 2:]

    arc = (B - A).norm(dim=-1) + (C - B).norm(dim=-1)  # [B, L-2]
    chord = (C - A).norm(dim=-1)                      # [B, L-2]

    curvature = (arc / (chord + eps)) - 1
    return curvature