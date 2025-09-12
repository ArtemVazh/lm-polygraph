import torch
import torch.nn.functional as F
import scipy.fftpack


def layerwise_total_energy(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute squared ℓ2 norm (total energy) for each layer's activation.

    Args:
        embeddings (Tensor): shape [B, L, D]

    Returns:
        Tensor: shape [B, L] — total energy per layer
    """
    return embeddings.pow(2).sum(dim=-1)


# FFT Low-Frequency Energy Ratio:
# Measures how smooth the layer-wise signal is by computing the proportion
# of FFT energy in the first k frequencies.
def fft_energy_ratio(embeddings: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """
    Compute FFT low-frequency energy ratio for each hidden dim over L layers.

    Args:
        embeddings (Tensor): shape [B, L, D]
        top_k (int): how many low frequencies to keep

    Returns:
        Tensor: shape [B, D] — energy ratio in top-k frequencies
    """
    fft_vals = torch.fft.fft(embeddings.float(), dim=1)  # [B, L, D], complex
    power = fft_vals.real ** 2 + fft_vals.imag ** 2  # magnitude squared
    total_energy = power.sum(dim=1)  # [B, D]
    low_freq_energy = power[:, :top_k].sum(dim=1)  # [B, D]
    return (low_freq_energy / total_energy.clamp(min=1e-8))  # [B, D]


# DCT Compression Energy
# Measures how well the signal can be compressed with low-frequency components.
def dct_energy_ratio(embeddings: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """
    Apply DCT along the layer dimension and compute energy ratio in top-k components.

    Args:
        embeddings (Tensor): shape [B, L, D]
        top_k (int): number of low-freq components to keep

    Returns:
        Tensor: shape [B, D]
    """
    B, L, D = embeddings.shape
    x = embeddings.float().cpu().numpy()  # DCT not native in PyTorch

    dct_vals = scipy.fftpack.dct(x, axis=1, norm='ortho')  # [B, L, D]
    power = dct_vals ** 2
    total = power.sum(axis=1)  # [B, D]
    low = power[:, :top_k].sum(axis=1)
    return torch.from_numpy(low / (total + 1e-8))


def spectral_entropy(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute spectral entropy for each sample's layerwise trajectory.

    Args:
        embeddings (Tensor): shape [B, L, D]
        eps (float): small number for numerical stability

    Returns:
        Tensor: shape [B]
    """
    # Compute FFT over the layer axis
    fft_vals = torch.fft.fft(embeddings.float(), dim=1)         # [B, L, D]
    power_spectrum = fft_vals.abs().pow(2)              # [B, L, D]

    # Sum over hidden dim → 1D power spectrum per sample
    power = power_spectrum.sum(dim=-1)                  # [B, L]

    # Normalize to get distribution
    power = power + eps                                 # avoid division by 0
    prob = power / power.sum(dim=-1, keepdim=True)      # [B, L]

    # Shannon entropy
    entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1)  # [B]

    return entropy