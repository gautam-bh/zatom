import torch
import torch.nn as nn
from torch import Tensor


class BandGapEmbedder(nn.Module):
    """Embeds a scalar band gap value into the model's conditioning space.

    Maps a band gap (eV, scalar per sample) to a hidden_size embedding via a 2-layer MLP.
    NaN values (missing/unknown labels) are treated as the null condition (zero embedding).
    Supports CFG-ready dropout: during training the embedding is randomly zeroed to enable
    unconditional generation at inference time.

    Args:
        hidden_size: Output embedding dimension (= model hidden_size).
        dropout_prob: Probability of zeroing the embedding during training (CFG null token).
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        band_gap: Tensor,  # (B,) float
        train: bool,
        force_uncond: bool = False,
    ) -> Tensor:  # (B, hidden_size)
        """Embed band gap values.

        Args:
            band_gap: Per-sample band gap in eV, shape (B,). NaN = unknown/missing.
            train: Whether the model is in training mode (enables dropout).
            force_uncond: If True, return zeros regardless of input (unconditional pass).

        Returns:
            Embedding tensor of shape (B, hidden_size).
        """
        if force_uncond:
            dummy = self.mlp(band_gap.unsqueeze(-1).float().nan_to_num(0.0))
            return torch.zeros_like(dummy)

        valid = ~torch.isnan(band_gap)
        x = band_gap.clone().nan_to_num(0.0)
        emb = self.mlp(x.unsqueeze(-1).float())

        # Zero out embeddings for samples without a label
        emb = emb * valid.unsqueeze(-1).float()

        # CFG dropout: randomly drop conditioning during training
        if train and self.dropout_prob > 0:
            drop = torch.rand(emb.shape[0], device=emb.device) < self.dropout_prob
            emb = emb * (~drop).unsqueeze(-1).float()

        return emb
