"""Contrastive spike encoder."""
import torch
from torch import nn


class SpikeEncoder(nn.Module):
    def __init__(self, n_channels=64, embed_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, embed_dim, kernel_size=5, padding=2)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, waveforms):
        features = torch.relu(self.conv(waveforms))
        pooled = features.mean(dim=-1)
        return torch.nn.functional.normalize(self.proj(pooled), dim=-1)
