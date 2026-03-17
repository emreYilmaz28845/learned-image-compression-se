"""Squeeze-and-Excitation channel attention block (Hu et al., CVPR 2018)."""

import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Applies global average pooling to squeeze spatial dimensions,
    then a two-layer FC bottleneck to learn per-channel scaling factors.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s
