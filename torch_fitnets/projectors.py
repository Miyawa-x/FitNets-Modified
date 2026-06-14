from __future__ import annotations

import torch
from torch import nn


class GlobalAverageProjection(nn.Module):
    """1x1 convolution followed by global average pooling.

    The head maps middle convolutional features to class logits while keeping
    projection capacity intentionally small.
    """

    def __init__(self, in_channels: int, num_classes: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits_map = self.proj(features)
        return logits_map.mean(dim=(2, 3))
