from __future__ import annotations

import torch
from torch import nn


class GlobalAverageProjection(nn.Module):
    """1x1 convolution followed by global average pooling.

    The head maps middle convolutional features to class logits while keeping
    projection capacity intentionally small.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        bias: bool = False,
        irange: float = 0.005,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=bias)
        nn.init.uniform_(self.proj.weight, -irange, irange)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits_map = self.proj(features)
        return logits_map.mean(dim=(2, 3))
