from __future__ import annotations

import torch
from torch import nn


class ConvHintRegressor(nn.Module):
    """Convolutional regressor for the original FitNets hint stage.

    Maps a student guided feature map onto the teacher hint feature map so the
    two can be compared with an MSE loss. The kernel size is chosen so that a
    valid (no-padding) convolution turns the student spatial size into the
    teacher spatial size, mirroring ``generateConvRegressor`` in the original
    pylearn2 code (``ks = student_shape - teacher_shape + 1``).
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        student_hw: tuple[int, int],
        teacher_hw: tuple[int, int],
        num_pieces: int = 2,
        irange: float = 0.05,
        max_kernel_norm: float = 0.9,
    ) -> None:
        super().__init__()
        if num_pieces < 1:
            raise ValueError("num_pieces must be positive")
        self.teacher_channels = teacher_channels
        self.num_pieces = num_pieces
        kh = student_hw[0] - teacher_hw[0] + 1
        kw = student_hw[1] - teacher_hw[1] + 1
        if kh < 1 or kw < 1:
            raise ValueError(
                "Teacher hint is larger than the student guided feature map "
                f"(student={student_hw}, teacher={teacher_hw}); a convolutional "
                "regressor needs the student to be at least as large. Pick a "
                "different student/teacher middle index."
            )
        self.conv = nn.Conv2d(
            student_channels,
            teacher_channels * num_pieces,
            kernel_size=(kh, kw),
            bias=True,
        )
        nn.init.uniform_(self.conv.weight, -irange, irange)
        nn.init.zeros_(self.conv.bias)
        # Honour the same max-norm constraint applied to FitNet conv layers.
        self.conv.fitnet_max_kernel_norm = max_kernel_norm

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(features)
        batch, _, height, width = outputs.shape
        outputs = outputs.view(
            batch,
            self.teacher_channels,
            self.num_pieces,
            height,
            width,
        )
        return outputs.max(dim=2).values
