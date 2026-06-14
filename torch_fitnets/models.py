from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FitNetCNN(nn.Module):
    """FitNet-style CNN with addressable middle features."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Iterable[int],
        pool_after: Iterable[int],
    ) -> None:
        super().__init__()
        channels = list(channels)
        self.pool_after = set(pool_after)
        self.feature_channels = channels

        blocks = []
        prev = input_channels
        for out_channels in channels:
            blocks.append(ConvBlock(prev, out_channels))
            prev = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    @property
    def num_feature_layers(self) -> int:
        return len(self.blocks)

    def forward_until(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= len(self.blocks):
            raise IndexError(f"middle layer index {layer_index} is out of range")
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.pool_after:
                x = self.pool(x)
            if idx == layer_index:
                return x
        raise RuntimeError("unreachable layer index")

    def forward(
        self,
        x: torch.Tensor,
        return_feature_index: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        middle = None
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.pool_after:
                x = self.pool(x)
            if idx == return_feature_index:
                middle = x

        logits = self.classifier(x)
        if return_feature_index is None:
            return logits
        if middle is None:
            raise IndexError(
                f"middle layer index {return_feature_index} is out of range"
            )
        return logits, middle


@dataclass(frozen=True)
class ModelSpec:
    channels: tuple[int, ...]
    pool_after: tuple[int, ...]
    default_middle: int


CIFAR_STUDENT_19 = ModelSpec(
    channels=(
        32,
        32,
        32,
        48,
        48,
        80,
        80,
        80,
        80,
        80,
        80,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    pool_after=(4, 10),
    default_middle=10,
)

CIFAR_TEACHER_19 = ModelSpec(
    channels=(
        96,
        96,
        96,
        128,
        128,
        192,
        192,
        192,
        192,
        192,
        192,
        256,
        256,
        256,
        256,
        256,
        256,
    ),
    pool_after=(4, 10),
    default_middle=10,
)

MNIST_STUDENT_6 = ModelSpec(
    channels=(16, 16, 16, 16, 12, 12),
    pool_after=(1, 3),
    default_middle=3,
)

MNIST_TEACHER_6 = ModelSpec(
    channels=(32, 32, 48, 48, 64, 64),
    pool_after=(1, 3),
    default_middle=3,
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "fitnet19_cifar_student": CIFAR_STUDENT_19,
    "fitnet19_cifar_teacher": CIFAR_TEACHER_19,
    "fitnet6_mnist_student": MNIST_STUDENT_6,
    "fitnet6_mnist_teacher": MNIST_TEACHER_6,
}


def build_model(
    arch: str,
    input_channels: int,
    num_classes: int,
) -> FitNetCNN:
    try:
        spec = MODEL_SPECS[arch]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model arch '{arch}'. Known: {known}") from exc
    return FitNetCNN(
        input_channels=input_channels,
        num_classes=num_classes,
        channels=spec.channels,
        pool_after=spec.pool_after,
    )


def default_middle_index(arch: str) -> int:
    return MODEL_SPECS[arch].default_middle
