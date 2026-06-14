from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def _init_uniform(module: nn.Module, irange: float) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.uniform_(module.weight, -irange, irange)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


class MaxoutConvBlock(nn.Module):
    """PyTorch counterpart of pylearn2 MaxoutConvC01B for NCHW tensors."""

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_pieces: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_shape: tuple[int, int] = (1, 1),
        pool_stride: tuple[int, int] = (1, 1),
        irange: float = 0.005,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_pieces = num_pieces
        self.conv = nn.Conv2d(
            in_channels,
            num_channels * num_pieces,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        _init_uniform(self.conv, irange)

        if pool_shape == (1, 1) and pool_stride == (1, 1):
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(kernel_size=pool_shape, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch, _, height, width = x.shape
        x = x.view(batch, self.num_channels, self.num_pieces, height, width)
        x = x.max(dim=2).values
        return self.pool(x)


class MaxoutLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_units: int,
        num_pieces: int,
        irange: float = 0.005,
    ) -> None:
        super().__init__()
        self.num_units = num_units
        self.num_pieces = num_pieces
        self.linear = nn.Linear(in_features, num_units * num_pieces)
        _init_uniform(self.linear, irange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.view(x.shape[0], self.num_units, self.num_pieces)
        return x.max(dim=2).values


@dataclass(frozen=True)
class ConvLayerSpec:
    num_channels: int
    num_pieces: int = 2
    kernel_size: int = 3
    padding: int = 1
    pool_shape: tuple[int, int] = (1, 1)
    pool_stride: tuple[int, int] = (1, 1)


@dataclass(frozen=True)
class ModelSpec:
    conv_layers: tuple[ConvLayerSpec, ...]
    default_middle: int
    fc_units: int | None = None
    fc_pieces: int = 5


class FitNetCNN(nn.Module):
    """FitNet-style Maxout CNN with addressable convolutional features."""

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        num_classes: int,
        spec: ModelSpec,
        irange: float = 0.005,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.feature_channels = [layer.num_channels for layer in spec.conv_layers]

        blocks: list[nn.Module] = []
        prev = input_channels
        for layer in spec.conv_layers:
            blocks.append(
                MaxoutConvBlock(
                    in_channels=prev,
                    num_channels=layer.num_channels,
                    num_pieces=layer.num_pieces,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    pool_shape=layer.pool_shape,
                    pool_stride=layer.pool_stride,
                    irange=irange,
                )
            )
            prev = layer.num_channels
        self.blocks = nn.ModuleList(blocks)

        flat_dim = self._infer_flat_dim(input_channels, image_size)
        if spec.fc_units is None:
            self.classifier = nn.Linear(flat_dim, num_classes)
            _init_uniform(self.classifier, irange)
        else:
            self.classifier = nn.Sequential(
                MaxoutLinear(flat_dim, spec.fc_units, spec.fc_pieces, irange=irange),
                nn.Linear(spec.fc_units, num_classes),
            )
            _init_uniform(self.classifier[-1], irange)

    @property
    def num_feature_layers(self) -> int:
        return len(self.blocks)

    def _forward_blocks(
        self,
        x: torch.Tensor,
        return_feature_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        middle = None
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == return_feature_index:
                middle = x
        return x, middle

    def _infer_flat_dim(self, input_channels: int, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            features, _ = self._forward_blocks(dummy)
            return int(features.flatten(1).shape[1])

    def forward_until(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= len(self.blocks):
            raise IndexError(f"middle layer index {layer_index} is out of range")
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == layer_index:
                return x
        raise RuntimeError("unreachable layer index")

    def forward(
        self,
        x: torch.Tensor,
        return_feature_index: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features, middle = self._forward_blocks(x, return_feature_index)
        logits = self.classifier(features.flatten(1))
        if return_feature_index is None:
            return logits
        if middle is None:
            raise IndexError(
                f"middle layer index {return_feature_index} is out of range"
            )
        return logits, middle


def _cifar_student_layers() -> tuple[ConvLayerSpec, ...]:
    channels = (
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
    )
    layers = []
    for idx, channels_i in enumerate(channels):
        if idx in (4, 10):
            layers.append(
                ConvLayerSpec(
                    num_channels=channels_i,
                    pool_shape=(2, 2),
                    pool_stride=(2, 2),
                )
            )
        elif idx == 16:
            layers.append(
                ConvLayerSpec(
                    num_channels=channels_i,
                    pool_shape=(8, 8),
                    pool_stride=(1, 1),
                )
            )
        else:
            layers.append(ConvLayerSpec(num_channels=channels_i))
    return tuple(layers)


def _cifar_teacher_layers() -> tuple[ConvLayerSpec, ...]:
    channels = (
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
    )
    layers = []
    for idx, channels_i in enumerate(channels):
        if idx in (4, 10):
            layers.append(
                ConvLayerSpec(
                    num_channels=channels_i,
                    pool_shape=(2, 2),
                    pool_stride=(2, 2),
                )
            )
        elif idx == 16:
            layers.append(
                ConvLayerSpec(
                    num_channels=channels_i,
                    pool_shape=(8, 8),
                    pool_stride=(1, 1),
                )
            )
        else:
            layers.append(ConvLayerSpec(num_channels=channels_i))
    return tuple(layers)


def _mnist_student_layers() -> tuple[ConvLayerSpec, ...]:
    return (
        ConvLayerSpec(16, kernel_size=5, padding=0),
        ConvLayerSpec(16, kernel_size=3, padding=0, pool_shape=(4, 4), pool_stride=(2, 2)),
        ConvLayerSpec(16, kernel_size=5, padding=3),
        ConvLayerSpec(16, kernel_size=3, padding=0, pool_shape=(4, 4), pool_stride=(2, 2)),
        ConvLayerSpec(12, num_pieces=4, kernel_size=3, padding=3),
        ConvLayerSpec(12, num_pieces=4, kernel_size=3, padding=0, pool_shape=(2, 2), pool_stride=(2, 2)),
    )


def _mnist_teacher_layers() -> tuple[ConvLayerSpec, ...]:
    return (
        ConvLayerSpec(32, kernel_size=5, padding=0),
        ConvLayerSpec(32, kernel_size=3, padding=0, pool_shape=(4, 4), pool_stride=(2, 2)),
        ConvLayerSpec(48, kernel_size=5, padding=3),
        ConvLayerSpec(48, kernel_size=3, padding=0, pool_shape=(4, 4), pool_stride=(2, 2)),
        ConvLayerSpec(64, num_pieces=4, kernel_size=3, padding=3),
        ConvLayerSpec(64, num_pieces=4, kernel_size=3, padding=0, pool_shape=(2, 2), pool_stride=(2, 2)),
    )


CIFAR_STUDENT_19 = ModelSpec(
    conv_layers=_cifar_student_layers(),
    default_middle=10,
    fc_units=500,
    fc_pieces=5,
)

CIFAR_TEACHER_19 = ModelSpec(
    conv_layers=_cifar_teacher_layers(),
    default_middle=1,
    fc_units=500,
    fc_pieces=5,
)

MNIST_STUDENT_6 = ModelSpec(
    conv_layers=_mnist_student_layers(),
    default_middle=3,
    fc_units=None,
)

MNIST_TEACHER_6 = ModelSpec(
    conv_layers=_mnist_teacher_layers(),
    default_middle=1,
    fc_units=None,
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
    image_size: int,
) -> FitNetCNN:
    try:
        spec = MODEL_SPECS[arch]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model arch '{arch}'. Known: {known}") from exc
    return FitNetCNN(
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
        spec=spec,
    )


def default_middle_index(arch: str) -> int:
    return MODEL_SPECS[arch].default_middle
