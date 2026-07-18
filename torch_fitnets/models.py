from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
        lr_scale: float = 0.05,
        max_kernel_norm: float = 0.9,
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
        self.conv.fitnet_lr_scale = lr_scale
        self.conv.fitnet_max_kernel_norm = max_kernel_norm

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
        max_col_norm: float = 1.9,
    ) -> None:
        super().__init__()
        self.num_units = num_units
        self.num_pieces = num_pieces
        self.linear = nn.Linear(in_features, num_units * num_pieces)
        _init_uniform(self.linear, irange)
        self.linear.fitnet_max_col_norm = max_col_norm

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
    lr_scale: float = 0.05
    max_kernel_norm: float = 0.9


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
                    lr_scale=layer.lr_scale,
                    max_kernel_norm=layer.max_kernel_norm,
                )
            )
            prev = layer.num_channels
        self.blocks = nn.ModuleList(blocks)

        flat_dim = self._infer_flat_dim(input_channels, image_size)
        if spec.fc_units is None:
            self.classifier = nn.Linear(flat_dim, num_classes)
            _init_uniform(self.classifier, irange)
            self.classifier.fitnet_max_col_norm = 1.9365
        else:
            self.classifier = nn.Sequential(
                MaxoutLinear(flat_dim, spec.fc_units, spec.fc_pieces, irange=irange),
                nn.Linear(spec.fc_units, num_classes),
            )
            _init_uniform(self.classifier[-1], irange)
            self.classifier[-1].fitnet_max_col_norm = 1.9365

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


class CifarResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


@dataclass(frozen=True)
class CifarResNetSpec:
    depth: int
    filters: tuple[int, int, int, int]
    default_middle: int = 2


class CifarResNet(nn.Module):
    """CIFAR ResNet compatible with common RepDistiller checkpoints."""

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        num_classes: int,
        spec: CifarResNetSpec,
    ) -> None:
        super().__init__()
        if input_channels != 3 or image_size != 32:
            raise ValueError("CIFAR ResNet teachers expect 3x32x32 inputs.")
        if (spec.depth - 2) % 6 != 0:
            raise ValueError("CIFAR basic-block ResNet depth must be 6n+2.")

        self.spec = spec
        self.feature_channels = list(spec.filters)
        self.inplanes = spec.filters[0]
        blocks_per_group = (spec.depth - 2) // 6

        self.conv1 = nn.Conv2d(
            3,
            spec.filters[0],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(spec.filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(spec.filters[1], blocks_per_group)
        self.layer2 = self._make_layer(spec.filters[2], blocks_per_group, stride=2)
        self.layer3 = self._make_layer(spec.filters[3], blocks_per_group, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(spec.filters[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_feature_layers(self) -> int:
        return 4

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [CifarResNetBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(CifarResNetBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features

    def forward_until(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= self.num_feature_layers:
            raise IndexError(f"middle layer index {layer_index} is out of range")
        return self._forward_features(x)[layer_index]

    def forward(
        self,
        x: torch.Tensor,
        return_feature_index: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_features(x)
        x = self.avgpool(features[-1])
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        if return_feature_index is None:
            return logits
        if return_feature_index < 0 or return_feature_index >= len(features):
            raise IndexError(
                f"middle layer index {return_feature_index} is out of range"
            )
        return logits, features[return_feature_index]


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


def _cifar_teacher_maxout_layers() -> tuple[ConvLayerSpec, ...]:
    """Goodfellow Maxout-CNN used as the FitNets CIFAR teacher.

    Three maxout convolutional layers (96-192-192, 2 pieces) with 4x4/4x4/2x2
    max-pooling, matching the architecture distilled in the FitNets paper.
    """
    return (
        ConvLayerSpec(
            num_channels=96,
            num_pieces=2,
            kernel_size=8,
            padding=4,
            pool_shape=(4, 4),
            pool_stride=(2, 2),
            max_kernel_norm=0.9,
        ),
        ConvLayerSpec(
            num_channels=192,
            num_pieces=2,
            kernel_size=8,
            padding=3,
            pool_shape=(4, 4),
            pool_stride=(2, 2),
            max_kernel_norm=1.9365,
        ),
        ConvLayerSpec(
            num_channels=192,
            num_pieces=2,
            kernel_size=5,
            padding=3,
            pool_shape=(2, 2),
            pool_stride=(2, 2),
            max_kernel_norm=1.9365,
        ),
    )


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

CIFAR_TEACHER_MAXOUT = ModelSpec(
    conv_layers=_cifar_teacher_maxout_layers(),
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
    "maxout_cifar_teacher": CIFAR_TEACHER_MAXOUT,
    "fitnet6_mnist_student": MNIST_STUDENT_6,
    "fitnet6_mnist_teacher": MNIST_TEACHER_6,
}

CIFAR_RESNET_SPECS: dict[str, CifarResNetSpec] = {
    "resnet32x4": CifarResNetSpec(depth=32, filters=(32, 64, 128, 256)),
}


def build_model(
    arch: str,
    input_channels: int,
    num_classes: int,
    image_size: int,
) -> FitNetCNN:
    try:
        spec = MODEL_SPECS[arch]
    except KeyError:
        spec = None

    if spec is not None:
        return FitNetCNN(
            input_channels=input_channels,
            image_size=image_size,
            num_classes=num_classes,
            spec=spec,
        )

    try:
        resnet_spec = CIFAR_RESNET_SPECS[arch]
    except KeyError as exc:
        known = ", ".join(sorted(set(MODEL_SPECS) | set(CIFAR_RESNET_SPECS)))
        raise ValueError(f"Unknown model arch '{arch}'. Known: {known}") from exc

    return CifarResNet(
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
        spec=resnet_spec,
    )


def default_middle_index(arch: str) -> int:
    if arch in MODEL_SPECS:
        return MODEL_SPECS[arch].default_middle
    if arch in CIFAR_RESNET_SPECS:
        return CIFAR_RESNET_SPECS[arch].default_middle
    known = ", ".join(sorted(set(MODEL_SPECS) | set(CIFAR_RESNET_SPECS)))
    raise ValueError(f"Unknown model arch '{arch}'. Known: {known}")


def _renorm_rows_(weight: torch.Tensor, max_norm: float) -> None:
    rows = weight.view(weight.shape[0], -1)
    norms = rows.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norms, max=1.0)
    rows.mul_(scale)


@torch.no_grad()
def apply_fitnet_constraints(model: nn.Module) -> None:
    """Apply the max-norm constraints used by the original FitNet YAML."""
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        max_kernel_norm = getattr(module, "fitnet_max_kernel_norm", None)
        if max_kernel_norm is not None:
            _renorm_rows_(weight, float(max_kernel_norm))
            continue
        max_col_norm = getattr(module, "fitnet_max_col_norm", None)
        if max_col_norm is not None:
            _renorm_rows_(weight, float(max_col_norm))
