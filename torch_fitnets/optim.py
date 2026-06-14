from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def scaled_param_groups(
    modules: Iterable[nn.Module],
    base_lr: float,
    weight_decay: float | None = None,
) -> list[dict[str, Any]]:
    """Build param groups that honor FitNet per-layer lr scales."""
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            params = []
            for param in module.parameters(recurse=False):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
            if not params:
                continue
            group: dict[str, Any] = {
                "params": params,
                "lr": base_lr * float(getattr(module, "fitnet_lr_scale", 1.0)),
            }
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            groups.append(group)
    return groups


def build_optimizer(
    param_groups: Any,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    rmsprop_alpha: float = 0.9,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            param_groups,
            lr=lr,
            alpha=rmsprop_alpha,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer must be 'rmsprop' or 'sgd'")
