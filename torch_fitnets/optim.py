from __future__ import annotations

from typing import Any

import torch


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
