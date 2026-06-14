from __future__ import annotations

import copy
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .losses import accuracy, kd_kl_loss, logits_entropy, logits_std
from .models import apply_fitnet_constraints


@dataclass
class EpochStats:
    loss: float
    ce: float
    kd: float
    acc: float
    entropy: float
    logit_std: float


class Meter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def student_front_parameters(model: nn.Module, middle_index: int) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx, block in enumerate(model.blocks):
        if idx <= middle_index:
            params.extend(block.parameters())
    return params


def student_tail_parameters(model: nn.Module, middle_index: int) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx, block in enumerate(model.blocks):
        if idx > middle_index:
            params.extend(block.parameters())
    params.extend(model.classifier.parameters())
    return params


def set_student_stage1_trainable(model: nn.Module, middle_index: int) -> None:
    freeze(model)
    for idx, block in enumerate(model.blocks):
        if idx <= middle_index:
            unfreeze(block)


def _new_stats() -> dict[str, Meter]:
    return {
        "loss": Meter(),
        "ce": Meter(),
        "kd": Meter(),
        "acc": Meter(),
        "entropy": Meter(),
        "logit_std": Meter(),
    }


def _to_epoch_stats(stats: dict[str, Meter]) -> EpochStats:
    return EpochStats(
        loss=stats["loss"].avg,
        ce=stats["ce"].avg,
        kd=stats["kd"].avg,
        acc=stats["acc"].avg,
        entropy=stats["entropy"].avg,
        logit_std=stats["logit_std"].avg,
    )


def _update_projection_stats(
    stats: dict[str, Meter],
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    ce_loss: torch.Tensor,
    kd_loss_value: torch.Tensor | None = None,
) -> None:
    batch = targets.size(0)
    kd_value = 0.0 if kd_loss_value is None else float(kd_loss_value.detach())
    stats["loss"].update(float(loss.detach()), batch)
    stats["ce"].update(float(ce_loss.detach()), batch)
    stats["kd"].update(kd_value, batch)
    stats["acc"].update(float(accuracy(logits, targets).detach()), batch)
    stats["entropy"].update(float(logits_entropy(logits).detach()), batch)
    stats["logit_std"].update(float(logits_std(logits).detach()), batch)


def _autocast(enabled: bool):
    if enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def _backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def run_stage0_epoch(
    teacher: nn.Module,
    teacher_proj: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    middle_index: int,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochStats:
    training = optimizer is not None
    teacher.eval()
    teacher_proj.train(training)
    stats = _new_stats()
    scaler_enabled = amp and device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            features = teacher.forward_until(inputs, middle_index)

        with _autocast(scaler_enabled):
            logits = teacher_proj(features)
            ce_loss = F.cross_entropy(logits, targets)
            loss = ce_loss

        if training:
            _backward_step(loss, optimizer, scaler)

        _update_projection_stats(stats, logits, targets, loss, ce_loss)

    return _to_epoch_stats(stats)


def run_stage1_epoch(
    teacher: nn.Module,
    teacher_proj: nn.Module,
    student: nn.Module,
    student_proj: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    teacher_middle_index: int,
    student_middle_index: int,
    temperature: float,
    ce_weight: float,
    kd_weight: float,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochStats:
    training = optimizer is not None
    teacher.eval()
    teacher_proj.eval()
    student.train(training)
    student_proj.train(training)
    stats = _new_stats()
    scaler_enabled = amp and device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(scaler_enabled):
            if kd_weight != 0:
                with torch.no_grad():
                    teacher_features = teacher.forward_until(inputs, teacher_middle_index)
                    teacher_mid_logits = teacher_proj(teacher_features)
            else:
                teacher_mid_logits = None

            student_features = student.forward_until(inputs, student_middle_index)
            student_mid_logits = student_proj(student_features)
            ce_loss = F.cross_entropy(student_mid_logits, targets)
            if teacher_mid_logits is not None:
                kd_loss_value = kd_kl_loss(
                    student_mid_logits,
                    teacher_mid_logits,
                    temperature=temperature,
                )
                loss = ce_weight * ce_loss + kd_weight * kd_loss_value
            else:
                kd_loss_value = None
                loss = ce_weight * ce_loss

        if training:
            _backward_step(loss, optimizer, scaler)
            apply_fitnet_constraints(student)

        _update_projection_stats(
            stats,
            student_mid_logits,
            targets,
            loss,
            ce_loss,
            kd_loss_value,
        )

    return _to_epoch_stats(stats)


def run_stage2_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    temperature: float,
    ce_weight: float,
    kd_weight: float,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochStats:
    training = optimizer is not None
    teacher.eval()
    student.train(training)
    stats = _new_stats()
    scaler_enabled = amp and device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(scaler_enabled):
            if kd_weight != 0:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
            else:
                teacher_logits = None

            student_logits = student(inputs)
            ce_loss = F.cross_entropy(student_logits, targets)
            if teacher_logits is not None:
                kd_loss_value = kd_kl_loss(
                    student_logits,
                    teacher_logits,
                    temperature=temperature,
                )
                loss = ce_weight * ce_loss + kd_weight * kd_loss_value
            else:
                kd_loss_value = None
                loss = ce_weight * ce_loss

        if training:
            _backward_step(loss, optimizer, scaler)
            apply_fitnet_constraints(student)

        _update_projection_stats(
            stats,
            student_logits,
            targets,
            loss,
            ce_loss,
            kd_loss_value,
        )

    return _to_epoch_stats(stats)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy(module.state_dict())
