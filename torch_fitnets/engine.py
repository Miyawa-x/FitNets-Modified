from __future__ import annotations

import copy
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .losses import (
    accuracy,
    feature_energy_loss,
    hint_mse_loss,
    kd_kl_loss,
    legacy_kd_cross_entropy_loss,
    logits_entropy,
    logits_std,
    relation_distance_loss,
    relation_similarity_loss,
)
from .models import apply_fitnet_constraints


@dataclass
class EpochStats:
    loss: float
    ce: float
    kd: float
    acc: float
    entropy: float
    logit_std: float


@dataclass
class HintEpochStats:
    loss: float


@dataclass
class RelationEpochStats:
    loss: float
    distance: float
    similarity: float
    energy: float
    student_feature_rms: float
    teacher_feature_rms: float


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


def _clip_gradients(optimizer: torch.optim.Optimizer, max_norm: float) -> None:
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, max_norm)


def _backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip: float | None = None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            _clip_gradients(optimizer, grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip:
            _clip_gradients(optimizer, grad_clip)
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
    grad_clip: float | None = None,
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

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, _autocast(scaler_enabled):
            logits = teacher_proj(features)
            ce_loss = F.cross_entropy(logits, targets)
            loss = ce_loss

        if training:
            _backward_step(loss, optimizer, scaler, grad_clip)

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
    grad_clip: float | None = None,
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

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, _autocast(scaler_enabled):
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
            _backward_step(loss, optimizer, scaler, grad_clip)
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
    grad_clip: float | None = None,
    kd_loss_mode: str = "modern",
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

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, _autocast(scaler_enabled):
            if kd_weight != 0:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
            else:
                teacher_logits = None

            student_logits = student(inputs)
            ce_loss = F.cross_entropy(student_logits, targets)
            if teacher_logits is not None:
                if kd_loss_mode == "modern":
                    kd_loss_value = kd_kl_loss(
                        student_logits,
                        teacher_logits,
                        temperature=temperature,
                    )
                elif kd_loss_mode == "legacy":
                    kd_loss_value = legacy_kd_cross_entropy_loss(
                        student_logits,
                        teacher_logits,
                        temperature=temperature,
                    )
                else:
                    raise ValueError(f"unknown KD loss mode: {kd_loss_mode}")
                loss = ce_weight * ce_loss + kd_weight * kd_loss_value
            else:
                kd_loss_value = None
                loss = ce_weight * ce_loss

        if training:
            _backward_step(loss, optimizer, scaler, grad_clip)
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


@torch.no_grad()
def feature_shape(
    model: nn.Module,
    middle_index: int,
    input_channels: int,
    image_size: int,
    device: torch.device,
) -> tuple[int, int, int]:
    """Return (channels, height, width) of a model's middle feature map."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, input_channels, image_size, image_size, device=device)
    feat = model.forward_until(dummy, middle_index)
    model.train(was_training)
    return int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])


def run_fitnet_hint_epoch(
    teacher: nn.Module,
    student: nn.Module,
    regressor: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    teacher_middle_index: int,
    student_middle_index: int,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = None,
    hint_loss_reduction: str = "full",
) -> HintEpochStats:
    """Original FitNets Stage 1: regress student guided features onto teacher hints."""
    training = optimizer is not None
    teacher.eval()
    student.train(training)
    regressor.train(training)
    meter = Meter()
    scaler_enabled = amp and device.type == "cuda"

    for inputs, _targets in loader:
        inputs = inputs.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_hint = teacher.forward_until(inputs, teacher_middle_index)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, _autocast(scaler_enabled):
            student_feat = student.forward_until(inputs, student_middle_index)
            student_hint = regressor(student_feat)
            loss = hint_mse_loss(
                student_hint,
                teacher_hint,
                reduction=hint_loss_reduction,
            )

        if training:
            _backward_step(loss, optimizer, scaler, grad_clip)
            apply_fitnet_constraints(student)
            apply_fitnet_constraints(regressor)

        meter.update(float(loss.detach()), inputs.size(0))

    return HintEpochStats(loss=meter.avg)


def run_relation_hint_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    teacher_middle_index: int,
    student_middle_index: int,
    distance_weight: float,
    similarity_weight: float,
    energy_weight: float,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = None,
) -> RelationEpochStats:
    """Train the student front by matching teacher batch geometry directly."""
    training = optimizer is not None
    teacher.eval()
    student.train(training)
    loss_meter = Meter()
    distance_meter = Meter()
    similarity_meter = Meter()
    energy_meter = Meter()
    student_rms_meter = Meter()
    teacher_rms_meter = Meter()

    for inputs, _targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        if inputs.size(0) < 2:
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_feature = teacher.forward_until(inputs, teacher_middle_index)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            # The deep, tiny-initialized Maxout front underflows in FP16 before h10.
            student_feature = student.forward_until(inputs, student_middle_index)

            distance = relation_distance_loss(student_feature, teacher_feature)
            similarity = relation_similarity_loss(student_feature, teacher_feature)
            energy = feature_energy_loss(student_feature, teacher_feature)
            loss = (
                distance_weight * distance
                + similarity_weight * similarity
                + energy_weight * energy
            )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("relation hint loss became non-finite")

        if training:
            _backward_step(loss, optimizer, scaler, grad_clip)
            apply_fitnet_constraints(student)

        batch = inputs.size(0)
        loss_meter.update(float(loss.detach()), batch)
        distance_meter.update(float(distance.detach()), batch)
        similarity_meter.update(float(similarity.detach()), batch)
        energy_meter.update(float(energy.detach()), batch)
        student_rms_meter.update(
            float(student_feature.detach().float().pow(2).mean().sqrt()),
            batch,
        )
        teacher_rms_meter.update(
            float(teacher_feature.detach().float().pow(2).mean().sqrt()),
            batch,
        )

    return RelationEpochStats(
        loss=loss_meter.avg,
        distance=distance_meter.avg,
        similarity=similarity_meter.avg,
        energy=energy_meter.avg,
        student_feature_rms=student_rms_meter.avg,
        teacher_feature_rms=teacher_rms_meter.avg,
    )


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy(module.state_dict())
