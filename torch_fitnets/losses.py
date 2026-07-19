from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student) for logits."""
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (
        temperature ** 2
    )


def hint_mse_loss(
    student_hint: torch.Tensor,
    teacher_hint: torch.Tensor,
) -> torch.Tensor:
    """Original FitNets hint objective: 0.5 * ||teacher - student||^2.

    The squared error is summed over the feature dimensions and averaged over
    the batch, matching ``HintCost`` from the legacy pylearn2 implementation.
    """
    diff = student_hint - teacher_hint
    return 0.5 * diff.pow(2).flatten(1).sum(dim=1).mean()


def _off_diagonal_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("relational distillation requires at least two samples")
    return ~torch.eye(batch_size, dtype=torch.bool, device=device)


def relation_distance_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Match normalized pairwise distances without aligning feature dimensions."""
    student = student_feature.flatten(1).float()
    teacher = teacher_feature.detach().flatten(1).float()
    mask = _off_diagonal_mask(student.shape[0], student.device)

    student_distance = torch.cdist(student, student, p=2)
    teacher_distance = torch.cdist(teacher, teacher, p=2)
    student_scale = student_distance[mask].mean().clamp_min(eps)
    teacher_scale = teacher_distance[mask].mean().clamp_min(eps)

    return F.smooth_l1_loss(
        student_distance[mask] / student_scale,
        teacher_distance[mask] / teacher_scale,
    )


def relation_similarity_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Match pairwise cosine geometry in each model's native feature space."""
    student = student_feature.flatten(1).float()
    teacher = teacher_feature.detach().flatten(1).float()
    student = F.normalize(student - student.mean(dim=0, keepdim=True), dim=1, eps=eps)
    teacher = F.normalize(teacher - teacher.mean(dim=0, keepdim=True), dim=1, eps=eps)
    mask = _off_diagonal_mask(student.shape[0], student.device)
    student_similarity = student @ student.t()
    teacher_similarity = teacher @ teacher.t()
    return F.smooth_l1_loss(
        student_similarity[mask],
        teacher_similarity[mask],
    )


def feature_energy_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Anchor relation learning with per-sample log-RMS feature energy."""
    student = student_feature.flatten(1).float()
    teacher = teacher_feature.detach().flatten(1).float()
    student_log_rms = 0.5 * torch.log(student.pow(2).mean(dim=1) + eps)
    teacher_log_rms = 0.5 * torch.log(teacher.pow(2).mean(dim=1) + eps)
    return F.smooth_l1_loss(student_log_rms, teacher_log_rms)


def logits_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


def logits_std(logits: torch.Tensor) -> torch.Tensor:
    centered = logits - logits.mean(dim=1, keepdim=True)
    return centered.pow(2).mean(dim=1).sqrt().mean()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1).eq(targets).float().mean()


@dataclass
class LossWeights:
    ce: float
    kd: float
