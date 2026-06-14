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
