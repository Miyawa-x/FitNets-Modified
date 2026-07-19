"""Relation-based FitNets for heterogeneous feature distillation.

Stage 1 matches pairwise distances and cosine similarities computed directly
from teacher and student middle features. Feature dimensions may differ and no
learnable projection is used. Stage 2 trains the complete student with the same
cross-entropy plus final-logit KD objective used by the FitNets baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from torch_fitnets.data import build_dataloaders
from torch_fitnets.engine import (
    clone_state_dict,
    freeze,
    run_relation_hint_epoch,
    run_stage2_epoch,
    set_student_stage1_trainable,
    unfreeze,
)
from torch_fitnets.models import build_model, default_middle_index
from torch_fitnets.optim import build_optimizer, scaled_param_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Relation FitNets without a feature projection."
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar10", "cifar100", "mnist", "fake-cifar10", "fake-cifar100"],
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Use GCN+ZCA whitening (must match the teacher's preprocessing).",
    )
    parser.add_argument("--output-dir", default="./runs/relation_fitnets_cifar100")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--teacher-arch", default="auto")
    parser.add_argument("--student-arch", default="auto")
    parser.add_argument("--teacher-ckpt", default=None)
    parser.add_argument("--student-init-ckpt", default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--allow-random-teacher",
        action="store_true",
        help="Allow training without --teacher-ckpt. Useful only for smoke tests.",
    )
    parser.add_argument("--teacher-mid-index", type=int, default=None)
    parser.add_argument("--student-mid-index", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Maximum gradient norm (0 disables clipping).",
    )

    parser.add_argument("--relation-epochs", type=int, default=40)
    parser.add_argument("--kd-epochs", type=int, default=288)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--similarity-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--lr-relation-front", type=float, default=0.005)
    parser.add_argument("--lr-kd", type=float, default=0.005)
    parser.add_argument("--kd-front-lr-scale", type=float, default=1.0)
    parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "sgd"])
    parser.add_argument("--rmsprop-alpha", type=float, default=0.9)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--kd-ce-weight", type=float, default=1.0)
    parser.add_argument("--kd-kd-weight", type=float, default=4.0)
    return parser.parse_args()


def resolve_arches(dataset: str, teacher_arch: str, student_arch: str) -> tuple[str, str]:
    if dataset == "mnist":
        default_teacher = "fitnet6_mnist_teacher"
        default_student = "fitnet6_mnist_student"
    else:
        default_teacher = "maxout_cifar_teacher"
        default_student = "fitnet19_cifar_student"
    return (
        default_teacher if teacher_arch == "auto" else teacher_arch,
        default_student if student_arch == "auto" else student_arch,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict", "teacher", "student"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload
    raise TypeError("checkpoint does not contain a state dict")


def load_module(
    module: nn.Module,
    path: str | None,
    device: torch.device,
    strict: bool,
    key: str | None = None,
) -> None:
    if path is None:
        return
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if key is not None and isinstance(payload, dict) and key in payload:
        state = payload[key]
    else:
        state = checkpoint_state(payload)
    if any(name.startswith("module.") for name in state):
        state = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in state.items()
        }
    result = module.load_state_dict(state, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(
            f"warning: non-strict load from {path}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def validate_middle_index(model: nn.Module, middle_index: int, name: str) -> None:
    if middle_index < 0 or middle_index >= model.num_feature_layers:
        raise ValueError(
            f"{name} middle index {middle_index} is out of range for "
            f"{model.num_feature_layers} feature layers."
        )


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def print_kd_stats(epoch: int, split: str, stats: Any) -> None:
    print(
        f"kd epoch {epoch:03d} {split}: "
        f"loss={stats.loss:.4f} ce={stats.ce:.4f} kd={stats.kd:.4f} "
        f"acc={stats.acc:.4f} entropy={stats.entropy:.4f} "
        f"logit_std={stats.logit_std:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for relational losses")
    if min(args.distance_weight, args.similarity_weight, args.energy_weight) < 0:
        raise ValueError("relation loss weights must be non-negative")
    if (
        args.relation_epochs > 0
        and args.distance_weight == 0
        and args.similarity_weight == 0
        and args.energy_weight == 0
    ):
        raise ValueError("at least one relation loss weight must be positive")
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_needed = args.relation_epochs > 0 or (
        args.kd_epochs > 0 and args.kd_kd_weight != 0
    )
    if args.teacher_ckpt is None and teacher_needed and not args.allow_random_teacher:
        raise ValueError(
            "--teacher-ckpt is required for Relation FitNets. "
            "Use --allow-random-teacher only for smoke tests."
        )

    teacher_arch, student_arch = resolve_arches(
        args.dataset, args.teacher_arch, args.student_arch
    )
    train_loader, eval_loader, dataset_info = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        whiten=args.whiten,
    )
    teacher = build_model(
        teacher_arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
        image_size=dataset_info.image_size,
    ).to(device)
    student = build_model(
        student_arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
        image_size=dataset_info.image_size,
    ).to(device)

    teacher_mid_index = (
        default_middle_index(teacher_arch)
        if args.teacher_mid_index is None
        else args.teacher_mid_index
    )
    student_mid_index = (
        default_middle_index(student_arch)
        if args.student_mid_index is None
        else args.student_mid_index
    )
    validate_middle_index(teacher, teacher_mid_index, "teacher")
    validate_middle_index(student, student_mid_index, "student")
    load_module(teacher, args.teacher_ckpt, device, strict=args.strict_load)
    load_module(student, args.student_init_ckpt, device, strict=args.strict_load)
    if args.teacher_ckpt is None:
        print("warning: teacher starts randomly; results are only a smoke test.")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    metadata = {
        "dataset": args.dataset,
        "teacher_arch": teacher_arch,
        "student_arch": student_arch,
        "teacher_mid_index": teacher_mid_index,
        "student_mid_index": student_mid_index,
        "num_classes": dataset_info.num_classes,
        "method": "relation_fitnets",
        "relation": "distance_plus_centered_cosine_plus_log_rms_energy",
    }
    run_config = vars(args).copy()
    run_config.update(metadata)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    freeze(teacher)
    set_student_stage1_trainable(student, student_mid_index)

    if args.relation_epochs > 0:
        print("Stage 1 (relations): match feature distances and similarities")
        optimizer = build_optimizer(
            scaled_param_groups(
                student.blocks[: student_mid_index + 1],
                args.lr_relation_front,
                args.weight_decay,
            ),
            args.optimizer,
            lr=args.lr_relation_front,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rmsprop_alpha=args.rmsprop_alpha,
            rmsprop_eps=args.rmsprop_eps,
        )
        best_loss = float("inf")
        for epoch in range(1, args.relation_epochs + 1):
            train_stats = run_relation_hint_epoch(
                teacher,
                student,
                train_loader,
                optimizer,
                device,
                teacher_mid_index,
                student_mid_index,
                args.distance_weight,
                args.similarity_weight,
                args.energy_weight,
                args.amp,
                scaler,
                grad_clip,
            )
            eval_stats = run_relation_hint_epoch(
                teacher,
                student,
                eval_loader,
                None,
                device,
                teacher_mid_index,
                student_mid_index,
                args.distance_weight,
                args.similarity_weight,
                args.energy_weight,
                args.amp,
            )
            print(
                f"relation epoch {epoch:03d} train: loss={train_stats.loss:.6f} "
                f"distance={train_stats.distance:.6f} "
                f"similarity={train_stats.similarity:.6f} "
                f"energy={train_stats.energy:.6f} "
                f"student_rms={train_stats.student_feature_rms:.6e} "
                f"teacher_rms={train_stats.teacher_feature_rms:.6e}"
            )
            print(
                f"relation epoch {epoch:03d} eval:  loss={eval_stats.loss:.6f} "
                f"distance={eval_stats.distance:.6f} "
                f"similarity={eval_stats.similarity:.6f} "
                f"energy={eval_stats.energy:.6f} "
                f"student_rms={eval_stats.student_feature_rms:.6e} "
                f"teacher_rms={eval_stats.teacher_feature_rms:.6e}"
            )
            if eval_stats.loss < best_loss:
                best_loss = eval_stats.loss
                save_checkpoint(
                    output_dir / "stage1_student_front_best.pt",
                    student=clone_state_dict(student),
                    metadata=metadata,
                    epoch=epoch,
                    eval_relation_loss=eval_stats.loss,
                    eval_distance_loss=eval_stats.distance,
                    eval_similarity_loss=eval_stats.similarity,
                    eval_energy_loss=eval_stats.energy,
                )
        load_module(
            student,
            str(output_dir / "stage1_student_front_best.pt"),
            device,
            strict=True,
            key="student",
        )

    freeze(teacher)
    unfreeze(student)

    if args.kd_epochs > 0:
        print("Stage 2 (KD): train full student with CE + teacher final-logit KD")
        kd_param_groups = scaled_param_groups(
            student.blocks[: student_mid_index + 1],
            args.lr_kd * args.kd_front_lr_scale,
            args.weight_decay,
        )
        kd_param_groups.extend(
            scaled_param_groups(
                list(student.blocks[student_mid_index + 1 :]) + [student.classifier],
                args.lr_kd,
                args.weight_decay,
            )
        )
        optimizer = build_optimizer(
            kd_param_groups,
            args.optimizer,
            lr=args.lr_kd,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rmsprop_alpha=args.rmsprop_alpha,
            rmsprop_eps=args.rmsprop_eps,
        )
        best_acc = -1.0
        for epoch in range(1, args.kd_epochs + 1):
            train_stats = run_stage2_epoch(
                teacher,
                student,
                train_loader,
                optimizer,
                device,
                args.kd_temperature,
                args.kd_ce_weight,
                args.kd_kd_weight,
                args.amp,
                scaler,
                grad_clip,
            )
            eval_stats = run_stage2_epoch(
                teacher,
                student,
                eval_loader,
                None,
                device,
                args.kd_temperature,
                args.kd_ce_weight,
                args.kd_kd_weight,
                args.amp,
            )
            print_kd_stats(epoch, "train", train_stats)
            print_kd_stats(epoch, "eval", eval_stats)
            if eval_stats.acc > best_acc:
                best_acc = eval_stats.acc
                save_checkpoint(
                    output_dir / "stage2_student_best.pt",
                    student=clone_state_dict(student),
                    metadata=metadata,
                    epoch=epoch,
                    eval_acc=eval_stats.acc,
                )
        print(f"done. best student eval_acc={best_acc:.4f}")

    print(f"done. checkpoints are in {output_dir}")


if __name__ == "__main__":
    main()
