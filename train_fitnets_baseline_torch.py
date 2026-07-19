"""Original FitNets baseline in PyTorch.

Stage 1 (hints): regress the student guided feature map onto the teacher hint
feature map with an MSE loss (via a convolutional regressor).
Stage 2 (KD): train the full student with cross-entropy + knowledge distillation
from the teacher's final logits.

The defaults follow the legacy source where practical. Explicit modern KD and
Kaiming-tail options provide a controlled baseline whose Stage 2 can be shared
with newer hint methods.
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
    feature_shape,
    freeze,
    run_fitnet_hint_epoch,
    run_stage2_epoch,
    set_student_stage1_trainable,
    unfreeze,
)
from torch_fitnets.models import (
    build_model,
    default_middle_index,
    initialize_fitnet_tail_for_stage2,
)
from torch_fitnets.losses import fitnet_teacher_weight
from torch_fitnets.optim import build_optimizer, scaled_param_groups
from torch_fitnets.regressors import ConvHintRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the original FitNets baseline in PyTorch.")
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
    parser.add_argument("--output-dir", default="./runs/fitnets_baseline_cifar100")
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
        help="Max gradient norm (0 = off). Stabilizes the deep unnormalized student front.",
    )

    parser.add_argument("--hint-epochs", type=int, default=40)
    parser.add_argument("--kd-epochs", type=int, default=288)
    parser.add_argument(
        "--hint-loss-reduction",
        default="legacy_c01b",
        choices=["legacy_c01b", "full"],
    )

    parser.add_argument("--lr-hint-front", type=float, default=0.005)
    parser.add_argument("--lr-hint-reg", type=float, default=0.005)
    parser.add_argument("--lr-kd", type=float, default=0.005)
    parser.add_argument("--kd-front-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--stage2-tail-init",
        default="fitnet",
        choices=["fitnet", "kaiming"],
        help="Initialization for the untrained student tail at the Stage 2 boundary.",
    )
    parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "sgd"])
    parser.add_argument("--rmsprop-alpha", type=float, default=0.9)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    # FitNets relies on max-norm constraints rather than L2 weight decay.
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--kd-ce-weight", type=float, default=1.0)
    parser.add_argument("--kd-kd-weight", type=float, default=4.0)
    parser.add_argument(
        "--kd-loss-mode",
        default="legacy",
        choices=["legacy", "modern"],
    )
    parser.add_argument(
        "--kd-weight-schedule",
        default="fitnets",
        choices=["fitnets", "fixed"],
    )
    parser.add_argument("--kd-final-weight", type=float, default=1.0)
    parser.add_argument("--kd-decay-start", type=int, default=5)
    parser.add_argument("--kd-decay-saturate", type=int, default=400)

    return parser.parse_args()


def resolve_arches(dataset: str, teacher_arch: str, student_arch: str) -> tuple[str, str]:
    if dataset == "mnist":
        default_teacher = "fitnet6_mnist_teacher"
        default_student = "fitnet6_mnist_student"
    else:
        default_teacher = "maxout_cifar_teacher"
        default_student = "fitnet19_cifar_student"

    if teacher_arch == "auto":
        teacher_arch = default_teacher
    if student_arch == "auto":
        student_arch = default_student
    return teacher_arch, student_arch


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


def resolve_checkpoint_state(payload: Any, key: str | None) -> dict[str, Any]:
    if key is not None and isinstance(payload, dict) and key in payload:
        return payload[key]
    return checkpoint_state(payload)


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
    state = resolve_checkpoint_state(payload, key)
    if any(name.startswith("module.") for name in state):
        state = {(name[7:] if name.startswith("module.") else name): value for name, value in state.items()}
    result = module.load_state_dict(state, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(f"warning: non-strict load from {path}: missing={result.missing_keys}, unexpected={result.unexpected_keys}")


def validate_middle_index(model: nn.Module, middle_index: int, name: str) -> None:
    if middle_index < 0 or middle_index >= model.num_feature_layers:
        raise ValueError(
            f"{name} middle index {middle_index} is out of range for "
            f"{model.num_feature_layers} feature layers."
        )


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def print_kd_stats(epoch: int, split: str, stats: Any, kd_weight: float) -> None:
    print(
        f"kd epoch {epoch:03d} {split}: "
        f"loss={stats.loss:.4f} ce={stats.ce:.4f} kd={stats.kd:.4f} "
        f"acc={stats.acc:.4f} entropy={stats.entropy:.4f} "
        f"logit_std={stats.logit_std:.4f} kd_weight={kd_weight:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.kd_weight_schedule == "fitnets":
        if args.kd_decay_saturate <= args.kd_decay_start:
            raise ValueError("--kd-decay-saturate must exceed --kd-decay-start")
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    grad_clip = args.grad_clip or None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_needed = args.hint_epochs > 0 or (args.kd_epochs > 0 and args.kd_kd_weight != 0)
    if args.teacher_ckpt is None and teacher_needed and not args.allow_random_teacher:
        raise ValueError(
            "--teacher-ckpt is required for the FitNets baseline. "
            "Use --allow-random-teacher only for smoke tests."
        )

    teacher_arch, student_arch = resolve_arches(args.dataset, args.teacher_arch, args.student_arch)

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
        default_middle_index(teacher_arch) if args.teacher_mid_index is None else args.teacher_mid_index
    )
    student_mid_index = (
        default_middle_index(student_arch) if args.student_mid_index is None else args.student_mid_index
    )
    validate_middle_index(teacher, teacher_mid_index, "teacher")
    validate_middle_index(student, student_mid_index, "student")

    load_module(teacher, args.teacher_ckpt, device, strict=args.strict_load)
    load_module(student, args.student_init_ckpt, device, strict=args.strict_load)
    if args.teacher_ckpt is None:
        print("warning: --teacher-ckpt was not provided; teacher starts randomly.")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    teacher_c, teacher_h, teacher_w = feature_shape(
        teacher, teacher_mid_index, dataset_info.input_channels, dataset_info.image_size, device
    )
    student_c, student_h, student_w = feature_shape(
        student, student_mid_index, dataset_info.input_channels, dataset_info.image_size, device
    )
    regressor = ConvHintRegressor(
        student_channels=student_c,
        teacher_channels=teacher_c,
        student_hw=(student_h, student_w),
        teacher_hw=(teacher_h, teacher_w),
        num_pieces=getattr(teacher.blocks[teacher_mid_index], "num_pieces", 1),
    ).to(device)

    metadata = {
        "dataset": args.dataset,
        "teacher_arch": teacher_arch,
        "student_arch": student_arch,
        "teacher_mid_index": teacher_mid_index,
        "student_mid_index": student_mid_index,
        "teacher_hint_shape": [teacher_c, teacher_h, teacher_w],
        "student_guided_shape": [student_c, student_h, student_w],
        "num_classes": dataset_info.num_classes,
        "method": "fitnets_baseline",
        "baseline_variant": (
            "legacy_source_compatible"
            if args.hint_loss_reduction == "legacy_c01b"
            and args.kd_loss_mode == "legacy"
            and args.kd_weight_schedule == "fitnets"
            and args.stage2_tail_init == "fitnet"
            and args.grad_clip == 0
            and not args.amp
            and args.optimizer == "rmsprop"
            and args.lr_hint_front == 0.005
            and args.lr_hint_reg == 0.005
            and args.lr_kd == 0.005
            and args.weight_decay == 0
            and args.batch_size == 128
            and args.whiten
            else "controlled"
        ),
    }
    run_config = vars(args).copy()
    run_config.update(metadata)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # Stage 1: hint regression on the student front + regressor.
    freeze(teacher)
    set_student_stage1_trainable(student, student_mid_index)
    unfreeze(regressor)

    if args.hint_epochs > 0:
        print("Stage 1 (hints): regress student guided features onto teacher hints")
        hint_param_groups = scaled_param_groups(
            student.blocks[: student_mid_index + 1],
            args.lr_hint_front,
            args.weight_decay,
        )
        hint_param_groups.append(
            {
                "params": list(regressor.parameters()),
                "lr": args.lr_hint_reg,
                "weight_decay": args.weight_decay,
            }
        )
        optimizer = build_optimizer(
            hint_param_groups,
            args.optimizer,
            lr=args.lr_hint_front,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rmsprop_alpha=args.rmsprop_alpha,
            rmsprop_eps=args.rmsprop_eps,
        )
        best_loss = float("inf")
        for epoch in range(1, args.hint_epochs + 1):
            train_stats = run_fitnet_hint_epoch(
                teacher, student, regressor, train_loader, optimizer, device,
                teacher_mid_index, student_mid_index, args.amp, scaler, grad_clip,
                args.hint_loss_reduction,
            )
            eval_stats = run_fitnet_hint_epoch(
                teacher, student, regressor, eval_loader, None, device,
                teacher_mid_index, student_mid_index, args.amp, None, None,
                args.hint_loss_reduction,
            )
            print(f"stage1 epoch {epoch:03d} train: hint_mse={train_stats.loss:.4f}")
            print(f"stage1 epoch {epoch:03d} eval:  hint_mse={eval_stats.loss:.4f}")
            if eval_stats.loss < best_loss:
                best_loss = eval_stats.loss
                save_checkpoint(
                    output_dir / "stage1_student_front_best.pt",
                    student=clone_state_dict(student),
                    regressor=clone_state_dict(regressor),
                    metadata=metadata,
                    epoch=epoch,
                    eval_hint_mse=eval_stats.loss,
                )
        load_module(
            student,
            str(output_dir / "stage1_student_front_best.pt"),
            device,
            strict=True,
            key="student",
        )

    # Stage 2: full-student knowledge distillation.
    freeze(teacher)
    unfreeze(student)
    if args.kd_epochs > 0 and args.stage2_tail_init == "kaiming":
        initialize_fitnet_tail_for_stage2(student, student_mid_index)
        print("Stage 2: initialized the untrained student tail with Kaiming weights")

    if args.kd_epochs > 0:
        print("Stage 2 (KD): train full student with CE + KD from teacher logits")
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
            kd_weight = args.kd_kd_weight
            if args.kd_weight_schedule == "fitnets":
                kd_weight = fitnet_teacher_weight(
                    epoch,
                    initial_weight=args.kd_kd_weight,
                    final_weight=args.kd_final_weight,
                    start=args.kd_decay_start,
                    saturate=args.kd_decay_saturate,
                )
            train_stats = run_stage2_epoch(
                teacher, student, train_loader, optimizer, device,
                args.kd_temperature, args.kd_ce_weight, kd_weight, args.amp,
                scaler, grad_clip, args.kd_loss_mode,
            )
            eval_stats = run_stage2_epoch(
                teacher, student, eval_loader, None, device,
                args.kd_temperature, args.kd_ce_weight, kd_weight, args.amp,
                None, None, args.kd_loss_mode,
            )
            print_kd_stats(epoch, "train", train_stats, kd_weight)
            print_kd_stats(epoch, "eval", eval_stats, kd_weight)
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
