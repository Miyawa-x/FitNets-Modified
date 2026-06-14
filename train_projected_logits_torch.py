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
    run_stage0_epoch,
    run_stage1_epoch,
    run_stage2_epoch,
    set_student_stage1_trainable,
    student_front_parameters,
    student_tail_parameters,
    unfreeze,
)
from torch_fitnets.models import build_model, default_middle_index
from torch_fitnets.projectors import GlobalAverageProjection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train projected-logit FitNets in PyTorch."
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar10", "cifar100", "mnist", "fake-cifar10", "fake-cifar100"],
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default="./runs/projected_fitnets_cifar100")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--teacher-arch", default="auto")
    parser.add_argument("--student-arch", default="auto")
    parser.add_argument("--teacher-ckpt", default=None)
    parser.add_argument("--teacher-proj-ckpt", default=None)
    parser.add_argument("--student-init-ckpt", default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--allow-random-teacher",
        action="store_true",
        help="Allow distillation stages to run without --teacher-ckpt. Useful only for smoke tests.",
    )

    parser.add_argument("--teacher-mid-index", type=int, default=None)
    parser.add_argument("--student-mid-index", type=int, default=None)
    parser.add_argument("--proj-bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--stage0-epochs", type=int, default=20)
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage2-epochs", type=int, default=160)

    parser.add_argument("--lr-stage0", type=float, default=0.01)
    parser.add_argument("--lr-stage1-front", type=float, default=0.01)
    parser.add_argument("--lr-stage1-proj", type=float, default=0.005)
    parser.add_argument("--lr-stage2", type=float, default=0.05)
    parser.add_argument("--stage2-front-lr-scale", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--proj-weight-decay", type=float, default=1e-3)

    parser.add_argument("--stage1-temperature", type=float, default=2.0)
    parser.add_argument("--stage1-ce-weight", type=float, default=0.5)
    parser.add_argument("--stage1-kd-weight", type=float, default=1.0)
    parser.add_argument("--stage2-temperature", type=float, default=4.0)
    parser.add_argument("--stage2-ce-weight", type=float, default=1.0)
    parser.add_argument("--stage2-kd-weight", type=float, default=1.0)

    return parser.parse_args()


def resolve_arches(dataset: str, teacher_arch: str, student_arch: str) -> tuple[str, str]:
    if dataset == "mnist":
        default_teacher = "fitnet6_mnist_teacher"
        default_student = "fitnet6_mnist_student"
    else:
        default_teacher = "fitnet19_cifar_teacher"
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
        for key in (
            "model",
            "model_state_dict",
            "state_dict",
            "teacher",
            "student",
            "teacher_proj",
            "student_proj",
        ):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
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
        state = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in state.items()
        }
    load_result = module.load_state_dict(state, strict=strict)
    if not strict and (load_result.missing_keys or load_result.unexpected_keys):
        print(
            f"warning: non-strict load from {path}: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
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


def print_stats(stage: str, epoch: int, split: str, stats: Any) -> None:
    print(
        f"{stage} epoch {epoch:03d} {split}: "
        f"loss={stats.loss:.4f} ce={stats.ce:.4f} kd={stats.kd:.4f} "
        f"acc={stats.acc:.4f} entropy={stats.entropy:.4f} "
        f"logit_std={stats.logit_std:.4f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_needed = (
        args.stage0_epochs > 0
        or (args.stage1_epochs > 0 and args.stage1_kd_weight != 0)
        or (args.stage2_epochs > 0 and args.stage2_kd_weight != 0)
    )
    if args.teacher_ckpt is None and teacher_needed and not args.allow_random_teacher:
        raise ValueError(
            "--teacher-ckpt is required for projected-logit distillation. "
            "Use --allow-random-teacher only for smoke tests."
        )

    teacher_arch, student_arch = resolve_arches(
        args.dataset,
        args.teacher_arch,
        args.student_arch,
    )

    train_loader, eval_loader, dataset_info = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    teacher = build_model(
        teacher_arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
    ).to(device)
    student = build_model(
        student_arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
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
        print("warning: --teacher-ckpt was not provided; teacher starts randomly.")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    teacher_proj = GlobalAverageProjection(
        in_channels=teacher.feature_channels[teacher_mid_index],
        num_classes=dataset_info.num_classes,
        bias=args.proj_bias,
    ).to(device)
    student_proj = GlobalAverageProjection(
        in_channels=student.feature_channels[student_mid_index],
        num_classes=dataset_info.num_classes,
        bias=args.proj_bias,
    ).to(device)

    metadata = {
        "dataset": args.dataset,
        "teacher_arch": teacher_arch,
        "student_arch": student_arch,
        "teacher_mid_index": teacher_mid_index,
        "student_mid_index": student_mid_index,
        "num_classes": dataset_info.num_classes,
    }
    run_config = vars(args).copy()
    run_config.update(metadata)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    freeze(teacher)
    teacher_proj_ready = False
    if args.teacher_proj_ckpt is not None:
        load_module(teacher_proj, args.teacher_proj_ckpt, device, strict=True, key="teacher_proj")
        teacher_proj_ready = True
    elif args.stage0_epochs <= 0 and args.stage1_epochs > 0 and args.stage1_kd_weight != 0:
        print("warning: Stage 0 is skipped and no --teacher-proj-ckpt was provided.")

    if args.stage0_epochs > 0 and args.teacher_proj_ckpt is None:
        print("Stage 0: train teacher projection")
        optimizer = torch.optim.SGD(
            teacher_proj.parameters(),
            lr=args.lr_stage0,
            momentum=args.momentum,
            weight_decay=args.proj_weight_decay,
        )
        best_acc = -1.0
        for epoch in range(1, args.stage0_epochs + 1):
            train_stats = run_stage0_epoch(
                teacher,
                teacher_proj,
                train_loader,
                optimizer,
                device,
                teacher_mid_index,
                args.amp,
                scaler,
            )
            eval_stats = run_stage0_epoch(
                teacher,
                teacher_proj,
                eval_loader,
                None,
                device,
                teacher_mid_index,
                args.amp,
            )
            print_stats("stage0", epoch, "train", train_stats)
            print_stats("stage0", epoch, "eval", eval_stats)

            if eval_stats.acc > best_acc:
                best_acc = eval_stats.acc
                save_checkpoint(
                    output_dir / "stage0_teacher_proj_best.pt",
                    teacher_proj=clone_state_dict(teacher_proj),
                    metadata=metadata,
                    epoch=epoch,
                    eval_acc=eval_stats.acc,
                )

        load_module(
            teacher_proj,
            str(output_dir / "stage0_teacher_proj_best.pt"),
            device,
            strict=True,
            key="teacher_proj",
        )
        teacher_proj_ready = True

    if args.stage1_epochs > 0 and args.stage1_kd_weight != 0 and not teacher_proj_ready:
        raise ValueError(
            "Stage 1 needs a trained teacher projection. "
            "Run Stage 0 or pass --teacher-proj-ckpt."
        )

    freeze(teacher)
    freeze(teacher_proj)
    set_student_stage1_trainable(student, student_mid_index)
    unfreeze(student_proj)

    if args.stage1_epochs > 0:
        print("Stage 1: train student front and student projection")
        optimizer = torch.optim.SGD(
            [
                {
                    "params": student_front_parameters(student, student_mid_index),
                    "lr": args.lr_stage1_front,
                },
                {
                    "params": student_proj.parameters(),
                    "lr": args.lr_stage1_proj,
                    "weight_decay": args.proj_weight_decay,
                },
            ],
            lr=args.lr_stage1_front,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        best_acc = -1.0
        for epoch in range(1, args.stage1_epochs + 1):
            train_stats = run_stage1_epoch(
                teacher,
                teacher_proj,
                student,
                student_proj,
                train_loader,
                optimizer,
                device,
                teacher_mid_index,
                student_mid_index,
                args.stage1_temperature,
                args.stage1_ce_weight,
                args.stage1_kd_weight,
                args.amp,
                scaler,
            )
            eval_stats = run_stage1_epoch(
                teacher,
                teacher_proj,
                student,
                student_proj,
                eval_loader,
                None,
                device,
                teacher_mid_index,
                student_mid_index,
                args.stage1_temperature,
                args.stage1_ce_weight,
                args.stage1_kd_weight,
                args.amp,
            )
            print_stats("stage1", epoch, "train", train_stats)
            print_stats("stage1", epoch, "eval", eval_stats)

            if eval_stats.acc > best_acc:
                best_acc = eval_stats.acc
                save_checkpoint(
                    output_dir / "stage1_student_front_best.pt",
                    student=clone_state_dict(student),
                    student_proj=clone_state_dict(student_proj),
                    metadata=metadata,
                    epoch=epoch,
                    eval_acc=eval_stats.acc,
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

    if args.stage2_epochs > 0:
        print("Stage 2: train full student with CE + final KD")
        optimizer = torch.optim.SGD(
            [
                {
                    "params": student_front_parameters(student, student_mid_index),
                    "lr": args.lr_stage2 * args.stage2_front_lr_scale,
                },
                {
                    "params": student_tail_parameters(student, student_mid_index),
                    "lr": args.lr_stage2,
                },
            ],
            lr=args.lr_stage2,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        best_acc = -1.0
        for epoch in range(1, args.stage2_epochs + 1):
            train_stats = run_stage2_epoch(
                teacher,
                student,
                train_loader,
                optimizer,
                device,
                args.stage2_temperature,
                args.stage2_ce_weight,
                args.stage2_kd_weight,
                args.amp,
                scaler,
            )
            eval_stats = run_stage2_epoch(
                teacher,
                student,
                eval_loader,
                None,
                device,
                args.stage2_temperature,
                args.stage2_ce_weight,
                args.stage2_kd_weight,
                args.amp,
            )
            print_stats("stage2", epoch, "train", train_stats)
            print_stats("stage2", epoch, "eval", eval_stats)

            if eval_stats.acc > best_acc:
                best_acc = eval_stats.acc
                save_checkpoint(
                    output_dir / "stage2_student_best.pt",
                    student=clone_state_dict(student),
                    metadata=metadata,
                    epoch=epoch,
                    eval_acc=eval_stats.acc,
                )

    print(f"done. checkpoints are in {output_dir}")


if __name__ == "__main__":
    main()
