from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torch_fitnets.data import build_dataloaders
from torch_fitnets.losses import accuracy
from torch_fitnets.models import apply_fitnet_constraints, build_model
from torch_fitnets.optim import build_optimizer, scaled_param_groups


class Meter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR teacher in PyTorch.")
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
        help="Use GCN+ZCA whitening (matches the original FitNets/Maxout preprocessing).",
    )
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--output", default="checkpoints/cifar100_teacher.pt")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a teacher checkpoint. Legacy checkpoints restore model and epoch only.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=288)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "sgd"])
    parser.add_argument("--rmsprop-alpha", type=float, default=0.9)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    # The FitNets/Maxout teacher relies on max-norm constraints rather than L2
    # weight decay, so the default is 0 to stay aligned with the original recipe.
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--milestones", type=int, nargs="*", default=[])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Maximum gradient norm (0 disables clipping).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_arch(dataset: str, arch: str) -> str:
    if arch != "auto":
        return arch
    if dataset == "mnist":
        return "fitnet6_mnist_teacher"
    return "maxout_cifar_teacher"


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def make_scaler(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
    scaler: Any,
    grad_clip: float | None,
) -> tuple[float, float, float, float, int]:
    training = optimizer is not None
    model.train(training)
    loss_meter = Meter()
    acc_meter = Meter()
    logit_abs_meter = Meter()
    grad_norm_meter = Meter()
    amp_skips = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, autocast_context(device, amp):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("teacher produced non-finite logits")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("teacher produced a non-finite CE loss")

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip if grad_clip is not None else float("inf"),
                    error_if_nonfinite=False,
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale() < scale_before
                if step_skipped:
                    amp_skips += 1
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip if grad_clip is not None else float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                step_skipped = False
            if not step_skipped:
                apply_fitnet_constraints(model)
            if bool(torch.isfinite(grad_norm)):
                grad_norm_meter.update(float(grad_norm.detach()), targets.size(0))

        batch = targets.size(0)
        loss_meter.update(float(loss.detach()), batch)
        acc_meter.update(float(accuracy(logits, targets).detach()), batch)
        logit_abs_meter.update(float(logits.detach().abs().max()), batch)

    return (
        loss_meter.avg,
        acc_meter.avg,
        logit_abs_meter.avg,
        grad_norm_meter.avg,
        amp_skips,
    )


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint {path} is not a dictionary")
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, eval_loader, dataset_info = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        whiten=args.whiten,
    )
    arch = resolve_arch(args.dataset, args.arch)
    model = build_model(
        arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
        image_size=dataset_info.image_size,
    ).to(device)

    optimizer = build_optimizer(
        scaled_param_groups([model], args.lr, args.weight_decay),
        args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rmsprop_alpha=args.rmsprop_alpha,
        rmsprop_eps=args.rmsprop_eps,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )
    scaler = make_scaler(device, args.amp)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    output = Path(args.output)
    config_path = output.with_suffix(".json")
    config = vars(args).copy()
    config.update(
        {
            "arch": arch,
            "num_classes": dataset_info.num_classes,
            "input_channels": dataset_info.input_channels,
        }
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))

    start_epoch = 1
    best_acc = -1.0
    if args.resume is not None:
        resume_payload = load_checkpoint(args.resume, device)
        resume_arch = resume_payload.get("arch")
        if resume_arch is not None and resume_arch != arch:
            raise ValueError(
                f"resume checkpoint arch is {resume_arch!r}, expected {arch!r}"
            )
        teacher_state = resume_payload.get("teacher", resume_payload)
        model.load_state_dict(teacher_state, strict=True)
        resume_epoch = int(resume_payload.get("epoch", 0))
        start_epoch = resume_epoch + 1
        best_acc = float(
            resume_payload.get("best_acc", resume_payload.get("eval_acc", -1.0))
        )

        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        else:
            print("warning: legacy checkpoint has no optimizer state; RMSProp restarts.")
        if "scheduler" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler"])
        elif args.milestones:
            print("warning: legacy checkpoint has no scheduler state.")
        if "scaler" in resume_payload and scaler is not None:
            scaler.load_state_dict(resume_payload["scaler"])

        if output.exists() and output.resolve() != Path(args.resume).resolve():
            best_payload = load_checkpoint(output, device)
            best_acc = max(best_acc, float(best_payload.get("eval_acc", -1.0)))
        print(
            f"resumed teacher from {args.resume} at epoch {resume_epoch}; "
            f"continuing with epoch {start_epoch}."
        )

    if start_epoch > args.epochs:
        raise ValueError(
            f"resume checkpoint already reached epoch {start_epoch - 1}, "
            f"but --epochs is {args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        (
            train_loss,
            train_acc,
            train_logit_abs,
            train_grad_norm,
            train_amp_skips,
        ) = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.amp,
            scaler,
            grad_clip,
        )
        eval_loss, eval_acc, eval_logit_abs, _, _ = run_epoch(
            model,
            eval_loader,
            None,
            device,
            args.amp,
            scaler,
            None,
        )
        scheduler.step()

        print(
            f"teacher epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
            f"train_logit_abs={train_logit_abs:.4f} "
            f"eval_logit_abs={eval_logit_abs:.4f} "
            f"grad_norm={train_grad_norm:.4f} "
            f"amp_skips={train_amp_skips} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        is_best = eval_acc > best_acc
        if is_best:
            best_acc = eval_acc

        checkpoint = {
            "teacher": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "arch": arch,
            "epoch": epoch,
            "eval_acc": eval_acc,
            "best_acc": best_acc,
            "config": config,
        }
        save_checkpoint(
            output.with_name(output.stem + "_last.pt"),
            **checkpoint,
        )
        if is_best:
            save_checkpoint(
                output,
                **checkpoint,
            )

    print(f"done. best teacher checkpoint: {output} (best eval_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
