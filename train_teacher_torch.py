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
from torch_fitnets.models import build_model
from torch_fitnets.optim import build_optimizer


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
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--output", default="checkpoints/cifar100_teacher.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=288)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "sgd"])
    parser.add_argument("--rmsprop-alpha", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="*", default=[])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
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
    return "fitnet19_cifar_teacher"


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
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_meter = Meter()
    acc_meter = Meter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch = targets.size(0)
        loss_meter.update(float(loss.detach()), batch)
        acc_meter.update(float(accuracy(logits, targets).detach()), batch)

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    )
    arch = resolve_arch(args.dataset, args.arch)
    model = build_model(
        arch,
        input_channels=dataset_info.input_channels,
        num_classes=dataset_info.num_classes,
        image_size=dataset_info.image_size,
    ).to(device)

    optimizer = build_optimizer(
        model.parameters(),
        args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rmsprop_alpha=args.rmsprop_alpha,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )
    scaler = make_scaler(device, args.amp)

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

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.amp,
            scaler,
        )
        eval_loss, eval_acc = run_epoch(
            model,
            eval_loader,
            None,
            device,
            args.amp,
            scaler,
        )
        scheduler.step()

        print(
            f"teacher epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        save_checkpoint(
            output.with_name(output.stem + "_last.pt"),
            teacher=model.state_dict(),
            arch=arch,
            epoch=epoch,
            eval_acc=eval_acc,
        )
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(
                output,
                teacher=model.state_dict(),
                arch=arch,
                epoch=epoch,
                eval_acc=eval_acc,
            )

    print(f"done. best teacher checkpoint: {output}")


if __name__ == "__main__":
    main()
