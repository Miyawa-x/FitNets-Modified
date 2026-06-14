from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DatasetInfo:
    input_channels: int
    image_size: int
    num_classes: int


DATASET_INFO: dict[str, DatasetInfo] = {
    "cifar10": DatasetInfo(input_channels=3, image_size=32, num_classes=10),
    "cifar100": DatasetInfo(input_channels=3, image_size=32, num_classes=100),
    "mnist": DatasetInfo(input_channels=1, image_size=28, num_classes=10),
    "fake-cifar10": DatasetInfo(input_channels=3, image_size=32, num_classes=10),
    "fake-cifar100": DatasetInfo(input_channels=3, image_size=32, num_classes=100),
}


CIFAR_NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fake-cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "fake-cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def build_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> tuple[DataLoader, DataLoader, DatasetInfo]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for dataset loading") from exc

    dataset = dataset.lower()
    if dataset not in DATASET_INFO:
        known = ", ".join(sorted(DATASET_INFO))
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {known}")

    info = DATASET_INFO[dataset]

    if dataset in CIFAR_NORMALIZATION:
        mean, std = CIFAR_NORMALIZATION[dataset]
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        eval_transform = train_transform

    if dataset == "cifar10":
        train_set = datasets.CIFAR10(data_root, train=True, transform=train_transform, download=download)
        eval_set = datasets.CIFAR10(data_root, train=False, transform=eval_transform, download=download)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(data_root, train=True, transform=train_transform, download=download)
        eval_set = datasets.CIFAR100(data_root, train=False, transform=eval_transform, download=download)
    elif dataset == "mnist":
        train_set = datasets.MNIST(data_root, train=True, transform=train_transform, download=download)
        eval_set = datasets.MNIST(data_root, train=False, transform=eval_transform, download=download)
    else:
        train_set = datasets.FakeData(
            size=1024,
            image_size=(info.input_channels, info.image_size, info.image_size),
            num_classes=info.num_classes,
            transform=train_transform,
        )
        eval_set = datasets.FakeData(
            size=256,
            image_size=(info.input_channels, info.image_size, info.image_size),
            num_classes=info.num_classes,
            transform=eval_transform,
        )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader, info
