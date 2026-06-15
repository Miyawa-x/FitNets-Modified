from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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


def _global_contrast_normalize(
    flat: np.ndarray,
    scale: float = 55.0,
    min_divisor: float = 1e-8,
) -> np.ndarray:
    """Per-image GCN matching pylearn2 (subtract mean, divide by L2 norm / scale)."""
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.sqrt((flat ** 2).sum(axis=1)) / scale
    norms[norms < min_divisor] = 1.0
    return flat / norms[:, None]


def _fit_zca(flat: np.ndarray, filter_bias: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ZCA whitening transform (pylearn2 defaults: filter_bias=0.1)."""
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = centered.T @ centered / centered.shape[0]
    eigs, eigv = np.linalg.eigh(cov)
    eigs = np.maximum(eigs, 0.0)
    whitening = (eigv * (1.0 / np.sqrt(eigs + filter_bias))) @ eigv.T
    return mean.astype(np.float32), whitening.astype(np.float32)


class _WhitenedImageDataset(Dataset):
    """Holds GCN+ZCA whitened images and applies train-time crop/flip on tensors."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
        image_size: int,
    ) -> None:
        from torchvision import transforms

        self.images = images
        self.labels = labels
        if train:
            self.augment = transforms.Compose(
                [
                    transforms.RandomCrop(image_size, padding=8),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.augment = None

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if self.augment is not None:
            image = self.augment(image)
        return image, int(self.labels[index])


def _stack_dataset(raw_set, info: DatasetInfo) -> tuple[np.ndarray, np.ndarray]:
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    count = len(raw_set)
    flat = np.empty(
        (count, info.input_channels * info.image_size * info.image_size),
        dtype=np.float64,
    )
    labels = np.empty(count, dtype=np.int64)
    for i in range(count):
        image, label = raw_set[i]
        flat[i] = to_tensor(image).numpy().reshape(-1)
        labels[i] = int(label)
    return flat, labels


def _build_whitened_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    info: DatasetInfo,
) -> tuple[DataLoader, DataLoader, DatasetInfo]:
    from torchvision import datasets

    if dataset == "cifar10":
        train_raw = datasets.CIFAR10(data_root, train=True, download=download)
        eval_raw = datasets.CIFAR10(data_root, train=False, download=download)
    elif dataset == "cifar100":
        train_raw = datasets.CIFAR100(data_root, train=True, download=download)
        eval_raw = datasets.CIFAR100(data_root, train=False, download=download)
    else:
        raise ValueError(f"GCN+ZCA whitening is only implemented for CIFAR, got '{dataset}'.")

    print(f"whiten: computing GCN+ZCA for {dataset} (this runs once per launch)...")
    train_flat, train_labels = _stack_dataset(train_raw, info)
    eval_flat, eval_labels = _stack_dataset(eval_raw, info)

    train_flat = _global_contrast_normalize(train_flat)
    eval_flat = _global_contrast_normalize(eval_flat)
    mean, whitening = _fit_zca(train_flat)
    train_flat = (train_flat - mean) @ whitening
    eval_flat = (eval_flat - mean) @ whitening

    shape = (info.input_channels, info.image_size, info.image_size)
    train_images = torch.from_numpy(train_flat.astype(np.float32)).reshape(-1, *shape)
    eval_images = torch.from_numpy(eval_flat.astype(np.float32)).reshape(-1, *shape)

    train_set = _WhitenedImageDataset(
        train_images, torch.from_numpy(train_labels), train=True, image_size=info.image_size
    )
    eval_set = _WhitenedImageDataset(
        eval_images, torch.from_numpy(eval_labels), train=False, image_size=info.image_size
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


def build_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    whiten: bool = False,
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

    if whiten:
        if dataset not in ("cifar10", "cifar100"):
            raise ValueError("--whiten (GCN+ZCA) is only supported for cifar10/cifar100.")
        return _build_whitened_loaders(
            dataset, data_root, batch_size, num_workers, download, info
        )

    if dataset in CIFAR_NORMALIZATION:
        mean, std = CIFAR_NORMALIZATION[dataset]
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=8),
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
