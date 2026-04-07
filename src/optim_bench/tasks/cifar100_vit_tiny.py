from __future__ import annotations

import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader

from optim_bench.models.vit import vit_tiny
from optim_bench.registry import TASK_REGISTRY
from optim_bench.seed import create_generator, worker_init_fn
from optim_bench.tasks.base import Task
from optim_bench.types import Mode, Variant

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _build_transform(variant: Variant, train: bool) -> T.Compose:
    transforms: list[T.Transform] = []
    if train and variant == Variant.FULL:
        transforms.extend([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    transforms.extend([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return T.Compose(transforms)


@TASK_REGISTRY.register("cifar100_vit_tiny")
class CIFAR100ViTTiny(Task):
    def create_model(self) -> nn.Module:
        return vit_tiny(num_classes=100)

    def create_train_loader(
        self,
        seed: int,
        mode: Mode,
        variant: Variant,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        transform = _build_transform(variant, train=True)
        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform,
        )

        if mode == Mode.OPTIMIZATION:
            test_set = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform,
            )
            train_set = ConcatDataset([train_set, test_set])

        return DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=create_generator(seed),
            persistent_workers=num_workers > 0,
        )

    def create_val_loader(
        self, batch_size: int, num_workers: int,
    ) -> DataLoader | None:
        transform = _build_transform(Variant.RAW, train=False)
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform,
        )
        return DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()
