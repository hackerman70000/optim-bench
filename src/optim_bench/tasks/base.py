from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader

from optim_bench.types import Mode, Variant


class Task(ABC):
    @abstractmethod
    def create_model(self) -> nn.Module: ...

    @abstractmethod
    def create_train_loader(
        self,
        seed: int,
        mode: Mode,
        variant: Variant,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader: ...

    @abstractmethod
    def create_val_loader(
        self,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader | None: ...

    @abstractmethod
    def loss_fn(self) -> nn.Module: ...
