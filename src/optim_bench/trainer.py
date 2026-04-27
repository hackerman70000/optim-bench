from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from optim_bench.config import ExperimentConfig
from optim_bench.csv_logger import CSVLogger
from optim_bench.registry import OPTIMIZER_REGISTRY, TASK_REGISTRY
from optim_bench.seed import set_seed
from optim_bench.types import Mode, Variant


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    epoch_time_s: float
    peak_memory_mb: float
    lr: float


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def run(self, seed: int) -> list[EpochMetrics]:
        run_dir = self.config.run_dir(seed)

        if self._is_run_complete(run_dir):
            logger.info(f"Skipping completed run: {run_dir}")
            return []

        set_seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)

        task = TASK_REGISTRY.build(self.config.task.name)
        model = task.create_model().to(self.device)

        train_loader = task.create_train_loader(
            seed=seed,
            mode=self.config.mode,
            variant=self.config.variant,
            batch_size=self.config.task.batch_size,
            num_workers=self.config.task.num_workers,
        )
        val_loader = (
            task.create_val_loader(self.config.task.batch_size, self.config.task.num_workers)
            if self.config.mode == Mode.GENERALIZATION
            else None
        )
        criterion = task.loss_fn().to(self.device)

        optimizer = OPTIMIZER_REGISTRY.build(
            self.config.optimizer.name,
            params=model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            **self.config.optimizer.extra,
        )

        self._create_graph = self.config.optimizer.name == "sophia"
        if self._create_graph:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        self._save_config(run_dir)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Starting: {self.config.task.name} | {self.config.optimizer.name} | "
            f"mode={self.config.mode.value} | variant={self.config.variant.value} | "
            f"seed={seed} | params={param_count:,} | lr={self.config.optimizer.lr}"
        )

        metrics: list[EpochMetrics] = []
        try:
            with CSVLogger(run_dir / "metrics.csv") as csv_logger:
                for epoch in range(self.config.task.epochs):
                    epoch_metrics = self._train_epoch(
                        model, train_loader, criterion, optimizer, scheduler, epoch,
                    )
                    if val_loader is not None:
                        val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                        epoch_metrics.val_loss = val_loss
                        epoch_metrics.val_accuracy = val_acc

                    csv_logger.log(epoch_metrics)
                    metrics.append(epoch_metrics)

                    val_str = f"val_acc={epoch_metrics.val_accuracy:.2%}" if epoch_metrics.val_accuracy is not None else "no_val"
                    logger.info(
                        f"[{epoch + 1}/{self.config.task.epochs}] "
                        f"loss={epoch_metrics.train_loss:.4f} "
                        f"{val_str} "
                        f"lr={epoch_metrics.lr:.2e} "
                        f"time={epoch_metrics.epoch_time_s:.1f}s"
                    )
        except KeyboardInterrupt:
            logger.warning(f"Interrupted at epoch {len(metrics)}. Partial results saved.")
        finally:
            if self._create_graph:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        return metrics

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.config.variant == Variant.RAW:
            return None

        warmup_steps = self.config.scheduler.warmup_epochs * steps_per_epoch
        total_steps = self.config.task.epochs * steps_per_epoch

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=self.config.scheduler.min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
    ) -> EpochMetrics:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self._sync_device()
        t_start = perf_counter()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph=self._create_graph)
            optimizer.step()

            if self._create_graph:
                for p in model.parameters():
                    p.grad = None

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += inputs.size(0)

        self._sync_device()
        epoch_time = perf_counter() - t_start
        peak_mem = self._get_memory_mb()

        return EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / total,
            train_accuracy=correct / total,
            val_loss=None,
            val_accuracy=None,
            epoch_time_s=round(epoch_time, 2),
            peak_memory_mb=round(peak_mem, 1),
            lr=optimizer.param_groups[0]["lr"],
        )

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += inputs.size(0)

        return total_loss / total, correct / total

    def _sync_device(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def _get_memory_mb(self) -> float:
        if self.device.type == "mps":
            return torch.mps.driver_allocated_memory() / 1024**2
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def _save_config(self, run_dir: Path) -> None:
        config_data = {
            "task": asdict(self.config.task),
            "optimizer": asdict(self.config.optimizer),
            "mode": self.config.mode.value,
            "variant": self.config.variant.value,
            "hp_setting": self.config.hp_setting.value,
            "device": self.config.device,
            "seeds": self.config.seeds,
        }
        with (run_dir / "config.json").open("w") as f:
            json.dump(config_data, f, indent=2)

    def _is_run_complete(self, run_dir: Path) -> bool:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return False
        line_count = sum(1 for _ in metrics_path.open()) - 1
        return line_count >= self.config.task.epochs
