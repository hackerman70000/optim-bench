from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dacite import from_dict

from optim_bench.types import HPSetting, Mode, Variant


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    sweep_range: list[float] = field(default_factory=list)


@dataclass
class TaskConfig:
    name: str
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 2


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class ExperimentConfig:
    task: TaskConfig
    optimizer: OptimizerConfig
    mode: Mode
    variant: Variant
    hp_setting: HPSetting
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256])
    output_dir: Path = field(default_factory=lambda: Path("results"))
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = "mps"

    def run_dir(self, seed: int) -> Path:
        return (
            self.output_dir
            / self.task.name
            / self.optimizer.name
            / f"{self.mode.value}_{self.variant.value}_{self.hp_setting.value}"
            / f"seed{seed}"
        )


def _find_conf_dir() -> Path:
    candidates = [Path("conf"), Path(__file__).resolve().parent.parent.parent / "conf"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError("Cannot find conf/ directory")


def load_optimizer_config(name: str, conf_dir: Path | None = None) -> OptimizerConfig:
    conf_dir = conf_dir or _find_conf_dir()
    path = conf_dir / "optimizer" / f"{name}.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return from_dict(data_class=OptimizerConfig, data=data)


def load_task_config(name: str, conf_dir: Path | None = None) -> TaskConfig:
    conf_dir = conf_dir or _find_conf_dir()
    path = conf_dir / "task" / f"{name}.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return from_dict(data_class=TaskConfig, data=data)


def load_sweep_best(
    task: str,
    optimizer: str,
    variant: Variant,
    output_dir: Path = Path("results"),
) -> float:
    import json
    path = output_dir / "sweeps" / task / optimizer / f"{variant.value}_best.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No sweep results found at {path}. "
            f"Run 'optim-bench sweep {task} {optimizer} --variant {variant.value}' first."
        )
    with path.open() as f:
        data = json.load(f)
    return float(data["best_lr"])


def build_experiment_config(
    task: str,
    optimizer: str,
    mode: str,
    variant: str,
    hp_setting: str,
    seeds: list[int] | None = None,
    lr_override: float | None = None,
    epochs_override: int | None = None,
    output_dir: str = "results",
    device: str = "mps",
) -> ExperimentConfig:
    task_cfg = load_task_config(task)
    opt_cfg = load_optimizer_config(optimizer)
    mode_enum = Mode(mode)
    variant_enum = Variant(variant)
    hp_enum = HPSetting(hp_setting)

    if epochs_override is not None:
        task_cfg.epochs = epochs_override

    if lr_override is not None:
        opt_cfg.lr = lr_override
    elif hp_enum == HPSetting.OPTIMIZED:
        opt_cfg.lr = load_sweep_best(task, optimizer, variant_enum, Path(output_dir))

    return ExperimentConfig(
        task=task_cfg,
        optimizer=opt_cfg,
        mode=mode_enum,
        variant=variant_enum,
        hp_setting=hp_enum,
        seeds=seeds or [42, 137, 256],
        output_dir=Path(output_dir),
        device=device,
    )
