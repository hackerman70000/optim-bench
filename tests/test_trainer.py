import csv
import json
from pathlib import Path

import torch

import optim_bench.optimizers  # noqa: F401
import optim_bench.tasks  # noqa: F401
from optim_bench.config import ExperimentConfig, OptimizerConfig, SchedulerConfig, TaskConfig
from optim_bench.trainer import EpochMetrics, Trainer
from optim_bench.types import HPSetting, Mode, Variant


def _make_config(
    tmp_path: Path,
    mode: Mode = Mode.GENERALIZATION,
    variant: Variant = Variant.RAW,
    epochs: int = 2,
    task: str = "cifar10_resnet20",
    optimizer: str = "adamw",
    lr: float = 1e-3,
) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(name=task, epochs=epochs, batch_size=32, num_workers=0),
        optimizer=OptimizerConfig(name=optimizer, lr=lr, weight_decay=0.01),
        mode=mode,
        variant=variant,
        hp_setting=HPSetting.DEFAULT,
        seeds=[42],
        output_dir=tmp_path / "results",
        device="cpu",
    )


def test_trainer_generalization_raw(tmp_path: Path):
    config = _make_config(tmp_path, mode=Mode.GENERALIZATION, variant=Variant.RAW)
    trainer = Trainer(config)
    metrics = trainer.run(42)

    assert len(metrics) == 2
    assert all(isinstance(m, EpochMetrics) for m in metrics)
    assert all(m.val_loss is not None for m in metrics)
    assert all(m.val_accuracy is not None for m in metrics)
    assert metrics[1].train_loss < metrics[0].train_loss


def test_trainer_optimization_mode(tmp_path: Path):
    config = _make_config(tmp_path, mode=Mode.OPTIMIZATION, variant=Variant.RAW)
    trainer = Trainer(config)
    metrics = trainer.run(42)

    assert len(metrics) == 2
    assert all(m.val_loss is None for m in metrics)
    assert all(m.val_accuracy is None for m in metrics)


def test_trainer_full_variant_creates_scheduler(tmp_path: Path):
    config = _make_config(tmp_path, variant=Variant.FULL, epochs=5)
    config.scheduler = SchedulerConfig(warmup_epochs=2, min_lr=1e-6)
    trainer = Trainer(config)
    metrics = trainer.run(42)

    assert len(metrics) == 5
    lrs = [m.lr for m in metrics]
    assert lrs[-1] < lrs[2], "LR should decay after warmup via cosine annealing"


def test_trainer_saves_csv(tmp_path: Path):
    config = _make_config(tmp_path, epochs=3)
    trainer = Trainer(config)
    trainer.run(42)

    csv_path = config.run_dir(42) / "metrics.csv"
    assert csv_path.exists()

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert "epoch" in rows[0]
    assert "train_loss" in rows[0]
    assert "val_accuracy" in rows[0]


def test_trainer_saves_config_json(tmp_path: Path):
    config = _make_config(tmp_path)
    trainer = Trainer(config)
    trainer.run(42)

    config_path = config.run_dir(42) / "config.json"
    assert config_path.exists()

    with config_path.open() as f:
        data = json.load(f)

    assert data["task"]["name"] == "cifar10_resnet20"
    assert data["optimizer"]["name"] == "adamw"
    assert data["mode"] == "generalization"
    assert data["variant"] == "raw"


def test_trainer_skips_completed_run(tmp_path: Path):
    config = _make_config(tmp_path, epochs=2)
    trainer = Trainer(config)

    metrics_first = trainer.run(42)
    assert len(metrics_first) == 2

    metrics_second = trainer.run(42)
    assert len(metrics_second) == 0


def test_trainer_run_dir_structure(tmp_path: Path):
    config = _make_config(tmp_path)
    run_dir = config.run_dir(42)
    expected = tmp_path / "results" / "cifar10_resnet20" / "adamw" / "generalization_raw_default" / "seed42"
    assert run_dir == expected


def test_trainer_seed_reproducibility(tmp_path: Path):
    config1 = _make_config(tmp_path / "run1", epochs=1)
    config2 = _make_config(tmp_path / "run2", epochs=1)

    metrics1 = Trainer(config1).run(42)
    metrics2 = Trainer(config2).run(42)

    assert abs(metrics1[0].train_loss - metrics2[0].train_loss) < 1e-5


def test_trainer_different_seeds_differ(tmp_path: Path):
    config = _make_config(tmp_path, epochs=1)
    config.seeds = [42, 99]
    trainer = Trainer(config)

    m1 = trainer.run(42)
    m2 = trainer.run(99)

    assert abs(m1[0].train_loss - m2[0].train_loss) > 1e-5


def test_trainer_all_optimizers(tmp_path: Path):
    default_lrs = {"adamw": 1e-3, "sgd": 0.1, "lion": 1e-4, "sophia": 1e-4}
    for opt_name, lr in default_lrs.items():
        config = _make_config(tmp_path / opt_name, optimizer=opt_name, epochs=1, lr=lr)
        trainer = Trainer(config)
        metrics = trainer.run(42)
        assert len(metrics) == 1
        assert metrics[0].train_loss > 0


def test_trainer_vit_task(tmp_path: Path):
    config = _make_config(tmp_path, task="cifar100_vit_tiny", epochs=1)
    trainer = Trainer(config)
    metrics = trainer.run(42)

    assert len(metrics) == 1
    assert metrics[0].train_loss > 0
    assert metrics[0].val_accuracy is not None


def test_epoch_metrics_fields():
    m = EpochMetrics(
        epoch=0, train_loss=1.5, train_accuracy=0.3,
        val_loss=1.2, val_accuracy=0.4,
        epoch_time_s=10.5, peak_memory_mb=256.0, lr=0.001,
    )
    assert m.epoch == 0
    assert m.train_loss == 1.5
    assert m.peak_memory_mb == 256.0
