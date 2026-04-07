import csv
from pathlib import Path

import pandas as pd

from optim_bench.evaluation import aggregate_seeds, load_all_results, load_run


def _write_metrics_csv(path: Path, epochs: int = 3, val: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy",
            "epoch_time_s", "peak_memory_mb", "lr",
        ])
        writer.writeheader()
        for i in range(epochs):
            writer.writerow({
                "epoch": i,
                "train_loss": 2.0 - i * 0.3,
                "train_accuracy": 0.3 + i * 0.1,
                "val_loss": 2.1 - i * 0.2 if val else "",
                "val_accuracy": 0.25 + i * 0.08 if val else "",
                "epoch_time_s": 10.0,
                "peak_memory_mb": 256.0,
                "lr": 0.001,
            })


def test_load_run(tmp_path: Path):
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)

    df = load_run(tmp_path)
    assert df is not None
    assert len(df) == 3
    assert "train_loss" in df.columns


def test_load_run_missing(tmp_path: Path):
    df = load_run(tmp_path)
    assert df is None


def test_load_all_results(tmp_path: Path):
    for opt in ["adamw", "sgd"]:
        for seed in [42, 137]:
            path = tmp_path / "cifar10_resnet20" / opt / "generalization_raw_default" / f"seed{seed}" / "metrics.csv"
            _write_metrics_csv(path)

    df = load_all_results(tmp_path)
    assert len(df) == 12  # 2 optimizers × 2 seeds × 3 epochs
    assert "task" in df.columns
    assert "optimizer" in df.columns
    assert "seed" in df.columns
    assert set(df["optimizer"].unique()) == {"adamw", "sgd"}


def test_load_all_results_empty(tmp_path: Path):
    df = load_all_results(tmp_path)
    assert df.empty


def test_aggregate_seeds(tmp_path: Path):
    for seed in [42, 137, 256]:
        path = tmp_path / "cifar10_resnet20" / "adamw" / "generalization_raw_default" / f"seed{seed}" / "metrics.csv"
        _write_metrics_csv(path)

    df = load_all_results(tmp_path)
    agg = aggregate_seeds(df)

    assert "train_loss_mean" in agg.columns
    assert "train_loss_std" in agg.columns
    assert len(agg) == 3  # 3 epochs, aggregated across 3 seeds


def test_load_results_parses_config_from_path(tmp_path: Path):
    path = tmp_path / "cifar100_vit_tiny" / "lion" / "optimization_full_optimized" / "seed99" / "metrics.csv"
    _write_metrics_csv(path, val=False)

    df = load_all_results(tmp_path)
    assert len(df) == 3
    row = df.iloc[0]
    assert row["task"] == "cifar100_vit_tiny"
    assert row["optimizer"] == "lion"
    assert row["mode"] == "optimization"
    assert row["variant"] == "full"
    assert row["hp_setting"] == "optimized"
    assert row["seed"] == "99"
