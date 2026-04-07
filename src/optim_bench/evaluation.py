from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

OPTIMIZER_COLORS = {
    "adamw": "#1f77b4",
    "sgd": "#ff7f0e",
    "lion": "#2ca02c",
    "sophia": "#d62728",
}


def load_run(run_dir: Path) -> pd.DataFrame | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def load_all_results(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("sweeps", "plots"):
            continue
        for opt_dir in sorted(task_dir.iterdir()):
            if not opt_dir.is_dir():
                continue
            for config_dir in sorted(opt_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                parts = config_dir.name.split("_")
                if len(parts) < 3:
                    continue
                mode, variant, hp_setting = parts[0], parts[1], parts[2]

                for seed_dir in sorted(config_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    df = load_run(seed_dir)
                    if df is None:
                        continue
                    df["task"] = task_dir.name
                    df["optimizer"] = opt_dir.name
                    df["mode"] = mode
                    df["variant"] = variant
                    df["hp_setting"] = hp_setting
                    df["seed"] = seed_dir.name.replace("seed", "")
                    rows.append(df)

    if not rows:
        logger.warning("No results found.")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["task", "optimizer", "mode", "variant", "hp_setting", "epoch"]
    numeric_cols = [
        "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "epoch_time_s", "peak_memory_mb",
    ]
    existing_numeric = [c for c in numeric_cols if c in df.columns]

    agg_funcs = {}
    for col in existing_numeric:
        agg_funcs[col] = ["mean", "std"]

    result = df.groupby(group_cols, as_index=False).agg(agg_funcs)
    result.columns = [
        f"{a}_{b}" if b else a
        for a, b in result.columns
    ]
    return result


def plot_training_curves(
    df: pd.DataFrame,
    task: str,
    mode: str,
    variant: str,
    hp_setting: str,
    metric: str = "train_loss",
    output_dir: Path | None = None,
) -> plt.Figure:
    subset = df[
        (df["task"] == task)
        & (df["mode"] == mode)
        & (df["variant"] == variant)
        & (df["hp_setting"] == hp_setting)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name, group in subset.groupby("optimizer"):
        agg = group.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
        color = OPTIMIZER_COLORS.get(str(opt_name), None)
        ax.plot(agg["epoch"], agg["mean"], label=opt_name, color=color)
        ax.fill_between(
            agg["epoch"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{task} | {mode} | {variant} | {hp_setting}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{task}_{mode}_{variant}_{hp_setting}_{metric}.png",
            dpi=150,
        )

    return fig


def plot_summary_bars(
    df: pd.DataFrame,
    metric: str = "val_accuracy",
    output_dir: Path | None = None,
) -> plt.Figure:
    if metric not in df.columns:
        logger.warning(f"Metric '{metric}' not in data.")
        return plt.figure()

    last_epoch = (
        df.groupby(["task", "optimizer", "mode", "variant", "hp_setting"])["epoch"]
        .max()
        .reset_index()
        .rename(columns={"epoch": "max_epoch"})
    )
    df_last = df.merge(last_epoch, on=["task", "optimizer", "mode", "variant", "hp_setting"])
    df_last = df_last[df_last["epoch"] == df_last["max_epoch"]]

    summary = (
        df_last.groupby(["task", "optimizer", "mode", "variant", "hp_setting"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    tasks = summary["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)

    for idx, task_name in enumerate(tasks):
        ax = axes[0, idx]
        task_data = summary[summary["task"] == task_name]
        optimizers = task_data["optimizer"].unique()
        x = range(len(optimizers))
        colors = [OPTIMIZER_COLORS.get(o, "#999999") for o in optimizers]

        ax.bar(x, task_data.groupby("optimizer")["mean"].first(), yerr=task_data.groupby("optimizer")["std"].first(), color=colors, capsize=4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(optimizers)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(task_name)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"summary_{metric}.png", dpi=150)

    return fig


def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
    task_filter: str | None = None,
) -> None:
    df = load_all_results(results_dir)
    if df.empty:
        return

    if task_filter:
        df = df[df["task"] == task_filter]

    configs = df.groupby(["task", "mode", "variant", "hp_setting"]).size().reset_index()

    for _, row in configs.iterrows():
        task, mode, variant, hp_setting = row["task"], row["mode"], row["variant"], row["hp_setting"]

        plot_training_curves(df, task, mode, variant, hp_setting, "train_loss", output_dir)

        if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
            plot_training_curves(df, task, mode, variant, hp_setting, "val_accuracy", output_dir)

    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        plot_summary_bars(df, "val_accuracy", output_dir)

    plot_summary_bars(df, "train_loss", output_dir)

    plt.close("all")
    logger.info(f"Generated plots in {output_dir}")
