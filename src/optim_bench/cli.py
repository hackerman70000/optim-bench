from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

app = typer.Typer(name="optim-bench", help="PyTorch optimizer benchmarking framework")


def _detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_registries() -> None:
    import optim_bench.optimizers  # noqa: F401
    import optim_bench.tasks  # noqa: F401


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task name (e.g. cifar10_resnet20)")],
    optimizer: Annotated[str, typer.Argument(help="Optimizer name (e.g. adamw)")],
    mode: Annotated[str, typer.Option(help="optimization or generalization")] = "generalization",
    variant: Annotated[str, typer.Option(help="raw or full")] = "raw",
    hp_setting: Annotated[str, typer.Option(help="default or optimized")] = "default",
    lr: Annotated[Optional[float], typer.Option(help="Override learning rate")] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "42,137,256",
    epochs: Annotated[Optional[int], typer.Option(help="Override number of epochs")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
    device: Annotated[Optional[str], typer.Option(help="Device (auto-detected if omitted)")] = None,
) -> None:
    _ensure_registries()
    from optim_bench.config import build_experiment_config
    from optim_bench.trainer import Trainer

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    device_str = device or _detect_device()

    config = build_experiment_config(
        task=task,
        optimizer=optimizer,
        mode=mode,
        variant=variant,
        hp_setting=hp_setting,
        seeds=seed_list,
        lr_override=lr,
        epochs_override=epochs,
        output_dir=output_dir,
        device=device_str,
    )

    trainer = Trainer(config)
    for seed in config.seeds:
        trainer.run(seed)

    logger.info("All runs complete.")


@app.command()
def sweep(
    task: Annotated[str, typer.Argument(help="Task name")],
    optimizer: Annotated[str, typer.Argument(help="Optimizer name")],
    mode: Annotated[str, typer.Option(help="optimization or generalization")] = "generalization",
    variant: Annotated[str, typer.Option(help="raw or full")] = "raw",
    seed: Annotated[int, typer.Option(help="Seed for sweep runs")] = 42,
    epochs: Annotated[Optional[int], typer.Option(help="Override number of epochs")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
    device: Annotated[Optional[str], typer.Option(help="Device")] = None,
) -> None:
    import json

    _ensure_registries()
    from optim_bench.config import build_experiment_config, load_optimizer_config
    from optim_bench.trainer import Trainer

    device_str = device or _detect_device()
    opt_cfg = load_optimizer_config(optimizer)

    if not opt_cfg.sweep_range:
        logger.error(f"No sweep_range defined for optimizer '{optimizer}'")
        raise typer.Exit(1)

    logger.info(f"Starting LR sweep for {optimizer} on {task}: {opt_cfg.sweep_range}")
    best_lr = opt_cfg.lr
    best_metric = float("-inf")

    for lr_val in opt_cfg.sweep_range:
        config = build_experiment_config(
            task=task,
            optimizer=optimizer,
            mode=mode,
            variant=variant,
            hp_setting="default",
            seeds=[seed],
            lr_override=lr_val,
            epochs_override=epochs,
            output_dir=f"{output_dir}/sweeps",
            device=device_str,
        )

        trainer = Trainer(config)
        metrics = trainer.run(seed)

        if not metrics:
            continue

        last = metrics[-1]
        if mode == "generalization" and last.val_accuracy is not None:
            metric_val = last.val_accuracy
        else:
            metric_val = -last.train_loss

        logger.info(f"  lr={lr_val:.2e} -> metric={metric_val:.4f}")

        if metric_val > best_metric:
            best_metric = metric_val
            best_lr = lr_val

    result = {
        "best_lr": best_lr,
        "best_metric": best_metric,
        "task": task,
        "optimizer": optimizer,
        "mode": mode,
        "variant": variant,
        "sweep_range": opt_cfg.sweep_range,
    }

    result_dir = Path(output_dir) / "sweeps" / task / optimizer
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{variant}_best.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Best LR: {best_lr:.2e} (metric={best_metric:.4f}). Saved to {result_path}")


@app.command()
def compare(
    results_dir: Annotated[str, typer.Option(help="Results directory")] = "results",
    task: Annotated[Optional[str], typer.Option(help="Filter by task")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory for plots")] = "plots",
) -> None:
    from optim_bench.evaluation import generate_all_plots

    generate_all_plots(
        results_dir=Path(results_dir),
        output_dir=Path(output_dir),
        task_filter=task,
    )
    logger.info(f"Plots saved to {output_dir}/")


@app.command()
def run_all(
    mode: Annotated[str, typer.Option(help="optimization or generalization")] = "generalization",
    variant: Annotated[str, typer.Option(help="raw or full")] = "raw",
    hp_setting: Annotated[str, typer.Option(help="default or optimized")] = "default",
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "42,137,256",
    epochs: Annotated[Optional[int], typer.Option(help="Override number of epochs")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
    device: Annotated[Optional[str], typer.Option(help="Device")] = None,
) -> None:
    """Run all optimizer x task combinations."""
    _ensure_registries()
    from optim_bench.config import build_experiment_config
    from optim_bench.registry import OPTIMIZER_REGISTRY, TASK_REGISTRY
    from optim_bench.trainer import Trainer

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    device_str = device or _detect_device()
    tasks = TASK_REGISTRY.list_available()
    optimizers = OPTIMIZER_REGISTRY.list_available()
    total = len(tasks) * len(optimizers)

    logger.info(f"Running {total} task x optimizer combinations ({mode}, {variant}, {hp_setting})")

    for i, task_name in enumerate(tasks):
        for j, opt_name in enumerate(optimizers):
            idx = i * len(optimizers) + j + 1
            logger.info(f"[{idx}/{total}] {task_name} + {opt_name}")

            config = build_experiment_config(
                task=task_name,
                optimizer=opt_name,
                mode=mode,
                variant=variant,
                hp_setting=hp_setting,
                seeds=seed_list,
                lr_override=None,
                epochs_override=epochs,
                output_dir=output_dir,
                device=device_str,
            )

            trainer = Trainer(config)
            for seed in config.seeds:
                trainer.run(seed)

    logger.info("All combinations complete.")


@app.command()
def run_pipeline(
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "42,137,256",
    epochs: Annotated[Optional[int], typer.Option(help="Override number of epochs")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
    device: Annotated[Optional[str], typer.Option(help="Device")] = None,
    skip_sweeps: Annotated[bool, typer.Option(help="Skip LR sweeps")] = False,
) -> None:
    """Run the full experiment pipeline: sweeps -> default runs -> optimized runs -> plots."""
    import json

    _ensure_registries()
    from optim_bench.config import build_experiment_config, load_optimizer_config
    from optim_bench.registry import OPTIMIZER_REGISTRY, TASK_REGISTRY
    from optim_bench.trainer import Trainer

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    device_str = device or _detect_device()
    tasks = TASK_REGISTRY.list_available()
    optimizers = OPTIMIZER_REGISTRY.list_available()
    modes = ["generalization", "optimization"]
    variants = ["raw", "full"]

    total_sweeps = len(tasks) * len(optimizers) * len(variants)
    total_runs = len(tasks) * len(optimizers) * len(modes) * len(variants) * 2 * len(seed_list)
    logger.info(
        f"Full pipeline: {total_sweeps} sweeps + {total_runs} runs "
        f"({len(tasks)} tasks x {len(optimizers)} optimizers x "
        f"{len(modes)} modes x {len(variants)} variants x 2 HP x {len(seed_list)} seeds)"
    )

    # Phase 1: LR sweeps
    if not skip_sweeps:
        logger.info("=" * 60)
        logger.info("PHASE 1: LR SWEEPS")
        logger.info("=" * 60)
        sweep_idx = 0
        for task_name in tasks:
            for opt_name in optimizers:
                for variant in variants:
                    sweep_idx += 1
                    logger.info(f"[Sweep {sweep_idx}/{total_sweeps}] {task_name} + {opt_name} ({variant})")

                    opt_cfg = load_optimizer_config(opt_name)
                    if not opt_cfg.sweep_range:
                        continue

                    best_lr = opt_cfg.lr
                    best_metric = float("-inf")

                    for lr_val in opt_cfg.sweep_range:
                        config = build_experiment_config(
                            task=task_name, optimizer=opt_name,
                            mode="generalization", variant=variant,
                            hp_setting="default", seeds=[seed_list[0]],
                            lr_override=lr_val, epochs_override=epochs,
                            output_dir=f"{output_dir}/sweeps", device=device_str,
                        )
                        trainer = Trainer(config)
                        metrics = trainer.run(seed_list[0])
                        if not metrics:
                            continue
                        last = metrics[-1]
                        metric_val = last.val_accuracy if last.val_accuracy is not None else -last.train_loss
                        if metric_val > best_metric:
                            best_metric = metric_val
                            best_lr = lr_val

                    result_dir = Path(output_dir) / "sweeps" / task_name / opt_name
                    result_dir.mkdir(parents=True, exist_ok=True)
                    with (result_dir / f"{variant}_best.json").open("w") as f:
                        json.dump({"best_lr": best_lr, "best_metric": best_metric}, f, indent=2)
                    logger.info(f"  Best LR: {best_lr:.2e} (metric={best_metric:.4f})")

    # Phase 2: Full experiments (default + optimized)
    for hp_setting in ["default", "optimized"]:
        logger.info("=" * 60)
        logger.info(f"PHASE 2: FULL EXPERIMENTS ({hp_setting.upper()})")
        logger.info("=" * 60)
        run_idx = 0
        for task_name in tasks:
            for opt_name in optimizers:
                for mode in modes:
                    for variant in variants:
                        run_idx += 1
                        logger.info(
                            f"[Run {run_idx}] {task_name} + {opt_name} "
                            f"({mode}, {variant}, {hp_setting})"
                        )
                        try:
                            config = build_experiment_config(
                                task=task_name, optimizer=opt_name,
                                mode=mode, variant=variant,
                                hp_setting=hp_setting, seeds=seed_list,
                                epochs_override=epochs, output_dir=output_dir,
                                device=device_str,
                            )
                        except FileNotFoundError as e:
                            logger.warning(f"  Skipping: {e}")
                            continue

                        trainer = Trainer(config)
                        for seed in seed_list:
                            trainer.run(seed)

    # Phase 3: Plots
    logger.info("=" * 60)
    logger.info("PHASE 3: GENERATING PLOTS")
    logger.info("=" * 60)
    from optim_bench.evaluation import generate_all_plots

    generate_all_plots(
        results_dir=Path(output_dir),
        output_dir=Path("plots"),
    )

    logger.info("Pipeline complete!")


@app.command(name="list")
def list_components() -> None:
    _ensure_registries()
    from optim_bench.registry import OPTIMIZER_REGISTRY, TASK_REGISTRY

    typer.echo("Tasks:")
    for name in TASK_REGISTRY.list_available():
        typer.echo(f"  - {name}")

    typer.echo("\nOptimizers:")
    for name in OPTIMIZER_REGISTRY.list_available():
        typer.echo(f"  - {name}")
