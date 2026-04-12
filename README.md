# optim-bench

PyTorch optimizer benchmarking framework. Compares optimizers across tasks, training modes, and hyperparameter settings.

## Optimizers

**Built-in:** AdamW, SGD  
**External:** Lion, Sophia (via `pytorch-optimizer`)

## Tasks

| Task | Model | Dataset | Params |
|------|-------|---------|--------|
| `cifar10_resnet20` | ResNet-20 | CIFAR-10 | ~0.27M |
| `cifar100_vit_tiny` | ViT-Tiny | CIFAR-100 | ~1M |

## Setup

```bash
uv sync
```

## Usage

```bash
# List available components
optim-bench list

# Single experiment
optim-bench run cifar10_resnet20 adamw --mode generalization --variant raw

# LR sweep
optim-bench sweep cifar10_resnet20 adamw --variant raw

# Full pipeline (sweeps -> default -> optimized -> plots)
optim-bench run-pipeline --device cuda

# Generate comparison plots
optim-bench compare
```

### Modes

- **generalization** — train/test split, optimizes validation accuracy
- **optimization** — full dataset, minimizes training loss

### Variants

- **raw** — no scheduler, no augmentation (isolates optimizer behavior)
- **full** — cosine annealing + warmup + augmentation

### HP settings

- **default** — hyperparameters from original papers
- **optimized** — best LR from sweep

## Tests

```bash
uv run pytest tests/ -v
```

## Project structure

```
conf/               YAML configs (optimizer defaults, task settings)
src/optim_bench/
  models/           ResNet-20, ViT-Tiny
  tasks/            Task definitions (model + dataset + transforms)
  optimizers/       Optimizer factory wrappers
  trainer.py        Training loop with timing and memory tracking
  evaluation.py     Result aggregation and plotting
  cli.py            Typer CLI
docs/               Project plan
```
