from __future__ import annotations

from typing import Any

from pytorch_optimizer import Lion

from optim_bench.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("lion")
def create_lion(params: Any, lr: float, weight_decay: float = 0.1, **kwargs: Any) -> Lion:
    betas = kwargs.pop("betas", (0.9, 0.99))
    return Lion(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))
