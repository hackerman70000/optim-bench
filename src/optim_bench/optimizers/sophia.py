from __future__ import annotations

from typing import Any

from pytorch_optimizer import SophiaH

from optim_bench.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("sophia")
def create_sophia(params: Any, lr: float, weight_decay: float = 0.01, **kwargs: Any) -> SophiaH:
    rho = kwargs.pop("rho", 0.04)
    betas = kwargs.pop("betas", (0.965, 0.99))
    return SophiaH(params, lr=lr, weight_decay=weight_decay, rho=rho, betas=tuple(betas))
