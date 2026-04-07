from __future__ import annotations

from typing import Any

import torch.optim

from optim_bench.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("adamw")
def create_adamw(params: Any, lr: float, weight_decay: float = 0.01, **kwargs: Any) -> torch.optim.AdamW:
    betas = kwargs.pop("betas", (0.9, 0.999))
    eps = kwargs.pop("eps", 1e-8)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)
