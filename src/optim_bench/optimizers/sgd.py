from __future__ import annotations

from typing import Any

import torch.optim

from optim_bench.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("sgd")
def create_sgd(params: Any, lr: float, weight_decay: float = 5e-4, **kwargs: Any) -> torch.optim.SGD:
    momentum = kwargs.pop("momentum", 0.9)
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
