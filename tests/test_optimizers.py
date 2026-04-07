import torch
import torch.nn as nn

from optim_bench.registry import OPTIMIZER_REGISTRY
import optim_bench.optimizers  # noqa: F401


def _make_params():
    model = nn.Linear(10, 2)
    return model.parameters()


def test_all_optimizers_registered():
    expected = {"adamw", "sgd", "lion", "sophia"}
    available = set(OPTIMIZER_REGISTRY.list_available())
    assert expected == available


def test_adamw_builds():
    opt = OPTIMIZER_REGISTRY.build("adamw", params=_make_params(), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_sgd_builds():
    opt = OPTIMIZER_REGISTRY.build("sgd", params=_make_params(), lr=0.1)
    assert isinstance(opt, torch.optim.SGD)


def test_lion_builds():
    opt = OPTIMIZER_REGISTRY.build("lion", params=_make_params(), lr=1e-4)
    assert opt is not None


def test_sophia_builds():
    opt = OPTIMIZER_REGISTRY.build("sophia", params=_make_params(), lr=1e-4)
    assert opt is not None


def test_optimizer_step():
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    target = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    for name in OPTIMIZER_REGISTRY.list_available():
        opt = OPTIMIZER_REGISTRY.build(name, params=model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        opt.step()
