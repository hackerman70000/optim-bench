import pytest

from optim_bench.config import load_optimizer_config, load_task_config


def test_load_adamw_config():
    cfg = load_optimizer_config("adamw")
    assert cfg.name == "adamw"
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 0.01
    assert len(cfg.sweep_range) == 5


def test_load_sgd_config():
    cfg = load_optimizer_config("sgd")
    assert cfg.name == "sgd"
    assert cfg.lr == 0.1
    assert cfg.extra["momentum"] == 0.9


def test_load_lion_config():
    cfg = load_optimizer_config("lion")
    assert cfg.name == "lion"
    assert cfg.weight_decay == 0.1


def test_load_sophia_config():
    cfg = load_optimizer_config("sophia")
    assert cfg.name == "sophia"
    assert cfg.extra["rho"] == 0.04


def test_load_cifar10_task():
    cfg = load_task_config("cifar10_resnet20")
    assert cfg.name == "cifar10_resnet20"
    assert cfg.epochs == 100
    assert cfg.batch_size == 128


def test_load_cifar100_task():
    cfg = load_task_config("cifar100_vit_tiny")
    assert cfg.name == "cifar100_vit_tiny"
    assert cfg.epochs == 100


def test_load_missing_optimizer_raises():
    with pytest.raises(FileNotFoundError):
        load_optimizer_config("nonexistent")


def test_load_missing_task_raises():
    with pytest.raises(FileNotFoundError):
        load_task_config("nonexistent")
