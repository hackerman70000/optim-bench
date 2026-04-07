import optim_bench.tasks  # noqa: F401

from optim_bench.registry import TASK_REGISTRY
from optim_bench.types import Mode, Variant


def test_all_tasks_registered():
    expected = {"cifar10_resnet20", "cifar100_vit_tiny"}
    available = set(TASK_REGISTRY.list_available())
    assert expected == available


def test_cifar10_creates_model():
    task = TASK_REGISTRY.build("cifar10_resnet20")
    model = task.create_model()
    params = sum(p.numel() for p in model.parameters())
    assert 250_000 < params < 300_000


def test_cifar100_creates_model():
    task = TASK_REGISTRY.build("cifar100_vit_tiny")
    model = task.create_model()
    params = sum(p.numel() for p in model.parameters())
    assert 800_000 < params < 1_500_000


def test_cifar10_loss_fn():
    task = TASK_REGISTRY.build("cifar10_resnet20")
    loss = task.loss_fn()
    assert loss is not None


def test_cifar10_train_loader_generalization():
    task = TASK_REGISTRY.build("cifar10_resnet20")
    loader = task.create_train_loader(
        seed=42, mode=Mode.GENERALIZATION, variant=Variant.RAW,
        batch_size=16, num_workers=0,
    )
    assert len(loader.dataset) == 50_000


def test_cifar10_train_loader_optimization():
    task = TASK_REGISTRY.build("cifar10_resnet20")
    loader = task.create_train_loader(
        seed=42, mode=Mode.OPTIMIZATION, variant=Variant.RAW,
        batch_size=16, num_workers=0,
    )
    assert len(loader.dataset) == 60_000


def test_cifar10_val_loader():
    task = TASK_REGISTRY.build("cifar10_resnet20")
    loader = task.create_val_loader(batch_size=16, num_workers=0)
    assert loader is not None
    assert len(loader.dataset) == 10_000
