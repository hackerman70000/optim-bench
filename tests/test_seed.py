import torch

from optim_bench.seed import create_generator, set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(42)
    a = torch.randn(10)
    set_seed(99)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_create_generator_reproducibility():
    g1 = create_generator(42)
    g2 = create_generator(42)
    a = torch.randn(10, generator=g1)
    b = torch.randn(10, generator=g2)
    assert torch.equal(a, b)
