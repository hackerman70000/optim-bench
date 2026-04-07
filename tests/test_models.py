import torch

from optim_bench.models.resnet import resnet20
from optim_bench.models.vit import vit_tiny


def test_resnet20_output_shape():
    model = resnet20(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_resnet20_param_count():
    model = resnet20(num_classes=10)
    count = sum(p.numel() for p in model.parameters())
    assert 250_000 < count < 300_000


def test_vit_tiny_output_shape():
    model = vit_tiny(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 100)


def test_vit_tiny_param_count():
    model = vit_tiny(num_classes=100)
    count = sum(p.numel() for p in model.parameters())
    assert 800_000 < count < 1_500_000


def test_resnet20_different_classes():
    model = resnet20(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_vit_tiny_different_classes():
    model = vit_tiny(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
