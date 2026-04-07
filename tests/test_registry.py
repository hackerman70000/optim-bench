import pytest

from optim_bench.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register("foo")
    def foo_fn():
        return "foo"

    assert "foo" in reg
    assert reg.get("foo") is foo_fn


def test_register_auto_name():
    reg = Registry("test")

    @reg.register()
    def my_func():
        return 42

    assert "my_func" in reg


def test_build():
    reg = Registry("test")

    @reg.register("adder")
    def adder(a, b):
        return a + b

    assert reg.build("adder", a=2, b=3) == 5


def test_duplicate_raises():
    reg = Registry("test")

    @reg.register("dup")
    def first():
        pass

    with pytest.raises(ValueError, match="already registered"):
        @reg.register("dup")
        def second():
            pass


def test_get_missing_raises():
    reg = Registry("test")

    with pytest.raises(KeyError, match="not found"):
        reg.get("nonexistent")


def test_list_available():
    reg = Registry("test")

    @reg.register("b")
    def b():
        pass

    @reg.register("a")
    def a():
        pass

    assert reg.list_available() == ["a", "b"]


def test_len():
    reg = Registry("test")
    assert len(reg) == 0

    @reg.register("x")
    def x():
        pass

    assert len(reg) == 1
