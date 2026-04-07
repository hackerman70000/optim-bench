from typer.testing import CliRunner

from optim_bench.cli import app

runner = CliRunner()


def test_list_command():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "cifar10_resnet20" in result.output
    assert "cifar100_vit_tiny" in result.output
    assert "adamw" in result.output
    assert "sgd" in result.output
    assert "lion" in result.output
    assert "sophia" in result.output


def test_run_command_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Task name" in result.output
    assert "Optimizer name" in result.output


def test_sweep_command_help():
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    assert "Task name" in result.output


def test_run_pipeline_command_help():
    result = runner.invoke(app, ["run-pipeline", "--help"])
    assert result.exit_code == 0
    assert "sweeps" in result.output.lower()


def test_run_all_command_help():
    result = runner.invoke(app, ["run-all", "--help"])
    assert result.exit_code == 0
    assert "optimizer" in result.output.lower()


def test_compare_command_help():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "Results directory" in result.output
