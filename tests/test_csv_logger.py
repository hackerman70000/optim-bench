import csv
from dataclasses import dataclass
from pathlib import Path

from optim_bench.csv_logger import CSVLogger


@dataclass
class SampleMetrics:
    epoch: int
    loss: float
    accuracy: float


def test_csv_logger_writes_header_and_rows(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    with CSVLogger(path) as log:
        log.log(SampleMetrics(epoch=0, loss=1.5, accuracy=0.3))
        log.log(SampleMetrics(epoch=1, loss=1.0, accuracy=0.5))

    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["epoch"] == "0"
    assert rows[0]["loss"] == "1.5"
    assert rows[1]["accuracy"] == "0.5"


def test_csv_logger_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "metrics.csv"
    with CSVLogger(path) as log:
        log.log(SampleMetrics(epoch=0, loss=1.0, accuracy=0.1))

    assert path.exists()
