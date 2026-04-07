from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any


class CSVLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> CSVLogger:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._file:
            self._file.close()

    def log(self, metrics: Any) -> None:
        row = asdict(metrics) if hasattr(metrics, "__dataclass_fields__") else dict(metrics)
        if self._writer is None and self._file is not None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        if self._writer is not None:
            self._writer.writerow(row)
        if self._file is not None:
            self._file.flush()
