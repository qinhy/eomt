from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from training.runtime import to_scalar


class MetricCSVLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.csv"
        self._file = self.metrics_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=("timestamp", "epoch", "step", "phase", "name", "value"),
        )
        if self.metrics_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def save_hparams(self, args) -> None:
        with (self.log_dir / "hparams.json").open("w", encoding="utf-8") as file:
            json.dump(vars(args), file, indent=2, default=str)

    def log_metrics(
        self,
        phase: str,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        timestamp = time.time()
        for name, value in sorted(metrics.items()):
            scalar = to_scalar(value)
            if scalar is None:
                continue
            self._writer.writerow(
                {
                    "timestamp": f"{timestamp:.6f}",
                    "epoch": epoch,
                    "step": step,
                    "phase": phase,
                    "name": name,
                    "value": f"{scalar:.10f}",
                }
            )
        self._file.flush()

    def close(self) -> None:
        self._file.close()

