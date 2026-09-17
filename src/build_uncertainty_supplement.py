#!/usr/bin/env python3
"""Create the machine-readable S1 uncertainty table from revision outputs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/uncertainty"
OUT = RESULTS / "S1_Table_bootstrap_uncertainty.csv"


def pct(value: str) -> str:
    return f"{100 * float(value):.2f}"


rows: list[dict[str, str]] = []
with (RESULTS / "task_b1_uncertainty.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        rows.append(
            {
                "task": "B1",
                "model": row["model"],
                "n_core": "47",
                "n_metric": "47",
                "metric": row["metric"],
                "estimate_percent": pct(row["estimate"]),
                "ci95_low_percent": pct(row["ci95_low"]),
                "ci95_high_percent": pct(row["ci95_high"]),
            }
        )

with (RESULTS / "task_c_core_metrics.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        task = {"fault_testset": "C1", "fault_exclude": "C2", "fault_reason": "C3"}[row["task"]]
        rows.append(
            {
                "task": task,
                "model": row["model"],
                "n_core": row["n"],
                "n_metric": "95" if task == "C1" and row["metric"] == "missing_information_accuracy" else row["n"],
                "metric": row["metric"],
                "estimate_percent": pct(row["estimate"]),
                "ci95_low_percent": pct(row["ci95_low"]),
                "ci95_high_percent": pct(row["ci95_high"]),
            }
        )

with OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
