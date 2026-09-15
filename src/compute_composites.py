#!/usr/bin/env python3
"""Compute component metrics and the paper's descriptive composites."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator():
    source = ROOT / "src/evaluate_all.py"
    spec = importlib.util.spec_from_file_location("evaluator", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "results/parsed_predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "results/component_metrics_and_composites.csv")
    args = parser.parse_args()
    evaluator = load_evaluator()
    fields = [
        "model", "A1_recall", "A1_f2", "A2_accuracy", "B1_accuracy", "B1_macro_f1",
        "B2_accuracy", "B2_jaccard", "C1_missing_information_accuracy", "C1_macro_f1",
        "C2_accuracy", "C3_jaccard", "S_A", "S_B", "S_C", "S_overall",
    ]
    output_rows = []
    for model_dir in sorted(path for path in args.predictions_root.iterdir() if path.is_dir()):
        results = {}
        for path in evaluator.prediction_files(model_dir):
            for result in evaluator.evaluate_file(path):
                results[result["subtask"]] = result
        missing = sorted(set(("A1", "A2", "B1", "B2", "C1", "C2", "C3")) - set(results))
        if missing:
            raise ValueError(f"{model_dir.name} is missing tasks: {', '.join(missing)}")

        row = {
            "model": model_dir.name,
            "A1_recall": results["A1"]["recall"],
            "A1_f2": results["A1"]["f2"],
            "A2_accuracy": results["A2"]["accuracy"],
            "B1_accuracy": results["B1"]["exact_accuracy"],
            "B1_macro_f1": results["B1"]["macro_f1"],
            "B2_accuracy": results["B2"]["exact_accuracy"],
            "B2_jaccard": results["B2"]["jaccard"],
            "C1_missing_information_accuracy": results["C1"]["missing_information_accuracy"],
            "C1_macro_f1": results["C1"]["macro_f1"],
            "C2_accuracy": results["C2"]["exact_accuracy"],
            "C3_jaccard": results["C3"]["jaccard"],
        }
        # Figures 2--3 were generated from the two-decimal percentage values
        # printed in the manuscript tables. Apply that same rounding before
        # the descriptive composite calculation.
        for key in tuple(row):
            if key != "model":
                row[key] = round(100 * row[key], 2)
        row["S_A"] = ((row["A1_recall"] + row["A1_f2"]) / 2 + row["A2_accuracy"]) / 2
        row["S_B"] = (
            (row["B1_accuracy"] + row["B1_macro_f1"]) / 2
            + (row["B2_accuracy"] + row["B2_jaccard"]) / 2
        ) / 2
        row["S_C"] = (
            (row["C1_missing_information_accuracy"] + row["C1_macro_f1"]) / 2
            + row["C2_accuracy"]
            + row["C3_jaccard"]
        ) / 3
        row["S_overall"] = (row["S_A"] + row["S_B"] + row["S_C"]) / 3
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: row[key] if key == "model" else f"{row[key]:.2f}" for key in fields})
    print(f"Wrote {len(output_rows)} models to {args.output}")


if __name__ == "__main__":
    main()
