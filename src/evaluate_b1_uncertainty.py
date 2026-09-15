#!/usr/bin/env python3
"""Evaluate B1 with class distribution and bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260907
BOOTSTRAPS = 2_000


def evaluator_module():
    spec = importlib.util.spec_from_file_location("evaluator", ROOT / "src/evaluate_all.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def macro_f1(gold_sets, predicted_sets, labels):
    scores = []
    for label in labels:
        tp = sum(label in gold and label in pred for gold, pred in zip(gold_sets, predicted_sets))
        fp = sum(label not in gold and label in pred for gold, pred in zip(gold_sets, predicted_sets))
        fn = sum(label in gold and label not in pred for gold, pred in zip(gold_sets, predicted_sets))
        denom = 2 * tp + fp + fn
        # Keep the complete nine-category label space in every bootstrap
        # resample.  A category absent from a resample receives F1 = 0 rather
        # than being dropped from the macro average.
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def prediction_file(model_dir):
    for name in ("single_choice_v1.jsonl", "single_choice.jsonl"):
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No B1 prediction file in {model_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="Directory containing one subdirectory per model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/uncertainty")
    args = parser.parse_args()
    output_json = args.output_dir / "task_b1_uncertainty.json"
    output_csv = args.output_dir / "task_b1_uncertainty.csv"
    evaluator = evaluator_module()
    model_dirs = sorted(path for path in args.predictions_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise SystemExit("No model subdirectories found in --predictions-root")
    first_rows = [json.loads(line) for line in prediction_file(model_dirs[0]).read_text(encoding="utf-8").splitlines() if line]
    distribution = dict(sorted(Counter(row["answer"] for row in first_rows).items()))
    labels = sorted(distribution)
    report = {"protocol": {"n": len(first_rows), "bootstrap_replicates": BOOTSTRAPS,
                             "random_seed": SEED, "interval": "two-sided percentile 95% confidence interval"},
              "class_distribution": distribution, "models": {}}
    csv_rows = []
    rng = np.random.default_rng(SEED)
    for model_dir in model_dirs:
        rows = [json.loads(line) for line in prediction_file(model_dir).read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != len(first_rows):
            raise RuntimeError(f"{model_dir.name} has {len(rows)} B1 rows, expected {len(first_rows)}")
        gold_sets = [{row["answer"]} for row in rows]
        predicted_sets = [set(evaluator.normalize_choice_answer(evaluator.extract_boxed_content(row.get("output", "")))) for row in rows]
        exact = np.asarray([gold == pred for gold, pred in zip(gold_sets, predicted_sets)])
        sample_indices = rng.integers(0, len(rows), size=(BOOTSTRAPS, len(rows)))
        acc_boot = exact[sample_indices].mean(axis=1)
        f1_boot = np.asarray([
            macro_f1([gold_sets[i] for i in idx], [predicted_sets[i] for i in idx], labels)
            for idx in sample_indices
        ])
        metrics = {
            "accuracy": float(exact.mean()),
            "accuracy_ci95": [float(np.quantile(acc_boot, 0.025)), float(np.quantile(acc_boot, 0.975))],
            "macro_f1": macro_f1(gold_sets, predicted_sets, labels),
            "macro_f1_ci95": [float(np.quantile(f1_boot, 0.025)), float(np.quantile(f1_boot, 0.975))],
        }
        report["models"][model_dir.name] = metrics
        for metric in ("accuracy", "macro_f1"):
            csv_rows.append([model_dir.name, metric, metrics[metric], *metrics[f"{metric}_ci95"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", "estimate", "ci95_low", "ci95_high"])
        writer.writerows(csv_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
