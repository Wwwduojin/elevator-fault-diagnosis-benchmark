#!/usr/bin/env python3
"""Evaluate the unique-stem C2/C3 core sets with bootstrap confidence bounds."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260907
BOOTSTRAPS = 2_000


def load_evaluator():
    source = ROOT / "src/evaluate_all.py"
    spec = importlib.util.spec_from_file_location("evaluator", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confidence_interval(values, rng):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def outcomes(rows, task, evaluator):
    exact, jaccards = [], []
    for row in rows:
        prediction = set(evaluator.normalize_choice_answer(
            evaluator.extract_boxed_content(row.get("output", ""))
        ))
        gold = set(row["answer"])
        exact.append(float(prediction == gold))
        if task == "fault_reason":
            jaccards.append(evaluator.jaccard(prediction, gold))
    return exact, jaccards


def c1_statistics(rows, evaluator):
    correct = []
    missing = []
    label_stats = {}
    for row in rows:
        gold = row["fault_id"]
        pred = evaluator.extract_fault_id(row.get("output", ""))
        is_correct = float(pred == gold)
        correct.append(is_correct)
        if row.get("completeness", 1.0) < 1.0:
            missing.append(is_correct)
        for label in {gold, pred} - {None}:
            label_stats.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        if pred == gold:
            label_stats[gold]["tp"] += 1
        else:
            if pred:
                label_stats[pred]["fp"] += 1
            label_stats[gold]["fn"] += 1
    fixed_labels = sorted({row["fault_id"] for row in rows})
    f1_values = []
    for label in fixed_labels:
        values = label_stats.get(label, {"tp": 0, "fp": 0, "fn": 0})
        denominator = 2 * values["tp"] + values["fp"] + values["fn"]
        f1_values.append(0.0 if denominator == 0 else 2 * values["tp"] / denominator)
    return {
        "accuracy": float(np.mean(correct)),
        "missing_information_accuracy": float(np.mean(missing)),
        "macro_f1": float(np.mean(f1_values)),
    }


def c1_confidence_intervals(rows, evaluator, rng):
    """Vectorised non-parametric bootstrap for C1 accuracy, MI-Acc and macro-F1."""
    labels = sorted({row["fault_id"] for row in rows})
    label_index = {label: index for index, label in enumerate(labels)}
    truth = np.asarray([label_index[row["fault_id"]] for row in rows])
    prediction = np.asarray([
        label_index.get(evaluator.extract_fault_id(row.get("output", "")), -1) for row in rows
    ])
    correct = truth == prediction
    missing = np.asarray([row.get("completeness", 1.0) < 1.0 for row in rows])
    sampled = rng.integers(0, len(rows), size=(BOOTSTRAPS, len(rows)))
    sampled_correct = correct[sampled]
    sampled_missing = missing[sampled]
    accuracy = sampled_correct.mean(axis=1)
    missing_accuracy = (sampled_correct * sampled_missing).sum(axis=1) / sampled_missing.sum(axis=1)

    f1_columns = []
    for label in range(len(labels)):
        tp = ((truth == label) & (prediction == label))[sampled].sum(axis=1)
        fp = ((truth != label) & (prediction == label))[sampled].sum(axis=1)
        fn = ((truth == label) & (prediction != label))[sampled].sum(axis=1)
        denominator = 2 * tp + fp + fn
        f1_columns.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
    f1_array = np.stack(f1_columns)
    macro_f1 = f1_array.mean(axis=0)
    def ci(values):
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {"accuracy_ci95": ci(accuracy),
            "missing_information_accuracy_ci95": ci(missing_accuracy),
            "macro_f1_ci95": ci(macro_f1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="Directory containing one subdirectory per model with the three Task-C core outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/uncertainty")
    args = parser.parse_args()
    output_json = args.output_dir / "task_c_core_metrics.json"
    output_csv = args.output_dir / "task_c_core_metrics.csv"
    evaluator = load_evaluator()
    rng = np.random.default_rng(SEED)
    results = {
        "protocol": {
            "source": "published common unique C1 core and unique-stem C2/C3 core sets",
            "test_items": {"C1": 105, "C2": 15, "C3": 15},
            "bootstrap_replicates": BOOTSTRAPS,
            "random_seed": SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "caution": "The small core sets avoid repeated stems but yield wide uncertainty intervals.",
        },
        "models": {},
    }
    rows_for_csv = []
    model_dirs = sorted(path for path in args.predictions_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise SystemExit("No model subdirectories found in --predictions-root")
    for model_dir in model_dirs:
        item = {}
        c1_rows = [json.loads(line) for line in (model_dir / "fault_testset.jsonl").read_text(encoding="utf-8").splitlines() if line]
        c1 = c1_statistics(c1_rows, evaluator)
        c1["n"] = len(c1_rows)
        c1.update(c1_confidence_intervals(c1_rows, evaluator, rng))
        item["fault_testset"] = c1
        for metric in ("accuracy", "missing_information_accuracy", "macro_f1"):
            rows_for_csv.append([model_dir.name, "fault_testset", c1["n"], metric, c1[metric], *c1[f"{metric}_ci95"]])

        for task in ("fault_exclude", "fault_reason"):
            rows = [json.loads(line) for line in (model_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines() if line]
            exact, jaccards = outcomes(rows, task, evaluator)
            metrics = {
                "n": len(rows),
                "exact_accuracy": float(np.mean(exact)),
                "exact_accuracy_ci95": confidence_interval(exact, rng),
            }
            if jaccards:
                metrics["jaccard"] = float(np.mean(jaccards))
                metrics["jaccard_ci95"] = confidence_interval(jaccards, rng)
            item[task] = metrics
            for metric in ("exact_accuracy", "jaccard"):
                if metric in metrics:
                    rows_for_csv.append([model_dir.name, task, metrics["n"], metric, metrics[metric], *metrics[f"{metric}_ci95"]])
        results["models"][model_dir.name] = item
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task", "n", "metric", "estimate", "ci95_low", "ci95_high"])
        writer.writerows(rows_for_csv)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
