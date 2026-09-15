#!/usr/bin/env python3
"""Run dependency-light supervised reference baselines for Task A.

The models use only the eight released numerical sensor fields.  Evaluation is
grouped by anonymised elevator ID, so no elevator contributes records to both a
training and a test fold.  These supervised references are reported separately
from the zero-shot LLM benchmark and are not included in its composite score.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/task_a/alarm_judge.jsonl"
OUT = ROOT / "results/task_a_supervised_baselines"
FEATURES = [
    "speed", "pressure", "temperature_in_lift", "temperature_in_room",
    "temperature_upon_lift", "x_acceleration", "y_acceleration", "z_acceleration",
]
SEED = 20260907
FOLDS = 5


def load_data():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64)
    y_alarm = np.asarray([int(row["is_alarm"]) for row in rows], dtype=np.int64)
    codes = sorted({str(row["alarm_type"]) for row in rows if int(row["is_alarm"]) == 1})
    code_index = {code: i for i, code in enumerate(codes)}
    faulty = y_alarm == 1
    y_code = np.asarray([code_index[str(row["alarm_type"])] for row in rows if int(row["is_alarm"]) == 1], dtype=np.int64)
    groups = np.asarray([row["elevator_id"] for row in rows], dtype=object)
    return rows, x, y_alarm, faulty, y_code, groups, codes


def make_group_folds(groups, y, n_folds=FOLDS):
    """Greedy, deterministic group allocation balancing size and positive count."""
    group_to_indices = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)
    stats = []
    for group, indices in group_to_indices.items():
        labels = y[indices]
        stats.append((group, len(indices), int(labels.sum())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(stats)
    stats.sort(key=lambda item: (item[1], item[2]), reverse=True)
    buckets = [{"groups": [], "n": 0, "pos": 0} for _ in range(n_folds)]
    # Seed every fold with one large group before greedy balancing.  Without
    # this guard a very large early group can leave later folds empty.
    for index, (group, count, positive) in enumerate(stats[:n_folds]):
        buckets[index]["groups"].append(group)
        buckets[index]["n"] += count
        buckets[index]["pos"] += positive
    for group, count, positive in stats[n_folds:]:
        # Standard longest-processing-time allocation: device isolation is the
        # essential property, and this keeps fold record counts comparable.
        chosen = min(range(n_folds), key=lambda i: (buckets[i]["n"], buckets[i]["pos"]))
        buckets[chosen]["groups"].append(group)
        buckets[chosen]["n"] += count
        buckets[chosen]["pos"] += positive
    return [set(bucket["groups"]) for bucket in buckets]


def scaler_fit(x):
    median = np.median(x, axis=0)
    q1 = np.quantile(x, 0.25, axis=0)
    q3 = np.quantile(x, 0.75, axis=0)
    scale = q3 - q1
    scale[scale == 0] = 1.0
    return median, scale


def scaler_apply(x, median, scale):
    # Robust centring prevents large encoded values from dominating all fields.
    return (x - median) / scale


def macro_f1(y_true, y_pred, classes):
    scores = []
    for cls in classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def binary_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    beta_sq = 4.0
    f2 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) if precision + recall else 0.0
    return {"accuracy": float((y_true == y_pred).mean()), "recall": recall, "precision": precision, "f2": f2,
            "tp": tp, "fp": fp, "fn": fn}


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_logistic(x, y, n_classes, epochs=700, learning_rate=0.08, l2=1e-3):
    rng = np.random.default_rng(SEED + n_classes)
    weights = rng.normal(0, 0.01, size=(x.shape[1], n_classes))
    bias = np.zeros(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sample_weight = len(y) / (n_classes * counts[y])
    target = np.eye(n_classes)[y]
    for _ in range(epochs):
        probabilities = softmax(x @ weights + bias)
        residual = (probabilities - target) * sample_weight[:, None]
        weights -= learning_rate * ((x.T @ residual) / len(y) + l2 * weights)
        bias -= learning_rate * residual.mean(axis=0)
    return weights, bias


def predict_logistic(model, x):
    weights, bias = model
    return np.argmax(x @ weights + bias, axis=1)


def predict_knn(x_train, y_train, x_test, n_classes, k=7):
    predictions = []
    # Batches keep the distance matrix bounded without changing results.
    for start in range(0, len(x_test), 256):
        batch = x_test[start:start + 256]
        distances = ((batch[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        votes = y_train[nearest]
        predictions.extend(np.bincount(row, minlength=n_classes).argmax() for row in votes)
    return np.asarray(predictions, dtype=np.int64)


def fit_mlp(x, y, n_classes, hidden=32, epochs=450, learning_rate=0.01, l2=1e-4):
    rng = np.random.default_rng(SEED + 100 + n_classes)
    w1 = rng.normal(0, np.sqrt(2 / x.shape[1]), size=(x.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, n_classes))
    b2 = np.zeros(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sample_weight = len(y) / (n_classes * counts[y])
    target = np.eye(n_classes)[y]
    for _ in range(epochs):
        h_pre = x @ w1 + b1
        h = np.maximum(h_pre, 0)
        probabilities = softmax(h @ w2 + b2)
        d_logits = (probabilities - target) * sample_weight[:, None] / len(y)
        d_w2 = h.T @ d_logits + l2 * w2
        d_b2 = d_logits.sum(axis=0)
        d_hidden = (d_logits @ w2.T) * (h_pre > 0)
        d_w1 = x.T @ d_hidden + l2 * w1
        d_b1 = d_hidden.sum(axis=0)
        w1 -= learning_rate * d_w1
        b1 -= learning_rate * d_b1
        w2 -= learning_rate * d_w2
        b2 -= learning_rate * d_b2
    return w1, b1, w2, b2


def predict_mlp(model, x):
    w1, b1, w2, b2 = model
    return np.argmax(np.maximum(x @ w1 + b1, 0) @ w2 + b2, axis=1)


def evaluate_task(x, y, groups, n_classes, task_name):
    folds = make_group_folds(groups, y)
    specs = {
        "Logistic regression": (lambda a, b: fit_logistic(a, b, n_classes), predict_logistic),
        "k-nearest neighbours (k=7)": (None, None),
        "MLP (8-32-output)": (lambda a, b: fit_mlp(a, b, n_classes), predict_mlp),
    }
    all_results = {}
    for model_name in specs:
        prediction = np.empty(len(y), dtype=np.int64)
        fold_rows = []
        for fold_no, test_groups in enumerate(folds, 1):
            test_mask = np.asarray([group in test_groups for group in groups])
            train_mask = ~test_mask
            median, scale = scaler_fit(x[train_mask])
            x_train = scaler_apply(x[train_mask], median, scale)
            x_test = scaler_apply(x[test_mask], median, scale)
            if model_name.startswith("k-nearest"):
                fold_prediction = predict_knn(x_train, y[train_mask], x_test, n_classes)
            else:
                fit, predict = specs[model_name]
                fold_prediction = predict(fit(x_train, y[train_mask]), x_test)
            prediction[test_mask] = fold_prediction
            fold_rows.append({"fold": fold_no, "test_records": int(test_mask.sum()),
                              "test_elevators": len(test_groups)})
        if task_name == "A1":
            metrics = binary_metrics(y, prediction)
        else:
            metrics = {"accuracy": float((y == prediction).mean()),
                       "macro_f1": macro_f1(y, prediction, range(n_classes))}
        all_results[model_name] = {"metrics": metrics, "predictions": prediction, "folds": fold_rows}
    return all_results


def main():
    rows, x, y_alarm, faulty, y_code, groups, codes = load_data()
    a1 = evaluate_task(x, y_alarm, groups, 2, "A1")
    a2 = evaluate_task(x[faulty], y_code, groups[faulty], len(codes), "A2")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "data": str(DATA.relative_to(ROOT)), "data_records": len(rows), "features": FEATURES,
            "random_seed": SEED, "folds": FOLDS,
            "split": "grouped five-fold cross-validation by anonymised elevator_id",
            "preprocessing": "fold-specific robust centring by median and scaling by interquartile range",
            "scope": "supervised Task A reference baselines; excluded from zero-shot LLM composite ranking",
        },
        "fault_codes": codes,
        "A1": {name: {"metrics": result["metrics"], "folds": result["folds"]} for name, result in a1.items()},
        "A2": {name: {"metrics": result["metrics"], "folds": result["folds"]} for name, result in a2.items()},
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "out_of_fold_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "model", "sample_index", "elevator_id", "truth", "prediction"])
        for name, result in a1.items():
            for index, pred in enumerate(result["predictions"]):
                writer.writerow(["A1", name, index, groups[index], int(y_alarm[index]), int(pred)])
        faulty_indices = np.flatnonzero(faulty)
        for name, result in a2.items():
            for local_index, pred in enumerate(result["predictions"]):
                index = faulty_indices[local_index]
                writer.writerow(["A2", name, index, groups[index], codes[y_code[local_index]], codes[pred]])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
