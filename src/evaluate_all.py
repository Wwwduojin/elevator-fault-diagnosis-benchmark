#!/usr/bin/env python3
"""Evaluate one model's predictions for the elevator diagnosis benchmark.

Prediction JSONL files contain the released task record plus an ``output``
field. The model's final closed-set answer is parsed from ``\\boxed{...}``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


PREDICTION_CANDIDATES = {
    "A": ("alarm_judge_v1.jsonl", "alarm_judge.jsonl"),
    "B1": ("single_choice_v1.jsonl", "single_choice.jsonl"),
    "B2": ("multi_choice_v1.jsonl", "multi_choice.jsonl"),
    "C1": ("fault_testset.jsonl",),
    "C2": ("fault_exclude.jsonl",),
    "C3": ("fault_reason.jsonl",),
}

BOX_PATTERN = re.compile(r"\\?boxed\{((?:[^{}]|\{[^{}]*\})+)\}")
TEXT_WRAPPER_PATTERN = re.compile(r"\\(?:text|textbf|mathrm|textrm)\{([^}]+)\}")


def prediction_files(prediction_dir: Path) -> List[Path]:
    """Select at most one prediction file for each task."""
    paths = []
    for names in PREDICTION_CANDIDATES.values():
        match = next((prediction_dir / name for name in names if (prediction_dir / name).is_file()), None)
        if match is not None:
            paths.append(match)
    return paths


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_latex_wrapper(value: str) -> str:
    value = value.strip()
    while True:
        match = TEXT_WRAPPER_PATTERN.fullmatch(value)
        if not match:
            return value
        value = match.group(1).strip()


def extract_boxed_content(text: str) -> str:
    match = BOX_PATTERN.search(text or "")
    return clean_latex_wrapper(match.group(1)) if match else ""


def normalize_choice_answer(answer: str) -> List[str]:
    """Return the unique A--Z choice labels found in a boxed answer."""
    if not answer:
        return []
    answer = answer.upper().strip()
    match = re.match(r"^([A-Z])[.\-\s:\uff1a].*", answer)
    if match:
        return [match.group(1)]
    if re.fullmatch(r"[A-Z]", answer):
        return [answer]
    answer = answer.replace("\\,", ",").replace("\\", " ")
    answer = answer.replace("，", ",").replace("、", ",")
    if re.search(r"[,\s]", answer):
        parts = [part for part in re.split(r"[,\s]+", answer) if re.fullmatch(r"[A-Z]", part)]
    else:
        parts = re.findall(r"[A-Z]", answer)
    return sorted(set(parts))


def answer_set(value: object) -> Set[str]:
    if isinstance(value, str):
        return set(normalize_choice_answer(value))
    if isinstance(value, Sequence):
        return {str(item).upper() for item in value}
    raise TypeError(f"Unsupported answer value: {value!r}")


def macro_f1(
    gold_sets: Sequence[Set[str]],
    predicted_sets: Sequence[Set[str]],
    labels: Iterable[str],
) -> float:
    """Compute Macro-F1 over a fixed label space with zero_division=0."""
    scores = []
    for label in labels:
        tp = sum(label in gold and label in pred for gold, pred in zip(gold_sets, predicted_sets))
        fp = sum(label not in gold and label in pred for gold, pred in zip(gold_sets, predicted_sets))
        fn = sum(label in gold and label not in pred for gold, pred in zip(gold_sets, predicted_sets))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def jaccard(predicted: Set[str], gold: Set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    return len(predicted & gold) / len(predicted | gold)


def parse_alarm_output(text: str):
    matches = list(re.finditer(r"Status:\s*(\d)", text or ""))
    if not matches:
        return None, None
    status = int(matches[-1].group(1))
    fault = None
    if status == 1:
        match = re.search(r"Fault_code:\s*(\d+)", text[matches[-1].end():])
        if match:
            fault = match.group(1)
    return status, fault


def extract_fault_id(output: str) -> Optional[str]:
    boxed = extract_boxed_content(output)
    match = re.search(r"F\d+", boxed.upper()) if boxed else None
    return match.group(0) if match else None


def infer_subtask(path: Path) -> str:
    for subtask, names in PREDICTION_CANDIDATES.items():
        if path.name in names:
            return subtask
    raise ValueError(f"Unsupported prediction filename: {path.name}")


def evaluate_task_a(rows: Sequence[dict]) -> List[Dict[str, object]]:
    tp = fp = fn = tn = 0
    a2_total = a2_correct = 0
    for row in rows:
        gold_status = int(row["is_alarm"])
        predicted_status, predicted_fault = parse_alarm_output(row.get("output", ""))
        # The released prompt requires Status 0 or 1. Preserve the published
        # evaluator's convention of mapping an unparseable answer to Status 0.
        if predicted_status is None:
            predicted_status = 0
        if gold_status == predicted_status == 1:
            tp += 1
        elif gold_status == 0 and predicted_status == 1:
            fp += 1
        elif gold_status == 1 and predicted_status == 0:
            fn += 1
        else:
            tn += 1
        if gold_status == 1:
            a2_total += 1
            if predicted_status == 1 and predicted_fault == str(row["alarm_type"]):
                a2_correct += 1

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    beta_squared = 4.0
    f2 = (
        (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        if precision + recall else 0.0
    )
    return [
        {
            "task": "A",
            "subtask": "A1",
            "num_samples": len(rows),
            "recall": recall,
            "f2": f2,
            "precision": precision,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        {
            "task": "A",
            "subtask": "A2",
            "num_samples": a2_total,
            "accuracy": a2_correct / a2_total if a2_total else 0.0,
        },
    ]


def evaluate_choice_task(rows: Sequence[dict], subtask: str) -> Dict[str, object]:
    gold_sets = [answer_set(row["answer"]) for row in rows]
    predicted_sets = [
        set(normalize_choice_answer(extract_boxed_content(row.get("output", ""))))
        for row in rows
    ]
    exact = [gold == predicted for gold, predicted in zip(gold_sets, predicted_sets)]
    result: Dict[str, object] = {
        "task": "B" if subtask.startswith("B") else "C",
        "subtask": subtask,
        "num_samples": len(rows),
        "exact_accuracy": sum(exact) / len(exact) if exact else 0.0,
    }
    if subtask == "B1":
        labels = sorted(set().union(*gold_sets))
        result["macro_f1"] = macro_f1(gold_sets, predicted_sets, labels)
        result["label_space"] = labels
    if subtask in {"B2", "C3"}:
        scores = [jaccard(predicted, gold) for gold, predicted in zip(gold_sets, predicted_sets)]
        result["jaccard"] = sum(scores) / len(scores) if scores else 0.0
    return result


def evaluate_c1(rows: Sequence[dict]) -> Dict[str, object]:
    gold_sets = [{str(row["fault_id"])} for row in rows]
    predicted_sets = []
    for row in rows:
        prediction = extract_fault_id(row.get("output", ""))
        predicted_sets.append({prediction} if prediction else set())
    exact = [gold == predicted for gold, predicted in zip(gold_sets, predicted_sets)]
    missing = [index for index, row in enumerate(rows) if float(row.get("completeness", 1.0)) < 1.0]
    labels = sorted(set().union(*gold_sets))
    return {
        "task": "C",
        "subtask": "C1",
        "num_samples": len(rows),
        "accuracy": sum(exact) / len(exact) if exact else 0.0,
        "missing_information_samples": len(missing),
        "missing_information_accuracy": (
            sum(exact[index] for index in missing) / len(missing) if missing else None
        ),
        "macro_f1": macro_f1(gold_sets, predicted_sets, labels),
        "label_space": labels,
    }


def evaluate_file(path: Path) -> List[Dict[str, object]]:
    rows = load_jsonl(path)
    subtask = infer_subtask(path)
    if subtask == "A":
        return evaluate_task_a(rows)
    if subtask == "C1":
        return [evaluate_c1(rows)]
    return [evaluate_choice_task(rows, subtask)]


def fmt_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Directory containing one model's JSONL files")
    parser.add_argument("--json-output", type=Path, help="Optional path for machine-readable results")
    args = parser.parse_args()

    paths = prediction_files(args.predictions)
    if not paths:
        raise SystemExit("No supported prediction JSONL files found in --predictions")

    all_results = []
    for path in paths:
        for result in evaluate_file(path):
            all_results.append(result)
            print(f"{result['task']} {result['subtask']} | N={result['num_samples']}")
            for key in ("accuracy", "exact_accuracy", "recall", "f2", "macro_f1", "jaccard", "missing_information_accuracy"):
                if result.get(key) is not None:
                    print(f"  {key}: {fmt_percent(float(result[key]))}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
