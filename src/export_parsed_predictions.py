#!/usr/bin/env python3
"""Export score-verification records without prompts or free-form responses.

The output retains only row identifiers, gold labels, and parsed closed-set
predictions. It intentionally excludes raw device identifiers, prompt text,
chain-of-thought-style content, provider metadata, and API responses.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator():
    source = ROOT / "src/evaluate_all.py"
    spec = importlib.util.spec_from_file_location("evaluator", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def find_file(directory: Path, names):
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names} found in {directory}")


def boxed(labels) -> str:
    value = "".join(labels)
    return f"\\boxed{{{value}}}" if value else ""


def item_hash(row: dict) -> str:
    value = json.dumps(
        {key: row[key] for key in ("question", "choice", "answer")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def export_model(model: str, original_dir: Path, task_c_dir: Path, output_dir: Path, evaluator) -> None:
    alarm_rows = load_jsonl(find_file(original_dir, ("alarm_judge_v1.jsonl", "alarm_judge.jsonl")))
    alarm_export = []
    for index, row in enumerate(alarm_rows):
        status, fault = evaluator.parse_alarm_output(row.get("output", ""))
        parsed = "" if status is None else f"Status: {status}"
        if status == 1 and fault:
            parsed += f"\nFault_code: {fault}"
        alarm_export.append({
            "row_index": index,
            "is_alarm": int(row["is_alarm"]),
            "alarm_type": str(row.get("alarm_type", "")),
            "output": parsed,
        })
    write_jsonl(output_dir / "alarm_judge_v1.jsonl", alarm_export)

    for filename, candidates in {
        "single_choice_v1.jsonl": ("single_choice_v1.jsonl", "single_choice.jsonl"),
        "multi_choice_v1.jsonl": ("multi_choice_v1.jsonl", "multi_choice.jsonl"),
    }.items():
        rows = load_jsonl(find_file(original_dir, candidates))
        exported = []
        for index, row in enumerate(rows):
            labels = evaluator.normalize_choice_answer(evaluator.extract_boxed_content(row.get("output", "")))
            exported.append({"row_index": index, "answer": row["answer"], "output": boxed(labels)})
        write_jsonl(output_dir / filename, exported)

    for filename in ("fault_testset.jsonl", "fault_exclude.jsonl", "fault_reason.jsonl"):
        rows = load_jsonl(task_c_dir / filename)
        exported = []
        for index, row in enumerate(rows):
            if filename == "fault_testset.jsonl":
                prediction = evaluator.extract_fault_id(row.get("output", ""))
                exported.append({
                    "row_index": index,
                    "sample_id": row.get("sample_id"),
                    "fault_id": row["fault_id"],
                    "completeness": row.get("completeness", 1.0),
                    "output": boxed([prediction]) if prediction else "",
                })
            else:
                labels = evaluator.normalize_choice_answer(evaluator.extract_boxed_content(row.get("output", "")))
                exported.append({
                    "row_index": index,
                    "item_hash": item_hash(row),
                    "answer": row["answer"],
                    "output": boxed(labels),
                })
        write_jsonl(output_dir / filename, exported)
    print(f"Exported parsed predictions for {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-outputs-root", type=Path, required=True)
    parser.add_argument("--task-c-core-outputs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/parsed_predictions")
    args = parser.parse_args()
    evaluator = load_evaluator()

    original_models = {path.name: path for path in args.original_outputs_root.iterdir() if path.is_dir()}
    core_models = {path.name: path for path in args.task_c_core_outputs_root.iterdir() if path.is_dir()}
    if set(original_models) != set(core_models):
        raise ValueError("Original and Task-C core output roots must contain the same model directories")
    for model in sorted(original_models):
        export_model(model, original_models[model], core_models[model], args.output_root / model, evaluator)


if __name__ == "__main__":
    main()
