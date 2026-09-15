#!/usr/bin/env python3
"""Derive deterministic deduplicated Task-C evaluation sets.

C1 keeps one record per unique ``(fault_id, observed_symptoms)`` input. When
``--archived-outputs-root`` is supplied, it is further restricted to inputs
present for every archived model, reproducing the common-core procedure used
in the paper. C2 and C3 keep one canonical option set per unique question
stem. Source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_input(row: dict) -> str:
    if "fault_id" in row:
        value = {"fault_id": row["fault_id"], "observed_symptoms": row["observed_symptoms"]}
    else:
        value = {"question": row["question"], "choice": row["choice"], "answer": row["answer"]}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_c1(rows):
    selected = {}
    for row in rows:
        key = (row["fault_id"], tuple(row["observed_symptoms"]))
        if key not in selected or row["sample_id"] < selected[key]["sample_id"]:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def unique_stems(rows):
    selected = []
    for question in sorted({row["question"] for row in rows}):
        candidates = [row for row in rows if row["question"] == question]
        selected.append(min(candidates, key=canonical_input))
    return selected


def common_c1(rows, outputs_root: Path):
    common = {canonical_input(row) for row in rows}
    model_dirs = sorted(path for path in outputs_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise ValueError("No model subdirectories found in --archived-outputs-root")
    for model_dir in model_dirs:
        path = model_dir / "fault_testset.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        common &= {canonical_input(row) for row in load_jsonl(path)}
    return [row for row in rows if canonical_input(row) in common]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/task_c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_task_c_core")
    parser.add_argument("--archived-outputs-root", type=Path,
                        help="Optional model-output archive used to obtain an all-model common C1 core")
    args = parser.parse_args()

    specifications = {
        "fault_testset": unique_c1,
        "fault_exclude": unique_stems,
        "fault_reason": unique_stems,
    }
    manifest = {"sets": {}}
    for name, derive in specifications.items():
        source = args.source_dir / f"{name}.jsonl"
        source_rows = load_jsonl(source)
        core_rows = derive(source_rows)
        initial_unique = len(core_rows)
        selection = (
            "lowest sample_id per unique (fault_id, observed_symptoms)"
            if name == "fault_testset"
            else "lexicographically canonical option set per unique question stem"
        )
        if name == "fault_testset" and args.archived_outputs_root:
            core_rows = common_c1(core_rows, args.archived_outputs_root)
            selection += ", then intersection across all archived model outputs"
        output = args.output_dir / f"{name}.jsonl"
        write_jsonl(output, core_rows)
        manifest["sets"][name] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "source_rows": len(source_rows),
            "initial_unique_rows": initial_unique,
            "core": str(output),
            "core_sha256": sha256(output),
            "core_rows": len(core_rows),
            "selection": selection,
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
