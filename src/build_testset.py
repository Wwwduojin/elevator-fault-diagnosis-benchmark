#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-generate elevator fault test set from a Markdown fault document.

Input:
  - Markdown like: "## （8）xxx" + "**主要原因：**" + numbered list
Output:
  - JSONL test set
  - CSV test set (optional)
  - Stats JSON + CSV summary

Usage example:
  python build_testset.py \
    --md_path "/mnt/data/故障.md" \
    --out_dir "./out" \
    --total_samples 1000 \
    --seed 42 \
    --complete_ratio 0.15 \
    --mild_ratio 0.55 \
    --severe_ratio 0.30 \
    --mild_drop 1,2 \
    --severe_keep 1,2

Notes:
  - "现象完备": observed_symptoms == all symptoms of that fault
  - "轻度缺失": drop k symptoms (k in mild_drop)
  - "重度缺失": keep k symptoms (k in severe_keep)
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class FaultEntry:
    fault_id: str
    fault_name: str
    symptoms: List[str]


# ----------------------------
# Markdown parsing
# ----------------------------
HEADER_RE = re.compile(r"^##\s*（\s*(\d+)\s*）\s*(.+?)\s*$")
REASONS_MARK_RE = re.compile(r"^\*\*主要原因：\*\*\s*$")
BULLET_RE = re.compile(r"^\s*(?:\d+[\.\、]\s*|\-\s*|\*\s*)(.+?)\s*$")


def normalize_text(s: str) -> str:
    s = s.strip()
    # unify punctuation/whitespace lightly
    s = re.sub(r"\s+", " ", s)
    # remove trailing Chinese/English semicolons/periods
    s = re.sub(r"[；;。\.\s]+$", "", s)
    return s


def parse_fault_md(md_text: str) -> List[FaultEntry]:
    """
    Parse markdown to fault entries.

    Strategy:
      - Split by headings: ## （id）name
      - In each block, locate '**主要原因：**' and collect subsequent list lines
        until another bold section or heading or blank-line-run ends list.
    """
    lines = md_text.splitlines()
    entries: List[FaultEntry] = []

    i = 0
    current_id = None
    current_name = None
    current_block: List[str] = []

    def flush_block(fid: Optional[str], fname: Optional[str], block_lines: List[str]):
        if not fid or not fname:
            return
        symptoms = extract_reasons(block_lines)
        if symptoms:
            entries.append(FaultEntry(fault_id=fid, fault_name=fname, symptoms=symptoms))

    while i < len(lines):
        m = HEADER_RE.match(lines[i].strip())
        if m:
            # flush previous
            flush_block(current_id, current_name, current_block)
            current_id = m.group(1).zfill(2)
            current_name = normalize_text(m.group(2))
            current_block = []
        else:
            if current_id is not None:
                current_block.append(lines[i])
        i += 1

    flush_block(current_id, current_name, current_block)
    return entries


def extract_reasons(block_lines: List[str]) -> List[str]:
    """
    Extract '主要原因' list items from a fault block.
    """
    symptoms: List[str] = []
    in_reasons = False

    for raw in block_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # start reasons section
        if REASONS_MARK_RE.match(stripped):
            in_reasons = True
            continue

        # stop reasons section when reach another bold heading like **排除方法：**
        if in_reasons and stripped.startswith("**") and stripped.endswith("**") and ("主要原因" not in stripped):
            break

        if in_reasons:
            if not stripped:
                # allow blank lines inside, but do not stop immediately
                continue
            bm = BULLET_RE.match(stripped)
            if bm:
                item = normalize_text(bm.group(1))
                if item:
                    symptoms.append(item)
            # If not matching bullet, ignore (some md may have wrapped lines)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in symptoms:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ----------------------------
# Sampling
# ----------------------------
def parse_int_list(s: str) -> List[int]:
    # "1,2,3" -> [1,2,3]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    return out


def choose_k_drop(n: int, candidates: List[int], rng: random.Random) -> int:
    # choose k in candidates but must be < n
    valid = [k for k in candidates if 0 < k < n]
    if not valid:
        # fallback: drop 1 if possible
        return 1 if n > 1 else 0
    return rng.choice(valid)


def choose_k_keep(n: int, candidates: List[int], rng: random.Random) -> int:
    # choose k in candidates but must be <= n and >=1
    valid = [k for k in candidates if 1 <= k <= n]
    if not valid:
        return 1
    return rng.choice(valid)


def generate_samples(
    entries: List[FaultEntry],
    total_samples: int,
    seed: int,
    complete_ratio: float,
    mild_ratio: float,
    severe_ratio: float,
    mild_drop: List[int],
    severe_keep: List[int],
    ensure_each_label_at_least: int = 0,
) -> List[Dict]:
    """
    Generate dataset samples across labels.

    Allocation strategy:
      - Optionally ensure each label has at least N samples.
      - Remaining samples: label chosen proportionally to symptom-count (or uniform).
      - For each sample: choose completeness level by ratios and sample symptoms accordingly.
    """
    rng = random.Random(seed)
    if not entries:
        raise ValueError("No fault entries parsed from markdown.")

    # Normalize ratios
    rsum = complete_ratio + mild_ratio + severe_ratio
    if rsum <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    complete_ratio, mild_ratio, severe_ratio = (complete_ratio / rsum, mild_ratio / rsum, severe_ratio / rsum)

    # Prepare label weights (you may switch to uniform by setting all to 1)
    weights = [max(1, len(e.symptoms)) for e in entries]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    samples: List[Dict] = []

    # Ensure each label at least N
    if ensure_each_label_at_least > 0:
        needed = ensure_each_label_at_least * len(entries)
        if needed > total_samples:
            # If user sets too high, we'll cap at floor(total_samples/num_labels)
            ensure_each_label_at_least = total_samples // len(entries)
        for e in entries:
            for _ in range(ensure_each_label_at_least):
                samples.append(make_one_sample(e, rng, complete_ratio, mild_ratio, severe_ratio, mild_drop, severe_keep))

    # Remaining samples
    remaining = total_samples - len(samples)
    if remaining > 0:
        # Precompute cumulative for weighted choice (avoid numpy dependency)
        cum = []
        acc = 0.0
        for p in probs:
            acc += p
            cum.append(acc)

        for _ in range(remaining):
            u = rng.random()
            idx = next(i for i, c in enumerate(cum) if u <= c)
            e = entries[idx]
            samples.append(make_one_sample(e, rng, complete_ratio, mild_ratio, severe_ratio, mild_drop, severe_keep))

    # Add unique sample_id
    for i, s in enumerate(samples):
        s["sample_id"] = f"S{i+1:07d}"

    return samples


def make_one_sample(
    entry: FaultEntry,
    rng: random.Random,
    complete_ratio: float,
    mild_ratio: float,
    severe_ratio: float,
    mild_drop: List[int],
    severe_keep: List[int],
) -> Dict:
    S = entry.symptoms[:]
    n = len(S)

    # pick completeness bucket
    u = rng.random()
    if u <= complete_ratio:
        observed = S
        level = "complete"
    elif u <= complete_ratio + mild_ratio:
        # mild missing: drop k
        k = choose_k_drop(n, mild_drop, rng)
        if k <= 0:
            observed = S
            level = "complete"
        else:
            missing_idx = set(rng.sample(range(n), k))
            observed = [S[i] for i in range(n) if i not in missing_idx]
            level = "mild_missing"
    else:
        # severe missing: keep k
        k = choose_k_keep(n, severe_keep, rng)
        if k >= n:
            observed = S
            level = "complete"
        else:
            observed = rng.sample(S, k)
            level = "severe_missing"

    observed = [normalize_text(x) for x in observed if normalize_text(x)]
    observed = list(dict.fromkeys(observed))  # dedup preserve order
    completeness = len(observed) / n if n > 0 else 0.0

    return {
        "fault_id": f"F{entry.fault_id}",
        "label": entry.fault_name,               # classification label
        "fault_name": entry.fault_name,          # explicit
        "observed_symptoms": observed,           # model input
        "all_symptoms_count": n,
        "observed_count": len(observed),
        "completeness": round(completeness, 6),
        "level": level,                          # complete / mild_missing / severe_missing
    }


# ----------------------------
# Stats & Export
# ----------------------------
def compute_stats(samples: List[Dict]) -> Dict:
    total = len(samples)
    by_level: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    by_label_level: Dict[str, Dict[str, int]] = {}

    for s in samples:
        lvl = s["level"]
        lbl = s["label"]
        by_level[lvl] = by_level.get(lvl, 0) + 1
        by_label[lbl] = by_label.get(lbl, 0) + 1
        by_label_level.setdefault(lbl, {})
        by_label_level[lbl][lvl] = by_label_level[lbl].get(lvl, 0) + 1

    complete_cnt = by_level.get("complete", 0)

    return {
        "total_samples": total,
        "complete_samples": complete_cnt,
        "complete_ratio": (complete_cnt / total) if total else 0.0,
        "by_level": dict(sorted(by_level.items(), key=lambda x: (-x[1], x[0]))),
        "by_label": dict(sorted(by_label.items(), key=lambda x: (-x[1], x[0]))),
        "by_label_level": {k: dict(sorted(v.items())) for k, v in sorted(by_label_level.items(), key=lambda x: (-sum(x[1].values()), x[0]))},
    }


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict]):
    if pd is None:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def stats_to_label_table(stats: Dict) -> List[Dict]:
    # Flatten label stats for CSV
    out = []
    for lbl, cnt in stats["by_label"].items():
        row = {"label": lbl, "total": cnt}
        lvls = stats["by_label_level"].get(lbl, {})
        for k, v in lvls.items():
            row[k] = v
        out.append(row)
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_path", type=str, required=True, help="Path to markdown file, e.g., /mnt/data/故障.md")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--total_samples", type=int, default=1000, help="Total samples to generate")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--complete_ratio", type=float, default=0.15)
    ap.add_argument("--mild_ratio", type=float, default=0.55)
    ap.add_argument("--severe_ratio", type=float, default=0.30)

    ap.add_argument("--mild_drop", type=str, default="1,2", help="Drop k symptoms for mild missing, e.g., 1,2")
    ap.add_argument("--severe_keep", type=str, default="1,2", help="Keep k symptoms for severe missing, e.g., 1,2")

    ap.add_argument("--ensure_each_label_at_least", type=int, default=0, help="Ensure each label has at least N samples")

    ap.add_argument("--write_csv", action="store_true", help="Also write CSV outputs (requires pandas)")
    args = ap.parse_args()

    with open(args.md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    entries = parse_fault_md(md_text)
    if not entries:
        raise RuntimeError("Parsed 0 fault entries. Please check markdown format.")

    mild_drop = parse_int_list(args.mild_drop)
    severe_keep = parse_int_list(args.severe_keep)

    samples = generate_samples(
        entries=entries,
        total_samples=args.total_samples,
        seed=args.seed,
        complete_ratio=args.complete_ratio,
        mild_ratio=args.mild_ratio,
        severe_ratio=args.severe_ratio,
        mild_drop=mild_drop,
        severe_keep=severe_keep,
        ensure_each_label_at_least=args.ensure_each_label_at_least,
    )

    stats = compute_stats(samples)

    safe_mkdir(args.out_dir)
    jsonl_path = os.path.join(args.out_dir, "fault_testset.jsonl")
    stats_path = os.path.join(args.out_dir, "fault_testset.stats.json")

    write_jsonl(jsonl_path, samples)
    write_json(stats_path, stats)

    print("=== Generated Test Set ===")
    print(f"Fault entries parsed: {len(entries)}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Complete samples: {stats['complete_samples']} ({stats['complete_ratio']:.4f})")
    print(f"By level: {stats['by_level']}")
    print("Top-10 labels by count:")
    for i, (lbl, cnt) in enumerate(list(stats["by_label"].items())[:10], start=1):
        print(f"  {i:02d}. {cnt:5d}  {lbl}")

    if args.write_csv:
        if pd is None:
            print("pandas not installed; skip CSV.")
        else:
            csv_path = os.path.join(args.out_dir, "fault_testset.csv")
            write_csv(csv_path, samples)

            label_stats_csv = os.path.join(args.out_dir, "fault_testset.label_stats.csv")
            write_csv(label_stats_csv, stats_to_label_table(stats))

            print(f"CSV saved: {csv_path}")
            print(f"Label stats CSV saved: {label_stats_csv}")

    print(f"JSONL saved: {jsonl_path}")
    print(f"Stats JSON saved: {stats_path}")


if __name__ == "__main__":
    main()




# python build_testset.py \
#   --md_path "./故障_8.md" \
#   --out_dir "./out_fault_trainset" \
#   --total_samples 00 \
#   --seed 2026 \
#   --complete_ratio 0.2 \
#   --mild_ratio 0.6 \
#   --severe_ratio 0.2 \
#   --mild_drop 1,2 \
#   --severe_keep 1,2 \
#   --write_csv