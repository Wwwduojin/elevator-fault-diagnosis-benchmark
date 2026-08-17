import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

def prediction_files(prediction_dir: str) -> List[str]:
    """Discover supported prediction files without relying on local absolute paths."""
    root = Path(prediction_dir)
    names = [
        "alarm_judge.jsonl", "alarm_judge_v1.jsonl",
        "single_choice.jsonl", "single_choice_v1.jsonl",
        "multi_choice.jsonl", "multi_choice_v1.jsonl",
        "fault_testset.jsonl", "fault_exclude.jsonl", "fault_reason.jsonl",
    ]
    return [str(root / name) for name in names if (root / name).is_file()]

# ================== LaTeX ==================
BOX_PATTERN = re.compile(r"\\?boxed\{((?:[^{}]|\{[^{}]*\})+)\}")
TEXT_WRAPPER_PATTERN = re.compile(r"\\(?:text|textbf|mathrm|textrm)\{([^}]+)\}")

def clean_latex_wrapper(s: str) -> str:
    s = s.strip()
    while True:
        m = TEXT_WRAPPER_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1).strip()
    return s

def extract_boxed_content(text: str) -> str:
    m = BOX_PATTERN.search(text or "")
    return clean_latex_wrapper(m.group(1)) if m else ""

# ================== Choice ==================
def normalize_choice_answer(ans_str: str) -> List[str]:
    if not ans_str:
        return []
    ans_str = ans_str.upper().strip()

    m = re.match(r"^([A-Z])[\.\-\s:：].*", ans_str)
    if m:
        return [m.group(1)]

    if re.fullmatch(r"[A-Z]", ans_str):
        return [ans_str]

    ans_str = ans_str.replace("\\,", ",").replace("\\", " ")
    ans_str = ans_str.replace("，", ",").replace("、", ",")

    if re.search(r"[,\s]", ans_str):
        parts = re.split(r"[,\s]+", ans_str)
        parts = [p for p in parts if re.fullmatch(r"[A-Z]", p)]
    else:
        parts = re.findall(r"[A-Z]", ans_str)

    return sorted(set(parts))

# ================== Metrics ==================
def compute_macro_f1(stats: Dict[str, Dict[str, int]]) -> float:
    f1s = []
    for s in stats.values():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        if tp == fp == fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        if p + r:
            f1s.append(2 * p * r / (p + r))
    return sum(f1s) / len(f1s) if f1s else None

def jaccard(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred and gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)

# ================== taskA 解析 ==================
def parse_alarm_output(text: str):
    if not text:
        return None, None
    matches = list(re.finditer(r"Status:\s*(\d)", text))
    if not matches:
        return None, None
    status = int(matches[-1].group(1))
    fault = None
    if status == 1:
        rest = text[matches[-1].end():]
        m = re.search(r"Fault_code:\s*(\d+)", rest)
        if m:
            fault = m.group(1)
    return status, fault

# ================== taskC C1 ==================
def extract_fault_id(output: str) -> Optional[str]:
    raw = extract_boxed_content(output)
    m = re.search(r"F\d+", raw.upper()) if raw else None
    return m.group(0) if m else None

# ================== task 推断 ==================
def infer_task_from_path(path: str):
    name = path.split("/")[-1]
    if name == "alarm_judge.jsonl" or name == "alarm_judge_v1.jsonl":
        return "taskA"
    if name in {"single_choice.jsonl", "single_choice_v1.jsonl"}:
        return "taskB", "B1"
    if name in {"multi_choice.jsonl", "multi_choice_v1.jsonl"}:
        return "taskB", "B2"
    if name == "fault_testset.jsonl":
        return "taskC", "C1"
    if name == "fault_exclude.jsonl":
        return "taskC", "C2"
    if name == "fault_reason.jsonl":
        return "taskC", "C3"
    raise ValueError(name)

# ================== 单文件评测 ==================
def evaluate_file(path: str) -> List[Dict]:
    task_info = infer_task_from_path(path)
    results = []

    # -------- taskA --------
    # -------- taskA --------
    if task_info == "taskA":
        a1_tp = a1_fp = a1_fn = a1_tn = 0
        a2_total = a2_correct = 0
        a1_total = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)

                gold_status = int(d["is_alarm"])
                pred_status, pred_fault = parse_alarm_output(d.get("output", ""))

                # 若无法解析，视为预测为 0（更合理）
                if pred_status is None:
                    pred_status = 0

                a1_total += 1

                # ---- 混淆矩阵完整统计 ----
                if gold_status == 1 and pred_status == 1:
                    a1_tp += 1
                elif gold_status == 0 and pred_status == 1:
                    a1_fp += 1
                elif gold_status == 1 and pred_status == 0:
                    a1_fn += 1
                elif gold_status == 0 and pred_status == 0:
                    a1_tn += 1

                # ---- A2 ----
                if gold_status == 1:
                    a2_total += 1
                    if pred_status == 1 and pred_fault == str(d["alarm_type"]):
                        a2_correct += 1

        precision = a1_tp / (a1_tp + a1_fp) if (a1_tp + a1_fp) else None
        recall = a1_tp / (a1_tp + a1_fn) if (a1_tp + a1_fn) else None
        alarm_rate = (a1_tp + a1_fp) / a1_total if a1_total else None

        results.append({
            "task": "taskA",
            "subtask": "A1",
            "num_samples": a1_total,
            "precision_fault": precision,
            "recall_fault": recall,
            "alarm_rate": alarm_rate,
            "tp": a1_tp,
            "fp": a1_fp,
            "fn": a1_fn,
            "tn": a1_tn,
            "accuracy": None,
            "macro_f1": None,
            "fault_coverage": None,
            "jaccard": None,
            "missing_robustness": None,
        })

        results.append({
            "task": "taskA",
            "subtask": "A2",
            "num_samples": a2_total,
            "accuracy": a2_correct / a2_total if a2_total else None,
            "precision_fault": None,
            "recall_fault": None,
            "macro_f1": None,
            "fault_coverage": None,
            "jaccard": None,
            "missing_robustness": None,
        })

        return results

    # -------- taskB / taskC --------
    task, subtask = task_info
    total = correct = 0
    coverage_sum = jaccard_sum = 0.0
    missing_total = missing_correct = 0
    label_stats = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            total += 1

            # ===== taskB =====
            if task == "taskB":
                pred = set(normalize_choice_answer(extract_boxed_content(d.get("output", ""))))
                gold = set(d["answer"])

                if pred == gold:
                    correct += 1

                for l in pred | gold:
                    label_stats.setdefault(l, {"tp": 0, "fp": 0, "fn": 0})
                    if l in pred and l in gold:
                        label_stats[l]["tp"] += 1
                    elif l in pred:
                        label_stats[l]["fp"] += 1
                    else:
                        label_stats[l]["fn"] += 1

                if subtask == "B2":
                    coverage_sum += len(pred & gold) / len(gold) if gold else 0.0
                    jaccard_sum += jaccard(pred, gold)

            # ===== taskC =====
            elif task == "taskC":
                if subtask == "C1":
                    gold = d["fault_id"]
                    pred = extract_fault_id(d.get("output", ""))

                    if pred == gold:
                        correct += 1

                    for l in {gold, pred} - {None}:
                        label_stats.setdefault(l, {"tp": 0, "fp": 0, "fn": 0})

                    if pred == gold:
                        label_stats[gold]["tp"] += 1
                    else:
                        if pred:
                            label_stats[pred]["fp"] += 1
                        label_stats[gold]["fn"] += 1

                    if d.get("completeness", 1.0) < 1.0:
                        missing_total += 1
                        if pred == gold:
                            missing_correct += 1

                elif subtask in {"C2", "C3"}:
                    pred = set(normalize_choice_answer(extract_boxed_content(d.get("output", ""))))
                    gold = set(d["answer"])

                    if pred == gold:
                        correct += 1

                    if subtask == "C3":
                        coverage_sum += len(pred & gold) / len(gold) if gold else 0.0
                        jaccard_sum += jaccard(pred, gold)

    return [{
        "task": task,
        "subtask": subtask,
        "num_samples": total,
        "accuracy": correct / total if total else None,
        "precision_fault": None,
        "recall_fault": None,
        "macro_f1": compute_macro_f1(label_stats) if label_stats else None,
        "fault_coverage": coverage_sum / total if coverage_sum else None,
        "jaccard": jaccard_sum / total if jaccard_sum else None,
        "missing_robustness": (
            missing_correct / missing_total if missing_total else None
        ),
    }]

# ================== Output ==================
def fmt_percent(x: Optional[float], digits: int = 2) -> Optional[str]:
    if x is None:
        return None
    return f"{x * 100:.{digits}f}%"


# ================== Main ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate elevator benchmark predictions.")
    parser.add_argument("--predictions", required=True, help="Directory containing prediction JSONL files.")
    args = parser.parse_args()
    print("\n========== 批量评测开始 ==========\n")

    all_results = []

    paths = prediction_files(args.predictions)
    if not paths:
        raise SystemExit("No supported prediction JSONL files found in --predictions")

    for path in paths:
        rows = evaluate_file(path)
        for r in rows:
            all_results.append(r)
            print(f"{r['task']} {r['subtask']} | N={r['num_samples']}")
            for k in [
                "accuracy",
                "precision_fault",
                "recall_fault",
                "macro_f1",
                "fault_coverage",
                "jaccard",
                "missing_robustness",
            ]:
                if r.get(k) is not None:
                    v = r.get(k)
                    print(f"  {k}: {fmt_percent(v)}")
            print("-" * 50)

    print("\n========== 统一结果表 ==========\n")
    for r in all_results:
        print(r)
