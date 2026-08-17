# Elevator Fault Diagnosis Benchmark

Official benchmark resources for the paper **A Benchmark for Elevator Fault Diagnosis: Multi-Level Tasks from Perception to Maintenance Decisions**.

The benchmark evaluates elevator fault diagnosis at three levels: sensor-based perception, taxonomy-grounded knowledge understanding, and reasoning under incomplete information.

## Benchmark tasks

| Task group | Description | Released resources |
|---|---|---|
| A | Sensor-based fault detection and fault-code classification | Anonymized data in `data/task_a/` |
| B1 | Single-choice fault-category classification | `data/task_b/single_choice.jsonl` — 47 instances |
| B2 | Multiple-choice fault-cause identification | `data/task_b/multi_choice.jsonl` — 500 instances |
| C1 | Fault prediction with missing phenomena | `data/task_c/fault_testset.jsonl` — 200 instances |
| C2 | Fault-handling or exclusion-method selection | `data/task_c/fault_exclude.jsonl` — 250 instances |
| C3 | Multi-cause inference under incomplete evidence | `data/task_c/fault_reason.jsonl` — 245 instances |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation

Place model predictions in a local directory, for example:

```text
predictions/
├── single_choice.jsonl
├── multi_choice.jsonl
├── fault_testset.jsonl
├── fault_exclude.jsonl
└── fault_reason.jsonl
```

Each JSONL record should contain the task fields, the ground-truth fields, and an `output` field. The final model answer should be enclosed in `\\boxed{}` so that the evaluator can extract the predicted label or fault ID.

Run the unified evaluator:

```bash
python src/evaluate_all.py --predictions predictions
```

The evaluator reports task-specific accuracy, Macro-F1, Jaccard similarity, fault coverage, and missing-information robustness where applicable.

## Test-set construction

`src/build_testset.py` generates C1-style missing-phenomena test sets from a Markdown fault document:

```bash
python src/build_testset.py \
  --md_path path/to/fault.md \
  --out_dir generated_testset \
  --total_samples 200 \
  --seed 42
```

## Data availability

Task A is released in anonymized form. The original logs and the original identifier-to-brand mapping are not included because they are subject to partner confidentiality and equipment-security restrictions.

The B/C files are derived benchmark instances rather than the original technical manuals. Before public distribution, users should confirm that the derived knowledge graph and benchmark instances are compatible with the licenses and agreements governing their source materials.

Do not upload raw maintenance manuals, industrial logs, API keys, private model outputs, or personally identifiable information.

## Repository structure

```text
elevator_github_release/
├── data/
│   ├── task_a/
│   ├── task_b/
│   ├── task_c/
│   └── knowledge_graph/
├── src/
│   ├── build_testset.py
│   └── evaluate_all.py
├── predictions/          # local model predictions; ignored by Git
├── requirements.txt
├── README.md
└── README_zh.md
```

## Citation

Please add the final DOI, repository URL, and citation format after publication.

For a Chinese version, see [README_zh.md](README_zh.md).
