# Elevator Fault Diagnosis Benchmark

Data, evaluation code, and score-verification records for the paper **A benchmark for elevator fault diagnosis: Multi-level tasks from perception to diagnostic reasoning**.

The benchmark evaluates three complementary levels: (A) sensor-based fault detection and fault-code classification, (B) taxonomy-grounded fault classification and cause identification, and (C) reasoning with incomplete or constrained information. It is a closed-set research benchmark, not a validated maintenance or safety-decision system.

## Tasks and released data

| Subtask | Definition | Original release | Revised comparison set |
|---|---|---:|---:|
| A1 | Fault detection from serialized sensor records | 5,000 | 5,000 |
| A2 | Fault-code classification on faulty A1 records | 1,168 | 1,168 |
| B1 | Single-label fault-category classification | 47 | 47 |
| B2 | Multi-label fault-cause identification | 500 | 500 |
| C1 | Fault prediction with deliberately omitted symptoms | 200 generated records / 131 unique inputs | 105-input all-model common unique core |
| C2 | Resolution-method selection from a constrained candidate set | 250 records / 15 unique stems | 15 canonical stems |
| C3 | Multi-cause inference from a constrained candidate set | 245 records / 15 unique stems | 15 canonical stems |

Original Task-C releases remain in `data/task_c/`. The exact deduplicated comparison sets used in the revised paper are in `data/task_c_core/`, with hashes and selection details in `data/task_c_core/manifest.json`.

## Installation

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Metrics

| Subtask | Primary reported metrics |
|---|---|
| A1 | Recall and F2 |
| A2 | Accuracy |
| B1 | Accuracy and fixed-label-space Macro-F1 over all nine categories |
| B2 | Exact-match accuracy and Jaccard similarity |
| C1 | Missing-information accuracy and fixed-label-space Macro-F1 |
| C2 | Exact-match accuracy |
| C3 | Jaccard similarity |

All values are computed deterministically from parsed closed-set predictions. Unparseable choice and fault-ID answers are incorrect; for Task A, an unparseable status follows the published evaluator's convention and is mapped to non-alarm. The paper's cross-task composites are descriptive equal-weight summaries; component metrics and their confidence intervals are the primary results.

## Evaluate one model

Place one model's JSONL files in a directory. Each row must contain the task's gold fields and an `output` field. Choice or fault-ID answers are parsed from `\\boxed{...}`; Task A uses `Status:` and `Fault_code:`.

```text
my_predictions/
├── alarm_judge_v1.jsonl
├── single_choice_v1.jsonl
├── multi_choice_v1.jsonl
├── fault_testset.jsonl
├── fault_exclude.jsonl
└── fault_reason.jsonl
```

```bash
python src/evaluate_all.py \
  --predictions my_predictions \
  --json-output results/my_model_metrics.json
```

For Task C, use the records in `data/task_c_core/` when comparing against the revised paper.

## Reproduce reported scores and intervals

`results/parsed_predictions/` contains only row identifiers, gold labels, and parsed prediction labels for the 14 evaluated models. Raw prompts, free-form responses, device identifiers, provider metadata, and chain-of-thought-style text are excluded. These files can be scored directly:

```bash
python src/evaluate_all.py \
  --predictions results/parsed_predictions/gemini-2.5-pro
```

The paper uses 2,000 non-parametric bootstrap resamples, seed `20260907`, and two-sided percentile 95% confidence intervals:

```bash
python src/evaluate_b1_uncertainty.py \
  --predictions-root results/parsed_predictions

python src/evaluate_task_c_core.py \
  --predictions-root results/parsed_predictions

python src/build_uncertainty_supplement.py
```

Published aggregate files and the machine-readable S1 table are under `results/uncertainty/`.

The component values and descriptive composite formulas used for Figs 2--3 can be regenerated with:

```bash
python src/compute_composites.py
```

## Task-A supervised references

The logistic-regression, 7-nearest-neighbour, and shallow-MLP references use the eight released numerical features. Evaluation uses deterministic five-fold cross-validation grouped by anonymized `elevator_id`, with fold-specific median/IQR scaling. These supervised references are reported separately from the zero-shot LLM comparisons.

```bash
python src/run_task_a_baselines.py
```

Results are written to `results/task_a_supervised_baselines/`.

## Derive Task-C cores

The source-level unique sets can be regenerated without model outputs. Supplying the historical output archive additionally reproduces the all-model common C1 intersection:

```bash
python src/derive_task_c_core_sets.py \
  --output-dir generated_task_c_core

python src/derive_task_c_core_sets.py \
  --archived-outputs-root path/to/model_outputs \
  --output-dir generated_task_c_common_core
```

## Model evaluation protocol

The evaluated identifiers are `claude-sonnet-4-5-20250929`, `deepseek-r1`, `deepseek-v3`, `gemini-2.5-pro`, `gemini-3-pro-preview`, `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-5.2`, `qwen3-235b-a22b-instruct-2507`, `qwen3-8B`, `qwen3-max`, and `qwen3-next-80b-a3b-instruct`.

`qwen3-8B` and `qwen3-next-80b-a3b-instruct` were locally deployed; the other twelve were accessed through APIs. Zero-shot inference used temperature 0.6, top-p 0.95, and a maximum generation length of 16,384 tokens. The historical archive does not preserve the provider endpoint or run date for every API model, so unavailable metadata are not inferred.

## Data availability and responsible use

Task A is released in anonymized form. Original logs and the identifier-to-brand mapping are withheld because of partner confidentiality and equipment-security restrictions. The released numeric fields lack complete unit, calibration, and reference-range metadata and must not be treated as validated physical measurements.

Task B/C files are derived benchmark instances, not the original technical manuals. Users are responsible for ensuring that downstream redistribution remains compatible with source licences and agreements.

Do not upload raw maintenance manuals, industrial logs, API keys, non-anonymized identifiers, or personally identifiable information. Benchmark scores do not establish operational safety or suitability for real maintenance decisions.

## Repository structure

```text
.
├── data/
│   ├── task_a/
│   ├── task_b/
│   ├── task_c/
│   ├── task_c_core/
│   └── knowledge_graph/
├── results/
│   ├── parsed_predictions/
│   ├── task_a_supervised_baselines/
│   └── uncertainty/
├── src/
├── requirements.txt
├── README.md
└── README_zh.md
```

## Citation

Please add the article DOI and final citation after publication. Repository: <https://github.com/Wwwduojin/elevator-fault-diagnosis-benchmark>

For Chinese documentation, see [README_zh.md](README_zh.md).
