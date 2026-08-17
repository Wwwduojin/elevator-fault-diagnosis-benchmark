# Task A data

This directory contains anonymized Task A benchmark inputs:

- `alarm_judge.jsonl`: 5,000 records for fault detection and fault-code classification.
- `normal_data.jsonl`: 3,832 normal-operation records.

Anonymization applied to the released copies:

- `elevator_id` is replaced by a deterministic SHA-256-derived identifier.
- `brand_name` is replaced by an anonymous label such as `brand_01`.
- Sensor values and task labels are retained for benchmark evaluation.

The original industrial files and the original identifier-to-label mapping are not included.
