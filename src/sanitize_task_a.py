#!/usr/bin/env python3
"""Anonymize Task A JSONL files for public release."""

import argparse
import hashlib
import json
from pathlib import Path


def anonymize_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"elevator_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", nargs="+", required=True)
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        parser.error("--input and --output must contain the same number of files")

    records_by_file = []
    brands = set()
    for input_path in args.input:
        records = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        records_by_file.append(records)
        brands.update(record.get("brand_name", "") for record in records)

    brand_map = {brand: f"brand_{i:02d}" for i, brand in enumerate(sorted(brands), 1) if brand}
    for output_path, records in zip(args.output, records_by_file):
        output = []
        for record in records:
            record = dict(record)
            if record.get("elevator_id"):
                record["elevator_id"] = anonymize_id(record["elevator_id"])
            if record.get("brand_name"):
                record["brand_name"] = brand_map[record["brand_name"]]
            output.append(json.dumps(record, ensure_ascii=False))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
