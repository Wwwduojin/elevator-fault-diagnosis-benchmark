# Local raw prediction files

把待评测模型的 JSONL 放在这里，然后运行 `src/evaluate_all.py`。
该目录中的 `.json`/`.jsonl` 文件已被 `.gitignore` 排除，避免意外提交原始提示词、
API 回复、未脱敏设备编号或其他敏感数据。

Raw prediction files placed here are intentionally ignored by Git. The public
score-verification records used for the paper are stored separately under
`results/parsed_predictions/`; they contain only gold labels and parsed
closed-set predictions.
