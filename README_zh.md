# 电梯故障诊断 Benchmark

本仓库是论文 **A Benchmark for Elevator Fault Diagnosis: Multi-Level Tasks from Perception to Maintenance Decisions** 的公开 Benchmark 资源。

Benchmark 从三个层次评估电梯故障诊断能力：基于传感器的感知、基于故障分类体系的知识理解，以及不完整信息下的诊断推理。

## Benchmark 任务

| 任务组 | 内容 | 已提供资源 |
|---|---|---|
| A | 传感器故障检测与故障代码分类 | 已脱敏数据位于 `data/task_a/` |
| B1 | 故障类别单选 | `data/task_b/single_choice.jsonl`，47 条 |
| B2 | 故障原因多选 | `data/task_b/multi_choice.jsonl`，500 条 |
| C1 | 缺失故障现象下的故障识别 | `data/task_c/fault_testset.jsonl`，200 条 |
| C2 | 故障处理或排除方法选择 | `data/task_c/fault_exclude.jsonl`，250 条 |
| C3 | 不完整证据下的多原因推理 | `data/task_c/fault_reason.jsonl`，245 条 |

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 模型评测

将模型预测结果放入本地目录，例如：

```text
predictions/
├── single_choice.jsonl
├── multi_choice.jsonl
├── fault_testset.jsonl
├── fault_exclude.jsonl
└── fault_reason.jsonl
```

每行 JSONL 至少应包含任务输入字段、标准答案字段和 `output` 字段。模型最终答案应放在 `\\boxed{}` 中，评测脚本会据此解析选项或故障 ID。

运行统一评测脚本：

```bash
python src/evaluate_all.py --predictions predictions
```

脚本会根据任务输出准确率、Macro-F1、Jaccard、故障覆盖率和缺失信息鲁棒性等指标。

## 测试集构造

`src/build_testset.py` 可以根据 Markdown 格式的故障文档生成 C1 类缺失现象测试集：

```bash
python src/build_testset.py \
  --md_path path/to/fault.md \
  --out_dir generated_testset \
  --total_samples 200 \
  --seed 42
```

## 数据公开说明

Task A 已以脱敏形式公开。由于合作方保密和设备安全限制，本仓库不包含原始日志以及原始设备编号与品牌的对应关系。

B/C 文件是由技术资料派生得到的 Benchmark 实例，不是原始维修手册。在正式公开前，请确认派生知识图谱和测试集符合来源资料的许可证以及合作协议。

请勿上传原始维修手册、工业运行日志、API Key、私有模型输出或个人信息。

## 目录结构

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
├── predictions/          # 本地模型预测结果，已被 Git 忽略
├── requirements.txt
├── README.md
└── README_zh.md
```

## 引用

论文正式发表后，请在此处补充 DOI、仓库地址和正式引用格式。

英文版本见 [README.md](README.md)。
