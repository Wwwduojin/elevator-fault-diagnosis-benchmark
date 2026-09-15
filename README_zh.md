# 电梯故障诊断 Benchmark

本仓库提供论文 **A benchmark for elevator fault diagnosis: Multi-level tasks from perception to diagnostic reasoning** 的数据、评测代码和去敏后的分数复核记录。

Benchmark 包含三个互补层次：（A）基于传感器记录的故障检测与故障代码分类；（B）基于故障分类体系的类别判定与原因识别；（C）不完整或受约束信息下的诊断推理。它是闭集研究基准，不是经过实际维保或安全决策验证的系统。

## 任务与数据

| 子任务 | 定义 | 原始公开集 | 修订稿比较集 |
|---|---|---:|---:|
| A1 | 基于序列化传感器记录的故障检测 | 5,000 | 5,000 |
| A2 | A1 故障样本的故障代码分类 | 1,168 | 1,168 |
| B1 | 故障大类单标签分类 | 47 | 47 |
| B2 | 故障原因多标签识别 | 500 | 500 |
| C1 | 故意缺失部分现象时的故障预测 | 200 条生成记录 / 131 个唯一输入 | 105 个全模型共同唯一输入 |
| C2 | 受约束候选集中的排除方法选择 | 250 条 / 15 个唯一题干 | 15 个规范化题干 |
| C3 | 受约束候选集中的多原因推理 | 245 条 / 15 个唯一题干 | 15 个规范化题干 |

原始 Task C 数据保留在 `data/task_c/`，修订稿实际使用的去重核心集位于 `data/task_c_core/`，选择方法和 SHA-256 校验值见 `data/task_c_core/manifest.json`。

## 安装

建议使用 Python 3.8 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 评测指标

| 子任务 | 论文主要报告指标 |
|---|---|
| A1 | Recall 和 F2 |
| A2 | Accuracy |
| B1 | Accuracy，以及在全部 9 个固定类别上计算的 Macro-F1 |
| B2 | 严格匹配准确率和 Jaccard |
| C1 | 缺失信息准确率和固定标签空间 Macro-F1 |
| C2 | 严格匹配准确率 |
| C3 | Jaccard |

所有指标均由解析后的闭集答案确定性计算。无法解析的选项或故障 ID 记为错误；Task A 中无法解析的状态按已发表评测器的约定映射为“无故障”。论文中的跨任务综合分仅是等权的描述性摘要，主要结果仍是各组成指标及其置信区间。

## 评测单个模型

每个 JSONL 记录需包含该任务的真实标签字段和 `output` 字段。选项或故障 ID 从 `\\boxed{...}` 中解析；Task A 使用 `Status:` 和 `Fault_code:`。

```bash
python src/evaluate_all.py \
  --predictions my_predictions \
  --json-output results/my_model_metrics.json
```

与修订稿对比时，Task C 应使用 `data/task_c_core/` 中的记录。

## 复核论文结果

`results/parsed_predictions/` 仅保留 14 个模型的行索引、标准答案和解析后的预测标签，已移除原始设备编号、提示词、自由文本回复、推理文本和 API 元数据。可直接复核单个模型：

```bash
python src/evaluate_all.py \
  --predictions results/parsed_predictions/gemini-2.5-pro
```

论文使用 2,000 次非参数 bootstrap、随机种子 `20260907`和双侧百分位 95% 置信区间：

```bash
python src/evaluate_b1_uncertainty.py \
  --predictions-root results/parsed_predictions

python src/evaluate_task_c_core.py \
  --predictions-root results/parsed_predictions

python src/build_uncertainty_supplement.py
```

汇总结果与机器可读的 S1 表位于 `results/uncertainty/`。

用于图 2--3 的组成指标与描述性综合分可通过下列命令重新生成：

```bash
python src/compute_composites.py
```

## Task A 传统监督基线

Logistic regression、7-nearest-neighbour 和浅层 MLP 使用 8 个已公开数值特征，按脱敏 `elevator_id` 进行确定性五折分组交叉验证，并在每个训练折内使用中位数/IQR 缩放。它们与零样本 LLM 结果分开报告。

```bash
python src/run_task_a_baselines.py
```

## Task C 核心集生成

不提供模型输出时，脚本可生成来源数据层面的唯一集；若提供历史模型输出，还会取所有模型都存在的 C1 共同交集。

```bash
python src/derive_task_c_core_sets.py \
  --output-dir generated_task_c_core

python src/derive_task_c_core_sets.py \
  --archived-outputs-root path/to/model_outputs \
  --output-dir generated_task_c_common_core
```

## 模型与推理设置

`qwen3-8B` 和 `qwen3-next-80b-a3b-instruct` 为本地部署，其他 12 个模型通过 API 访问。零样本推理使用 temperature 0.6、top-p 0.95 和最大 16,384 tokens。历史输出并未为每个 API 模型保留服务商端点和运行日期，因此本仓库不推测缺失的元数据。

## 数据公开与使用边界

Task A 已脱敏公开。由于合作方保密与设备安全限制，本仓库不包含原始日志及原始设备编号与品牌的对应关系。已公开数值字段不具备完整的单位、校准和参考范围元数据，不应视为经验证的物理量。

Task B/C 文件是派生的 Benchmark 实例，不是原始技术手册。后续分发时应确保符合来源材料的许可与协议。

请勿上传原始维修手册、工业日志、API Key、未脱敏标识符或个人信息。Benchmark 分数不能证明模型适合实际维保或安全决策。

## 引用

论文发表后请补充正式 DOI 与引用信息。仓库：<https://github.com/Wwwduojin/elevator-fault-diagnosis-benchmark>

英文说明见 [README.md](README.md)。
