# Project Plan: Tabular Foundation Models

## 1. 项目目标

本项目是“人工智能大模型”期末作业 Project 2，主题是 Tabular Foundation Models。

项目要回答的问题是：

> 表格基础模型 TabPFN v2、TabICL 在真实表格数据上是否好用？它们和传统强模型 LightGBM、XGBoost 相比，在准确率、速度、规模适应能力、类别特征、缺失值鲁棒性上分别有什么优势和短板？

最终交付物：

- 实验代码：可下载数据、运行模型、保存结果、生成图表。
- 中文报告：包含背景、文献综述、实验设计、结果、分析、创新点和结论。
- 答辩 PPT：展示项目动机、方法、结果和贡献。

## 2. 已确定方案

### 2.1 任务类型

主线只做分类任务。

分类任务指预测固定类别，例如“是否违约”“是否患病”“收入是否超过 50K”。分类任务更容易跑通，也更适合 TabPFN v2 和 TabICL 的主线评测。

回归任务指预测连续数值，例如房价、销量。回归任务作为可选扩展，不影响主线完成。

### 2.2 模型

表格基础模型：

- TabPFN v2：主打小中规模表格数据上的强 zero-shot/in-context learning 能力。
- TabICL：主打更大规模数据上的 in-context learning 和 scalability。

传统 baseline：

- LightGBM：表格数据中非常强、速度快的梯度提升树模型。
- XGBoost：经典强 baseline，适合和 LightGBM 一起作为传统方法代表。

备用模型：

- 如果 TabICL 安装或运行失败超过半天，记录失败原因，并考虑 TabDPT 作为备用第二个表格基础模型。
- TabPFN-2.5、TabICLv2 默认用于文献综述和未来工作，不作为最低完成标准。

### 2.3 数据策略

OpenML、TALENT、TabArena 的分工：

- OpenML：实际取数平台，优先从这里下载数据集。
- TALENT：参考其任务分类和 benchmark 设计。
- TabArena：参考其公平比较、速度评估和排行榜思路。

不做全量复现：

- 不全测 OpenML，因为它是大型数据平台，数据集数量非常多。
- 不全测 TALENT，因为它包含大量分类、回归数据集，工作量超过课程项目范围。
- 不全测 TabArena，因为它是完整 benchmark 系统，包含大量 split、模型和实验结果。

本项目采用代表性数据集 + 子采样规模实验。

### 2.4 初始数据集

第一阶段使用 6 个 OpenML 分类数据集：

| 数据集 | 规模 | 特点 | 用途 |
| --- | ---: | --- | --- |
| credit-g | 约 1000 行 | 小数据，数值+类别特征 | 小样本分类 |
| mfeat-factors | 约 2000 行 | 数值特征，多分类 | 多分类能力 |
| kc1 | 约 2100 行 | 软件缺陷，偏不平衡 | 不平衡分类 |
| kr-vs-kp | 约 3196 行 | 类别特征多 | 类别特征处理 |
| adult | 约 48842 行 | 数值+类别特征，可能含缺失 | 较大数据、缺失值 |
| bank-marketing | 约 45211 行 | 数值+类别特征，商业场景 | 较大数据、真实业务 |

如果某个数据集下载、格式或许可出现问题，优先从 OpenML 中替换为相近规模和特征类型的数据集，并记录替换原因。

## 3. 实验设计

### 3.1 主实验

对 6 个数据集分别运行：

- TabPFN v2
- TabICL
- LightGBM
- XGBoost

每次实验记录：

- 数据集名称
- 模型名称
- 训练样本数
- 测试样本数
- 特征数
- 类别特征数
- accuracy
- balanced accuracy
- macro F1
- 推理总耗时
- 平均每条样本推理耗时
- 是否成功运行
- 失败原因

### 3.2 规模实验

在 adult 和 bank-marketing 上做子采样实验：

```text
500 / 1000 / 2000 / 5000 / 10000 / 30000
```

执行规则：

- TabPFN v2 跑到 10000 行以内。
- TabICL、LightGBM、XGBoost 尝试跑到 30000 行。
- 如果模型跑不动，记录失败点、耗时、内存/显存情况和错误信息。

这个实验用于回答 scalability，即数据变大后模型是否还能保持速度和性能。

### 3.3 类别特征实验

重点使用 kr-vs-kp、adult、bank-marketing。

分析问题：

- 类别特征多的数据上，TabPFN v2 和 TabICL 是否稳定？
- LightGBM/XGBoost 是否仍然明显占优？
- 不同编码方式是否影响表格基础模型表现？

### 3.4 缺失值鲁棒性实验

在 adult 和 bank-marketing 上人工加入缺失值：

```text
10% / 30% / 50%
```

分析问题：

- 缺失值增加后，各模型准确率下降多少？
- 哪些模型对缺失值更敏感？
- 传统树模型是否比表格基础模型更稳？

### 3.5 Big Plus 创新实验

创新点固定为 retrieval-based in-context learning。

含义：

- 普通方式：随机选训练样本作为上下文。
- 改进方式：对每个测试样本，先检索最相似的训练样本，再作为上下文。

第一阶段只在 3 个数据集上做：

- credit-g
- adult
- bank-marketing

比较内容：

- 随机上下文 vs 相似样本上下文
- accuracy、balanced accuracy、macro F1
- 推理耗时是否增加

## 4. 项目结构建议

建议后续创建以下结构：

```text
.
├── AGENTS.md
├── docs/
│   ├── literature_notes.md
│   └── project_plan.md
├── scripts/
│   ├── prepare_data.py
│   ├── run_benchmark.py
│   ├── run_missing_value_experiment.py
│   └── analyze_results.py
├── data/
│   └── openml_cache/
├── results/
│   ├── metrics.csv
│   └── figures/
├── report/
└── slides/
```

说明：

- `docs/` 保存策划案和文献笔记。
- `scripts/` 保存实验脚本。
- `data/` 保存下载缓存，不建议提交大数据文件。
- `results/` 保存结果表和图。
- `report/` 保存最终报告。
- `slides/` 保存答辩 PPT。

## 5. 时间计划

按 6 周执行。

第 1 周：环境和最小实验

- 安装 Python 环境和依赖。
- 下载 1 到 2 个 OpenML 小数据集。
- 跑通 TabPFN v2、LightGBM、XGBoost。
- 尝试安装并运行 TabICL。

第 2 周：主实验

- 完成 6 个数据集的主实验。
- 生成第一版 `results/metrics.csv`。
- 记录所有成功和失败情况。

第 3 周：规模实验

- 在 adult 和 bank-marketing 上运行不同样本规模。
- 生成准确率和耗时随样本数变化的图。
- 判断是否需要远端 GPU。

第 4 周：鲁棒性和创新实验

- 完成 10%、30%、50% 缺失值实验。
- 完成 retrieval-based in-context learning 小规模验证。

第 5 周：报告

- 写文献综述。
- 整理实验表格和图。
- 写模型优缺点分析。
- 写创新实验分析。

第 6 周：PPT 和复核

- 完成答辩 PPT。
- 复跑关键实验确认结果。
- 准备答辩问题，例如为什么不全测 TALENT/TabArena、为什么选这些数据集、为什么选 retrieval 改进。

## 6. 团队分工

成员 A：

- TabPFN v2
- LightGBM
- XGBoost
- OpenML 数据准备
- 主实验结果表

成员 B：

- TabICL
- 规模实验
- 缺失值实验
- retrieval-based in-context learning

共同完成：

- 报告结论
- 图表解释
- PPT
- 答辩准备

## 7. 验收标准

项目完成时至少满足：

- 成功运行 2 个表格基础模型：TabPFN v2、TabICL。
- 成功运行 2 个传统 baseline：LightGBM、XGBoost。
- 至少 6 个数据集有主实验结果。
- 至少包含准确率、balanced accuracy、macro F1、推理速度。
- 至少完成 1 个规模实验。
- 至少完成 1 个缺失值鲁棒性实验。
- 至少完成 1 个 Big Plus 创新实验。
- 报告解释 OpenML、TALENT、TabArena 的关系，以及为什么不全量复现。
