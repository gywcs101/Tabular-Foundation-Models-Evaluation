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

- TALENT：当前已下载到本地，作为第一阶段主要可运行数据来源。
- TabArena：当前已下载源码仓库和元数据，作为数据集筛选和 benchmark 设计参考；不下载完整大 artifact。
- OpenML：作为补充取数平台，当需要复现 TabArena 中的具体任务或补充缺失数据时再使用。

TALENT 的具体筛选规则见：

```text
docs/dataset_selection_criteria.md
```

不做全量复现：

- 不全测 OpenML，因为它是大型数据平台，数据集数量非常多。
- 不全测 TALENT，因为它包含大量分类、回归数据集，工作量超过课程项目范围。
- 不全测 TabArena，因为它是完整 benchmark 系统，包含大量 split、模型和实验结果。

本项目采用控制变量的数据选择策略。当前不全量复现 TALENT 或 TabArena，而是把扩展性拆成两条实验线：

- 样本数扩展性：控制特征数量尽量接近，只改变样本数。
- 特征数扩展性：控制样本数量尽量接近，只改变特征数。

### 2.4 初始数据集选择

当前优先使用 TALENT 分类数据集，不再把第一版 OpenML 数据集作为主实验默认清单。

样本数扩展性候选：

| 数据集 | 样本数 | 特征数 | 任务类型 | 用途 |
| --- | ---: | ---: | --- | --- |
| pc1 | 1109 | 21 | 二分类 | A 组，小样本 |
| kc1 | 2109 | 21 | 二分类 | B 组，中小样本 |
| jm1 | 10885 | 21 | 二分类 | C 组，中等规模 |
| default_of_credit_card_clients | 30000 | 23 | 二分类 | C 组上界 |

特征数扩展性候选：

| 数据集 | 样本数 | 特征数 | 任务类型 | 用途 |
| --- | ---: | ---: | --- | --- |
| mfeat-morphological | 2000 | 6 | 多分类 | 低维特征 |
| mfeat-zernike | 2000 | 47 | 多分类 | 中维特征 |
| mfeat-karhunen | 2000 | 64 | 多分类 | 中高维特征 |
| mfeat-fourier | 2000 | 76 | 多分类 | 中高维特征 |
| mfeat-factors | 2000 | 216 | 多分类 | 高维特征 |
| mfeat-pixel | 2000 | 240 | 多分类 | 高维特征 |

选择依据：

- 样本数扩展性实验中，特征数尽量控制在 20 到 30 附近。
- 特征数扩展性实验中，样本数尽量控制在 2000 附近。
- 所有模型使用同一训练集和同一测试集。
- 如果模型超时、内存不足或超出模型限制，记录为扩展性结果。

## 3. 实验设计

### 3.1 主实验

对样本数扩展性和特征数扩展性候选数据集分别运行：

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
- mean confidence / correct mean confidence / wrong mean confidence / confidence gap
- 拟合耗时
- 推理耗时
- 平均每条样本推理耗时
- 墙钟总时间
- 峰值内存消耗
- 训练吞吐量
- 是否成功运行
- 失败原因

后处理统一生成：

- `accuracy / macro_f1` 折线图
- `time` 柱状图
- `log(T)=αlog(N)+C` 的增长速率拟合图
- `Training Throughput` 对比图
- 置信度直方图和正确/错误置信度对比图
- 混淆矩阵热力图

### 3.2 样本数扩展性实验

使用 `pc1`、`kc1`、`jm1`、`default_of_credit_card_clients` 等候选，优先保持特征数接近：

```text
1000 / 3000 / 10000 / 30000
```

执行规则：

- 四个模型使用相同训练集和测试集。
- 不因为某个模型较慢就单独换更小数据。
- 如果模型跑不动，记录失败点、耗时、内存/显存情况和错误信息。

这个实验用于回答样本数变大后模型是否还能保持速度和性能。后续会用 `log(T)=αlog(N)+C` 拟合样本数增长速率，并用训练吞吐量辅助比较模型效率。

### 3.3 特征数扩展性实验

使用 `mfeat` 系列，优先保持样本数约 2000 行，只改变特征数：

```text
6 / 47 / 64 / 76 / 216 / 240 个特征
```

分析问题：

- 特征数增加后，各模型推理时间如何变化？
- 高维特征是否让某些模型更容易失败或变慢？
- 准确率提升来自更多特征，还是被高维噪声影响？

### 3.4 类别特征实验

从最终 TALENT 候选中选择类别特征较多的数据集。如果当前主候选类别特征不足，则作为补充实验单独增加一个类别特征数据集。

分析问题：

- 类别特征多的数据上，TabPFN v2 和 TabICL 是否稳定？
- LightGBM/XGBoost 是否仍然明显占优？
- 不同编码方式是否影响表格基础模型表现？

### 3.5 缺失值鲁棒性实验

在 1 到 2 个主实验候选数据集上人工加入缺失值：

```text
10% / 30% / 50%
```

分析问题：

- 缺失值增加后，各模型准确率下降多少？
- 哪些模型对缺失值更敏感？
- 传统树模型是否比表格基础模型更稳？

### 3.6 Big Plus 创新实验

创新点固定为 retrieval-based in-context learning。

含义：

- 普通方式：随机选训练样本作为上下文。
- 改进方式：对每个测试样本，先检索最相似的训练样本，再作为上下文。

第一阶段只在 2 到 3 个已选 TALENT 候选数据集上做，优先选择：

- 一个 A/B 组小中样本数据集。
- 一个 C 组中等规模数据集。
- 一个 mfeat 特征数实验数据集。

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
│   ├── raw/
│   ├── final/
│   └── figures/
├── report/
└── slides/
```

说明：

- `docs/` 保存策划案和文献笔记。
- `scripts/` 保存实验脚本。
- `data/` 保存下载缓存，不建议提交大数据文件。
- `results/raw/` 保存单次实验目录，每次运行包含 `metrics.csv`、`run_config.json`、`predictions.csv`、`confusion_matrix.csv`、`run.log`。
- `results/final/` 保存合并后的总表和跨实验统计，例如 `final_metrics.csv`、`scalability_alpha.csv`。
- `results/figures/` 保存最终图表。
- `report/` 保存最终报告。
- `slides/` 保存答辩 PPT。

## 5. 时间计划

按 6 周执行。

第 1 周：环境、仓库和资源核验

- GitHub 仓库整理和同步。
- 检查本地 TabPFN、TabICL、LightGBM、XGBoost 独立环境。
- 检查 TALENT 数据包和 TabArena 元数据。
- 跑通 `scripts/check_local_assets.py`。

第 2 周：主实验

- 完成样本数扩展性和特征数扩展性候选数据集的主实验。
- 生成第一版 `results/raw/{model}/.../metrics.csv`，并合并为 `results/final/final_metrics.csv`。
- 记录所有成功和失败情况。

第 3 周：扩展性实验

- 完成样本数扩展性图表。
- 完成特征数扩展性图表。
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
- TALENT 数据准备
- 主实验结果表

成员 B：

- TabICL
- 样本数扩展性和特征数扩展性实验
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
- 至少 6 个 TALENT 数据集有主实验结果。
- 至少包含准确率、balanced accuracy、macro F1、推理速度。
- 至少完成 1 个样本数扩展性实验。
- 至少完成 1 个特征数扩展性实验。
- 至少完成 1 个缺失值鲁棒性实验。
- 至少完成 1 个 Big Plus 创新实验。
- 报告解释 OpenML、TALENT、TabArena 的关系，以及为什么不全量复现。
