# 创新点计划书：Retrieval-based Context Selection

本文档是一个待审核的 Big Plus 创新点计划。它暂时不替代主实验，只作为在主 benchmark 之外增加的一组轻量实验。

## 1. 创新点一句话概括

在 TabICL / TabPFN 这类表格基础模型中，模型不是像传统模型那样从零训练参数，而是通过上下文样本进行 in-context learning。我们提出一个轻量改进：

> 对每个测试样本，不直接使用全部 `train + val`，而是从 `train + val` 中检索出与该测试样本最相似的 K 个样本作为上下文。

核心问题是：

> 对表格基础模型来说，上下文样本的“相关性”是否比“数量”更重要？

## 2. 为什么这个算创新点

这个实验不是单纯多跑一个模型，而是在改变表格基础模型的使用方式。

原始方式：

- 使用完整 `train + val` 作为上下文或拟合数据。
- 好处是样本数量多，模型能看到完整数据分布。
- 缺点是上下文里可能包含很多与当前测试样本关系不大的样本。

Retrieval 方式：

- 对每个测试样本单独检索最相似的 K 个训练样本。
- 好处是上下文更贴近当前测试样本。
- 缺点是上下文数量减少，可能损失全局分布信息。

因此这个实验能回答一个有研究意义的问题：

- 如果 Retrieval-K 优于 Random-K，说明“选更相关的上下文”确实有帮助。
- 如果 Retrieval-K 甚至优于 Full context，说明局部相似上下文可能比完整上下文更有效。
- 如果 Retrieval-K 不如 Full context，说明完整数据分布对模型更重要。
- 如果 Retrieval-K 和 Random-K 接近，说明简单最近邻检索没有带来额外价值。

无论结果正负，都可以写成我们对 in-context learning 机制的观察。

## 3. 推荐实验范围

为了不拖慢当前报告进度，创新点实验应控制在小范围内。

推荐最小版本：

- 模型：TabICL
- 数据集：1 个
- 方法：Full context、Random-K、Retrieval-K
- K 值：256

推荐完整但仍然可控的版本：

- 模型：TabICL
- 数据集：2 个
- 方法：Full context、Random-K、Retrieval-K
- K 值：128、256、512

暂不推荐：

- 对 12 个数据集全跑。
- 同时对 4 个模型全跑。
- 进行模型 fine-tuning。
- 加入复杂神经检索器。

原因：

- 当前主实验已经比较完整，加分项应该是“补充亮点”，不能冲击主线。
- Retrieval 每个测试样本都要单独选上下文，运行时间可能明显增加。
- TabICL 与 in-context learning 关系最直接，最适合作为第一版创新实验对象。

## 4. 数据集选择

推荐数据集 1：

```text
mfeat-morphological_2000rows_6feat_multiclass
```

选择原因：

- 是多分类任务，比二分类更容易观察错误类型变化。
- 特征数只有 6，属于低维特征表示。
- 当前混淆矩阵中存在明显类别混淆，例如某些数字类别容易互相混淆。
- 如果 retrieval 有用，可能改善局部相似类别的判断。
- 数据量不大，适合快速验证。

推荐数据集 2：

```text
pc1_1109rows_21feat_binclass
```

选择原因：

- 是小样本二分类任务。
- 数据量小，运行成本低。
- 可以测试 retrieval 在二分类、小样本任务中是否也有价值。
- 与 mfeat-morphological 形成对照：一个多分类，一个二分类。

如果只能选一个：

- 优先选 `mfeat-morphological_2000rows_6feat_multiclass`。

原因：

- 更容易在报告中解释“错误类型是否改变”。
- 与混淆矩阵分析联系更强。

## 5. 模型选择

推荐只做 TabICL。

原因：

- TabICL 的核心就是 tabular in-context learning。
- Retrieval-based context selection 和 TabICL 的机制最贴合。
- TabPFN 也可以尝试，但 CPU 上可能更慢，而且实现上不一定能方便地对每个测试样本单独换上下文。
- LightGBM 和 XGBoost 是传统训练模型，不适合作为 retrieval-context 改进对象。

报告表述：

> We apply the retrieval-based context selection only to TabICL because the method specifically targets in-context learning, not gradient-boosted tree training.

## 6. 对照组设计

必须至少包含三组：

### 6.1 Full context

含义：

- 使用完整 `train + val`。
- 这是当前主实验中的 TabICL 原始设置。

作用：

- 作为最重要的基线。
- 用来判断 retrieval 是否能超过原始完整上下文。

优点：

- 上下文数量最多。
- 保留完整数据分布。

缺点：

- 包含很多可能和当前测试样本无关的样本。
- 计算成本更高。

### 6.2 Random-K context

含义：

- 从 `train + val` 中随机选 K 个样本作为上下文。

作用：

- 控制“上下文数量变少”这一因素。
- 如果只比较 Full vs Retrieval，无法判断结果变化到底来自“样本变少”还是“检索更相关”。

优点：

- 实现简单。
- 是 Retrieval-K 的公平数量对照。

缺点：

- 随机结果有波动。
- 最好固定随机种子，必要时重复 3 次。

### 6.3 Retrieval-K context

含义：

- 对每个测试样本，从 `train + val` 中选出最相似的 K 个样本。

作用：

- 检验相似样本上下文是否优于随机上下文。

优点：

- 上下文更相关。
- 和 in-context learning 的机制高度相关。

缺点：

- 可能损失全局类别分布。
- 如果 K 太小，可能漏掉部分类别。
- 每个测试样本单独检索，运行时间可能增加。

## 7. K 值设置

推荐完整实验：

```text
K = 128, 256, 512
```

推荐最小实验：

```text
K = 256
```

设置原因：

- `128`：较小上下文，能测试“高相关但少样本”的极端情况。
- `256`：折中设置，通常足够覆盖一部分局部结构。
- `512`：较大上下文，更接近完整上下文，但仍显著少于 full context。

对于 `mfeat-morphological`：

- `train + val` 大约 1600 条。
- `K=256` 是 full context 的一小部分，能明显测试“少而精”的效果。
- `K=512` 仍比 full context 少很多，但类别覆盖更稳。

对于 `pc1`：

- 样本更少。
- 如果完整上下文不足 512，则 `K` 应自动取 `min(K, len(train_val))`。

## 8. 相似度计算方式

推荐第一版使用标准化后的欧氏距离。

步骤：

1. 对数值特征做标准化：

```text
z = (x - mean) / std
```

2. 对每个测试样本，计算它与所有 `train + val` 样本的欧氏距离。
3. 选择距离最近的 K 个样本作为上下文。

原因：

- 简单、透明、容易解释。
- 不需要训练额外模型。
- 适合当前小规模加分实验。

如果有类别特征：

- 当前推荐数据集主要是数值特征，可以先不处理复杂类别距离。
- 如果后续扩展到混合特征数据集，再考虑 one-hot 或 Gower distance。

不推荐第一版使用：

- 神经网络 embedding 检索。
- 复杂度量学习。
- FAISS 等向量库。

原因：

- 实现成本高。
- 报告解释复杂。
- 当前数据规模不需要。

## 9. 实验流程

对每个数据集执行：

1. 读取 TALENT 数据。
2. 合并 `train + val` 作为候选上下文池。
3. 保持原始 `test` 不变。
4. 对 TabICL 跑 Full context：
   - 使用完整 `train + val`。
   - 记录结果。
5. 对 TabICL 跑 Random-K：
   - 从 `train + val` 随机选 K 个样本。
   - 建议固定随机种子 `42`。
   - 如果时间允许，随机重复 3 次，取平均。
6. 对 TabICL 跑 Retrieval-K：
   - 对每个测试样本检索 K 个最相似训练样本。
   - 用这些样本作为上下文进行预测。
7. 保存结果。
8. 和主实验 TabICL 结果进行对比。

## 10. 结果保存建议

建议单独保存到：

```text
results/innovation/retrieval_context/
```

推荐文件：

```text
results/innovation/retrieval_context/metrics.csv
results/innovation/retrieval_context/predictions.csv
results/innovation/retrieval_context/run_config.json
results/innovation/retrieval_context/run.log
```

`metrics.csv` 建议包含：

- `dataset`
- `model`
- `context_strategy`
- `k`
- `random_seed`
- `accuracy`
- `macro_f1`
- `mean_confidence`
- `correct_mean_confidence`
- `wrong_mean_confidence`
- `confidence_gap`
- `fit_time_seconds`
- `predict_time_seconds`
- `wall_time_seconds`
- `seconds_per_test_sample`
- `peak_memory_mb`
- `success`
- `error_message`

`context_strategy` 建议取值：

- `full`
- `random_k`
- `retrieval_k`

## 11. 评价指标

主要指标：

- `accuracy`
- `macro_f1`
- `predict_time_seconds`
- `wall_time_seconds`

辅助指标：

- `mean_confidence`
- `correct_mean_confidence`
- `wrong_mean_confidence`
- `confidence_gap`
- `peak_memory_mb`

如果只做最小实验，至少保留：

- `accuracy`
- `macro_f1`
- `wall_time_seconds`

## 12. 结果解释方式

### 情况 A：Retrieval-K > Full context

说明：

- 相似样本上下文比完整上下文更有效。
- 上下文质量可能比上下文数量更重要。

报告写法：

> Retrieval-based context selection improves TabICL on the selected dataset, suggesting that local relevance of context examples can be more useful than simply using all available context.

### 情况 B：Retrieval-K > Random-K，但 Retrieval-K < Full context

说明：

- 检索确实比随机选择更好。
- 但完整上下文仍然提供了重要的全局分布信息。

报告写法：

> Retrieval improves over random context selection, but full context remains stronger, indicating a trade-off between context relevance and context coverage.

### 情况 C：Retrieval-K ≈ Random-K

说明：

- 简单欧氏距离检索没有明显帮助。
- 可能需要更好的相似度度量。

报告写法：

> Simple nearest-neighbor retrieval does not outperform random context selection, suggesting that naive distance-based retrieval may be insufficient for tabular in-context learning.

### 情况 D：Retrieval-K < Random-K

说明：

- 检索可能引入局部偏差。
- 最近邻样本未必是最有代表性的上下文。

报告写法：

> Retrieval can even hurt performance when the selected local neighborhood is biased, showing that context selection must balance relevance and class coverage.

## 13. 报告中的小节设计

建议放在 Evaluation 后面，或者作为 Big Plus 小节：

```text
6. Lightweight Improvement: Retrieval-based Context Selection
```

小节结构：

1. Motivation
   - TabICL depends on context examples.
   - Context selection may affect prediction.
2. Method
   - Full context
   - Random-K
   - Retrieval-K
3. Experiment Setup
   - Dataset
   - K values
   - Similarity metric
4. Results
   - 表格对比 accuracy / macro F1 / time
5. Discussion
   - 是否提升
   - 为什么提升或没提升
   - 局限性

## 14. PPT 中的展示方式

如果做出来，建议只放 1 页：

标题：

```text
Big Plus: Retrieval-based Context Selection
```

内容：

- 左侧：三种上下文策略示意图。
- 右侧：小表格展示 Full / Random-K / Retrieval-K 的 accuracy、macro F1、wall time。
- 底部：一句结论。

不要在 PPT 中放太多技术细节。

## 15. 预期工作量

最小版本：

- 1 个数据集
- 1 个 K 值
- 3 种策略
- 预计半天内可以完成脚本和初步结果

完整小版本：

- 2 个数据集
- 3 个 K 值
- Random-K 重复 3 次
- 预计 1 天左右

## 16. 风险和处理

风险 1：运行太慢。

处理：

- 只做 `mfeat-morphological`。
- 只做 `K=256`。
- 只做 TabICL。

风险 2：TabICL 接口不支持每个测试样本单独上下文。

处理：

- 写成循环：每次用一个测试样本对应的 K 个上下文拟合并预测。
- 如果全测试集太慢，可只抽取固定测试子集，例如 100 个测试样本。
- 报告中说明这是 pilot experiment。

风险 3：结果没有提升。

处理：

- 不把它写成“我们提出了更强方法”。
- 写成“我们发现简单 retrieval 不一定足够，未来需要考虑类别覆盖和更好的相似度度量”。

风险 4：只选最近邻导致类别不均衡。

处理：

- 在讨论中指出：未来可以尝试 class-balanced retrieval。
- 当前实验只作为轻量初探。

## 17. 推荐最终执行版本

如果时间允许，建议采用：

- 模型：TabICL
- 数据集：
  - `mfeat-morphological_2000rows_6feat_multiclass`
  - `pc1_1109rows_21feat_binclass`
- 策略：
  - Full context
  - Random-256
  - Retrieval-256
- 可选：
  - Random-256 重复 3 次

如果时间紧，采用最小版本：

- 模型：TabICL
- 数据集：
  - `mfeat-morphological_2000rows_6feat_multiclass`
- 策略：
  - Full context
  - Random-256
  - Retrieval-256

这个最小版本已经足够作为 Big Plus 的轻量探索。

