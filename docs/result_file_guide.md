# 实验结果文件说明

更新时间：2026-06-09

本文档说明正式实验结果应该怎么看。新口径下，单次实验能计算出的数字必须在运行时直接写入结果文件；后处理只负责合并结果、计算跨规模的 `alpha`、画图。

## 1. 结果目录

单次实验结果建议保存到：

```text
results/raw/{runner_name}/{model}/{experiment_axis}/{scale_group}/{dataset_name}/{run_id}/
```

示例：

```text
results/raw/gywcs101/tabicl/feature_scale/F1_6_20/mfeat-morphological_2000rows_6feat_multiclass/run_20260609_201500/
```

这样安排的好处是：

- `raw` 表示这是原始单次实验结果，不是最终汇总表。
- `runner_name` 表示是谁跑的这一批结果。
- 按模型分文件夹，方便两名成员分工跑实验。
- 按实验轴和规模组分文件夹，方便后续画样本量/特征量扩展性图。
- 每次运行有独立 `run_id`，不会覆盖之前结果。

## 2. 单次实验文件

每个运行目录应包含：

```text
metrics.csv
run_config.json
predictions.csv
confusion_matrix.csv
run.log
```

查看顺序建议：

1. `metrics.csv`：先看这次实验是否成功，以及所有核心数字。
2. `run.log`：如果失败或卡住，用它排查流程。
3. `run_config.json`：确认模型、数据、路径、随机种子和包版本。
4. `confusion_matrix.csv`：看类别混淆。
5. `predictions.csv`：做置信度直方图、错例分析和更细的错误检查。

## 3. `metrics.csv`

`metrics.csv` 是单次实验最重要的文件，通常只有一行。它必须直接保存最终数字，而不是要求后面再手算。

核心字段：

- `success`：是否成功完成。
- `error_type` / `error_message`：失败原因。
- `dataset`：数据集目录名。
- `experiment_axis`：实验轴，例如 `sample_scale` 或 `feature_scale`。
- `scale_group`：规模组，例如 `F1_6_20`。
- `model`：模型名。
- `fit_rows` / `test_rows` / `total_rows`：拟合、测试、总样本数。
- `n_features` / `n_num_features` / `n_cat_features`：特征数量。
- `n_classes`：类别数量。
- `accuracy`：整体准确率。
- `balanced_accuracy`：平衡准确率。
- `macro_f1`：宏平均 F1。
- `mean_confidence`：平均置信度。
- `correct_mean_confidence`：预测正确样本的平均置信度。
- `wrong_mean_confidence`：预测错误样本的平均置信度。
- `confidence_gap`：正确置信度均值减错误置信度均值。
- `fit_time_seconds`：拟合/上下文构建耗时。
- `predict_time_seconds`：测试集 `predict` 耗时。
- `wall_time_seconds`：整次运行总耗时。
- `seconds_per_test_sample`：每条测试样本平均预测耗时。
- `peak_memory_mb`：实验过程中的峰值内存。
- `training_throughput`：`fit_rows * n_features / fit_time_seconds`。
- `probabilities_available`：模型是否成功输出类别概率。

不再使用：

- `memory_before_mb`
- `memory_after_mb`

这两个字段只能说明运行前后两个时间点的内存，不代表峰值内存，容易误导扩展性分析。

## 4. `run_config.json`

`run_config.json` 保存复现信息，不作为主统计表。

它应该包含：

- 数据集原始 `info.json`
- 数据路径
- 模型名称
- 本地权重路径
- 是否使用 `train + val`
- 随机种子
- 编码方式
- 类别标签列表
- Python 版本
- 关键包版本
- 峰值内存采样间隔

写报告时，一般不用展示它；排查“为什么两个人跑出来不一样”时很有用。

## 5. `predictions.csv`

`predictions.csv` 每一行对应测试集中的一个样本。

必须包含：

- `sample_index`
- `y_true`
- `y_pred`
- `correct`
- `confidence`
- `prob_class_*`

其中：

- `confidence` 是模型对最终预测类别的概率。
- `prob_class_*` 是每个类别的概率列。

这个文件用于：

- 置信度直方图
- 正确/错误置信度对比
- 错例分析
- 查找高置信度错误样本

如果模型没有概率输出，则 `confidence` 写为 `NaN`，并在 `metrics.csv` 中记录 `probabilities_available=False`。

## 6. `confusion_matrix.csv`

混淆矩阵用于分析模型倾向。

读法：

- 行是真实类别。
- 列是预测类别。
- 数值表示真实类别被预测成某一类别的次数。

后续可以用它画混淆矩阵热力图。正文不需要放所有数据集的热力图，挑代表性结果即可。

## 7. `run.log`

`run.log` 只用于人工排错，不再作为正式统计指标来源。

它可以记录：

- 读数据
- 预处理
- 创建模型
- 拟合
- 预测
- 概率输出
- 保存结果
- 异常信息

正式统计以 `metrics.csv` 为准。比如 `wall_time_seconds` 必须直接写在 `metrics.csv`，不能要求后面从日志时间戳手算。

## 8. 后续汇总文件

完整实验结束后，再由合并脚本生成：

```text
results/final/final_metrics.csv
results/final/scalability_alpha.csv
```

含义：

- `final_metrics.csv`：合并所有单次实验的 `metrics.csv`。
- `scalability_alpha.csv`：基于多次实验拟合出的增长速率阶数 `alpha`。

图表统一保存到：

```text
results/figures/
```

## 9. 当前状态

旧实验结果已经清空。后续正式开跑前，实验脚本还需要按 `docs/measurement_plan.md` 调整结果字段和保存路径。
