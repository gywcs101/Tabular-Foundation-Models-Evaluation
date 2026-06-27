# results/final 表格说明

这个文件只用于组内查看 `results/final/` 目录下的 CSV 表格，不是项目总览 README。

当前 final 表格已经更新到 5 份实验结果：

- 2 份轻薄本结果
- 3 份游戏本结果
- 4 个模型
- 12 个选定数据集
- `final_metrics.csv` 中共 240 条最终实验记录

说明：如果某个实验被重复跑了，分析脚本会保留“最新一次成功运行”的结果。因此最终汇总表不会因为重复运行而重复计数。

## 1. input_completeness_report.csv

作用：

- 检查每个人是否跑完了所有计划实验。
- 用来确认实验结果是否完整。

怎么看：

- 每一行代表一个计划中的实验：某个人、某个模型、某个数据集。
- `status = complete` 表示该实验正常完成。
- `status = duplicate_success` 表示这个实验成功跑了不止一次，最终分析会取最新一次成功结果。
- 如果出现 `missing`、`no_success`、`incomplete_files`，才是真正需要处理的问题。

重要列：

- `runner_dir`：结果目录名，也就是谁跑的。
- `model`：模型名。
- `experiment_axis`：实验线，`sample_scale` 表示样本量实验，`feature_scale` 表示特征量实验。
- `scale_group`：数据集所属规模组。
- `dataset`：数据集名称。
- `run_count`：发现了多少次运行。
- `success_count`：成功运行次数。
- `latest_success_run`：最终采用的成功运行。
- `missing_files`：缺少哪些结果文件。
- `status`：完整性状态。

怎么用于报告：

- 可以用它说明：所有计划实验都已经完成。
- `duplicate_success` 只需要作为数据清理说明，不代表结果缺失。

## 2. final_metrics.csv

作用：

- 最重要的总表。
- 后续大部分统计、画图、结论都来自这个表。

怎么看：

- 每一行是一条最终采用的成功实验结果。
- 当前应该有 240 行：

```text
5 人 × 4 模型 × 12 数据集 = 240 行
```

重要列：

- 身份信息：
  - `runner_dir`：结果来自谁。
  - `runner_name_from_dir`：从目录名解析出来的人名。
  - `device_type`：设备类型，`light_laptop` 或 `gaming_laptop`。
  - `model`：模型。
  - `experiment_axis`：样本量实验或特征量实验。
  - `scale_group`：规模组。
  - `dataset`：数据集。
  - `run_id`：具体运行编号。
- 准确率相关：
  - `accuracy`
  - `balanced_accuracy`
  - `macro_f1`
- 置信度相关：
  - `mean_confidence`
  - `correct_mean_confidence`
  - `wrong_mean_confidence`
  - `confidence_gap`
- 时间和内存相关：
  - `fit_time_seconds`
  - `predict_time_seconds`
  - `wall_time_seconds`
  - `seconds_per_test_sample`
  - `peak_memory_mb`
  - `training_throughput`
- 数据集信息：
  - `fit_rows`
  - `test_rows`
  - `n_features`
  - `n_classes`

怎么用于报告：

- 查单个模型、单个数据集、单个人的具体结果时看这个表。
- 准确率可以跨设备汇总，因为同一模型同一数据集的预测结果理论上不应明显受设备影响。
- 时间和内存不要随便把轻薄本和游戏本混在一起平均；如果讨论硬件差异，要按 `device_type` 分开看。

## 3. model_dataset_summary.csv

作用：

- 按“模型 + 数据集”聚合后的表。
- 适合看每个模型在每个数据集上的平均表现。

怎么看：

- 每一行是一个 `(model, dataset)` 组合。
- 当前有 4 个模型、12 个数据集，所以应该有 48 行。
- 列名里带 `_mean`、`_std`、`_min`、`_max`、`_count` 的，分别表示平均值、标准差、最小值、最大值、计数。

常用列：

- `accuracy_mean`
- `macro_f1_mean`
- `predict_time_seconds_mean`
- `wall_time_seconds_mean`
- `peak_memory_mb_mean`

怎么用于报告：

- 适合写“哪个模型整体效果更好”。
- 适合支撑按数据集对比的图：
  - `performance_accuracy_by_dataset`
  - `performance_macro_f1_by_dataset`

## 4. device_type_summary.csv

作用：

- 按设备类型聚合时间和内存指标。
- 主要用于比较轻薄本和游戏本。

怎么看：

- 每一行是一个 `(device_type, model, dataset)` 组合。
- `device_type` 通常是：
  - `light_laptop`
  - `gaming_laptop`
- 带 `_mean`、`_std`、`_min`、`_max`、`_count` 的列表示该设备组内的统计结果。

常用列：

- `fit_time_seconds_mean`
- `predict_time_seconds_mean`
- `wall_time_seconds_mean`
- `seconds_per_test_sample_mean`
- `peak_memory_mb_mean`
- `training_throughput_mean`

怎么用于报告：

- 适合写硬件对时间、内存的影响。
- 时间图和内存图主要参考这个表。
- 不建议用这个表讨论模型准确率，因为准确率不应该主要由硬件决定。

## 5. device_consistency.csv

作用：

- 检查不同人、不同设备之间结果波动大不大。
- 用来判断结果是否稳定。

怎么看：

- 每一行是某种分组下的某个指标波动情况。
- 重点看标准差和相对波动。

怎么用于报告：

- 可以作为“实验可靠性”的补充证据。
- 不一定放正文主表，更适合写方法严谨性或附录。

## 6. scalability_alpha.csv

作用：

- 保存双对数拟合得到的扩展性斜率 `alpha`。
- 拟合公式是：

```text
log(T) = alpha * log(N) + C
```

其中：

- `T` 是时间或内存消耗。
- `N` 是样本数或特征数。

怎么看：

- `experiment_axis`：
  - `sample_scale`：样本量实验。
  - `feature_scale`：特征量实验。
- `device_type`：轻薄本或游戏本。
- `model`：模型名。
- `metric`：被拟合的指标，比如 `predict_time_seconds`、`wall_time_seconds`、`peak_memory_mb`。
- `x_metric`：横轴变量，比如 `fit_rows` 或 `n_features`。
- `alpha`：经验增长斜率。
- `n_points`：拟合用了几个点。

解释方式：

- `alpha` 越大，说明随着样本数或特征数增加，该指标增长越快。
- 这个 `alpha` 是经验趋势，不是严格的算法复杂度证明。
- 时间 `alpha` 可以用于正文解释扩展性。
- 内存 `alpha` 不建议作为正文重点，峰值内存柱状图更直观。

怎么用于报告：

- 样本量实验中，可以展示时间 `alpha`。
- 特征量实验中，时间 `alpha` 可以作为补充。
- 内存 `alpha` 更适合放附录或简单提及。

## 7. multiclass_confusion_top_errors.csv

作用：

- 多分类数据集的混淆矩阵错误统计。
- 用来找“哪个真实类别经常被预测成哪个类别”。

怎么看：

- 每一行是一个非对角线错误，也就是真实标签和预测标签不一样。
- `mean_count` 表示平均有多少测试样本出现这种混淆。
- `run_count` 表示有多少次运行参与统计。

重要列：

- `dataset`
- `model`
- `true_label`
- `predicted_label`
- `mean_count`
- `std_count`
- `min_count`
- `max_count`
- `run_count`

怎么用于报告：

- 写混淆矩阵分析时用这个表找具体结论。
- 不要只写“对角线很明显”，那和 accuracy 重复。
- 应该写具体错误，比如某些数字经常互相混淆。
- 如果多个模型都混淆同一对数字，更可能是数据特征表达不足或类别本身相似，不一定是某一个模型的问题。

## 8. multiclass_confusion_top5_by_model.csv

作用：

- `multiclass_confusion_top_errors.csv` 的精简版。
- 每个 `(dataset, model)` 只保留前 5 个最明显的混淆错误。

怎么看：

- 列和 `multiclass_confusion_top_errors.csv` 基本一样。
- 更适合人工快速查看。

怎么用于报告：

- 写报告或做 PPT 时，优先看这个表。
- 用它决定正文要放哪几张混淆矩阵图。
- 如果某个数据集的前几大错误很集中，就适合拿出来讲。

## 建议查看顺序

如果只是检查实验是否完整：

1. `input_completeness_report.csv`
2. `final_metrics.csv`

如果要写正文结果分析：

1. `model_dataset_summary.csv`
2. `device_type_summary.csv`
3. `scalability_alpha.csv`

如果要写错误分析：

1. `multiclass_confusion_top5_by_model.csv`
2. `multiclass_confusion_top_errors.csv`

如果要做更细的核对：

1. `device_consistency.csv`
2. `final_metrics.csv`

