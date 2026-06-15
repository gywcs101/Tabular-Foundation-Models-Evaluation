# 实验测量数据计划

更新时间：2026-06-09

本文档定义本项目正式实验必须记录的指标、结果文件结构和后续绘图口径。当前原则是：**跑实验时必须把单次实验能计算出的数字全部算出来并写入结果文件；只有图表和跨多次实验才能计算的聚合指标放到后处理阶段。**

## 1. 核心原则

本项目要比较 TabICL、TabPFN、XGBoost、LightGBM 在同一批 TALENT 分类数据集上的表现。结果文件必须同时支持：

- 预测效果比较
- 置信度分析
- 运行时间和峰值内存比较
- 样本数扩展性分析
- 特征数扩展性分析
- 失败原因记录

因此，后续脚本不能只保存原始预测再让人手动算指标。除图表外，所有单次实验指标都应在实验结束前直接计算并保存。

## 2. 单次实验必须保存的数字

### 2.1 正确率相关

必须保存：

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

含义：

- `accuracy`：整体预测正确比例，最直观。
- `balanced_accuracy`：先分别计算每个类别的正确率，再平均，能降低类别不均衡的影响。
- `macro_f1`：每个类别分别计算 F1 后再平均，适合多分类任务中观察各类别综合表现。

### 2.2 置信度相关

必须保存：

- `mean_confidence`
- `correct_mean_confidence`
- `wrong_mean_confidence`
- `confidence_gap`

定义：

- `confidence`：模型对自己最终预测类别给出的概率。
- `mean_confidence`：所有测试样本的平均置信度。
- `correct_mean_confidence`：预测正确样本的平均置信度。
- `wrong_mean_confidence`：预测错误样本的平均置信度。
- `confidence_gap = correct_mean_confidence - wrong_mean_confidence`

解释：

- 正确时置信度高是好事。
- 错误时置信度高是坏事，说明模型错得很自信。
- `confidence_gap` 越大，通常说明模型更能区分“自己做对”和“自己做错”的情况。

如果某个模型不能输出概率，则这些字段写为 `NaN`，同时在 `metrics.csv` 中记录 `probabilities_available=False`。

### 2.3 时间和内存相关

必须保存：

- `fit_time_seconds`
- `predict_time_seconds`
- `wall_time_seconds`
- `seconds_per_test_sample`
- `peak_memory_mb`
- `training_throughput`

定义：

- `fit_time_seconds`：模型拟合、训练或上下文构建耗时。
- `predict_time_seconds`：模型对测试集执行 `predict` 的耗时。
- `wall_time_seconds`：从实验开始到结果文件写完的总等待时间。
- `seconds_per_test_sample = predict_time_seconds / test_rows`
- `peak_memory_mb`：实验过程中当前 Python 进程 RSS 的峰值，单位 MB。
- `training_throughput = fit_rows * n_features / fit_time_seconds`

注意：

- `fit_time_seconds + predict_time_seconds` 不是整次实验总耗时，因为还存在数据读取、预处理、概率输出、指标计算和写文件时间。
- `wall_time_seconds` 是用户真实等待时间。
- `predict_time_seconds` 是更接近模型推理速度的核心时间。
- `training_throughput` 用于衡量单位拟合时间处理了多少“样本-特征单元”。

### 2.4 实验状态

必须保存：

- `success`
- `error_type`
- `error_message`

失败也是扩展性实验的重要结果。如果某个模型在较大样本或高维数据上因为限制、超时、内存不足、权重问题而失败，必须记录，而不是删除这次运行。

## 3. 峰值内存测量方案

正式脚本采用 `psutil` 采样当前 Python 进程 RSS：

```text
peak_memory_mb = max(psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
```

执行方式：

- 实验开始后启动后台采样线程。
- 每 `0.05` 到 `0.10` 秒采样一次当前进程 RSS。
- 实验结束或失败时停止采样。
- 将采样期间观察到的最大值写入 `metrics.csv` 的 `peak_memory_mb`。

选择理由：

- `tracemalloc` 主要追踪 Python 内存分配，不适合完整覆盖 NumPy、PyTorch、LightGBM、XGBoost 等原生扩展库的真实进程内存。
- `psutil` 能读取操作系统层面的进程内存信息，更接近本项目报告中需要的“CPU 笔记本实际资源占用”。
- 后台采样实现简单，依赖轻，适合两名成员在不同机器上统一复现。

限制：

- 采样方式可能漏掉极短瞬时峰值，但对本项目这种秒级到分钟级实验足够稳定。
- 默认只统计当前 Python 进程。如果某个模型额外启动子进程，后续可扩展为同时统计子进程 RSS。
- `peak_memory_mb` 不再使用 `memory_before_mb` 或 `memory_after_mb` 替代；这两个字段不属于本项目正式指标。

## 4. 扩展性指标

### 4.1 单次实验直接保存

单次运行必须保存以下扩展性所需字段：

- `experiment_axis`
- `scale_group`
- `fit_rows`
- `test_rows`
- `total_rows`
- `n_features`
- `n_num_features`
- `n_cat_features`
- `n_classes`
- `fit_time_seconds`
- `predict_time_seconds`
- `wall_time_seconds`
- `seconds_per_test_sample`
- `training_throughput`
- `peak_memory_mb`

这些字段足以支撑后续对样本数和特征数的趋势分析。

### 4.2 后处理阶段统一计算

增长速率阶数 `alpha` 不能由单次实验计算，因为它需要同一模型在多个规模点上的结果。完整实验结束后，再按以下公式统一拟合：

```text
log(T) = alpha * log(N) + C
```

约定：

- 样本量扩展性：`N = fit_rows` 或 `total_rows`，主报告建议使用 `fit_rows`。
- 特征量扩展性：`N = n_features`。
- `T` 可分别使用 `fit_time_seconds`、`predict_time_seconds`、`wall_time_seconds`。
- 使用最小二乘法拟合直线，得到 `alpha`。

后处理脚本应把 `alpha` 结果保存到 `results/final/scalability_alpha.csv`，图表保存到 `results/figures/`。

## 5. 正式结果文件结构

单次实验结果目录建议为：

```text
results/raw/{runner_name}/{model}/{experiment_axis}/{scale_group}/{dataset_name}/{run_id}/
```

每次运行保存 4 个核心文件：

```text
metrics.csv
run_config.json
predictions.csv
confusion_matrix.csv
```

另保留一个排错日志：

```text
run.log
```

### 5.1 `metrics.csv`

这是单次实验最重要的文件，通常只有一行。它保存所有可直接比较的数字和必要标识字段。

必须包含：

- 数据集标识：`dataset`、`experiment_axis`、`scale_group`
- 运行者标识：`runner_name`
- 模型标识：`model`、`device`、`model_path`
- 数据规模：`fit_rows`、`test_rows`、`total_rows`、`n_features`
- 正确率指标：`accuracy`、`balanced_accuracy`、`macro_f1`
- 置信度指标：`mean_confidence`、`correct_mean_confidence`、`wrong_mean_confidence`、`confidence_gap`
- 时间内存指标：`fit_time_seconds`、`predict_time_seconds`、`wall_time_seconds`、`seconds_per_test_sample`、`peak_memory_mb`、`training_throughput`
- 状态字段：`success`、`error_type`、`error_message`

### 5.2 `run_config.json`

保存复现信息，不作为主统计表。

建议包含：

- 原始数据集 `info.json`
- train/val/test 使用方式
- 随机种子
- 编码方式
- 类别标签列表
- Python 版本
- 关键包版本
- 本地权重路径
- 峰值内存采样间隔

### 5.3 `predictions.csv`

保存逐测试样本结果，用于置信度直方图、错例分析和混淆分析。

必须包含：

- `sample_index`
- `y_true`
- `y_pred`
- `correct`
- `confidence`
- `prob_class_*`

如果模型不能输出概率：

- `confidence` 写为 `NaN`
- 不写或留空 `prob_class_*`
- `metrics.csv` 中 `probabilities_available=False`

### 5.4 `confusion_matrix.csv`

保存混淆矩阵计数。

用途：

- 后续画混淆矩阵热力图
- 分析模型最容易混淆的类别对

### 5.5 `run.log`

只用于人工排错，不再作为计算指标的数据来源。

`run.log` 可以记录：

- 每一步开始和结束时间
- 失败异常
- 输出目录
- 关键进度

但正式统计必须以 `metrics.csv` 为准。

## 6. 后续图表清单

跑实验时不画图。所有图表在完整结果收集后统一生成：

1. `accuracy / macro_f1` 折线图
2. 时间柱状图
3. `log(T)` 对 `log(N)` 的双对数拟合图
4. 样本量扩展性 `alpha` 对比图
5. 特征量扩展性 `alpha` 对比图
6. `training_throughput` 对比图
7. `peak_memory_mb` 曲线图
8. 置信度直方图
9. 正确/错误置信度对比图
10. 混淆矩阵热力图

## 7. 当前脚本需要修改的点

正式开跑前，实验脚本需要按本文档调整：

- 删除正式结果中的 `memory_before_mb` 和 `memory_after_mb`。
- 增加后台 RSS 采样，写入 `peak_memory_mb`。
- 增加 `wall_time_seconds`。
- 增加 `mean_confidence`、`correct_mean_confidence`、`wrong_mean_confidence`、`confidence_gap`。
- 增加 `training_throughput`。
- 在 `predictions.csv` 中增加 `confidence`。
- 将结果输出到 `results/raw/{runner_name}/{model}/...`。
- 保证失败时也写出 `metrics.csv`、`run_config.json` 和 `run.log`。

## 8. 总结

新的测量计划遵循三个原则：

- 单次实验能算的数字当场算完并保存。
- 后处理只负责合并、拟合跨实验指标和画图。
- 结果文件少而清楚，避免把同一个指标散落在多个地方。
