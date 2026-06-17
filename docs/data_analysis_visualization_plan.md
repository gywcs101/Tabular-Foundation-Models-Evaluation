# 数据分析与绘图计划

更新时间：2026-06-16

本文档定义本项目在完成多设备重复实验之后，如何合并结果、解释指标、计算派生量、绘制图表，并说明哪些指标适合取平均，哪些指标需要按设备分开展示。

本项目当前实验对象为四类模型：

- TabICL
- TabPFN
- LightGBM
- XGBoost

实验数据来自 `data/selected_talent/`，分为两条主线：

- 样本数扩展性：控制特征数接近，观察样本量增加后的表现变化。
- 特征数扩展性：控制样本数接近，观察特征数增加后的表现变化。

## 1. 分析目标

本项目的分析不是只比较谁的准确率最高，而是回答四类问题：

1. 预测效果：哪个模型在不同数据集上更准确、更稳定。
2. 运行成本：哪个模型更快、更省内存，更适合普通笔记本运行。
3. 扩展性：样本数或特征数增加时，模型性能和计算成本如何变化。
4. 置信度与错误特征：模型是否能正确表达自己的不确定性，是否会“错得很自信”。

因此，最终报告应同时包含：

- 性能表格
- 时间和内存成本表格
- 样本数扩展性图
- 特征数扩展性图
- 置信度分布图
- 混淆矩阵热力图
- 多设备重复实验一致性分析

## 2. 多设备重复实验的处理原则

我们计划使用两台轻薄本和两台游戏本运行同一套实验。这样做的意义是提高实验可信度，并观察不同性能电脑上的运行成本差异。

建议结果目录命名：

```text
results/raw/{runner_name}_{device_type}/...
```

示例：

```text
results/raw/吴兆临_light_laptop/
results/raw/姚怀宇_light_laptop/
results/raw/吴兆临_gaming_laptop/
results/raw/姚怀宇_gaming_laptop/
```

如果已经有旧结果目录：

```text
results/raw/吴兆临/
results/raw/姚怀宇/
```

则后续分析脚本应把它们视为轻薄本结果，或者手动重命名为：

```text
results/raw/吴兆临_light_laptop/
results/raw/姚怀宇_light_laptop/
```

## 3. 指标分类

不同指标受设备影响程度不同，不能全部用同一种方式展示。

### 3.1 预测性能指标

包括：

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

这些指标理论上主要由模型、数据和随机种子决定，通常不应明显受电脑性能影响。

处理方式：

- 可以对多台设备结果取平均。
- 同时计算标准差，用来证明重复实验稳定。
- 如果标准差接近 0，报告中可以写“预测性能在不同设备上高度一致”。

推荐展示：

- 主图使用多设备均值。
- 误差线表示标准差。
- 如果误差线几乎看不见，可以在表格中补充标准差。

### 3.2 置信度指标

包括：

- `mean_confidence`
- `correct_mean_confidence`
- `wrong_mean_confidence`
- `confidence_gap`

这些指标同样主要由模型输出概率决定，一般不应强烈受设备性能影响。

处理方式：

- 可以对多设备结果取平均。
- 保留标准差。
- 如果某个模型概率输出不稳定，应单独说明。

注意：置信度直方图展示的是模型预测概率分布，不是内存占用。它回答的问题是“模型有多自信”，不是“模型占多少内存”。

### 3.3 时间指标

包括：

- `fit_time_seconds`
- `predict_time_seconds`
- `wall_time_seconds`
- `seconds_per_test_sample`
- `training_throughput`

这些指标明显受设备影响，例如 CPU 性能、内存速度、后台程序、散热状态都会影响时间。

处理方式：

- 不建议只给一个总平均值。
- 应同时展示不同设备类型。
- 可以计算轻薄本均值、游戏本均值，并展示整体均值。
- 报告中应明确说明时间结果是“实际本地运行成本”，不是模型绝对理论速度。

推荐展示：

- 同一张图中用不同颜色区分模型。
- 用不同线型或分面区分设备类型。
- 如果图太拥挤，则分成两张图：轻薄本一张，游戏本一张。

### 3.4 内存指标

主要指标：

- `peak_memory_mb`

这个指标表示实验过程中 Python 进程使用内存的峰值。它也会受设备、系统和依赖实现影响。

处理方式：

- 和时间一样，不建议只报告单一平均值。
- 应区分轻薄本和游戏本。
- 可以报告轻薄本均值、游戏本均值、整体均值和标准差。

推荐展示：

- 峰值内存柱状图。
- 样本数或特征数变化下的内存折线图。
- 双对数坐标缩放曲线图，用于观察内存随规模增长的趋势。

不推荐：

- 不应使用置信度直方图展示内存。直方图可以展示内存分布，但本项目只有少量设备重复结果，内存直方图意义不强。

## 4. 核心指标含义、公式和判断标准

### 4.1 Accuracy

公式：

```text
accuracy = 正确预测样本数 / 测试样本总数
```

含义：

- 最直观的正确率。
- 适合类别比较均衡的数据集。

判断：

- 越高越好。
- 但类别不均衡时不能只看 accuracy。

### 4.2 Balanced Accuracy

公式：

```text
balanced_accuracy = 每个类别召回率的平均值
```

二分类时：

```text
balanced_accuracy = (正类召回率 + 负类召回率) / 2
```

含义：

- 每个类别先单独算正确率，再平均。
- 可以减少类别不均衡带来的误导。

判断：

- 越高越好。
- 如果 accuracy 高但 balanced_accuracy 低，说明模型可能偏向多数类。

### 4.3 Macro F1

单个类别的 F1：

```text
F1 = 2 * precision * recall / (precision + recall)
```

宏平均：

```text
macro_f1 = 所有类别 F1 的平均值
```

含义：

- 每个类别同等重要。
- 适合多分类任务，也适合观察模型是否忽视少数类。

判断：

- 越高越好。
- 如果 macro_f1 明显低于 accuracy，说明模型可能对部分类别表现差。

### 4.4 Confidence

定义：

```text
confidence = 模型对最终预测类别给出的概率
```

例如模型预测某样本属于类别 3，并给类别 3 的概率为 0.86，则该样本的置信度为 0.86。

相关指标：

```text
mean_confidence = 所有测试样本 confidence 的平均值
correct_mean_confidence = 预测正确样本 confidence 的平均值
wrong_mean_confidence = 预测错误样本 confidence 的平均值
confidence_gap = correct_mean_confidence - wrong_mean_confidence
```

判断：

- `correct_mean_confidence` 高通常是好事。
- `wrong_mean_confidence` 高通常是坏事，表示模型错得很自信。
- `confidence_gap` 越大，通常说明模型更能区分自己什么时候可靠。

注意：

- 如果模型没有概率输出，置信度相关字段为 `NaN`。
- 本项目四个模型通常都能输出概率，因此应尽量保留置信度分析。

### 4.5 Fit Time

字段：

```text
fit_time_seconds
```

含义：

- 对 LightGBM / XGBoost 来说，是训练模型的时间。
- 对 TabPFN / TabICL 来说，是模型准备、上下文构建或调用 `fit` 接口的时间。

判断：

- 越低越好。
- 但不同模型的 `fit` 含义不完全一样，所以不能孤立比较，必须结合 `predict_time_seconds` 和 `wall_time_seconds`。

### 4.6 Predict Time

字段：

```text
predict_time_seconds
```

含义：

- 模型对测试集进行预测的耗时。
- 这是最接近题目中 inference speed 的指标。

判断：

- 越低越好。
- 应同时查看 `seconds_per_test_sample`，避免测试集大小不同导致误判。

### 4.7 Wall Time

字段：

```text
wall_time_seconds
```

含义：

- 从单次实验开始到结果保存完成的总耗时。
- 包含数据读取、预处理、fit、predict、保存文件等步骤。

判断：

- 越低越好。
- 适合展示“用户实际等待多久”。

### 4.8 Seconds Per Test Sample

公式：

```text
seconds_per_test_sample = predict_time_seconds / test_rows
```

含义：

- 平均预测一个测试样本需要多久。
- 可以更公平地比较测试集大小不同的数据集。

判断：

- 越低越好。

### 4.9 Peak Memory

字段：

```text
peak_memory_mb
```

含义：

- 单次实验过程中 Python 进程 RSS 内存峰值，单位 MB。
- RSS 可以理解为当前进程实际占用的物理内存近似值。

判断：

- 越低越好。
- 如果某模型峰值内存过高，说明它在普通笔记本上部署成本较高。
- 如果轻薄本和游戏本差异很大，应分设备解释。

### 4.10 Training Throughput

公式：

```text
training_throughput = fit_rows * n_features / fit_time_seconds
```

含义：

- 每秒处理多少“样本-特征单元”。
- 可以粗略衡量模型拟合阶段处理表格规模的效率。

判断：

- 越高越好。
- 该指标受设备影响明显，适合按设备类型分开展示。

注意：

- TabPFN / TabICL 的 `fit` 不等同于传统训练，因此这个指标更适合作为工程成本参考，不应被解释为模型真实训练效率。

## 5. 可扩展性指标

本项目的可扩展性主要看两个维度：

1. 样本数扩展性：样本数增加时，模型效果和成本如何变化。
2. 特征数扩展性：特征数增加时，模型效果和成本如何变化。

### 5.1 样本数扩展性

使用数据组：

```text
sample_scale/A_1000_3000
sample_scale/B_3000_10000
sample_scale/C_10000_30000
```

横轴可以使用：

```text
fit_rows
```

纵轴包括：

- `accuracy`
- `macro_f1`
- `predict_time_seconds`
- `wall_time_seconds`
- `peak_memory_mb`
- `seconds_per_test_sample`

判断：

- 如果样本数增加后 accuracy / macro_f1 提升，说明模型能从更多数据中受益。
- 如果样本数增加后时间急剧上升，说明计算扩展性较差。
- 如果准确率提升很小但时间大幅增加，说明性价比不高。

### 5.2 特征数扩展性

使用数据组：

```text
feature_scale/F1_6_20
feature_scale/F2_20_100
feature_scale/F3_100_300
```

横轴可以使用：

```text
n_features
```

纵轴包括：

- `accuracy`
- `macro_f1`
- `predict_time_seconds`
- `wall_time_seconds`
- `peak_memory_mb`
- `seconds_per_test_sample`

判断：

- 如果特征数增加后准确率提高，说明高维特征带来了有效信息。
- 如果特征数增加后准确率下降，可能说明模型对高维输入不够鲁棒。
- 如果时间和内存随特征数快速增加，说明模型对高维数据成本敏感。

### 5.3 增长速率阶数 Alpha

增长速率阶数用于描述成本随规模增长的速度。

公式：

```text
log(T) = alpha * log(N) + C
```

其中：

- `T`：时间或内存指标。
- `N`：规模变量。
- 样本数扩展性中，`N = fit_rows`。
- 特征数扩展性中，`N = n_features`。
- `alpha`：增长速率阶数。
- `C`：常数项。

计算方式：

```text
alpha = 对 log(N), log(T) 做最小二乘线性拟合后得到的斜率
```

建议计算：

- `alpha_sample_predict_time`
- `alpha_sample_wall_time`
- `alpha_sample_peak_memory`
- `alpha_feature_predict_time`
- `alpha_feature_wall_time`
- `alpha_feature_peak_memory`

判断：

- `alpha` 越小，说明扩展性越好。
- `alpha ≈ 0`：规模增加时成本几乎不变。
- `alpha ≈ 1`：成本近似线性增长。
- `alpha > 1`：成本增长快于线性，扩展性较差。

注意：

- 本项目每条扩展性曲线只有少量规模点，因此 alpha 只能作为趋势指标，不能当成严格复杂度证明。
- 时间和内存都可以计算 alpha，但报告中应说明它们受设备影响，需要按设备或设备类型分别计算，再汇总。

## 6. 多设备数据如何汇总

### 6.1 推荐分组字段

每条实验记录应拆出：

```text
runner_name
device_type
model
dataset
experiment_axis
scale_group
```

其中 `device_type` 推荐从目录名解析：

```text
light_laptop
gaming_laptop
```

如果目录名中没有设备类型，分析脚本应允许手动映射。

### 6.2 预测性能汇总

对每个：

```text
model + dataset
```

计算：

```text
mean_accuracy
std_accuracy
mean_balanced_accuracy
std_balanced_accuracy
mean_macro_f1
std_macro_f1
```

展示方式：

- 主报告使用均值。
- 标准差用于说明重复实验稳定性。
- 如果轻薄本和游戏本预测性能完全一致，可以不分设备画性能图。

### 6.3 时间和内存汇总

对每个：

```text
device_type + model + dataset
```

计算：

```text
mean_fit_time_seconds
std_fit_time_seconds
mean_predict_time_seconds
std_predict_time_seconds
mean_wall_time_seconds
std_wall_time_seconds
mean_peak_memory_mb
std_peak_memory_mb
mean_training_throughput
std_training_throughput
```

展示方式：

- 轻薄本和游戏本应分开展示或在同一图中明确区分。
- 不建议只给四台设备整体平均值。
- 如果必须给一个总值，应同时给标准差。

### 6.4 推荐报告口径

预测指标：

```text
使用四台设备重复实验的均值，误差线表示标准差。
```

时间和内存：

```text
分别报告轻薄本组和游戏本组的均值与标准差，并比较两类设备上的运行成本差异。
```

## 7. 图表清单

所有正式图表建议同时导出两种格式：

```text
.png
.svg
```

`.png` 用于快速预览、Word 报告和大多数 PPT 场景；`.svg` 用于后期编辑、无损放大和保持论文图质量。两种格式内容应完全一致，只是文件格式不同。

### 7.1 预测性能总览图

文件名：

```text
results/figures/performance_accuracy_by_model.png
results/figures/performance_macro_f1_by_model.png
```

图形：

- 柱状图。
- 横轴：模型。
- 纵轴：accuracy 或 macro_f1。
- 柱高：多设备均值。
- 误差线：标准差。

用途：

- 快速展示四个模型总体预测效果。

判断：

- 柱越高越好。
- 误差线越短，重复实验越稳定。

### 7.2 按数据集的性能图

文件名：

```text
results/figures/performance_accuracy_by_dataset.png
results/figures/performance_macro_f1_by_dataset.png
```

图形：

- 分组柱状图。
- 横轴：数据集。
- 纵轴：accuracy 或 macro_f1。
- 颜色：模型。
- 数据集名称较长时，横轴标签旋转 30 到 45 度。

用途：

- 展示不同模型在不同数据集上的强弱变化。

### 7.3 样本数扩展性性能图

文件名：

```text
results/figures/sample_scale_accuracy_line.png
results/figures/sample_scale_macro_f1_line.png
```

图形：

- 折线图。
- 横轴：`fit_rows` 或样本规模组。
- 纵轴：accuracy 或 macro_f1。
- 颜色：模型。
- 点：具体数据集。
- 如果同一规模组有多个数据集，先对同组数据集取平均。

用途：

- 展示样本数增加后模型效果趋势。

### 7.4 样本数扩展性时间图

文件名：

```text
results/figures/sample_scale_predict_time_line_by_device.png
results/figures/sample_scale_wall_time_line_by_device.png
```

图形：

- 折线图。
- 横轴：`fit_rows`。
- 纵轴：时间。
- 颜色：模型。
- 线型或分面：设备类型。

推荐：

- 如果一张图太乱，分成两张：
  - 轻薄本
  - 游戏本

用途：

- 展示样本规模增加后运行时间如何增长。

### 7.5 样本数扩展性内存图

文件名：

```text
results/figures/sample_scale_peak_memory_bar_by_device.png
results/figures/sample_scale_peak_memory_bar_by_device.svg
```

图形：

- 峰值内存柱状图。
- 横轴：样本规模组或数据集。
- 纵轴：`peak_memory_mb`。
- 颜色：模型。
- 分面或并列小图：设备类型。
- 误差线：同一设备类型下多台设备的标准差。

用途：

- 展示样本规模增加后内存占用如何变化。
- 直接比较不同模型的本地内存成本。

说明：

- 峰值内存柱状图是本文内存成本分析的主图。
- 如果柱状图过宽，可分别输出轻薄本和游戏本两张图。

### 7.6 特征数扩展性性能图

文件名：

```text
results/figures/feature_scale_accuracy_line.png
results/figures/feature_scale_macro_f1_line.png
```

图形：

- 折线图。
- 横轴：`n_features`。
- 纵轴：accuracy 或 macro_f1。
- 颜色：模型。

用途：

- 展示低维、中维、高维特征下模型效果变化。

### 7.7 特征数扩展性时间和内存图

文件名：

```text
results/figures/feature_scale_predict_time_line_by_device.png
results/figures/feature_scale_peak_memory_bar_by_device.png
results/figures/feature_scale_predict_time_line_by_device.svg
results/figures/feature_scale_peak_memory_bar_by_device.svg
```

图形：

- 时间图使用折线图，横轴为 `n_features`，纵轴为时间。
- 内存图使用峰值内存柱状图，横轴为特征规模组或数据集，纵轴为 `peak_memory_mb`。
- 颜色：模型。
- 线型或分面：设备类型。

用途：

- 展示高维表格对不同模型计算成本的影响。
- 峰值内存柱状图作为内存分析主图，折线图主要用于时间趋势。

### 7.8 双对数坐标缩放曲线图

文件名：

```text
results/figures/scalability_loglog_time.png
results/figures/scalability_loglog_memory.png
results/figures/scalability_loglog_time.svg
results/figures/scalability_loglog_memory.svg
```

图形：

- 双对数折线图或散点拟合图。
- 横轴：`fit_rows` 或 `n_features`，使用 log scale。
- 纵轴：`predict_time_seconds`、`wall_time_seconds` 或 `peak_memory_mb`，使用 log scale。
- 颜色：模型。
- 分面：样本数扩展性 / 特征数扩展性。
- 可进一步按设备类型分面。

用途：

- 支撑 alpha 增长速率阶数分析。
- 作为时间和内存扩展性分析的参考图。

注意：

- 双对数图适合展示时间和内存的规模变化。
- 不适合展示 accuracy，因为 accuracy 范围通常在 0 到 1，且不是规模成本。
- 双对数图不是内存成本的主图。内存成本主图使用峰值内存柱状图。
- 如果正文篇幅有限，双对数图可以放附录；如果报告重点讨论 alpha，则建议正文保留至少一张双对数图。

### 7.9 Alpha 表格和图

文件名：

```text
results/final/scalability_alpha.csv
results/figures/scalability_alpha_bar.png
```

表格字段：

```text
experiment_axis
device_type
model
metric
alpha
intercept
n_points
```

图形：

- 柱状图。
- 横轴：模型。
- 纵轴：alpha。
- 颜色：指标或设备类型。

用途：

- 直接比较哪个模型成本增长更快。

### 7.10 置信度直方图

文件名：

```text
results/figures/confidence_histogram_by_model.png
```

数据来源：

```text
predictions.csv
```

使用字段：

```text
confidence
is_correct
model
dataset
```

图形：

- 直方图或密度图。
- 横轴：confidence。
- 纵轴：样本数量或比例。
- 颜色：模型。

推荐做法：

- 先选择代表性数据集绘制，不要把全部数据集混成一张很乱的图。
- 可以按正确和错误分成两个小图。

用途：

- 查看模型是否整体过度自信。
- 查看错误预测的置信度是否偏高。

判断：

- 正确样本集中在高置信度区域是好事。
- 错误样本也大量集中在高置信度区域是坏事。

### 7.11 正确与错误置信度对比图

文件名：

```text
results/figures/confidence_correct_vs_wrong.png
```

图形：

- 分组柱状图或箱线图。
- 横轴：模型。
- 纵轴：confidence。
- 分组：正确样本 / 错误样本。

用途：

- 展示 `correct_mean_confidence` 和 `wrong_mean_confidence` 的差距。

判断：

- 正确样本柱子高、错误样本柱子低，比较理想。
- 两者接近，说明模型难以识别自己是否可靠。
- 错误样本置信度高，说明存在高风险错误。

### 7.12 混淆矩阵热力图

文件夹：

```text
results/figures/confusion_matrix_selected/
```

数据来源：

```text
confusion_matrix.csv
```

图形：

- 热力图。
- 横轴：预测类别。
- 纵轴：真实类别。
- 颜色越深表示数量越多。

建议选择代表性数据集：

```text
mfeat-morphological_2000rows_6feat_multiclass
mfeat-pixel_2000rows_240feat_multiclass
pc1_1109rows_21feat_binclass
default_of_credit_card_clients_30000rows_23feat_binclass
```

用途：

- 多分类任务中查看哪些类别容易混淆。
- 二分类任务中查看模型是否偏向某一类。

注意：

- 不建议把所有 48 个混淆矩阵都放入正文。
- 正文选代表性图，完整图可以放附录或结果文件夹。

## 8. 配色方案

为了报告和 PPT 统一，建议固定模型颜色：

```text
TabICL   : #0072B2 蓝色
TabPFN   : #D55E00 橙红色
LightGBM : #009E73 绿色
XGBoost  : #CC79A7 紫红色
```

这组颜色来自常用色盲友好配色，区分度较好，适合论文图和 PPT。

设备类型建议：

```text
light_laptop  : 实线
gaming_laptop : 虚线
```

或者：

```text
light_laptop  : 较浅色
gaming_laptop : 较深色
```

推荐优先使用线型区分设备，颜色保留给模型。这样读图更清楚。

## 9. 是否需要下载额外库

推荐使用 Python 生态中的常见库：

```text
pandas
numpy
matplotlib
seaborn
scipy
```

用途：

- `pandas`：读取和合并 CSV。
- `numpy`：计算均值、标准差、log 值。
- `matplotlib`：基础绘图和保存图片。
- `seaborn`：更方便地画柱状图、折线图、热力图。
- `scipy`：可用于线性拟合和统计检验。

如果不想额外安装 `scipy`，alpha 也可以用 `numpy.polyfit` 计算。

最小依赖：

```text
pandas
numpy
matplotlib
seaborn
```

检查命令：

```powershell
python -c "import pandas, numpy, matplotlib, seaborn; print('analysis libs ok')"
```

如缺少依赖，可安装：

```powershell
python -m pip install pandas numpy matplotlib seaborn
```

## 10. 输出文件设计

后处理脚本建议生成以下文件。

### 10.1 合并后的原始总表

```text
results/final/final_metrics.csv
```

每行是一条单次实验。

用途：

- 保留所有设备、所有成员、所有模型、所有数据集的结果。

### 10.2 设备一致性表

```text
results/final/device_consistency.csv
```

字段：

```text
model
dataset
experiment_axis
scale_group
metric
mean
std
min
max
relative_std
```

用途：

- 说明多设备重复实验是否稳定。

### 10.3 模型数据集汇总表

```text
results/final/model_dataset_summary.csv
```

字段：

```text
model
dataset
experiment_axis
scale_group
mean_accuracy
std_accuracy
mean_macro_f1
std_macro_f1
mean_predict_time_seconds
std_predict_time_seconds
mean_wall_time_seconds
std_wall_time_seconds
mean_peak_memory_mb
std_peak_memory_mb
```

用途：

- 大多数性能和成本图表的输入。

### 10.4 设备类型汇总表

```text
results/final/device_type_summary.csv
```

字段：

```text
device_type
model
dataset
experiment_axis
scale_group
mean_predict_time_seconds
std_predict_time_seconds
mean_wall_time_seconds
std_wall_time_seconds
mean_peak_memory_mb
std_peak_memory_mb
mean_training_throughput
std_training_throughput
```

用途：

- 专门支持轻薄本与游戏本对比。

### 10.5 Alpha 结果表

```text
results/final/scalability_alpha.csv
```

字段：

```text
experiment_axis
device_type
model
metric
alpha
intercept
n_points
```

用途：

- 支持增长速率阶数分析。

## 11. 推荐分析顺序

### 第一步：清理旧目录

只保留正式结果目录，例如：

```text
results/raw/吴兆临_light_laptop/
results/raw/姚怀宇_light_laptop/
results/raw/吴兆临_gaming_laptop/
results/raw/姚怀宇_gaming_laptop/
```

删除旧的调试目录，避免合并时污染结果。

### 第二步：合并所有 `metrics.csv`

生成：

```text
results/final/final_metrics.csv
```

同时从目录名解析：

- `runner_name`
- `device_type`

### 第三步：检查完整性

检查每个设备是否都有：

```text
12 数据集 * 4 模型 = 48 条成功结果
```

如果某台设备缺失结果，应先补跑。

### 第四步：检查预测稳定性

比较多设备下：

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

如果差异极小，可以说明预测结果可重复。

### 第五步：分析设备成本差异

比较轻薄本和游戏本的：

- `predict_time_seconds`
- `wall_time_seconds`
- `peak_memory_mb`
- `training_throughput`

重点观察：

- 游戏本是否显著更快。
- 哪些模型最受设备性能影响。
- TabPFN / TabICL 是否比树模型更吃硬件。

### 第六步：绘制核心图表

先画必需图：

1. 总体 accuracy 柱状图。
2. 总体 macro_f1 柱状图。
3. 样本数扩展性 accuracy 折线图。
4. 样本数扩展性时间折线图。
5. 特征数扩展性 accuracy 折线图。
6. 特征数扩展性时间折线图。
7. 峰值内存柱状图。
8. 双对数缩放曲线图，作为扩展性参考图。
9. 置信度直方图。
10. 代表性混淆矩阵热力图。

再根据报告空间选择补充图。

## 12. 报告中的推荐表述

### 12.1 多设备重复实验

可写：

```text
为了提高结论的可靠性，我们在两台轻薄本和两台游戏本上重复运行完整实验。预测性能指标主要用于检验模型结果的一致性；运行时间、吞吐量和峰值内存用于观察不同硬件条件下的实际计算成本。实验结果显示，accuracy 和 macro F1 在不同设备之间差异极小，说明模型预测结果具有较好的可重复性；而时间和内存指标受硬件条件影响更明显，因此本文对运行成本按设备类型分别展示，并报告均值和标准差。
```

### 12.2 时间和内存受设备影响

可写：

```text
时间和内存指标不仅反映模型本身的计算需求，也受到 CPU 性能、内存带宽、系统调度和后台进程影响。因此本文不将单台设备上的耗时视为模型的绝对速度，而是将其作为普通本地设备上的实际运行成本，并通过多设备重复实验展示其波动范围。
```

### 12.3 Alpha 的解释

可写：

```text
为了比较不同模型随数据规模增长时的成本变化趋势，我们在双对数坐标下拟合 log(T)=alpha log(N)+C，其中 alpha 表示成本随规模增长的速率。由于本项目规模点数量有限，alpha 不作为严格复杂度证明，而作为经验扩展性指标，用于比较模型在相同实验设置下的增长趋势。
```

### 12.4 置信度分析

可写：

```text
置信度分析用于观察模型是否能合理表达自身不确定性。理想情况下，模型在预测正确时应具有较高置信度，而在预测错误时置信度应较低。如果错误样本也具有很高置信度，说明模型存在过度自信问题，在实际应用中可能带来更高风险。
```

## 13. 需要注意的问题

1. 不要把置信度直方图用于展示内存。置信度和内存是两个不同概念。
2. 不要只用 accuracy 判断模型好坏，必须结合 macro_f1 和 balanced_accuracy。
3. 不要只用单台设备的时间结论代表模型绝对速度。
4. 不要把 TabPFN / TabICL 的 `fit_time_seconds` 解释成传统训练时间。
5. 不要把 alpha 当成严格算法复杂度，只能说是本实验中的经验增长趋势。
6. 混淆矩阵不要全部放正文，选择代表性数据集即可。
7. 图表颜色必须统一，否则报告和 PPT 会显得混乱。

## 14. 最终交付物

最终分析阶段应至少产出：

```text
results/final/final_metrics.csv
results/final/device_consistency.csv
results/final/model_dataset_summary.csv
results/final/device_type_summary.csv
results/final/scalability_alpha.csv
results/figures/*.png
results/figures/*.svg
```

报告中至少使用：

- 1 张总体性能图
- 1 张样本数扩展性图
- 1 张特征数扩展性图
- 1 张时间成本图
- 1 张峰值内存柱状图
- 1 张双对数缩放图，若正文篇幅不足可放附录
- 1 张置信度分析图
- 1 到 2 张混淆矩阵热力图
