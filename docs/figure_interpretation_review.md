# 图表含义与可用性审阅

更新时间：2026-06-16

本文档解释 `results/figures/` 中当前已生成图表的含义，并逐类判断：

1. 图表格式是否规范，是否有文字或图案重叠影响阅读。
2. 图中是否能总结出有效结论。
3. 是否建议放入最终报告或 PPT。

注意：当前图表只基于两台轻薄本结果：

```text
results/raw/吴兆临_light_laptop
results/raw/姚怀宇_light_laptop
```

两台游戏本结果尚未加入，因此本文档中的数值结论是第一版结论。游戏本结果补齐后，需要重新运行：

```powershell
python scripts\analysis\run_full_analysis.py
```

然后重新检查本文档中的结论是否仍成立。

## 1. 总体质量判断

当前图表整体已经能支撑初步分析：

- `.png` 和 `.svg` 均已生成，格式符合后续报告和 PPT 使用需求。
- 颜色已经按模型固定：TabICL 蓝色、TabPFN 橙色、LightGBM 绿色、XGBoost 紫红色。
- 大部分图使用白底、无顶部/右侧边框、简洁网格，基本符合科研图风格。

但当前也有一些需要注意的问题：

- 部分横轴数据集名称较长，按数据集柱状图和内存柱状图略显拥挤。
- 部分图的图例标题显示为 `model_label`，不够正式，后续可以改成空标题或 `Model`。
- 当前只有 `light_laptop` 一种设备类型，设备分面图暂时只能展示轻薄本；游戏本结果加入后，图的信息会更完整。
- 双对数图适合作为扩展性参考图，不建议作为读者理解成本差异的第一张主图。
- 混淆矩阵图数量较多，正文不应全部放入，只选最有代表性的几张。

## 2. 当前总体数值结论

基于当前两台轻薄本结果，四个模型平均表现如下：

| 模型 | 平均 accuracy | 平均 macro F1 | 平均 predict time 秒 | 平均 wall time 秒 | 平均峰值内存 MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabICL | 0.9131 | 0.8506 | 121.55 | 239.55 | 7032.11 |
| TabPFN | 0.9062 | 0.8440 | 680.41 | 1339.23 | 2223.94 |
| LightGBM | 0.8829 | 0.8244 | 0.015 | 1.38 | 178.32 |
| XGBoost | 0.8829 | 0.8245 | 0.006 | 1.71 | 177.70 |

可以初步总结：

- TabICL 当前预测效果最好，TabPFN 次之。
- LightGBM 和 XGBoost 预测效果略低，但速度和内存成本远低于 TabICL / TabPFN。
- TabPFN 在 CPU 上非常慢，尤其在样本数增加时。
- TabICL 的峰值内存最高，尤其在大样本和高维特征任务中。
- 传统树模型的速度和内存非常稳定，适合作为低成本 baseline。

## 3. 预测性能图

### 3.1 `performance_accuracy_by_model`

文件：

```text
results/figures/performance_accuracy_by_model.png
results/figures/performance_accuracy_by_model.svg
```

图表含义：

- 比较四个模型在所有已选数据集上的平均 accuracy。
- 柱高越高，整体预测正确率越高。
- 误差线表示不同数据集和设备重复结果带来的波动。

格式审阅：

- 图形简洁，文字无明显重叠。
- 横轴模型名称清楚。
- 适合报告和 PPT。

图中结论：

- TabICL 平均 accuracy 最高，约 0.913。
- TabPFN 次之，约 0.906。
- LightGBM 和 XGBoost 接近，约 0.883。
- 从 accuracy 看，两个表格基础模型整体优于两个树模型 baseline。

建议：

- 建议放入报告正文。
- PPT 中也可以使用，作为“整体性能结论”的第一张图。

### 3.2 `performance_macro_f1_by_model`

文件：

```text
results/figures/performance_macro_f1_by_model.png
results/figures/performance_macro_f1_by_model.svg
```

图表含义：

- 比较四个模型在所有数据集上的平均 macro F1。
- macro F1 更关注各类别是否都预测得好，能减少类别不均衡对 accuracy 的误导。

格式审阅：

- 版式清楚，无明显重叠。
- 适合报告和 PPT。

图中结论：

- TabICL 仍然最高，约 0.851。
- TabPFN 次之，约 0.844。
- XGBoost 和 LightGBM 非常接近，约 0.824。
- accuracy 和 macro F1 的模型排序基本一致，说明 TabICL / TabPFN 的优势不是只来自多数类预测。

建议：

- 建议放入报告正文。
- 如果篇幅有限，可与 accuracy 总览图二选一；报告中更建议保留 macro F1，因为它更能说明类别公平性。

### 3.3 `performance_accuracy_by_dataset`

文件：

```text
results/figures/performance_accuracy_by_dataset.png
results/figures/performance_accuracy_by_dataset.svg
```

图表含义：

- 展示每个数据集上四个模型的 accuracy。
- 用于观察模型是否在所有数据集上都稳定领先，还是只在部分数据集上有优势。

格式审阅：

- 图例清楚。
- 横轴数据集名称较长，虽然没有严重重叠，但阅读压力较大。
- 适合报告附录或宽版页面，不太适合 PPT 单页直接展示。

图中结论：

- 在 mfeat 系列多分类任务中，TabICL 和 TabPFN 通常表现较好。
- 在部分二分类任务上，四个模型差距很小。
- 在 `credit default` 和 `jm1` 等较大二分类任务上，accuracy 都在约 0.81 到 0.83，模型间差距不明显。
- 该图能说明“表格基础模型不是每个数据集都大幅领先”，优势具有数据集依赖性。

建议：

- 报告中可放，但最好作为宽图或附录图。
- PPT 不建议直接放全图，建议改成只展示代表性数据集或拆成两张图。

### 3.4 `performance_macro_f1_by_dataset`

文件：

```text
results/figures/performance_macro_f1_by_dataset.png
results/figures/performance_macro_f1_by_dataset.svg
```

图表含义：

- 展示每个数据集上四个模型的 macro F1。
- 用于观察类别不均衡或多分类任务下模型表现。

格式审阅：

- 横轴同样较拥挤。
- 信息量比 accuracy by dataset 更大，但阅读难度也更高。

图中结论：

- `pc1`、`kc1`、`credit default`、`jm1` 等二分类任务中，macro F1 明显低于 accuracy，说明这些数据集可能存在类别不均衡或少数类更难预测。
- mfeat 高维数据上各模型 macro F1 较高，说明多分类识别较稳定。
- TabICL / TabPFN 在多数数据集上保持较高 macro F1。

建议：

- 报告正文可以优先放这张而不是 accuracy by dataset，因为它更能解释不均衡影响。
- PPT 建议只截取或重画代表性数据集，否则文字过多。

## 4. 特征数扩展性图

### 4.1 `feature_scale_accuracy_line`

文件：

```text
results/figures/feature_scale_accuracy_line.png
results/figures/feature_scale_accuracy_line.svg
```

图表含义：

- 横轴是特征数，纵轴是 accuracy。
- 展示低维、中维、高维 mfeat 特征下，不同模型预测效果如何变化。

格式审阅：

- 整体清楚，无明显重叠。
- 图例在左上角，未遮挡关键曲线。
- 适合报告和 PPT。

图中结论：

- 特征数增加后，四个模型 accuracy 总体提高。
- 低维 6 特征时，LightGBM 和 XGBoost 表现较弱，TabICL / TabPFN 更好。
- 高维 216/240 特征时，四个模型都达到较高 accuracy，TabICL 仍略占优。
- 这说明高维像素/统计特征为 mfeat 任务提供了更多有效信息。

建议：

- 建议放入报告正文。
- PPT 中也适合使用，能直观展示特征数扩展性。

### 4.2 `feature_scale_macro_f1_line`

文件：

```text
results/figures/feature_scale_macro_f1_line.png
results/figures/feature_scale_macro_f1_line.svg
```

图表含义：

- 横轴是特征数，纵轴是 macro F1。
- 与 accuracy 图配合，验证特征数增加是否也改善各类别整体表现。

格式审阅：

- 版式清楚。
- 与 accuracy 图高度相似。

图中结论：

- 结论与 accuracy 图一致：特征数增加后 macro F1 明显提高。
- TabICL 和 TabPFN 在低维和高维任务中均保持较强表现。

建议：

- 报告正文中可与 accuracy 图二选一。
- 如果只能放一张，建议保留 macro F1 图，因为它更适合多分类任务。

### 4.3 `feature_scale_predict_time_seconds_line_by_device`

文件：

```text
results/figures/feature_scale_predict_time_seconds_line_by_device.png
results/figures/feature_scale_predict_time_seconds_line_by_device.svg
```

图表含义：

- 展示特征数增加时，各模型预测耗时如何变化。
- 当前只有轻薄本结果，所以只显示 light laptop。

格式审阅：

- 图形清楚。
- 但图名和分面标题中仍有 `device_type = light_laptop` 这类程序字段，后续应改成更正式的 `Light laptop`。
- 适合分析，不建议当前版本直接放最终 PPT。

图中结论：

- TabPFN 预测耗时随特征数增加增长最快。
- TabICL 也随特征数增加明显变慢。
- LightGBM 和 XGBoost 几乎贴近横轴，预测耗时远低于两个基础模型。

建议：

- 报告中建议保留，但正式版需要清理标题和图例。
- PPT 中可以放，前提是改掉程序化标题。

### 4.4 `feature_scale_wall_time_seconds_line_by_device`

文件：

```text
results/figures/feature_scale_wall_time_seconds_line_by_device.png
results/figures/feature_scale_wall_time_seconds_line_by_device.svg
```

图表含义：

- 展示特征数增加时，完整实验总耗时如何变化。

格式审阅：

- 版式基本清楚。
- 同样存在分面标题程序化的问题。

图中结论：

- TabPFN 总耗时最高，随特征数增加明显上升。
- TabICL 次之。
- LightGBM 和 XGBoost 总耗时很低。
- 与 predict time 图结论一致，说明基础模型主要成本来自推理/上下文处理。

建议：

- 可作为报告中的计算成本图。
- 如果篇幅有限，和 predict time 图二选一；题目强调 inference speed，所以优先保留 predict time。

### 4.5 `feature_scale_peak_memory_bar_by_device`

文件：

```text
results/figures/feature_scale_peak_memory_bar_by_device.png
results/figures/feature_scale_peak_memory_bar_by_device.svg
```

图表含义：

- 展示特征数实验中各模型峰值内存占用。
- 单位是 GB。

格式审阅：

- 标题和图例位置略拥挤，顶部有一定重叠风险。
- 横轴标签较长，但仍可读。
- 当前版本适合分析，不建议不修改就放 PPT。

图中结论：

- TabICL 峰值内存远高于其他模型，高维 mfeat 上可达到十几 GB。
- TabPFN 内存低于 TabICL，但仍明显高于树模型。
- LightGBM 和 XGBoost 内存接近，基本维持在很低水平。

建议：

- 这是内存成本主图之一，建议保留。
- 正式报告前建议优化标题/图例位置。

## 5. 样本数扩展性图

### 5.1 `sample_scale_accuracy_line`

文件：

```text
results/figures/sample_scale_accuracy_line.png
results/figures/sample_scale_accuracy_line.svg
```

图表含义：

- 横轴是训练/上下文样本数，纵轴是 accuracy。
- 展示样本量增加时，各模型效果变化。

格式审阅：

- 图表清楚，无明显重叠。
- 适合报告和 PPT。

图中结论：

- A 组到 B 组 accuracy 明显提高。
- C 组 accuracy 反而下降，原因不是简单“样本越多越好”，而是 C 组数据集本身任务更难。
- 这说明样本数扩展性实验中，不同数据集难度仍会影响结果，需要在报告里解释。

建议：

- 可以放报告，但解释时必须强调“这是按代表性数据集分组，不是同一数据集的连续子采样”。
- 如果要严格讲样本数变化趋势，后续可补充同一数据集子采样实验。

### 5.2 `sample_scale_macro_f1_line`

文件：

```text
results/figures/sample_scale_macro_f1_line.png
results/figures/sample_scale_macro_f1_line.svg
```

图表含义：

- 展示样本量增加时，各模型 macro F1 变化。

格式审阅：

- 图表清楚。
- 适合报告。

图中结论：

- B 组表现最好。
- C 组 macro F1 明显低于 B 组，说明大样本数据集存在更强类别不均衡或任务难度。
- TabICL / TabPFN 在 B 组非常强，但在 C 组优势减小。

建议：

- 建议与 accuracy 图配合使用。
- 若报告篇幅有限，保留 macro F1 图更好。

### 5.3 `sample_scale_predict_time_seconds_line_by_device`

文件：

```text
results/figures/sample_scale_predict_time_seconds_line_by_device.png
results/figures/sample_scale_predict_time_seconds_line_by_device.svg
```

图表含义：

- 展示样本量增加时预测耗时变化。

格式审阅：

- 趋势清楚。
- 分面标题有程序字段，后续应美化。

图中结论：

- TabPFN 随样本量增加耗时增长最剧烈。
- TabICL 也随样本量明显增长，但低于 TabPFN。
- XGBoost 和 LightGBM 几乎不随样本量显著增长，成本极低。

建议：

- 强烈建议放入报告，因为它直接回应题目中的 inference speed 和 scalability。
- PPT 可使用，但建议先美化标题。

### 5.4 `sample_scale_wall_time_seconds_line_by_device`

文件：

```text
results/figures/sample_scale_wall_time_seconds_line_by_device.png
results/figures/sample_scale_wall_time_seconds_line_by_device.svg
```

图表含义：

- 展示样本量增加时完整实验总耗时变化。

格式审阅：

- 清楚但标题需要美化。

图中结论：

- TabPFN 在 C 组 wall time 超过 5000 秒，CPU 成本非常高。
- TabICL 也明显变慢。
- 树模型总耗时仍在秒级以内。

建议：

- 报告中可作为计算成本补充。
- 如果只放一张时间图，优先放 predict time；wall time 可放附录。

### 5.5 `sample_scale_peak_memory_bar_by_device`

文件：

```text
results/figures/sample_scale_peak_memory_bar_by_device.png
results/figures/sample_scale_peak_memory_bar_by_device.svg
```

图表含义：

- 展示样本数实验中各模型峰值内存占用。

格式审阅：

- 顶部标题和图例有重叠风险。
- 横轴标签较长但可读。
- 当前版本不建议直接放 PPT，需要调整图例位置或拆图。

图中结论：

- TabICL 在大样本 C 组内存最高，`credit default` 任务上峰值可达十几 GB。
- TabPFN 内存明显高于树模型，但低于 TabICL。
- LightGBM 和 XGBoost 内存稳定且很低。

建议：

- 作为内存成本主图，建议保留。
- 正式使用前应优化排版。

## 6. 双对数缩放图

### 6.1 `scalability_loglog_time`

文件：

```text
results/figures/scalability_loglog_time.png
results/figures/scalability_loglog_time.svg
```

图表含义：

- 横轴和纵轴都使用对数坐标。
- 用于观察时间成本随样本数或特征数增长的趋势。
- 是计算增长速率阶数 `alpha` 的参考图。

格式审阅：

- 趋势清楚。
- 但标题中有 `device_type = light_laptop | axis_label = ...` 这种程序化文字。
- 图例标题为 `model_label`，需要美化。
- 当前版本适合分析，不建议直接放最终 PPT。

图中结论：

- TabPFN 和 TabICL 的时间曲线明显高于树模型。
- 样本数扩展中，TabPFN / TabICL 增长斜率较大。
- 特征数扩展中，TabPFN 增长也明显，TabICL 次之。
- 树模型时间基本保持低水平。

建议：

- 如果报告讨论 alpha，建议保留一张双对数时间图。
- 如果篇幅紧，可以只放 `scalability_alpha.csv` 的结果表或柱状图，双对数图放附录。

### 6.2 `scalability_loglog_memory`

文件：

```text
results/figures/scalability_loglog_memory.png
results/figures/scalability_loglog_memory.svg
```

图表含义：

- 展示峰值内存随样本数或特征数增长的双对数趋势。
- 是内存扩展性参考图，不是内存成本主图。

格式审阅：

- 可读，但同样存在程序化标题和图例标题问题。
- 当前版本不建议直接放 PPT。

图中结论：

- TabICL 内存增长最明显，尤其在特征数和样本数增加时。
- TabPFN 内存增长中等。
- LightGBM 和 XGBoost 内存几乎不变。

建议：

- 不建议作为正文主图。
- 如果报告中要解释 `alpha_peak_memory`，可放附录或方法分析部分。

## 7. 置信度图

### 7.1 `confidence_histogram_by_model`

文件：

```text
results/figures/confidence_histogram_by_model.png
results/figures/confidence_histogram_by_model.svg
```

图表含义：

- 展示不同模型预测置信度分布。
- 横轴越接近 1，说明模型越自信。

格式审阅：

- 图形本身清楚。
- 图例标题显示为 `model_label`，需要改成空标题或 `Model`。
- 多条线在 0.9 到 1.0 附近集中，视觉上信息较密。

图中结论：

- 四个模型大量预测都集中在高置信度区间。
- LightGBM 平均置信度最高，但它的准确率不是最高，说明较高置信度不等于更可靠。
- 该图能引出“需要区分正确时置信度和错误时置信度”的分析。

建议：

- 可放报告，但最好配合下一张 correct vs wrong 图一起解释。
- 单独放这张图信息不够完整。

### 7.2 `confidence_correct_vs_wrong`

文件：

```text
results/figures/confidence_correct_vs_wrong.png
results/figures/confidence_correct_vs_wrong.svg
```

图表含义：

- 比较预测正确样本和预测错误样本的平均置信度。
- 正确样本高、错误样本低，是比较理想的状态。

格式审阅：

- 图形清楚，无明显重叠。
- 误差线较长，说明不同数据集间差异较大。
- 适合报告和 PPT。

图中结论：

- 四个模型正确样本置信度都高于错误样本。
- TabPFN 的 confidence gap 最大，约 0.245。
- TabICL 次之，约 0.221。
- LightGBM 的 mean confidence 最高，但 confidence gap 最小，说明它更容易在错误样本上也保持较高置信度。

建议：

- 建议放入报告。
- 这张图比置信度直方图更适合正文，因为结论更直接。

## 8. 混淆矩阵图

文件夹：

```text
results/figures/confusion_matrix_selected/
```

当前为 4 个模型 × 4 个代表性数据集生成混淆矩阵，共 16 张 PNG 和 16 张 SVG。

### 8.1 mfeat-pixel 混淆矩阵

代表文件：

```text
confusion_matrix_tabicl_mfeat-pixel_2000rows_240feat_multiclass.png
```

图表含义：

- 展示 10 类手写数字识别中真实类别和预测类别的对应关系。
- 对角线越深，说明预测越准确。
- 非对角线颜色越深，说明某些类别容易混淆。

格式审阅：

- 图形清楚，无明显重叠。
- 颜色条清楚。
- 标题中的数据集名称带下划线，略显程序化，后续可美化。

图中结论：

- mfeat-pixel 上对角线非常明显，说明模型对高维像素特征的多分类识别较好。
- 非对角线颜色较浅，错误较少。

建议：

- 建议选 1 到 2 张代表性混淆矩阵放入报告。
- mfeat-pixel 的 TabICL 或 TabPFN 矩阵适合作为“高维多分类表现好”的例子。

### 8.2 mfeat-morphological 混淆矩阵

图表含义：

- 低维 6 特征手写数字识别任务。

格式审阅：

- 图形清楚。
- 10 类标签可读。

图中结论：

- 相比 mfeat-pixel，低维特征下错误更明显。
- 适合和 mfeat-pixel 对比，说明特征表达方式会显著影响错误模式。

建议：

- 如果报告讨论特征数扩展性，可以放 mfeat-morphological 与 mfeat-pixel 各一张对比。

### 8.3 pc1 混淆矩阵

图表含义：

- 小样本二分类任务，适合观察模型是否偏向多数类。

格式审阅：

- 2×2 矩阵清楚。
- 图较简单，适合 PPT。

图中结论：

- 需要结合具体矩阵数值判断是否偏向某一类。
- 该图适合解释为什么 accuracy 和 macro F1 可能不同。

建议：

- 如果报告强调类别不均衡，建议放一张 pc1 混淆矩阵。
- 如果篇幅有限，可以只在附录保留。

### 8.4 credit default 混淆矩阵

图表含义：

- 大样本二分类金融风险任务。

格式审阅：

- 图形清楚。
- 类别数量少，适合展示。

图中结论：

- 适合观察大样本二分类中模型是否偏向多数类。
- 可与 pc1 形成“小样本 vs 大样本”的二分类对比。

建议：

- 报告中建议保留一张，尤其适合实际应用背景分析。

### 8.5 混淆矩阵整体建议

不建议把 16 张全部放入正文。推荐正文最多放 2 到 4 张：

1. `mfeat-morphological`：低维多分类。
2. `mfeat-pixel`：高维多分类。
3. `pc1`：小样本二分类。
4. `credit default`：大样本二分类。

如果篇幅更紧，建议只放：

1. `mfeat-pixel` 的最佳模型混淆矩阵。
2. `credit default` 的代表模型混淆矩阵。

## 9. 暂不建议放入正文的图

以下图目前不建议直接放入正文或 PPT，除非后续美化：

```text
performance_accuracy_by_dataset
performance_macro_f1_by_dataset
feature_scale_peak_memory_bar_by_device
sample_scale_peak_memory_bar_by_device
scalability_loglog_time
scalability_loglog_memory
confidence_histogram_by_model
```

原因：

- 有些横轴标签过长。
- 有些标题或图例仍带程序字段。
- 有些图信息量大，适合附录或分析过程，不适合 PPT 主线。

## 10. 建议优先放入报告/PPT的图

第一优先级：

```text
performance_macro_f1_by_model
feature_scale_macro_f1_line
sample_scale_predict_time_seconds_line_by_device
confidence_correct_vs_wrong
sample_scale_peak_memory_bar_by_device
```

第二优先级：

```text
performance_accuracy_by_model
sample_scale_macro_f1_line
feature_scale_predict_time_seconds_line_by_device
feature_scale_peak_memory_bar_by_device
selected confusion matrices
```

附录或补充分析：

```text
performance_macro_f1_by_dataset
scalability_loglog_time
scalability_loglog_memory
confidence_histogram_by_model
all confusion matrices
```

## 11. 下一步建议

在最终报告或 PPT 使用前，建议对绘图脚本做三类小优化：

1. 把所有图例标题 `model_label` 去掉或改成 `Model`。
2. 把分面标题 `device_type = light_laptop` 改成 `Light laptop`，游戏本加入后改成 `Light laptop` / `Gaming laptop`。
3. 对横轴很长的图，拆成多张图或改成横向条形图，避免数据集名称拥挤。

这些优化不影响数据结论，只提升展示质量。

