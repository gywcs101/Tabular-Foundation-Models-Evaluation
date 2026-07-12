# Tabular Foundation Models Project

这是“人工智能大模型 II”课程项目 2：Tabular Foundation Models 的最终仓库。

本项目围绕一个核心问题展开：在有限算力条件下，Tabular Foundation Models 是否真的比传统梯度提升树更实用。我们在 12 个来自 TALENT 的分类数据集上，对 4 个模型进行了对比：

- `TabICL`
- `TabPFN`
- `LightGBM`
- `XGBoost`

论文实际关注的维度包括：

- 预测性能
- 推理时间
- 峰值内存
- 样本数扩展性
- 特征数扩展性
- 置信度可靠性

## 主要结果

- TabICL 和 TabPFN 的平均预测性能高于 LightGBM 和 XGBoost。
- 但两类 TFM 的推理时间和内存开销明显更高，尤其在样本数增大时更突出。
- 在 CPU 和普通笔记本环境下，GBDT 仍然是更稳妥、更高效的基线。

## 目录说明

- `Tabular-Report/`：论文源文件和编译结果
- `results/final/`：最终汇总表
- `results/figures/`：最终分析图
- `results/raw/`：原始实验输出
- `data/selected_talent/`：筛选后的 TALENT 数据集
- `scripts/`：实验、分析和控制脚本

## 论文

论文源码位于 `Tabular-Report/main.tex`，编译后的 PDF 位于 `Tabular-Report/main.pdf`。

## 运行环境

项目默认使用本地相对路径管理环境和模型权重：

- `.local_envs/`
- `.local_models/`
- `.local_external/`

这些目录不纳入仓库主体内容。
