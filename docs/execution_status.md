# Execution Status

更新时间：2026-06-09

## 当前阶段

当前项目处于：

```text
GitHub 协作已就绪，正在统一正式结果格式，尚未开始新口径主实验
```

项目内容已经放到 GitHub。当前重点是确认两名成员的本地模型环境、数据集目录、运行脚本和结果合并规则一致。旧口径试跑结果已经清空，后续正式实验需要先按 `docs/measurement_plan.md` 修改脚本输出字段。

## 已完成

- 整理课程项目要求到 `AGENTS.md`。
- 完成项目策划案：`docs/project_plan.md`。
- 完成文献笔记：`docs/literature_notes.md`。
- 完成 GitHub 协作说明：`docs/github_collaboration.md`。
- 完成仓库结构说明：`docs/repository_structure.md`。
- 完成 TALENT 数据集筛选准则：`docs/dataset_selection_criteria.md`。
- 完成实验测量计划：`docs/measurement_plan.md`。
- 完成 GitHub 仓库上传。
- 建立实验脚本和共享工具模块：
  - `scripts/project_config.py`
  - `scripts/data_utils.py`
  - `scripts/model_utils.py`
  - `scripts/experiment_utils.py`
  - `scripts/prepare_data.py`
  - `scripts/run_benchmark.py`
  - `scripts/run_missing_value_experiment.py`
  - `scripts/run_retrieval_icl_experiment.py`
  - `scripts/analyze_results.py`
  - `scripts/merge_results.py`
  - `scripts/run_smoke_test.py`
  - `scripts/check_local_assets.py`
- 补强 `.gitignore`，避免上传本地环境、数据缓存、模型权重、Python 缓存和临时结果。

## 本地模型环境

团队采用“每个模型一个环境”的策略。当前统一使用项目内相对路径：

```text
.local_envs\tabpfn
.local_envs\tabicl
.local_envs\xgboost
.local_envs\lightgbm
```

当前检查结果：

| 环境 | Python | 状态 |
| --- | --- | --- |
| `tabpfn` | 3.11.15 | `tabpfn`、`torch`、`sklearn`、`pandas`、`numpy` 可导入 |
| `tabicl` | 3.11.15 | `tabicl`、`torch`、`sklearn`、`numpy` 可导入 |
| `xgboost` | 3.11.15 | `xgboost 3.2.0`、`sklearn`、`pandas`、`numpy` 可导入 |
| `lightgbm` | 3.11.15 | `lightgbm 4.6.0`、`sklearn`、`pandas`、`numpy` 可导入 |

后续默认使用 Python 3.11 环境，不使用 Anaconda base 的 Python 3.13 作为实验环境。

## 本地数据资源

当前主实验数据已经整理到项目内：

```text
data\selected_talent
```

该目录体积较小，允许上传 GitHub，保证两名成员使用完全相同的数据划分。

完整 TALENT 原始包和 TabArena 仓库如需保留，统一放在：

```text
.local_external\TALENT-tabular-benchmark
.local_external\tabarena
```

`.local_external/` 不上传 GitHub。`TALENT-tabular-benchmark` 是从 Google Drive 下载并改名后的 TALENT/表格 benchmark 数据包。当前检查结果：

- `benchmark_dataset\data` 存在。
- 共检测到 300 个数据集目录。
- 抽查数据集的 `info.json`、`y_train/val/test.npy`、数值/类别特征文件可正常读取。

`tabarena` 是 TabArena 的源码仓库和元数据，不是完整数据缓存。当前检查结果：

- `README.md`、`pyproject.toml`、`tabarena/`、`examples/` 存在。
- 检测到 TabArena 51 个数据集的元数据 CSV。
- 该目录可用于参考 TabArena 数据集列表和评测思路，但不能直接作为完整样本数据来训练/评测模型。

## 当前策略

- Codex 不主动下载数据集或模型。
- 模型环境和模型权重由用户与队友在项目内本地目录自行维护。
- Conda 环境、模型权重、OpenML/TALENT/TabArena 原始缓存不上传 GitHub。
- `data/selected_talent/` 上传 GitHub，作为两名成员共同运行的固定数据集。
- GitHub 仓库保存代码、文档、依赖说明、可复现配置和可共享的小型结果文件。
- TALENT 作为当前主要可运行数据来源，但不全量跑 300 个数据集。
- 数据集筛选遵守 `docs/dataset_selection_criteria.md`：当前保留样本数扩展性的轻量方案和稳健方案，待用户审核后定稿。
- 样本数扩展性候选以 `pc1`、`kc1`、`sylvine`、`ringnorm`、`jm1`、`default_of_credit_card_clients` 为主，目标是让特征数尽量接近。
- 特征数扩展性候选以 `mfeat-morphological`、`mfeat-zernike`、`mfeat-karhunen`、`mfeat-fourier`、`mfeat-factors`、`mfeat-pixel` 为主，目标是让样本数尽量固定在 2000 行附近。
- 后续结果统一写入 `results/raw/{model}/{experiment_axis}/{scale_group}/{dataset_name}/{run_id}/`，再合并到 `results/final/`。
- 结果分析口径已统一为：`accuracy`、`balanced_accuracy`、`macro_f1`、置信度指标、时间/峰值内存指标、实验状态和扩展性派生指标。
- 单次实验能计算出的数字必须当场写入 `metrics.csv`，包括置信度指标、`wall_time_seconds`、`peak_memory_mb`、`training_throughput`。
- `peak_memory_mb` 计划使用 `psutil` 后台采样当前 Python 进程 RSS 峰值。
- `memory_before_mb` 和 `memory_after_mb` 不再作为正式结果字段。
- 旧训练结果已删除；当前 `results/` 只保留 `raw/`、`final/`、`figures/` 目录和占位文件。

## 已验证命令

本地资源检查命令：

```powershell
cd "D:\Codex项目\人工智能2大作业"
python scripts\check_local_assets.py
```

当前输出结论：

```text
OVERALL=OK
```

## 下一步

1. 和队友确认是否采用同样的目录命名规则。
2. 按 `docs/dataset_selection_criteria.md` 验证样本数扩展性的轻量/稳健方案和特征数扩展性候选数据集。
3. 按 `docs/measurement_plan.md` 修改实验脚本结果字段、峰值内存采样和输出路径。
4. 用小数据集分别验证 TabPFN、TabICL、LightGBM、XGBoost 的单模型运行脚本。
5. 统一结果 CSV 字段后，再进入完整主实验。
